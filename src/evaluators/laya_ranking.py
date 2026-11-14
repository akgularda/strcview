"""Update Laya's own relevance scorer from reviewer decisions."""

from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, Lock

from src.evaluators.base import EvaluationCriteria
from src.models.article import Article

CHECKPOINT = "convaiinnovations/laya"
CHECKPOINT_REVISION = "85cc74ddae6109077ebe2853a9b3077fb6606a78"
RANKING_VERSION = f"{CHECKPOINT}@{CHECKPOINT_REVISION}:1"


@dataclass(frozen=True)
class LayaRanking:
    scores: dict[str, float]
    base_scores: dict[str, float]


def ranking_signature(criteria: EvaluationCriteria, labels: list[Article], pending: list[Article]) -> str:
    digest = sha256(repr((RANKING_VERSION, criteria)).encode())
    for article in (*labels, *pending):
        decision = article.decision if article.manual_override else ""
        digest.update(repr((article.id, article.title, article.abstract, decision)).encode())
    return digest.hexdigest()


class LayaRanker:
    """Keep the encoder fixed and update the output layer of Laya's scorer."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self._agent = None
        self._question = None
        self._features = {}
        self._lock = Lock()
        self._cache_path = cache_path

    def _read_cache(self, question_key: str) -> None:
        if self._cache_path is None:
            return
        import numpy as np
        import torch

        try:
            with np.load(self._cache_path, allow_pickle=False) as cache:
                keys, vectors = cache["keys"], cache["vectors"]
                width = self._agent.model.scorer[-1].in_features
                if (cache["question"].item() == question_key and vectors.shape == (len(keys), width)
                        and vectors.dtype == np.float32 and np.isfinite(vectors).all()):
                    self._features.update(zip(keys.tolist(), torch.from_numpy(vectors)))
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _write_cache(self, question_key: str) -> None:
        if self._cache_path is None:
            return
        import numpy as np
        import torch

        temporary = None
        try:
            with NamedTemporaryFile(dir=self._cache_path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                vectors = torch.stack(list(self._features.values())).numpy()
                np.savez(stream, question=question_key, keys=list(self._features), vectors=vectors)
            temporary.replace(self._cache_path)
        except OSError:
            pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _questions(criteria: EvaluationCriteria) -> dict:
        instructions = f"Should this study be included in this review? Research question: {criteria.research_question}"
        if criteria.inclusion_criteria:
            instructions += f" Include: {criteria.inclusion_criteria}"
        if criteria.exclusion_criteria:
            instructions += f" Exclude: {criteria.exclusion_criteria}"
        return {"relevance": {"type": "noul", "instructions": instructions}}

    def _feature_batch(self, articles: list[Article], questions: dict):
        import torch

        states = [{"title": article.title.strip(), "abstract": article.abstract.strip() or "(No abstract available)"}
                  for article in articles]
        parts = []

        def capture(_layer, inputs):
            markers = inputs[0].detach().float().cpu()
            parts.append((markers[:, 1, :] - markers[:, 0, :]).clone())

        layer = self._agent.model.scorer[-1]
        with layer.register_forward_pre_hook(capture):
            results = self._agent.predict_batch(states, questions, batch_size=6)
        features = torch.cat(parts)
        if len(features) != len(articles) or any("relevance" not in result["answers"] for result in results):
            raise RuntimeError("Laya did not score every publication in the batch.")
        return features

    def rank(
        self,
        labels: list[Article],
        pending: list[Article],
        criteria: EvaluationCriteria,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Event | None = None,
    ) -> LayaRanking:
        if {article.decision for article in labels} != {"Include", "Exclude"} or not pending:
            return LayaRanking({}, {})
        import torch
        from laya import load
        from laya.common import QTYPES, temp_bucket

        with self._lock:
            if cancelled and cancelled.is_set():
                raise CancelledError()
            if self._agent is None:
                self._agent = load(CHECKPOINT, revision=CHECKPOINT_REVISION)
            questions = self._questions(criteria)
            question_key = sha256(repr((RANKING_VERSION, questions)).encode()).hexdigest()
            if self._question != questions:
                self._features.clear()
                self._question = questions
                self._read_cache(question_key)

            articles = [*labels, *pending]
            keys = [sha256(repr((article.id, article.title, article.abstract)).encode()).hexdigest() for article in articles]
            live_keys = set(keys)
            self._features = {key: value for key, value in self._features.items() if key in live_keys}
            missing = [(key, article) for key, article in zip(keys, articles) if key not in self._features]
            for start in range(0, len(missing), 24):
                if cancelled and cancelled.is_set():
                    raise CancelledError()
                chunk = missing[start:start + 24]
                vectors = self._feature_batch([article for _, article in chunk], questions)
                self._features.update((key, vector) for (key, _), vector in zip(chunk, vectors))
                if start % 120 == 96 or start + len(chunk) == len(missing):
                    self._write_cache(question_key)
                if progress:
                    progress(len(self._features), len(articles))

            features = torch.stack([self._features[key] for key in keys])
            native_weight = self._agent.model.scorer[-1].weight.detach().float().cpu().flatten()
            qtype = QTYPES["noul"]
            temperature = self._agent.temperature_by_options.get(temp_bucket(qtype, 2), self._agent.temperature[qtype])
            train = features[:len(labels)]
            candidates = features[len(labels):]
            targets = torch.tensor([float(article.decision == "Include") for article in labels])
            weights = torch.where(targets == 1, 0.5 / targets.sum(), 0.5 / (1 - targets).sum())

            # Fit the existing final linear layer with a penalty for moving away from
            # the checkpoint. The common bias cancels between Laya's yes/no logits.
            with torch.enable_grad():
                adapted = native_weight.clone().requires_grad_()
                optimizer = torch.optim.LBFGS([adapted], max_iter=20, line_search_fn="strong_wolfe")

                def loss():
                    optimizer.zero_grad()
                    logits = train.mv(adapted) / temperature
                    fit = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")
                    objective = fit + 5.0 * (adapted - native_weight).square().sum()
                    objective.backward()
                    return objective

                optimizer.step(loss)

            with torch.no_grad():
                base = torch.sigmoid(candidates.mv(native_weight) / temperature).tolist()
                ranked = torch.sigmoid(candidates.mv(adapted) / temperature).tolist()
            return LayaRanking(dict(zip((a.id for a in pending), ranked)), dict(zip((a.id for a in pending), base)))

"""Initial sampling and ranking for human screening."""

from collections import defaultdict, deque
from dataclasses import replace
from hashlib import sha256

from src.evaluators.active_learning import ActiveLearningRanker
from src.evaluators.base import EvaluationCriteria
from src.evaluators.laya_evaluator import LayaEvaluator
from src.models.article import Article

SEED_COUNT = 8
CANDIDATE_COUNT = 32
BINARY_DECISIONS = {"Include", "Exclude"}


def pending_records(articles: list[Article]) -> list[Article]:
    return [article for article in articles if not article.manual_override]


def source_sample(articles: list[Article], limit: int) -> list[Article]:
    groups = defaultdict(list)
    for article in articles:
        groups[article.source_db or "Unknown"].append(article)
    queues = deque(
        deque(sorted(group, key=lambda article: sha256(article.id.encode()).digest()))
        for group in groups.values()
    )
    sample = []
    while queues and len(sample) < limit:
        queue = queues.popleft()
        sample.append(queue.popleft())
        if queue:
            queues.append(queue)
    return sample


def choose_seeds(scores: list[tuple[str, float]], threshold: float) -> list[str]:
    ordered = sorted(scores, key=lambda pair: (pair[1], pair[0]))
    if len(ordered) <= SEED_COUNT:
        return [record_id for record_id, _ in scores]
    selected = []
    for index in range(3):
        selected.extend((ordered.pop()[0], ordered.pop(0)[0]))
        if index < 2:
            nearest = min(range(len(ordered)), key=lambda i: abs(ordered[i][1] - threshold))
            selected.append(ordered.pop(nearest)[0])
    return selected


def laya_seeds(articles: list[Article], criteria: EvaluationCriteria) -> list[str]:
    if len(articles) <= SEED_COUNT:
        return [article.id for article in articles]
    evaluator = LayaEvaluator()
    scores = []
    for article in articles:
        result = evaluator.evaluate_article(article, criteria)
        if result.topic_score is None:
            raise RuntimeError("Laya could not score the initial records.")
        scores.append((article.id, result.topic_score))
    return choose_seeds(scores, criteria.threshold)


def human_labels(articles: list[Article]) -> list[Article]:
    return [article for article in articles if article.manual_override and article.decision in BINARY_DECISIONS]


def rank_pending(labels: list[Article], pending: list[Article]) -> dict[str, float]:
    if {article.decision for article in labels} != BINARY_DECISIONS:
        return {}
    snapshots = [replace(article, priority_score=None) for article in pending]
    ranked = ActiveLearningRanker().fit_and_rerank(labels, snapshots, preserve_model_scores=True)
    return {article.id: article.priority_score for article in ranked if article.priority_score is not None}

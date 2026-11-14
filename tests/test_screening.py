import json
from collections import Counter

import pytest

from src.evaluators.base import EvaluationCriteria, EvaluationResult
from src.models.article import Article
from src.screening import choose_seeds, human_labels, laya_seeds, pending_records, rank_pending, source_sample
from src.workspace import Workspace, save_workspace


def test_seed_selection_covers_high_low_and_near_threshold():
    scores = [(str(index), index / 20) for index in range(20)]
    assert choose_seeds(scores, 0.51) == ["19", "0", "10", "18", "1", "11", "17", "2"]
    assert choose_seeds(scores[:3], 0.5) == ["0", "1", "2"]


def test_laya_is_only_used_to_select_from_a_balanced_candidate_pool(monkeypatch):
    articles = [Article(id=str(i), title=f"Publication {i}", source_db="WoS" if i < 90 else "Scopus") for i in range(100)]
    sampled = source_sample(articles, 32)
    assert Counter(a.source_db for a in sampled) == {"WoS": 22, "Scopus": 10}

    class Evaluator:
        def evaluate_article(self, article, criteria):
            return EvaluationResult("Include", 1, topic_score=int(article.id) / 100)

    monkeypatch.setattr("src.screening.LayaEvaluator", Evaluator)
    ids = laya_seeds(sampled, EvaluationCriteria("Question"))
    assert len(set(ids)) == 8
    assert all(a.decision == "Unscreened" for a in articles)

    monkeypatch.setattr(Evaluator, "evaluate_article", lambda *args: EvaluationResult("Exclude", 0))
    with pytest.raises(RuntimeError, match="Laya"):
        laya_seeds(sampled, EvaluationCriteria("Question"))
    assert laya_seeds(sampled[:3], EvaluationCriteria("Question")) == [a.id for a in sampled[:3]]


def test_active_ranking_requires_two_human_classes_and_preserves_proposals():
    workspace = Workspace(articles=[
        Article(title="Earthquake seismic wave detection"), Article(title="Childhood dental implants"),
        Article(title="Seismic wave earthquake detection", model_decision="Exclude", relevance_score=0.12),
        Article(title="Children dental health and implants"), Article(title="Review later"),
    ])
    relevant, irrelevant, match, other, deferred = workspace.articles
    workspace.review(relevant, "Include", "")
    workspace.review(deferred, "Borderline", "Revisit")
    assert rank_pending(human_labels(workspace.articles), pending_records(workspace.articles)) == {}

    workspace.review(irrelevant, "Exclude", "")
    scores = rank_pending(human_labels(workspace.articles), pending_records(workspace.articles))
    assert scores[match.id] > scores[other.id]
    assert deferred.id not in scores
    assert match.model_decision == "Exclude" and match.relevance_score == 0.12


def test_seed_order_survives_reopening_and_older_backups_load(tmp_path):
    workspace = Workspace()
    workspace.add_articles([Article(title="One"), Article(title="Two")], "demo", "demo.xlsx")
    workspace.seed_ids = [article.id for article in reversed(workspace.articles)]
    workspace.seed_method = "Laya"
    workspace.review(workspace.articles[-1], "Exclude", "First decision")
    path = tmp_path / "data" / "workspace.json"
    save_workspace(workspace, path)
    restored = Workspace.from_json(path.read_bytes())
    assert restored.seed_ids == workspace.seed_ids
    assert restored.articles[-1].reviewer_notes == "First decision"
    old = json.loads(path.read_bytes())
    del old["seed_ids"], old["seed_method"]
    assert Workspace.from_json(json.dumps(old).encode()).seed_ids == []

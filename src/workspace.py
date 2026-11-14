"""Session-independent operations for a literature review workspace."""

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.models.article import Article
from src.parsers.deduplicator import apply_duplicate_resolutions, find_duplicate_candidates

DECISIONS = ("Include", "Borderline", "Exclude")


@dataclass
class Workspace:
    articles: list[Article] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    resolved_pairs: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    audit: list[dict] = field(default_factory=list)
    research_question: str = ""
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    model: str = "Laya"
    threshold: float = 0.5
    seed_ids: list[str] = field(default_factory=list)
    seed_method: str = ""
    laya_ranking_signature: str = ""

    @property
    def dedup_stats(self) -> dict:
        total = sum(self.source_counts.values())
        return {
            "total_imported": total,
            "duplicates_removed": total - len(self.articles),
            "wos_imported": self.source_counts.get("WoS", 0),
            "scopus_imported": self.source_counts.get("Scopus", 0),
        }

    def log(self, action: str, **details) -> None:
        self.audit.append({"time": datetime.now(timezone.utc).isoformat(), "action": action, **details})

    def add_articles(self, articles: list[Article], import_key: str, filename: str) -> None:
        if not articles or import_key in self.imports:
            return
        for index, article in enumerate(articles, start=1):
            article.id = f"{import_key[:24]}-{index}"
            article.source_file = filename
        self.articles.extend(articles)
        self.source_counts = dict(Counter(self.source_counts) + Counter(a.source_db for a in articles))
        self.imports.add(import_key)
        self.log("Import", file=filename, records=len(articles))

    def duplicate_candidates(self) -> list[dict]:
        return [c for c in find_duplicate_candidates(self.articles) if c["candidate_id"] not in self.resolved_pairs]

    def resolve_duplicates(self, candidates: list[dict], action: str) -> int:
        decisions = {c["candidate_id"]: action for c in candidates}
        self.articles, stats = apply_duplicate_resolutions(self.articles, decisions, candidates)
        if stats["total_removed"]:
            remaining = {article.id for article in self.articles}
            self.seed_ids = [record_id for record_id in self.seed_ids if record_id in remaining]
        self.resolved_pairs.update(decisions)
        self.log("Duplicate review", resolution=action, pairs=list(decisions), removed=stats["total_removed"])
        return stats["total_removed"]

    def review(self, article: Article, decision: str, notes: str, *, consensus: bool = False) -> None:
        previous = article.decision
        if consensus:
            article.consensus_decision = decision
        else:
            article.reviewer_decision = decision
            article.consensus_decision = ""
        article.decision = decision
        article.reviewer_notes = notes
        article.manual_override = True
        self.log("Consensus" if consensus else "Review", id=article.id, previous=previous, decision=decision, notes=notes)

    def exclusion_streak(self) -> int:
        count = 0
        seen = set()
        current_ids = {article.id for article in self.articles}
        for event in reversed(self.audit):
            record_id = event.get("id")
            if event["action"] != "Review" or record_id not in current_ids or record_id in seen:
                continue
            if event.get("decision") != "Exclude":
                break
            seen.add(record_id)
            count += 1
        return count

    def to_json(self) -> str:
        return json.dumps({
            **vars(self), "version": 1, "articles": [a.to_dict() for a in self.articles],
            "resolved_pairs": sorted(self.resolved_pairs), "imports": sorted(self.imports),
        }, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, content: bytes) -> "Workspace":
        data = json.loads(content)
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Select a version 1 sTRCview workspace file.")
        articles = data.get("articles")
        counts = data.get("source_counts")
        audit = data.get("audit", [])
        if not isinstance(articles, list) or not isinstance(counts, dict) or not isinstance(audit, list):
            raise ValueError("The workspace file is incomplete.")
        if any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) < len(articles):
            raise ValueError("The workspace source counts are invalid.")
        if any(not isinstance(event, dict) or not isinstance(event.get("action"), str)
               or not isinstance(event.get("time"), str) or not isinstance(event.get("id", ""), str)
               for event in audit):
            raise ValueError("The workspace activity history is invalid.")
        restored = []
        seen_ids = set()
        defaults = vars(Article())
        for record in articles:
            if not isinstance(record, dict) or not record.get("id") or not record.get("title"):
                raise ValueError("An article is missing its ID or title.")
            article = Article.from_dict(record)
            for name, default in defaults.items():
                value = getattr(article, name)
                if isinstance(default, str) and not isinstance(value, str):
                    raise ValueError(f"Invalid article field: {name}")
            if article.id in seen_ids or article.decision not in (*DECISIONS, "Unscreened"):
                raise ValueError("The workspace contains a repeated ID or invalid decision.")
            if not isinstance(article.manual_override, bool) or type(article.citations) is not int:
                raise ValueError("The workspace contains invalid article metadata.")
            if article.year is not None and type(article.year) is not int:
                raise ValueError("The workspace contains an invalid publication year.")
            scores = [article.relevance_score, article.confidence]
            if article.priority_score is not None:
                scores.append(article.priority_score)
            for score in scores:
                if type(score) not in (int, float) or not 0 <= score <= 1:
                    raise ValueError("The workspace contains an invalid score.")
            seen_ids.add(article.id)
            restored.append(article)
        workspace = cls(articles=restored, source_counts=counts, audit=audit)
        for name in ("research_question", "inclusion_criteria", "exclusion_criteria", "model", "laya_ranking_signature"):
            value = data.get(name, getattr(workspace, name))
            if not isinstance(value, str):
                raise ValueError(f"Invalid workspace field: {name}")
            setattr(workspace, name, value)
        for name in ("resolved_pairs", "imports"):
            values = data.get(name, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"Invalid workspace field: {name}")
            setattr(workspace, name, set(values))
        threshold = data.get("threshold", 0.5)
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("The workspace threshold is invalid.")
        workspace.threshold = threshold
        seed_ids = data.get("seed_ids", [])
        if not isinstance(seed_ids, list) or any(not isinstance(value, str) for value in seed_ids):
            raise ValueError("The initial screening queue is invalid.")
        seed_method = data.get("seed_method", "")
        if seed_method not in ("", "Laya", "Sources"):
            raise ValueError("The initial screening method is invalid.")
        workspace.seed_ids = list(dict.fromkeys(value for value in seed_ids if value in seen_ids))
        workspace.seed_method = seed_method
        return workspace


def desktop_workspace_path() -> Path | None:
    root = os.environ.get("STRCVIEW_DATA_DIR")
    return Path(root) / "workspace.json" if root else None


def save_workspace(workspace: Workspace, path: Path) -> None:
    payload = workspace.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

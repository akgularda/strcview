from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import uuid

@dataclass
class Article:
    """Unified representation of a research article from Web of Science or Scopus."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    abstract: str = ""
    authors: str = ""
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    keywords: str = ""
    source_db: str = ""  # 'WoS', 'Scopus', or 'WoS & Scopus'
    raw_id: str = ""     # WoS UT or Scopus EID
    link: str = ""
    citations: int = 0
    affiliations: str = ""
    references: str = ""  # Cited references (CR) for bibliometrics / VOSviewer
    source_file: str = ""
    
    # Screening and evaluation fields
    decision: str = "Unscreened"  # 'Include', 'Exclude', 'Borderline', 'Unscreened'
    relevance_score: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0       # 0.0 to 1.0
    reasoning: str = ""           # Detailed explanation or model probabilities
    model_used: str = ""          # e.g., 'Laya', 'LLaMA', 'Rule-Based'
    manual_override: bool = False # True if researcher manually modified the decision
    reviewer_notes: str = ""      # Notes added by the researcher
    model_decision: str = ""
    reviewer_decision: str = ""
    second_reviewer_decision: str = ""
    consensus_decision: str = ""
    priority_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Article":
        # Filter keys that match dataclass fields
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

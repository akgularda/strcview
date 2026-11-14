from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from src.models.article import Article

@dataclass
class EvaluationCriteria:
    research_question: str
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    threshold: float = 0.50  # Score threshold above which an article is deemed 'Include'
    borderline_window: float = 0.15 # e.g. score between threshold - 0.15 and threshold + 0.15 can be 'Borderline'

@dataclass
class EvaluationResult:
    decision: str  # 'Include', 'Exclude', 'Borderline'
    relevance_score: float  # 0.0 to 1.0
    confidence: float = 0.0 # 0.0 to 1.0
    reasoning: str = ""
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    topic_score: Optional[float] = None

class BaseEvaluator(ABC):
    """Abstract base class for all screening evaluators."""
    
    @abstractmethod
    def evaluate_article(self, article: Article, criteria: EvaluationCriteria) -> EvaluationResult:
        """Evaluates a single article's title and abstract against criteria."""
        pass
        
    def evaluate_batch(
        self,
        articles: List[Article],
        criteria: EvaluationCriteria,
        progress_callback: Optional[Callable[[int, int, Article], None]] = None
    ) -> List[Article]:
        """
        Evaluates a batch of articles, updating each article in-place.
        Returns the updated list of articles.
        """
        total = len(articles)
        for i, article in enumerate(articles):
            result = self.evaluate_article(article, criteria)
            self.apply_result(article, result)
            
            if progress_callback:
                progress_callback(i + 1, total, article)
                
        return articles

    def apply_result(self, article: Article, result: EvaluationResult) -> None:
        article.model_decision = result.decision
        if not article.manual_override:
            article.decision = result.decision
        article.relevance_score = result.relevance_score
        article.confidence = result.confidence
        article.reasoning = result.reasoning
        article.model_used = result.model_name or self.__class__.__name__

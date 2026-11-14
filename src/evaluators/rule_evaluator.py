import re
from typing import List, Set
from src.models.article import Article
from src.evaluators.base import BaseEvaluator, EvaluationCriteria, EvaluationResult

class RuleEvaluator(BaseEvaluator):
    """
    Fast rule-based and keyword-matching evaluator.
    Can be used instantly with zero download for rapid baseline tests.
    """
    def __init__(self, stop_words: Set[str] = None):
        self.stop_words = stop_words or {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "are", "was", "were",
            "what", "how", "why", "when", "where", "which", "who", "whom",
            "this", "that", "these", "those", "their", "our", "your"
        }

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in self.stop_words]

    def evaluate_article(self, article: Article, criteria: EvaluationCriteria) -> EvaluationResult:
        full_text = f"{article.title} {article.title} {article.abstract} {article.keywords}".lower()
        
        # Tokenize inputs
        rq_terms = self._tokenize(criteria.research_question)
        incl_terms = self._tokenize(criteria.inclusion_criteria)
        excl_terms = self._tokenize(criteria.exclusion_criteria)
        
        # Match RQ terms
        rq_matches = [w for w in rq_terms if w in full_text]
        rq_coverage = len(rq_matches) / max(len(rq_terms), 1)
        
        # Match Inclusion terms
        incl_matches = [w for w in incl_terms if w in full_text] if incl_terms else []
        incl_coverage = len(incl_matches) / max(len(incl_terms), 1) if incl_terms else rq_coverage
        
        # Match Exclusion terms
        excl_matches = [w for w in excl_terms if w in full_text] if excl_terms else []
        excl_penalty = (len(excl_matches) / max(len(excl_terms), 1)) if excl_terms else 0.0
        
        # Composite score
        raw_score = (rq_coverage * 0.5) + (incl_coverage * 0.5) - (excl_penalty * 0.4)
        relevance_score = min(max(raw_score, 0.0), 1.0)
        
        if excl_matches and excl_penalty >= 0.5:
            decision = "Exclude"
            reasoning = f"Matched exclusion terms: {', '.join(dict.fromkeys(excl_matches))}."
        elif relevance_score >= criteria.threshold + criteria.borderline_window:
            decision = "Include"
            reasoning = f"High keyword match with RQ & inclusion criteria ({len(rq_matches)} terms matched)."
        elif relevance_score < criteria.threshold - criteria.borderline_window:
            decision = "Exclude"
            reasoning = f"Low relevance to research question ({len(rq_matches)}/{len(rq_terms)} terms matched)."
        else:
            decision = "Borderline"
            reasoning = f"Moderate keyword match ({len(rq_matches)} terms matched); requires manual review."
            
        return EvaluationResult(
            decision=decision,
            relevance_score=round(relevance_score, 3),
            confidence=0.70,
            reasoning=reasoning,
            probabilities={"include": relevance_score, "exclude": 1.0 - relevance_score},
            model_name="Keyword Baseline"
        )

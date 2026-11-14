import os
import math
import re
import logging
from typing import Dict, Any, Optional, List, Set
from src.models.article import Article
from src.evaluators.base import BaseEvaluator, EvaluationCriteria, EvaluationResult

logger = logging.getLogger(__name__)

class LayaEvaluator(BaseEvaluator):
    """
    Enhanced Evaluator powered by Laya (non-autoregressive ModernBERT decision engine)
    combined with an explicit criteria verification gate.
    Evaluates research question and inclusion/exclusion criteria over Title and Abstract.
    """
    def __init__(self, preload: bool = False, model_override: Optional[str] = None):
        self.router = None
        self.model_override = model_override
        self._init_error = None
        
        try:
            from laya import Router
            self.router = Router(preload=preload)
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Laya Router could not be initialized: {e}")

    def evaluate_article(self, article: Article, criteria: EvaluationCriteria) -> EvaluationResult:
        if self.router is None:
            return self._heuristic_fallback(article, criteria, reason=f"Laya not initialized: {self._init_error}")
            
        title_text = article.title.strip()
        abstract_text = article.abstract.strip() if article.abstract else "(No abstract available)"
        full_text = f"{title_text} {abstract_text} {article.keywords}".lower()
        
        # 1. Check explicit inclusion & exclusion terms from criteria
        incl_terms = self._extract_key_terms(criteria.inclusion_criteria)
        excl_terms = self._extract_key_terms(criteria.exclusion_criteria)
        
        has_incl_terms = any(term in full_text for term in incl_terms) if incl_terms else True
        matched_excl = [term for term in excl_terms if term in full_text] if excl_terms else []
        
        # 2. Query Laya ModernBERT Router
        state = {
            "title": title_text,
            "abstract": abstract_text
        }
        
        incl_desc = criteria.inclusion_criteria if criteria.inclusion_criteria else "Relevant to the research question"
        excl_desc = criteria.exclusion_criteria if criteria.exclusion_criteria else "Irrelevant or out of scope"
        
        questions = {
            "topic_relevance": {
                "type": "noul",
                "instructions": f"Is this research paper directly focused on: '{criteria.research_question}'?"
            },
            "decision": {
                "type": "choice",
                "instructions": f"Evaluate whether this paper meets: '{criteria.research_question}'",
                "criteria": {
                    "include": f"Matches criteria: {incl_desc}",
                    "exclude": f"Matches exclusion criteria: {excl_desc}",
                    "borderline": "Marginal, partial match, or insufficient details"
                }
            }
        }
        
        try:
            kwargs = {}
            if self.model_override:
                kwargs["model"] = self.model_override
                
            pred = self.router.predict(state, questions, **kwargs)
            answers = pred.get("answers", {})
            
            raw_score = float(answers["topic_relevance"]["noul"])
            if not math.isfinite(raw_score) or not 0 <= raw_score <= 1:
                raise ValueError("Invalid topic relevance score")
            dec_info = answers.get("decision", {})
            probs = dec_info.get("probabilities", {})
            confidence = float(dec_info.get("confidence", 0.85))
            
            # 3. Hybrid Gating Calibration:
            # If paper matches hard exclusion criteria and lacks inclusion terms:
            if matched_excl and not has_incl_terms:
                decision = "Exclude"
                score = round(min(raw_score * 0.25, 0.20), 3)
                reasoning = f"Excluded: Matched exclusion criteria ({', '.join(matched_excl[:3])}) and lacks required inclusion methodology."
            # If paper lacks inclusion criteria terms despite high topic match:
            elif not has_incl_terms and incl_terms:
                score = round(raw_score * 0.45, 3)
                if score >= criteria.threshold:
                    decision = "Borderline"
                else:
                    decision = "Exclude"
                reasoning = f"Low inclusion match: General topic match ({raw_score:.2f}) but lacks explicit required methodology terms."
            # If paper matches both topic and inclusion criteria:
            else:
                score = round(raw_score, 3)
                if score >= criteria.threshold + criteria.borderline_window:
                    decision = "Include"
                elif score < criteria.threshold - criteria.borderline_window:
                    decision = "Exclude"
                else:
                    decision = "Borderline"
                reasoning = f"Laya Score: {score:.2f} (Include prob: {probs.get('include', 0.0):.2f}, Exclude prob: {probs.get('exclude', 0.0):.2f})."
                
            return EvaluationResult(
                decision=decision,
                relevance_score=score,
                confidence=round(confidence, 3),
                reasoning=reasoning,
                probabilities=probs,
                model_name="Laya (ModernBERT Hybrid)",
                topic_score=raw_score,
            )
            
        except Exception as e:
            logger.error(f"Error during Laya evaluation: {e}")
            return self._heuristic_fallback(article, criteria, reason=f"Laya error: {str(e)}")

    def _extract_key_terms(self, text: str) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r'[^\w\s-]', ' ', text.lower())
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 2]
        stopwords = {"and", "for", "the", "with", "studies", "using", "such", "from", "that", "this", "methods"}
        return [t for t in tokens if t not in stopwords]

    def _heuristic_fallback(self, article: Article, criteria: EvaluationCriteria, reason: str) -> EvaluationResult:
        full_text = f"{article.title} {article.abstract}".lower()
        rq_words = [w.lower() for w in criteria.research_question.split() if len(w) > 3]
        matches = sum(1 for w in rq_words if w in full_text)
        score = matches / max(len(rq_words), 1)
        score = min(max(score, 0.0), 1.0)
        
        decision = "Include" if score >= criteria.threshold else ("Exclude" if score < criteria.threshold - 0.2 else "Borderline")
        return EvaluationResult(
            decision=decision,
            relevance_score=round(score, 3),
            confidence=0.5,
            reasoning=f"Rule fallback ({reason}). Match score: {score:.2f}",
            probabilities={"include": score, "exclude": 1.0 - score},
            model_name="Laya (Fallback)"
        )

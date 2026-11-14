import json
import logging
import math
import re
from typing import Optional, Dict, Any
import requests
from src.models.article import Article
from src.evaluators.base import BaseEvaluator, EvaluationCriteria, EvaluationResult

logger = logging.getLogger(__name__)

class LlamaEvaluator(BaseEvaluator):
    """
    Evaluates articles using a local LLaMA instance via Ollama API
    or an OpenAI-compatible endpoint.
    Produces structured decisions and qualitative explanations.
    """
    def __init__(
        self,
        model_name: str = "llama3.2",
        endpoint_url: str = "http://localhost:11434/api/chat",
        timeout: int = 30
    ):
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Checks if Ollama service is reachable."""
        try:
            base_url = re.sub(r'/api/.*$', '', self.endpoint_url)
            r = requests.get(f"{base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def evaluate_article(self, article: Article, criteria: EvaluationCriteria) -> EvaluationResult:
        prompt = (
            f"You are a scientific research assistant conducting a systematic literature review screening.\n"
            f"Evaluate the following paper against the Research Question and Criteria.\n\n"
            f"RESEARCH QUESTION: {criteria.research_question}\n"
            f"INCLUSION CRITERIA: {criteria.inclusion_criteria or 'Directly addresses research question'}\n"
            f"EXCLUSION CRITERIA: {criteria.exclusion_criteria or 'Out of scope or irrelevant'}\n\n"
            f"PAPER TITLE: {article.title}\n"
            f"PAPER ABSTRACT: {article.abstract or '(No abstract available)'}\n\n"
            f"Provide your screening evaluation in valid JSON with these exact keys:\n"
            f'{{"decision": "Include" | "Exclude" | "Borderline", "relevance_score": <float between 0.0 and 1.0>, "confidence": <float between 0.0 and 1.0>, "reasoning": "<short justification in 2-3 sentences>"}}'
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a precise academic screener. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        try:
            response = requests.post(self.endpoint_url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")
                
                # Parse JSON response
                parsed = self._extract_json(content)
                if isinstance(parsed, dict) and parsed:
                    raw_dec = str(parsed.get("decision", "Borderline")).capitalize()
                    if raw_dec not in ["Include", "Exclude", "Borderline"]:
                        raw_dec = "Borderline"
                        
                    score = float(parsed.get("relevance_score", 0.5))
                    conf = float(parsed.get("confidence", 0.8))
                    if not (math.isfinite(score) and math.isfinite(conf)):
                        return self._connection_error_result("Invalid scores returned by Ollama")
                    reason = str(parsed.get("reasoning", "Evaluated by LLaMA."))
                    
                    return EvaluationResult(
                        decision=raw_dec,
                        relevance_score=round(min(1.0, max(0.0, score)), 3),
                        confidence=round(min(1.0, max(0.0, conf)), 3),
                        reasoning=reason,
                        model_name=f"LLaMA ({self.model_name})"
                    )

            detail = f"HTTP {response.status_code}" if response.status_code != 200 else "Invalid response from Ollama"
            return self._connection_error_result(detail)
            
        except Exception as e:
            return self._connection_error_result(str(e))

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find json block
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return None

    def _connection_error_result(self, error_msg: str) -> EvaluationResult:
        return EvaluationResult(
            decision="Borderline",
            relevance_score=0.5,
            confidence=0.0,
            reasoning=f"LLaMA connection error: {error_msg}. Please ensure Ollama is running.",
            model_name="LLaMA (Unavailable)"
        )

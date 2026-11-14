import pytest
from src.models.article import Article
from src.evaluators.base import EvaluationCriteria
from src.evaluators.rule_evaluator import RuleEvaluator

def test_rule_evaluator():
    evaluator = RuleEvaluator()
    criteria = EvaluationCriteria(
        research_question="How are deep learning models applied to earthquake early warning systems?",
        inclusion_criteria="Studies using convolutional neural networks or transformers for seismic wave detection",
        exclusion_criteria="Studies on volcanic eruptions or financial markets",
        threshold=0.40
    )
    
    art_relevant = Article(
        title="Deep learning with convolutional neural networks for seismic P-wave detection in earthquake early warning",
        abstract="We demonstrate that convolutional neural networks predict earthquake early warning signals rapidly."
    )
    res_rel = evaluator.evaluate_article(art_relevant, criteria)
    assert res_rel.decision in ["Include", "Borderline"]
    assert res_rel.relevance_score > 0.40
    
    art_irrelevant = Article(
        title="Deep learning for stock price prediction in cryptocurrency markets",
        abstract="We use recurrent networks to trade Bitcoin and financial assets."
    )
    res_irrel = evaluator.evaluate_article(art_irrelevant, criteria)
    assert res_irrel.decision == "Exclude"

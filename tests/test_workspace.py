import io

from openpyxl import load_workbook

from src.analytics.bibliometrics import compute_bibliometric_summary
from src.analytics.exporter import export_to_excel
from src.analytics.prisma_diagram import generate_prisma_2020_svg
from src.evaluators.base import EvaluationCriteria
from src.evaluators.rule_evaluator import RuleEvaluator
from src.models.article import Article
from src.ui.ingestion import identify_format
from src.ui.review import match_reviewer_file
from src.workspace import Workspace


def test_workspace_roundtrip_keeps_source_counts_and_rejected_duplicate_pairs():
    workspace = Workspace(research_question="Does the intervention work?")
    workspace.add_articles([Article(title="İklim değişikliği", doi="10.1234/sample", source_db="WoS")], "wos-1", "wos.txt")
    workspace.add_articles([Article(title="İklim değişikliği", doi="https://doi.org/10.1234/sample", source_db="Custom Excel")], "excel-1", "data.xlsx")
    candidates = workspace.duplicate_candidates()
    assert len(candidates) == 1
    assert workspace.resolve_duplicates(candidates, "keep_both") == 0

    restored = Workspace.from_json(workspace.to_json().encode())
    restored.add_articles([Article(title="Repeat", source_db="WoS")], "wos-1", "renamed.txt")
    assert restored.duplicate_candidates() == []
    assert len(restored.articles) == 2
    assert restored.dedup_stats["duplicates_removed"] == 0
    summary = compute_bibliometric_summary(restored.articles, restored.dedup_stats)["prisma"]
    assert summary["identified_total"] == 2
    assert summary["identified_other"] == 1
    assert summary["records_screened"] == 0
    assert "n = 0" in generate_prisma_2020_svg(summary)


def test_manual_and_consensus_decisions_survive_model_rerun_and_export():
    workspace = Workspace()
    article = Article(title="=1+1", abstract="A study of summer heat", source_db="Custom Excel")
    workspace.add_articles([article], "sheet-1", "data.xlsx")
    workspace.review(article, "Include", "First reviewer note")
    workspace.review(article, "Borderline", "Consensus note", consensus=True)
    RuleEvaluator().evaluate_batch([article], EvaluationCriteria("summer heat", exclusion_criteria="summer heat"))
    assert article.decision == "Borderline"
    assert article.reviewer_decision == "Include"
    assert article.model_decision == "Exclude"

    book = load_workbook(io.BytesIO(export_to_excel(workspace.articles)))
    sheet = book["Publication Library"]
    headers = [cell.value for cell in sheet[1]]
    title = sheet.cell(2, headers.index("Title") + 1)
    assert (title.value, title.data_type) == ("=1+1", "s")
    assert sheet.cell(2, headers.index("Reviewer Decision") + 1).value == "Include"
    assert sheet.cell(2, headers.index("Consensus Decision") + 1).value == "Borderline"


def test_file_detection_and_reviewer_doi_alignment():
    assert identify_format(b"FN Clarivate Analytics Web of Science\nPT J\nTI Example") == "WoS"
    assert identify_format(b"Title,Abstract,EID\nExample,Text,2-s2.0-123") == "Scopus"
    assert identify_format(b"Title,Abstract,DOI\nExample,Text,10.1234/example") == "Map columns"
    workspace = Workspace(articles=[Article(id="row-1", title="Example", doi="10.1234/example")])
    result = match_reviewer_file(io.StringIO("DOI,Decision\nhttps://doi.org/10.1234/EXAMPLE,exclude"), workspace)
    assert result == {"row-1": "Exclude"}

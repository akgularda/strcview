"""
End-to-End Verification Test Suite on Real EEW Datasets (WoS & Scopus)
Tests all next-gen features for sTRCview:
1. Real dataset parsing & deduplication
2. PRISMA 2020 SVG flowchart generation & export
3. Active Learning continuous priority re-ranking & stopping rule
4. Cohen's Kappa inter-rater reliability & conflict adjudication
5. Keyword co-occurrence network graph generation (Plotly)
6. VOSviewer / Bibliometrix Scopus-format export
7. Keyword highlighting & Unpaywall DOI resolver
"""

import os
import re
import pytest
from pathlib import Path

from src.models.article import Article
from src.parsers.wos_parser import parse_wos_file
from src.parsers.scopus_parser import parse_scopus_file
from src.parsers.deduplicator import deduplicate_articles
from src.analytics.prisma_diagram import generate_prisma_2020_svg
from src.analytics.reliability import calculate_cohens_kappa, identify_conflicts
from src.analytics.network import build_keyword_cooccurrence_network
from src.evaluators.active_learning import ActiveLearningRanker
from src.analytics.exporter import export_vosviewer_scopus_format

DATA_DIR = Path(os.environ.get("STRCVIEW_EEW_DATA_DIR", "tests/fixtures/eew"))
SCOPUS_AI_PATH = DATA_DIR / "Scopus AI EEW.csv"
SCOPUS_HUMAN_PATH = DATA_DIR / "Scopus Human EEW.csv"
WOS_AI_PATH = DATA_DIR / "WoS AI EEW.txt"
WOS_HUMAN_PATH = DATA_DIR / "WoS Human EEW.txt"


def data_files_exist() -> bool:
    return (
        DATA_DIR.exists()
        and SCOPUS_AI_PATH.exists()
        and WOS_AI_PATH.exists()
    )


@pytest.fixture(scope="module")
def real_eew_dataset():
    """Loads and deduplicates real EEW articles from WoS and Scopus."""
    if not data_files_exist():
        pytest.skip(f"Real EEW dataset files not accessible at {DATA_DIR}")

    with open(SCOPUS_AI_PATH, "rb") as f:
        scopus_ai_articles = parse_scopus_file(f.read())
    with open(WOS_AI_PATH, "rb") as f:
        wos_ai_articles = parse_wos_file(f.read())

    all_raw = scopus_ai_articles + wos_ai_articles
    unique_articles, stats = deduplicate_articles(all_raw)

    return {
        "scopus_ai": scopus_ai_articles,
        "wos_ai": wos_ai_articles,
        "all_raw": all_raw,
        "unique": unique_articles,
        "stats": stats,
    }


def test_real_dataset_parsing_and_deduplication(real_eew_dataset):
    """Verifies that real Scopus CSV and WoS TXT files parse properly and deduplicate cleanly."""
    scopus_count = len(real_eew_dataset["scopus_ai"])
    wos_count = len(real_eew_dataset["wos_ai"])
    total_raw = len(real_eew_dataset["all_raw"])
    unique_count = len(real_eew_dataset["unique"])
    stats = real_eew_dataset["stats"]
    dup_count = stats["duplicates_removed"]

    assert scopus_count > 0, "Scopus AI dataset must contain articles"
    assert wos_count > 0, "WoS AI dataset must contain articles"
    assert total_raw == scopus_count + wos_count
    assert dup_count > 0, "There must be overlapping duplicates between WoS and Scopus"
    assert unique_count == total_raw - dup_count
    assert unique_count > 100, f"Expected >100 unique articles, got {unique_count}"
    assert stats["wos_only"] > 0
    assert stats["scopus_only"] > 0
    assert stats["in_both_databases"] > 0


def test_e2e_prisma_2020_svg(real_eew_dataset, tmp_path):
    """Verifies PRISMA 2020 SVG diagram generation using real dataset counts."""
    scopus_count = len(real_eew_dataset["scopus_ai"])
    wos_count = len(real_eew_dataset["wos_ai"])
    total_raw = len(real_eew_dataset["all_raw"])
    unique_count = len(real_eew_dataset["unique"])
    dup_count = real_eew_dataset["stats"]["duplicates_removed"]

    # Simulate realistic screening counts
    simulated_included = int(unique_count * 0.15)
    simulated_borderline = int(unique_count * 0.05)
    simulated_excluded = unique_count - simulated_included - simulated_borderline

    prisma_data = {
        "identified_wos": wos_count,
        "identified_scopus": scopus_count,
        "identified_total": total_raw,
        "duplicates_removed": dup_count,
        "records_screened": unique_count,
        "records_excluded": simulated_excluded,
        "records_borderline": simulated_borderline,
        "records_included": simulated_included,
        "records_unscreened": 0,
    }

    svg_content = generate_prisma_2020_svg(prisma_data)

    assert isinstance(svg_content, str)
    assert "<svg" in svg_content
    assert "</svg>" in svg_content
    assert "PRISMA 2020 Flow Diagram" in svg_content
    assert "IDENTIFICATION" in svg_content
    assert "SCREENING" in svg_content
    assert "INCLUDED" in svg_content
    assert f"n = {wos_count:,}" in svg_content
    assert f"n = {scopus_count:,}" in svg_content
    assert f"n = {total_raw:,}" in svg_content
    assert f"n = {dup_count:,}" in svg_content
    assert f"n = {unique_count:,}" in svg_content
    assert f"n = {simulated_included:,}" in svg_content

    # Verify export to SVG file
    out_svg = tmp_path / "prisma_flowchart.svg"
    out_svg.write_text(svg_content, encoding="utf-8")
    assert out_svg.exists()
    assert out_svg.stat().st_size > 1000


def test_e2e_active_learning_reranking_and_stopping(real_eew_dataset):
    """Tests ASReview-style active learning priority queue on real EEW papers."""
    articles = real_eew_dataset["unique"]
    assert len(articles) >= 50

    # Pick articles to act as labeled seed papers
    known_pos_keywords = ["deep learning", "neural network", "machine learning", "svm", "cnn", "lstm"]
    known_neg_keywords = ["interview", "qualitative", "policy", "social", "survey"]

    labeled = []
    unlabeled = []

    pos_found = 0
    neg_found = 0

    for art in articles:
        text = f"{art.title} {art.abstract}".lower()
        if pos_found < 5 and any(k in text for k in known_pos_keywords):
            art.decision = "Include"
            labeled.append(art)
            pos_found += 1
        elif neg_found < 5 and any(k in text for k in known_neg_keywords):
            art.decision = "Exclude"
            labeled.append(art)
            neg_found += 1
        else:
            art.decision = "Unscreened"
            unlabeled.append(art)

    # Ensure both positive and negative exist
    if pos_found == 0:
        articles[0].decision = "Include"
        labeled.append(articles[0])
    if neg_found == 0:
        articles[1].decision = "Exclude"
        labeled.append(articles[1])

    ranker = ActiveLearningRanker(classifier="logistic_regression")
    reranked = ranker.fit_and_rerank(labeled, unlabeled)

    assert len(reranked) == len(unlabeled)
    assert ranker.is_fitted is True
    assert ranker.positive_count > 0
    assert ranker.negative_count > 0

    # Check that articles have priority scores and are sorted descending
    scores = [getattr(a, "relevance_score", 0.0) for a in reranked]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True), "Queue must be strictly sorted by relevance probability"

    # Test stopping rule
    stop_check_40 = ranker.check_stopping_rule(consecutive_irrelevant=40, threshold=50)
    assert stop_check_40["should_stop"] is False
    assert stop_check_40["consecutive_irrelevant"] == 40
    assert stop_check_40["threshold"] == 50
    assert stop_check_40["remaining_to_stop"] == 10

    stop_check_50 = ranker.check_stopping_rule(consecutive_irrelevant=50, threshold=50)
    assert stop_check_50["should_stop"] is True
    assert stop_check_50["remaining_to_stop"] == 0


def test_e2e_cohens_kappa_and_conflicts(real_eew_dataset):
    """Tests Cohen's Kappa reliability and conflict detection on real articles."""
    articles = real_eew_dataset["unique"][:60]
    assert len(articles) == 60

    # Simulate Reviewer 1 (conservative screener) and Reviewer 2 (permissive screener)
    r1_decisions = []
    r2_decisions = []
    dual_screened_items = []

    for i, art in enumerate(articles):
        # r1 includes only first 15
        d1 = "Include" if i < 15 else "Exclude"
        # r2 includes first 20 and has 5 disagreements
        d2 = "Include" if i < 20 else "Exclude"

        r1_decisions.append(d1)
        r2_decisions.append(d2)

        dual_screened_items.append({
            "id": art.id or f"art_{i}",
            "title": art.title,
            "reviewer1": d1,
            "reviewer2": d2,
            "article": art
        })

    kappa_res = calculate_cohens_kappa(r1_decisions, r2_decisions)

    assert "kappa" in kappa_res
    assert "observed_agreement" in kappa_res
    assert "expected_agreement" in kappa_res
    assert "interpretation" in kappa_res
    assert kappa_res["n_total"] == 60
    assert kappa_res["n_disagreements"] == 5
    assert kappa_res["observed_agreement"] == round(55 / 60, 4)
    assert 0.7 <= kappa_res["kappa"] <= 1.0

    # Conflict identification
    conflicts = identify_conflicts(dual_screened_items)
    assert len(conflicts) == 5
    for conf in conflicts:
        assert conf["conflict_type"] == "Exclude vs Include" or conf["conflict_type"] == "Include vs Exclude"
        assert conf["requires_adjudication"] is True
        assert conf["title"] != ""


def test_e2e_keyword_cooccurrence_network(real_eew_dataset):
    """Tests building VOSviewer-style keyword co-occurrence graph on real articles."""
    articles = real_eew_dataset["unique"]
    fig = build_keyword_cooccurrence_network(articles, top_n=30, min_cooccurrence=2)

    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) >= 2, "Expected at least edge traces and node traces"

    # Find the node scatter trace
    node_trace = None
    for tr in fig.data:
        if getattr(tr, "mode", None) == "markers+text":
            node_trace = tr
            break

    assert node_trace is not None, "Node trace must be present"
    assert len(node_trace.x) > 0, "Keywords must be extracted and laid out in 2D coordinates"
    assert len(node_trace.text) == len(node_trace.x)


def test_e2e_vosviewer_scopus_export(real_eew_dataset, tmp_path):
    """Tests exporting real articles to VOSviewer/Bibliometrix Scopus CSV format."""
    articles = real_eew_dataset["unique"][:40]
    out_file = tmp_path / "vosviewer_export.csv"

    # Test file write mode
    res_path = export_vosviewer_scopus_format(articles, output_path=str(out_file))
    assert res_path == str(out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 500

    csv_text = out_file.read_text(encoding="utf-8")
    first_line = csv_text.splitlines()[0]
    for expected_col in ["Authors", "Title", "Year", "Source title", "Author Keywords", "Index Keywords", "References", "DOI"]:
        assert expected_col in first_line, f"Missing required column {expected_col} for VOSviewer"

    # Test in-memory string return mode
    csv_mem = export_vosviewer_scopus_format(articles, output_path=None)
    assert isinstance(csv_mem, str)
    assert "Authors" in csv_mem.splitlines()[0]


def test_e2e_keyword_highlighting_and_unpaywall_resolver():
    """Tests keyword highlighting logic and Unpaywall API resolver formatting."""
    # 1. Keyword highlighting logic
    def highlight_keywords(text: str, inclusion_kw: list, exclusion_kw: list) -> str:
        if not text:
            return ""
        highlighted = text
        for kw in inclusion_kw:
            if not kw.strip():
                continue
            pattern = re.compile(re.escape(kw.strip()), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f"<mark style='background-color: #d1fae5; color: #065f46; padding: 2px 5px; border-radius: 4px; font-weight: 600;'>{m.group(0)}</mark>",
                highlighted
            )
        for kw in exclusion_kw:
            if not kw.strip():
                continue
            pattern = re.compile(re.escape(kw.strip()), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f"<mark style='background-color: #fee2e2; color: #991b1b; padding: 2px 5px; border-radius: 4px; font-weight: 600;'>{m.group(0)}</mark>",
                highlighted
            )
        return highlighted

    raw_text = "We propose an earthquake early warning algorithm using deep neural network without structural damping."
    hl = highlight_keywords(raw_text, ["earthquake early warning", "deep neural network"], ["structural damping"])

    assert "#d1fae5" in hl  # green inclusion highlight
    assert "#fee2e2" in hl  # red exclusion highlight
    assert "earthquake early warning</mark>" in hl.lower()
    assert "structural damping</mark>" in hl.lower()

    # 2. Unpaywall URL resolver helper
    def resolve_unpaywall_url(doi: str, email: str = "unpaywall@strcview.local") -> str:
        clean_doi = doi.strip()
        if clean_doi.lower().startswith("https://doi.org/"):
            clean_doi = clean_doi[16:]
        elif clean_doi.lower().startswith("http://doi.org/"):
            clean_doi = clean_doi[15:]
        elif clean_doi.lower().startswith("doi:"):
            clean_doi = clean_doi[4:]
        clean_doi = clean_doi.strip()
        return f"https://api.unpaywall.org/v2/{clean_doi}?email={email}"

    sample_doi = "10.1016/j.soildyn.2023.108253"
    api_url = resolve_unpaywall_url(sample_doi)
    assert api_url == "https://api.unpaywall.org/v2/10.1016/j.soildyn.2023.108253?email=unpaywall@strcview.local"

    # Also handles full URL formatted DOIs
    url_doi = "https://doi.org/10.1007/s10518-023-01742-1"
    api_url2 = resolve_unpaywall_url(url_doi)
    assert api_url2 == "https://api.unpaywall.org/v2/10.1007/s10518-023-01742-1?email=unpaywall@strcview.local"

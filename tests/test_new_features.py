"""
Comprehensive unit tests for sTRCview next-generation features:
1. PRISMA 2020 SVG diagram generator
2. Cohen's Kappa & conflict identification
3. Keyword co-occurrence network (Plotly)
4. Active Learning priority screening ranker & stopping rule
5. VOSviewer / Bibliometrix Scopus format exporter
"""

import os
import tempfile
from pathlib import Path
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.models.article import Article
from src.analytics.prisma_diagram import generate_prisma_2020_svg
from src.analytics.reliability import calculate_cohens_kappa, identify_conflicts
from src.analytics.network import build_keyword_cooccurrence_network
from src.evaluators.active_learning import ActiveLearningRanker
from src.analytics.exporter import export_vosviewer_scopus_format


# ==========================================
# 1. PRISMA 2020 SVG Diagram Tests
# ==========================================
class TestPrismaDiagram:
    def test_generate_prisma_svg_basic(self):
        data = {
            "identified_wos": 120,
            "identified_scopus": 85,
            "identified_total": 205,
            "duplicates_removed": 35,
            "records_screened": 170,
            "records_excluded": 140,
            "records_borderline": 10,
            "records_included": 30,
            "records_unscreened": 0,
        }
        svg = generate_prisma_2020_svg(data)
        assert isinstance(svg, str)
        assert svg.strip().startswith("<svg")
        assert svg.strip().endswith("</svg>")
        assert "PRISMA 2020" in svg
        assert "120" in svg
        assert "85" in svg
        assert "205" in svg
        assert "35" in svg
        assert "170" in svg
        assert "140" in svg
        assert "30" in svg

    def test_generate_prisma_svg_defaults_and_fallbacks(self):
        # Empty dict should not raise error and default gracefully to 0
        svg = generate_prisma_2020_svg({})
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "n = 0" in svg

    def test_generate_prisma_svg_auto_totals(self):
        # Total should auto-compute if omitted
        data = {
            "identified_wos": 50,
            "identified_scopus": 50,
            "duplicates_removed": 10,
            "records_excluded": 30,
            "records_included": 10,
        }
        svg = generate_prisma_2020_svg(data)
        assert "n = 100" in svg  # 50 + 50
        assert "n = 90" in svg   # 100 - 10 screened


# ==========================================
# 2. Inter-Rater Reliability Tests
# ==========================================
class TestReliability:
    def test_perfect_agreement(self):
        r1 = ["Include", "Exclude", "Include", "Exclude"]
        r2 = ["Include", "Exclude", "Include", "Exclude"]
        res = calculate_cohens_kappa(r1, r2)
        assert res["kappa"] == 1.0
        assert res["observed_agreement"] == 1.0
        assert "Almost Perfect" in res["interpretation"]
        assert res["n_agreements"] == 4
        assert res["n_disagreements"] == 0

    def test_partial_agreement(self):
        r1 = ["Include", "Include", "Exclude", "Exclude"]
        r2 = ["Include", "Exclude", "Exclude", "Exclude"]
        res = calculate_cohens_kappa(r1, r2)
        # Expected: 3 out of 4 agree = 0.75 observed
        assert res["observed_agreement"] == 0.75
        assert 0.40 <= res["kappa"] <= 0.60
        assert "Moderate" in res["interpretation"]
        assert res["n_agreements"] == 3
        assert res["n_disagreements"] == 1
        assert "matrix" in res["confusion_matrix"]
        assert "table" in res["confusion_matrix"]

    def test_poor_agreement(self):
        r1 = ["Include", "Include", "Exclude", "Exclude"]
        r2 = ["Exclude", "Exclude", "Include", "Include"]
        res = calculate_cohens_kappa(r1, r2)
        assert res["observed_agreement"] == 0.0
        assert res["kappa"] < 0.0
        assert "Poor" in res["interpretation"]

    def test_empty_decisions(self):
        res = calculate_cohens_kappa([], [])
        assert res["kappa"] == 0.0
        assert res["n_total"] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_cohens_kappa(["Include"], ["Include", "Exclude"])

    def test_identify_conflicts_dicts(self):
        data = [
            {"id": "art-1", "title": "Paper 1", "reviewer1": "Include", "reviewer2": "Include"},
            {"id": "art-2", "title": "Paper 2", "reviewer1": "Include", "reviewer2": "Exclude"},
            {"id": "art-3", "title": "Paper 3", "reviewer1": "Borderline", "reviewer2": "Include"},
        ]
        conflicts = identify_conflicts(data)
        assert len(conflicts) == 2
        assert conflicts[0]["id"] == "art-2"
        assert conflicts[0]["conflict_type"] == "Include vs Exclude"
        assert conflicts[0]["requires_adjudication"] is True
        assert conflicts[1]["id"] == "art-3"

    def test_identify_conflicts_articles(self):
        a1 = Article(id="1", title="Title 1")
        a1.reviewer1_decision = "Include"
        a1.reviewer2_decision = "Exclude"

        a2 = Article(id="2", title="Title 2")
        a2.reviewer1_decision = "Exclude"
        a2.reviewer2_decision = "Exclude"

        conflicts = identify_conflicts([a1, a2])
        assert len(conflicts) == 1
        assert conflicts[0]["id"] == "1"


# ==========================================
# 3. Keyword Co-occurrence Network Tests
# ==========================================
class TestKeywordNetwork:
    def test_build_network_basic(self):
        arts = [
            Article(title="A1", keywords="Deep Learning; Earthquake; Early Warning"),
            Article(title="A2", keywords="Deep Learning; Neural Network; Earthquake"),
            Article(title="A3", keywords="Early Warning; P-Wave; Machine Learning"),
            Article(title="A4", keywords="Machine Learning; Deep Learning; Seismic"),
        ]
        fig = build_keyword_cooccurrence_network(arts, top_n=10, min_cooccurrence=1)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should contain line traces and marker trace
        # Layout check
        assert "Keyword Co-occurrence Network" in fig.layout.title.text

    def test_build_network_empty(self):
        fig = build_keyword_cooccurrence_network([])
        assert isinstance(fig, go.Figure)
        assert "No articles provided" in fig.layout.annotations[0].text

    def test_build_network_no_keywords(self):
        arts = [Article(title="No kw 1"), Article(title="No kw 2")]
        fig = build_keyword_cooccurrence_network(arts)
        assert isinstance(fig, go.Figure)
        assert "No keywords found" in fig.layout.annotations[0].text


# ==========================================
# 4. Active Learning Priority Screening Tests
# ==========================================
class TestActiveLearning:
    def test_fit_and_rerank_basic(self):
        ranker = ActiveLearningRanker(classifier="logistic_regression", random_state=42)

        # 2 positive, 2 negative training examples
        l1 = Article(title="Deep learning seismic phase picking", abstract="Neural network model for P-wave detection", decision="Include")
        l2 = Article(title="Earthquake early warning with convolutional networks", abstract="AI framework predicting shaking intensity", decision="Include")
        l3 = Article(title="High school geography curriculum", abstract="Pedagogical techniques for high school earth science", decision="Exclude")
        l4 = Article(title="Economic impact of tourism in Italy", abstract="Hospitality and restaurant revenue trends", decision="Exclude")

        # Unlabeled candidates
        u1 = Article(title="Automated earthquake detection using deep neural networks", abstract="P-wave arrival picking with transformer AI")
        u2 = Article(title="Urban hotel hospitality management", abstract="Tourism marketing strategies in Europe")

        reranked = ranker.fit_and_rerank([l1, l2, l3, l4], [u1, u2])

        assert len(reranked) == 2
        # u1 (AI earthquake) should be prioritized first
        assert reranked[0].title == u1.title
        assert reranked[0].relevance_score >= reranked[1].relevance_score
        assert ranker.is_fitted is True

    def test_cold_start_only_includes(self):
        ranker = ActiveLearningRanker()
        l1 = Article(title="Deep learning earthquake early warning", decision="Include")
        u1 = Article(title="Deep learning earthquake seismic waves")
        u2 = Article(title="Culinary recipes and baking bread")

        reranked = ranker.fit_and_rerank([l1], [u1, u2])
        assert len(reranked) == 2
        assert reranked[0].title == u1.title
        assert reranked[0].relevance_score >= reranked[1].relevance_score

    def test_sgd_classifier_support(self):
        ranker = ActiveLearningRanker(classifier="sgd", random_state=42)
        l1 = Article(title="Seismic wave AI", decision="Include")
        l2 = Article(title="Ancient Roman philosophy", decision="Exclude")
        u1 = Article(title="Seismic wave neural model")

        reranked = ranker.fit_and_rerank([l1, l2], [u1])
        assert len(reranked) == 1
        assert 0.0 <= reranked[0].relevance_score <= 1.0

    def test_stopping_rule(self):
        # Under threshold
        res1 = ActiveLearningRanker.check_stopping_rule(consecutive_irrelevant=25, threshold=50)
        assert res1["should_stop"] is False
        assert res1["recommendation"] == "CONTINUE"
        assert res1["progress_pct"] == 50.0

        # At threshold
        res2 = ActiveLearningRanker.check_stopping_rule(consecutive_irrelevant=50, threshold=50)
        assert res2["should_stop"] is True
        assert res2["recommendation"] == "STOP"
        assert res2["progress_pct"] == 100.0


# ==========================================
# 5. VOSviewer / Scopus Exporter Tests
# ==========================================
class TestVOSviewerExporter:
    def test_export_scopus_format_file(self):
        articles = [
            Article(
                id="art1",
                raw_id="2-s2.0-85012345678",
                title="Deep Learning in Earthquake Early Warning",
                authors="Smith, J.; Doe, A.; Johnson, K.",
                year=2023,
                journal="Computers & Geosciences",
                doi="10.1016/j.cageo.2023.105000",
                keywords="Deep Learning | Earthquake | Early Warning",
                affiliations="Stanford University; MIT",
                citations=42,
                references="Allen RM, 2009, NATURE, V458, P123; Meier MA, 2019, SCIENCE, V366, P1234",
            ),
            Article(
                id="art2",
                title="Seismic Wave Detection",
                authors="Yamada, T.; Sato, H.",
                year=2022,
                journal="Bulletin of the Seismological Society of America",
                doi="10.1785/0120220000",
                keywords="Seismology; Neural Networks",
                references="Kanamori H, 1997, JGR, V102, P11853",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "vosviewer_export.csv"
            result_path = export_vosviewer_scopus_format(articles, output_path=out_file)
            assert str(result_path) == str(out_file)
            assert out_file.exists()

            # Read back with pandas
            df = pd.read_csv(out_file, dtype=str)
            assert len(df) == 2

            # Mandatory VOSviewer columns
            required_cols = [
                "Authors", "Title", "Year", "Source title", "Cited by",
                "DOI", "References", "Author Keywords", "Index Keywords", "EID", "Source"
            ]
            for col in required_cols:
                assert col in df.columns, f"Missing required VOSviewer column: {col}"

            # Check contents
            row0 = df.iloc[0]
            assert row0["Title"] == "Deep Learning in Earthquake Early Warning"
            assert "Allen RM, 2009" in row0["References"]
            assert "Meier MA, 2019" in row0["References"]
            assert row0["DOI"] == "10.1016/j.cageo.2023.105000"
            assert row0["Source"] == "Scopus"
            assert row0["EID"] == "2-s2.0-85012345678"

    def test_export_scopus_format_string(self):
        articles = [
            Article(
                id="test1",
                title="Single Paper",
                authors="Author, A.",
                year=2024,
                references="Ref 1; Ref 2"
            )
        ]
        csv_str = export_vosviewer_scopus_format(articles, output_path=None)
        assert isinstance(csv_str, str)
        assert "Authors,Author(s) ID,Title" in csv_str
        assert "Single Paper" in csv_str
        assert "Ref 1; Ref 2" in csv_str

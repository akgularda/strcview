"""
Unit tests for sTRCview UI helper components:
- highlight_abstract_keywords
- resolve_unpaywall_doi
"""

import pytest
from src.ui.app import highlight_abstract_keywords, resolve_unpaywall_doi


def test_highlight_abstract_keywords_basic():
    text = "We propose an earthquake early warning algorithm using deep neural network without structural damping."
    inc = ["earthquake early warning", "deep neural network"]
    exc = ["structural damping"]

    hl = highlight_abstract_keywords(text, inc, exc)

    assert "#d1fae5" in hl  # soft green
    assert "#fee2e2" in hl  # soft red
    assert "earthquake early warning</mark>" in hl.lower()
    assert "structural damping</mark>" in hl.lower()


def test_highlight_abstract_keywords_empty_and_escape():
    assert highlight_abstract_keywords("", ["kw"], ["kw"]) == ""

    # Check HTML escaping inside text
    unsafe_text = "Risk of <script>alert('xss')</script> in machine learning."
    hl = highlight_abstract_keywords(unsafe_text, ["machine learning"], [])
    assert "<script>" not in hl
    assert "&lt;script&gt;" in hl
    assert "#d1fae5" in hl


def test_resolve_unpaywall_doi_formatting():
    # Empty DOI
    res_empty = resolve_unpaywall_doi("")
    assert res_empty["has_doi"] is False
    assert res_empty["is_oa"] is False

    # Standard DOI
    sample_doi = "10.1016/j.soildyn.2023.108253"
    res = resolve_unpaywall_doi(sample_doi)
    assert res["has_doi"] is True
    assert res["clean_doi"] == sample_doi
    assert res["doi_url"] == f"https://doi.org/{sample_doi}"
    assert "api.unpaywall.org/v2/10.1016/j.soildyn.2023.108253" in res["api_url"]

    # Full URL DOI
    url_doi = "https://doi.org/10.1007/s10518-023-01742-1"
    res2 = resolve_unpaywall_doi(url_doi)
    assert res2["has_doi"] is True
    assert res2["clean_doi"] == "10.1007/s10518-023-01742-1"
    assert res2["doi_url"] == "https://doi.org/10.1007/s10518-023-01742-1"

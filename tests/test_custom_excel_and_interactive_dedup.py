import io
import pandas as pd
import pytest

from src.models.article import Article
from src.parsers.custom_excel_parser import (
    detect_column_mapping,
    get_excel_sheet_names,
    load_tabular_file,
    parse_custom_dataframe
)
from src.parsers.deduplicator import (
    find_duplicate_candidates,
    merge_two_articles,
    apply_duplicate_resolutions
)


def test_detect_column_mapping_turkish_and_english():
    cols_en = ["Article Title", "Abstract", "Author(s)", "Publication Year", "Source Title", "DOI", "Author Keywords"]
    map_en = detect_column_mapping(cols_en)
    assert map_en["title"] == "Article Title"
    assert map_en["abstract"] == "Abstract"
    assert map_en["authors"] == "Author(s)"
    assert map_en["year"] == "Publication Year"
    assert map_en["journal"] == "Source Title"
    assert map_en["doi"] == "DOI"
    assert map_en["keywords"] == "Author Keywords"

    cols_tr = ["Makale Başlığı", "Özet", "Yazarlar", "Yıl", "Dergi", "DOI Linki", "Anahtar Kelimeler"]
    map_tr = detect_column_mapping(cols_tr)
    assert map_tr["title"] == "Makale Başlığı"
    assert map_tr["abstract"] == "Özet"
    assert map_tr["authors"] == "Yazarlar"
    assert map_tr["year"] == "Yıl"
    assert map_tr["journal"] == "Dergi"
    assert map_tr["doi"] == "DOI Linki"
    assert map_tr["keywords"] == "Anahtar Kelimeler"


def test_parse_custom_dataframe_basic():
    df = pd.DataFrame([
        {
            "Title": "Novel Machine Learning in Earthquake Warning",
            "Abstract": "This study applies convolutional networks.",
            "Authors": "Yilmaz, A.; Kaya, B.",
            "Year": "2024",
            "Journal": "Journal of Seismology",
            "DOI": "10.1007/s10950-024-01234",
            "Keywords": "EEW; CNN; Deep Learning"
        },
        {
            "Title": "Traditional Structural Analysis of Bridges",
            "Abstract": "Static load testing on concrete bridges.",
            "Authors": "Demir, C.",
            "Year": 2021,
            "Journal": "Civil Engineering Review",
            "DOI": "",
            "Keywords": "Bridge; Concrete"
        },
        {
            "Title": None,  # Should be skipped
            "Abstract": "Missing title row."
        }
    ])

    mapping = {
        "title": "Title",
        "abstract": "Abstract",
        "authors": "Authors",
        "year": "Year",
        "journal": "Journal",
        "doi": "DOI",
        "keywords": "Keywords"
    }

    articles = parse_custom_dataframe(df, mapping, source_name="My Research Excel")
    assert len(articles) == 2
    assert articles[0].title == "Novel Machine Learning in Earthquake Warning"
    assert articles[0].year == 2024
    assert articles[0].source_db == "My Research Excel"
    assert articles[0].doi == "10.1007/s10950-024-01234"
    assert articles[1].year == 2021


def test_custom_excel_file_reading():
    df = pd.DataFrame([{"Başlık": "Örnek Makale", "Özet": "Özet metni", "Yıl": 2023}])
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Çalışma1", index=False)
    raw_bytes = excel_buf.getvalue()

    sheets = get_excel_sheet_names(raw_bytes)
    assert "Çalışma1" in sheets

    loaded_df = load_tabular_file(raw_bytes, "test_file.xlsx", sheet_name="Çalışma1")
    assert len(loaded_df) == 1
    assert loaded_df["Başlık"].iloc[0] == "Örnek Makale"


def test_find_duplicate_candidates_and_resolutions():
    art1 = Article(
        id="art_1",
        title="Deep learning based earthquake early warning system",
        abstract="Short abstract.",
        doi="10.1016/j.soildyn.2023.100",
        source_db="WoS"
    )
    art2 = Article(
        id="art_2",
        title="Deep Learning Based Earthquake Early Warning System",
        abstract="Very comprehensive abstract with rich methodology details.",
        doi="10.1016/j.soildyn.2023.100",
        source_db="Scopus"
    )
    art3 = Article(
        id="art_3",
        title="Seismic response analysis of isolated structures",
        abstract="Distinct paper.",
        doi="10.1016/j.soildyn.2023.200",
        source_db="Scopus"
    )

    candidates = find_duplicate_candidates([art1, art2, art3])
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["match_type"] == "Tam DOI Eşleşmesi"
    assert cand["article_a"].id == "art_1"
    assert cand["article_b"].id == "art_2"

    # Test merge resolution
    decisions = {cand["candidate_id"]: "merge"}
    resolved, stats = apply_duplicate_resolutions([art1, art2, art3], decisions, candidates)
    assert len(resolved) == 2
    assert stats["merged_count"] == 1
    merged_item = next(a for a in resolved if a.id == "art_1")
    assert merged_item.source_db == "WoS & Scopus"
    assert "Very comprehensive abstract" in merged_item.abstract

    # Test keep_both resolution
    decisions_both = {cand["candidate_id"]: "keep_both"}
    resolved_both, stats_both = apply_duplicate_resolutions([art1, art2, art3], decisions_both, candidates)
    assert len(resolved_both) == 3
    assert stats_both["merged_count"] == 0

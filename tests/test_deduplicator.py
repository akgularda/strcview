import pytest
from src.models.article import Article
from src.parsers.deduplicator import deduplicate_articles, normalize_doi, normalize_title

def test_deduplicator_exact_doi():
    art1 = Article(
        title="Deep learning for earthquake early warning",
        abstract="Short abstract from WoS.",
        doi="https://doi.org/10.1016/j.ijdrr.2025.105294",
        source_db="WoS",
        raw_id="WOS:001"
    )
    art2 = Article(
        title="Deep Learning For Earthquake Early Warning",
        abstract="Longer abstract with more detailed results from Scopus.",
        doi="10.1016/j.ijdrr.2025.105294",
        source_db="Scopus",
        raw_id="2-s2.0-85217944451"
    )
    
    unique, stats = deduplicate_articles([art1, art2])
    assert len(unique) == 1
    assert stats["duplicates_removed"] == 1
    assert stats["in_both_databases"] == 1
    
    merged = unique[0]
    assert merged.source_db == "WoS & Scopus"
    assert "Longer abstract" in merged.abstract
    assert "WOS:001" in merged.raw_id and "2-s2.0-85217944451" in merged.raw_id

def test_deduplicator_title_similarity():
    art1 = Article(
        title="Real-time seismic response prediction method of high-rise buildings based on deep learning",
        doi="",
        source_db="WoS"
    )
    art2 = Article(
        title="Real-time seismic response prediction method of high-rise buildings based on deep learning.",
        doi="",
        source_db="Scopus"
    )
    
    unique, stats = deduplicate_articles([art1, art2])
    assert len(unique) == 1
    assert stats["duplicates_removed"] == 1

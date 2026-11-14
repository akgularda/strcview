import os
import pytest
from src.parsers.wos_parser import parse_wos_tagged_text, parse_wos_file
from src.parsers.scopus_parser import parse_scopus_csv, parse_scopus_file
from src.models.article import Article

def test_wos_tagged_parser_synthetic():
    sample_wos = """FN Clarivate Analytics Web of Science
VR 1.0
PT J
AU Smith, John
   Doe, Jane
TI Deep Learning for Seismic Early Warning Systems: A Comprehensive
   Evaluation of Waveforms
SO JOURNAL OF SEISMOLOGY
PY 2024
DI 10.1007/s10950-024-001
AB This paper explores deep learning architectures applied to real-time seismic waves.
   Our model predicts peak ground acceleration within 3 seconds.
DE earthquake; deep learning; early warning
TC 15
UT WOS:000123456700001
ER
"""
    articles = parse_wos_tagged_text(sample_wos)
    assert len(articles) == 1
    art = articles[0]
    assert "Deep Learning for Seismic Early Warning Systems: A Comprehensive Evaluation of Waveforms" in art.title
    assert "Smith, John; Doe, Jane" == art.authors
    assert art.year == 2024
    assert art.doi == "10.1007/s10950-024-001"
    assert "real-time seismic waves" in art.abstract
    assert art.source_db == "WoS"
    assert art.raw_id == "WOS:000123456700001"
    assert art.citations == 15

def test_scopus_csv_parser_synthetic():
    sample_csv = """"Authors","Title","Year","Source title","DOI","Abstract","Author Keywords","Cited by","EID"
"Wang Y.; Zhao Q.","Cumulative absolute velocity prediction for earthquake early warning","2024","Computer-Aided Civil Engineering","10.1111/mice.13065","A rapid and accurate estimate of earthquake damage is key.","Deep learning; EEW","16","2-s2.0-85164111541"
"""
    articles = parse_scopus_csv(sample_csv)
    assert len(articles) == 1
    art = articles[0]
    assert art.title == "Cumulative absolute velocity prediction for earthquake early warning"
    assert art.year == 2024
    assert art.doi == "10.1111/mice.13065"
    assert art.raw_id == "2-s2.0-85164111541"
    assert art.source_db == "Scopus"
    assert art.citations == 16
    assert "Deep learning" in art.keywords

def test_real_user_files_if_available():
    wos_path = r"g:\My Drive\Life OS\Research\Beyond Epicenters\EEW\Ham Dosyalar\WoS AI EEW.txt"
    scopus_path = r"g:\My Drive\Life OS\Research\Beyond Epicenters\EEW\Ham Dosyalar\Scopus AI EEW.csv"
    
    if os.path.exists(wos_path):
        with open(wos_path, "rb") as f:
            wos_arts = parse_wos_file(f.read())
        assert len(wos_arts) > 0
        assert all(isinstance(a, Article) for a in wos_arts)
        assert any("deep learning" in a.title.lower() for a in wos_arts)
        
    if os.path.exists(scopus_path):
        with open(scopus_path, "rb") as f:
            scopus_arts = parse_scopus_file(f.read())
        assert len(scopus_arts) > 0
        assert all(isinstance(a, Article) for a in scopus_arts)

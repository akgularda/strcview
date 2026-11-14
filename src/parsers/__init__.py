from .wos_parser import parse_wos_file, parse_wos_tagged_text, parse_wos_tabular
from .scopus_parser import parse_scopus_file, parse_scopus_csv, parse_scopus_ris
from .deduplicator import deduplicate_articles, normalize_doi, normalize_title

__all__ = [
    "parse_wos_file",
    "parse_wos_tagged_text",
    "parse_wos_tabular",
    "parse_scopus_file",
    "parse_scopus_csv",
    "parse_scopus_ris",
    "deduplicate_articles",
    "normalize_doi",
    "normalize_title"
]

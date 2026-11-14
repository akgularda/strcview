import re
import io
import hashlib
import csv
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.models.article import Article

# Standard aliases for automatic column name mapping
COLUMN_ALIASES: Dict[str, List[str]] = {
    "title": [
        "title", "article title", "document title", "makale başlığı", "başlık",
        "baslik", "paper title", "work title", "name", "item title"
    ],
    "abstract": [
        "abstract", "özet", "ozet", "description", "summary", "ab",
        "abstract note", "executive summary", "overview"
    ],
    "authors": [
        "authors", "author", "author(s)", "yazarlar", "yazar", "creator",
        "creators", "au", "inventors", "contributor"
    ],
    "year": [
        "year", "yıl", "yil", "publication year", "pub year", "date",
        "tarih", "py", "pub date", "release year"
    ],
    "journal": [
        "journal", "source title", "dergi", "source", "publication",
        "so", "venue", "booktitle", "conference", "publisher"
    ],
    "doi": [
        "doi", "digital object identifier", "di", "link", "doi link",
        "article doi", "electronic identifier"
    ],
    "keywords": [
        "keywords", "author keywords", "anahtar kelimeler", "de",
        "index keywords", "tags", "subject terms", "mesh terms"
    ]
}


def get_excel_sheet_names(file_bytes: bytes) -> List[str]:
    """Returns the list of sheet names from an Excel workbook (.xlsx, .xls)."""
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        return xl.sheet_names
    except Exception:
        return ["Sheet1"]


def detect_column_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Intelligently suggests default column mappings based on known aliases.
    Matches case-insensitively with exact match first, then substring match.
    """
    mapping: Dict[str, Optional[str]] = {
        "title": None,
        "abstract": None,
        "authors": None,
        "year": None,
        "journal": None,
        "doi": None,
        "keywords": None
    }

    # Normalize column names for comparison
    def normalize(value):
        value = unicodedata.normalize('NFKD', str(value).casefold().replace('ı', 'i'))
        return re.sub(r'[^a-z0-9]', '', value)

    norm_to_orig = {normalize(column): column for column in columns if normalize(column)}

    for field, aliases in COLUMN_ALIASES.items():
        found_col = None
        
        # 1. Exact normalized match
        for alias in aliases:
            norm_alias = normalize(alias)
            if norm_alias in norm_to_orig:
                found_col = norm_to_orig[norm_alias]
                break

        # 2. Substring match fallback
        if not found_col:
            for alias in aliases:
                norm_alias = normalize(alias)
                if len(norm_alias) < 3:
                    continue
                for norm_c, orig_c in norm_to_orig.items():
                    if len(norm_c) >= 3 and (norm_alias in norm_c or norm_c in norm_alias):
                        found_col = orig_c
                        break
                if found_col:
                    break

        mapping[field] = found_col

    return mapping


def load_tabular_file(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Loads an Excel or generic CSV/TSV file into a pandas DataFrame.
    """
    fname = filename.lower()
    if fname.endswith((".xlsx", ".xls")):
        target_sheet = sheet_name if sheet_name else 0
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=target_sheet)
    
    # Text-based tabular format (CSV, TSV, TXT)
    # Auto-detect separator
    encoding = "utf-16" if file_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        text = file_bytes.decode(encoding)
    except UnicodeDecodeError:
        text = file_bytes.decode("cp1252", errors="replace")
    header = text.split("\n", 1)[0]
    try:
        sep = csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
    except csv.Error:
        sep = "\t" if fname.endswith(".tsv") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def parse_custom_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    source_name: str = "Custom Excel"
) -> List[Article]:
    """
    Transforms a custom DataFrame into a list of Article dataclass instances
    using user-confirmed column mappings.
    """
    articles: List[Article] = []

    title_col = mapping.get("title")
    if title_col is None or title_col not in df.columns:
        return []

    abstract_col = mapping.get("abstract") if mapping.get("abstract") in df.columns else None
    authors_col = mapping.get("authors") if mapping.get("authors") in df.columns else None
    year_col = mapping.get("year") if mapping.get("year") in df.columns else None
    journal_col = mapping.get("journal") if mapping.get("journal") in df.columns else None
    doi_col = mapping.get("doi") if mapping.get("doi") in df.columns else None
    keywords_col = mapping.get("keywords") if mapping.get("keywords") in df.columns else None

    for idx, row in df.iterrows():
        raw_title = row.get(title_col)
        if pd.isna(raw_title) or not str(raw_title).strip():
            continue  # Skip entries with no title

        title_str = str(raw_title).strip()

        # Abstract
        abstract_str = ""
        if abstract_col is not None and not pd.isna(row.get(abstract_col)):
            abstract_str = str(row.get(abstract_col)).strip()

        # Authors
        authors_str = ""
        if authors_col is not None and not pd.isna(row.get(authors_col)):
            authors_str = str(row.get(authors_col)).strip()

        # Year
        year_val = None
        if year_col is not None and not pd.isna(row.get(year_col)):
            y_raw = str(row.get(year_col)).strip()
            # Extract 4-digit year (1900-2099)
            m = re.search(r'\b(19\d\d|20\d\d)\b', y_raw)
            if m:
                year_val = int(m.group(1))

        # Journal / Source
        journal_str = ""
        if journal_col is not None and not pd.isna(row.get(journal_col)):
            journal_str = str(row.get(journal_col)).strip()

        # DOI
        doi_str = ""
        if doi_col is not None and not pd.isna(row.get(doi_col)):
            doi_raw = str(row.get(doi_col)).strip()
            # Clean DOI prefix if present
            clean_doi = re.sub(r'^https?://[^/]*doi\.org/', '', doi_raw)
            clean_doi = re.sub(r'^doi:\s*', '', clean_doi).strip()
            doi_str = clean_doi

        # Keywords
        kw_str = ""
        if keywords_col is not None and not pd.isna(row.get(keywords_col)):
            kw_str = str(row.get(keywords_col)).strip()

        # Generate deterministic, reproducible ID
        hash_seed = f"{title_str}_{doi_str}_{idx}"
        art_id = f"custom_{hashlib.md5(hash_seed.encode('utf-8')).hexdigest()[:10]}"

        art = Article(
            id=art_id,
            title=title_str,
            abstract=abstract_str,
            authors=authors_str,
            year=year_val,
            journal=journal_str,
            doi=doi_str,
            source_db=source_name,
            keywords=kw_str,
            raw_id=f"row_{idx+1}"
        )
        articles.append(art)

    return articles

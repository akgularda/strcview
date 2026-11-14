import io
import re
import csv
from typing import List, Dict, Any, Union
import pandas as pd
from src.models.article import Article

def parse_wos_tagged_text(content: str) -> List[Article]:
    """
    Parses a Web of Science plain text export file (tagged format with ER delimiters).
    Handles multi-line continuations for Title (TI), Abstract (AB), Authors (AU), etc.
    """
    articles: List[Article] = []
    lines = content.splitlines()
    
    current_record: Dict[str, List[str]] = {}
    current_tag = None
    
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue
            
        # Check for End of Record
        if line == "ER" or line.startswith("ER "):
            if current_record:
                article = _create_article_from_wos_record(current_record)
                if article.title:  # Only add records that have at least a title
                    articles.append(article)
                current_record = {}
                current_tag = None
            continue
            
        # Check if line starts with a 2-character tag
        # e.g., 'PT ', 'TI ', 'AB ', 'AU ', etc.
        if len(line) >= 3 and line[:2].isupper() and line[2] == ' ':
            current_tag = line[:2]
            val = line[3:].strip()
            if current_tag not in current_record:
                current_record[current_tag] = []
            if val:
                current_record[current_tag].append(val)
        elif current_tag and line.startswith((' ', '\t')):
            # Continuation line
            val = line.strip()
            if val:
                current_record[current_tag].append(val)
                
    # In case file didn't end with ER
    if current_record:
        article = _create_article_from_wos_record(current_record)
        if article.title:
            articles.append(article)
            
    return articles

def _create_article_from_wos_record(record: Dict[str, List[str]]) -> Article:
    """Converts a parsed WoS tagged record dictionary into an Article instance."""
    # Title
    title = " ".join(record.get("TI", [])).strip()
    
    # Abstract
    abstract = " ".join(record.get("AB", [])).strip()
    
    # Authors
    authors_list = record.get("AU", [])
    if not authors_list:
        authors_list = record.get("AF", [])
    authors = "; ".join(authors_list).strip()
    
    # Year
    year = None
    py_list = record.get("PY", [])
    if py_list:
        match = re.search(r'\b(19\d\d|20\d\d)\b', py_list[0])
        if match:
            year = int(match.group(1))
            
    # Source / Journal
    journal = " ".join(record.get("SO", [])).strip()
    
    # DOI
    doi = ""
    di_list = record.get("DI", [])
    if di_list:
        doi = di_list[0].strip()
        # Clean potential URL prefix
        doi = re.sub(r'^https?://[^/]*doi\.org/', '', doi, flags=re.IGNORECASE)
        
    # Keywords
    de_list = record.get("DE", [])
    id_list = record.get("ID", [])
    all_kw = []
    if de_list:
        all_kw.append("; ".join(de_list))
    if id_list:
        all_kw.append("; ".join(id_list))
    keywords = " | ".join(all_kw).strip()
    
    # Raw ID
    ut_list = record.get("UT", [])
    raw_id = ut_list[0].strip() if ut_list else ""
    
    # Citations
    citations = 0
    tc_list = record.get("TC", record.get("Z9", []))
    if tc_list:
        try:
            citations = int(tc_list[0].strip())
        except ValueError:
            citations = 0
            
    # Affiliations
    affiliations = " ".join(record.get("C1", [])).strip()
    
    # Cited references (CR)
    references = "; ".join(record.get("CR", [])).strip()
    
    return Article(
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        keywords=keywords,
        source_db="WoS",
        raw_id=raw_id,
        citations=citations,
        affiliations=affiliations,
        references=references
    )

def parse_wos_tabular(file_content: Union[str, bytes]) -> List[Article]:
    """
    Parses a Web of Science export in CSV or TSV format.
    """
    if isinstance(file_content, bytes):
        try:
            encoding = 'utf-16' if file_content.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            text = file_content.decode(encoding)
        except UnicodeDecodeError:
            text = file_content.decode('latin-1', errors='replace')
    else:
        text = file_content
        
    # Detect delimiter: comma vs tab
    sample = text[:4096]
    sep = '\t' if '\t' in sample.partition('\n')[0] else ','
    
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str).fillna("")
    
    # Normalize column names
    col_map = {}
    for col in df.columns:
        norm = col.strip().lower()
        if "article title" in norm or norm == "title":
            col_map["title"] = col
        elif "abstract" in norm:
            col_map["abstract"] = col
        elif norm in ("authors", "author", "author full names", "au", "af"):
            col_map.setdefault("authors", col)
        elif "publication year" in norm or norm == "year":
            col_map["year"] = col
        elif "source title" in norm or "journal" in norm:
            col_map["journal"] = col
        elif "doi" in norm:
            col_map["doi"] = col
        elif "keyword" in norm:
            col_map["keywords"] = col
        elif "cited reference" in norm or norm == "cr":
            col_map["references"] = col
        elif "accession number" in norm or norm == "ut":
            col_map["raw_id"] = col
            
    articles = []
    for _, row in df.iterrows():
        title = str(row.get(col_map.get("title", ""), "")).strip()
        if not title:
            continue
        
        abstract = str(row.get(col_map.get("abstract", ""), "")).strip()
        authors = str(row.get(col_map.get("authors", ""), "")).strip()
        journal = str(row.get(col_map.get("journal", ""), "")).strip()
        doi = str(row.get(col_map.get("doi", ""), "")).strip()
        doi = re.sub(r'^https?://[^/]*doi\.org/', '', doi, flags=re.IGNORECASE)
        keywords = str(row.get(col_map.get("keywords", ""), "")).strip()
        raw_id = str(row.get(col_map.get("raw_id", ""), "")).strip()
        references = str(row.get(col_map.get("references", ""), "")).strip()
        
        year = None
        yr_val = str(row.get(col_map.get("year", ""), "")).strip()
        match = re.search(r'\b(19\d\d|20\d\d)\b', yr_val)
        if match:
            year = int(match.group(1))
            
        articles.append(Article(
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            keywords=keywords,
            source_db="WoS",
            raw_id=raw_id,
            references=references
        ))
        
    return articles

def parse_wos_file(file_content: Union[str, bytes]) -> List[Article]:
    """
    Main entry point for Web of Science files.
    Automatically detects whether the file is tagged text or CSV/TSV.
    """
    if isinstance(file_content, bytes):
        try:
            encoding = 'utf-16' if file_content.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            text = file_content.decode(encoding)
        except UnicodeDecodeError:
            text = file_content.decode('latin-1', errors='replace')
    else:
        text = file_content
        
    first_lines = text[:2000].splitlines()
    is_tagged = any(line.startswith(("FN Clarivate", "VR 1.0", "PT ")) for line in first_lines)
    
    if is_tagged or "\nER\n" in text or text.endswith("\nER") or text.endswith("\r\nER"):
        return parse_wos_tagged_text(text)
    else:
        return parse_wos_tabular(text)

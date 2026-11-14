import io
import re
import csv
from typing import List, Dict, Any, Union
import pandas as pd
from src.models.article import Article

def parse_scopus_csv(file_content: Union[str, bytes]) -> List[Article]:
    """
    Parses a Scopus CSV export file.
    Maps Scopus specific headers (Title, Abstract, Authors, Year, DOI, EID, etc.)
    to the Article data structure.
    """
    if isinstance(file_content, bytes):
        try:
            encoding = 'utf-16' if file_content.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            text = file_content.decode(encoding)
        except UnicodeDecodeError:
            text = file_content.decode('latin-1', errors='replace')
    else:
        text = file_content
        
    # Read using pandas with string types to preserve IDs and DOIs
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    
    # Header normalization mapping
    col_map = {}
    for col in df.columns:
        clean = col.strip().lower()
        if clean == "title":
            col_map["title"] = col
        elif clean == "abstract":
            col_map["abstract"] = col
        elif clean == "authors":
            col_map["authors"] = col
        elif clean == "author full names" and "authors" not in col_map:
            col_map["authors"] = col
        elif clean == "year":
            col_map["year"] = col
        elif clean == "source title":
            col_map["journal"] = col
        elif clean == "doi":
            col_map["doi"] = col
        elif clean == "author keywords":
            col_map["author_keywords"] = col
        elif clean == "index keywords":
            col_map["index_keywords"] = col
        elif clean == "cited by":
            col_map["citations"] = col
        elif clean == "link":
            col_map["link"] = col
        elif clean == "affiliations":
            col_map["affiliations"] = col
        elif clean in ("references", "cited references"):
            col_map["references"] = col
        elif clean == "eid":
            col_map["raw_id"] = col
            
    articles: List[Article] = []
    
    for _, row in df.iterrows():
        title = str(row.get(col_map.get("title", ""), "")).strip()
        if not title or title.lower() == "[no title available]":
            continue
            
        abstract = str(row.get(col_map.get("abstract", ""), "")).strip()
        if abstract.lower() == "[no abstract available]":
            abstract = ""
            
        authors = str(row.get(col_map.get("authors", ""), "")).strip()
        journal = str(row.get(col_map.get("journal", ""), "")).strip()
        
        doi = str(row.get(col_map.get("doi", ""), "")).strip()
        doi = re.sub(r'^https?://[^/]*doi\.org/', '', doi, flags=re.IGNORECASE)
        
        raw_id = str(row.get(col_map.get("raw_id", ""), "")).strip()
        link = str(row.get(col_map.get("link", ""), "")).strip()
        affiliations = str(row.get(col_map.get("affiliations", ""), "")).strip()
        references = str(row.get(col_map.get("references", ""), "")).strip()
        
        # Keywords combination
        ak = str(row.get(col_map.get("author_keywords", ""), "")).strip()
        ik = str(row.get(col_map.get("index_keywords", ""), "")).strip()
        keywords_parts = [k for k in [ak, ik] if k]
        keywords = " | ".join(keywords_parts)
        
        # Citations
        citations = 0
        cit_str = str(row.get(col_map.get("citations", ""), "")).strip()
        if cit_str.isdigit():
            citations = int(cit_str)
            
        # Year
        year = None
        yr_str = str(row.get(col_map.get("year", ""), "")).strip()
        match = re.search(r'\b(19\d\d|20\d\d)\b', yr_str)
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
            source_db="Scopus",
            raw_id=raw_id,
            link=link,
            citations=citations,
            affiliations=affiliations,
            references=references
        ))
        
    return articles

def parse_scopus_ris(content: str) -> List[Article]:
    """
    Parses a RIS formatted file (commonly used by Scopus / EndNote).
    """
    articles: List[Article] = []
    lines = content.splitlines()
    
    current: Dict[str, List[str]] = {}
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
            
        if line.startswith("ER  -"):
            if current:
                article = _create_article_from_ris(current)
                if article.title:
                    articles.append(article)
                current = {}
            continue
            
        if len(line) >= 6 and line[2:6] == "  - ":
            tag = line[:2]
            val = line[6:].strip()
            if tag not in current:
                current[tag] = []
            if val:
                current[tag].append(val)
        elif current and tag:
            # Continuation line
            current[tag].append(line)
            
    if current:
        article = _create_article_from_ris(current)
        if article.title:
            articles.append(article)
            
    return articles

def _create_article_from_ris(record: Dict[str, List[str]]) -> Article:
    title_list = record.get("TI", record.get("T1", []))
    title = " ".join(title_list).strip()
    
    abstract_list = record.get("AB", record.get("N2", []))
    abstract = " ".join(abstract_list).strip()
    
    authors_list = record.get("AU", record.get("A1", []))
    authors = "; ".join(authors_list).strip()
    
    journal_list = record.get("JO", record.get("JF", record.get("T2", [])))
    journal = " ".join(journal_list).strip()
    
    doi_list = record.get("DO", [])
    doi = doi_list[0].strip() if doi_list else ""
    doi = re.sub(r'^https?://[^/]*doi\.org/', '', doi, flags=re.IGNORECASE)
    
    keywords_list = record.get("KW", [])
    keywords = "; ".join(keywords_list).strip()
    
    year = None
    py_list = record.get("PY", record.get("Y1", []))
    if py_list:
        match = re.search(r'\b(19\d\d|20\d\d)\b', py_list[0])
        if match:
            year = int(match.group(1))
            
    return Article(
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        keywords=keywords,
        source_db="Scopus"
    )

def parse_scopus_file(file_content: Union[str, bytes]) -> List[Article]:
    """
    Main entry point for Scopus files. Automatically detects CSV or RIS.
    """
    if isinstance(file_content, bytes):
        try:
            encoding = 'utf-16' if file_content.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            text = file_content.decode(encoding)
        except UnicodeDecodeError:
            text = file_content.decode('latin-1', errors='replace')
    else:
        text = file_content
        
    sample = text[:1000]
    if "TY  - " in sample or "\nER  -" in sample:
        return parse_scopus_ris(text)
    else:
        return parse_scopus_csv(text)

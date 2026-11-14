import io
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import pandas as pd
from src.models.article import Article

def export_to_dataframe(articles: List[Article]) -> pd.DataFrame:
    """Converts a list of Article objects into a clean pandas DataFrame."""
    records = []
    for a in articles:
        records.append({
            "Decision": a.decision,
            "Relevance Score": a.relevance_score,
            "Confidence": a.confidence,
            "Title": a.title,
            "Year": a.year,
            "Journal": a.journal,
            "Authors": a.authors,
            "DOI": a.doi,
            "Source DB": a.source_db,
            "Abstract": a.abstract,
            "Keywords": a.keywords,
            "Reasoning": a.reasoning,
            "Model Used": a.model_used,
            "Manual Override": "Yes" if a.manual_override else "No",
            "Reviewer Notes": a.reviewer_notes,
            "Citations": a.citations,
            "Link": a.link,
            "Raw ID": a.raw_id,
            "References": getattr(a, "references", ""),
            "ID": a.id,
            "Source File": a.source_file,
            "Model Proposal": a.model_decision,
            "Reviewer Decision": a.reviewer_decision,
            "Second Reviewer Decision": a.second_reviewer_decision,
            "Consensus Decision": a.consensus_decision,
            "Review Priority": a.priority_score,
        })
    return pd.DataFrame(records)

def export_to_excel(articles: List[Article], summary_stats: Dict[str, Any] = None) -> bytes:
    """Exports articles and PRISMA summary to a multi-sheet Excel file in memory."""
    output = io.BytesIO()
    df_all = export_to_dataframe(articles)
    if df_all.empty:
        df_all = export_to_dataframe([Article()]).iloc[:0]
    df_included = df_all[df_all["Decision"] == "Include"]
    df_borderline = df_all[df_all["Decision"] == "Borderline"]
    df_excluded = df_all[df_all["Decision"] == "Exclude"]
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name="Publication Library", index=False)
        df_included.to_excel(writer, sheet_name="Included Articles", index=False)
        df_borderline.to_excel(writer, sheet_name="Borderline Articles", index=False)
        df_excluded.to_excel(writer, sheet_name="Excluded Articles", index=False)
        
        if summary_stats and "prisma" in summary_stats:
            prisma_dict = summary_stats["prisma"]
            prisma_records = [
                {"PRISMA Step": "Records identified from Web of Science", "Count": prisma_dict.get("identified_wos", 0)},
                {"PRISMA Step": "Records identified from Scopus", "Count": prisma_dict.get("identified_scopus", 0)},
                {"PRISMA Step": "Records identified from other sources", "Count": prisma_dict.get("identified_other", 0)},
                {"PRISMA Step": "Total records identified before deduplication", "Count": prisma_dict.get("identified_total", 0)},
                {"PRISMA Step": "Duplicate records removed", "Count": prisma_dict.get("duplicates_removed", 0)},
                {"PRISMA Step": "Unique records screened (Title & Abstract)", "Count": prisma_dict.get("records_screened", 0)},
                {"PRISMA Step": "Records excluded (irrelevant or out of scope)", "Count": prisma_dict.get("records_excluded", 0)},
                {"PRISMA Step": "Records borderline / uncertain", "Count": prisma_dict.get("records_borderline", 0)},
                {"PRISMA Step": "Records awaiting screening", "Count": prisma_dict.get("records_unscreened", 0)},
                {"PRISMA Step": "Records advancing to full-text review", "Count": prisma_dict.get("records_included", 0)},
            ]
            pd.DataFrame(prisma_records).to_excel(writer, sheet_name="PRISMA Summary", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"
            
    return output.getvalue()

def export_to_csv(articles: List[Article]) -> str:
    """Exports all articles to a CSV string."""
    df = export_to_dataframe(articles)
    return df.to_csv(index=False, encoding='utf-8-sig')

def generate_prisma_report(summary_stats: Dict[str, Any], research_question: str) -> str:
    """Generates a markdown formatted PRISMA 2020 screening statement."""
    p = summary_stats.get("prisma", {})
    return f"""# PRISMA 2020 Screening Report - sTRCview

**Research Question:** {research_question}

### 1. Identification
- **Records identified from Web of Science:** {p.get('identified_wos', 0)}
- **Records identified from Scopus:** {p.get('identified_scopus', 0)}
- **Records identified from other sources:** {p.get('identified_other', 0)}
- **Total records identified:** {p.get('identified_total', 0)}
- **Duplicate records removed:** {p.get('duplicates_removed', 0)}

### 2. Screening
- **Unique records screened (Title & Abstract):** {p.get('records_screened', 0)}
- **Records excluded based on Title & Abstract:** {p.get('records_excluded', 0)}
- **Borderline records flagged for manual review:** {p.get('records_borderline', 0)}
- **Records awaiting screening:** {p.get('records_unscreened', 0)}

### 3. Full-text queue
- **Records advancing to full-text review:** {p.get('records_included', 0)}
"""


def export_vosviewer_scopus_format(
    articles: List[Union[Article, Dict[str, Any]]],
    output_path: Optional[Union[str, Path]] = None,
) -> Union[str, Path]:
    """
    Exports screened articles into a standardized Scopus CSV format tailored for
    direct ingestion into VOSviewer and R Bibliometrix.

    Preserves critical bibliometric mapping fields:
      - Authors: Author list (delimited by semicolons)
      - Title: Article title
      - Year: Publication year
      - Source title: Journal / Conference name
      - Cited by: Total citations count
      - DOI: Clean DOI identifier
      - Link: URL to article
      - Affiliations: Author affiliations
      - Abstract: Full text abstract
      - Author Keywords & Index Keywords: Semicolon-delimited keywords
      - References: Cited reference list (CR) essential for co-citation and bibliographic coupling
      - EID: Standard Scopus Electronic Identifier

    Parameters:
        articles: List of Article objects or dictionaries
        output_path: Destination file path (.csv). If None, returns the CSV string.

    Returns:
        output_path if written to file, or CSV string if output_path is None.
    """
    records = []

    for art in articles:
        if isinstance(art, dict):
            art_id = str(art.get("id") or "")
            raw_id = str(art.get("raw_id") or "")
            title = str(art.get("title") or "")
            authors = str(art.get("authors") or "")
            year = art.get("year")
            journal = str(art.get("journal") or "")
            doi = str(art.get("doi") or "")
            link = str(art.get("link") or "")
            abstract = str(art.get("abstract") or "")
            keywords = str(art.get("keywords") or "")
            affiliations = str(art.get("affiliations") or "")
            citations = art.get("citations") or 0
            references = str(art.get("references") or art.get("cited_references") or "")
        else:
            art_id = getattr(art, "id", "")
            raw_id = getattr(art, "raw_id", "")
            title = getattr(art, "title", "")
            authors = getattr(art, "authors", "")
            year = getattr(art, "year", None)
            journal = getattr(art, "journal", "")
            doi = getattr(art, "doi", "")
            link = getattr(art, "link", "")
            abstract = getattr(art, "abstract", "")
            keywords = getattr(art, "keywords", "")
            affiliations = getattr(art, "affiliations", "")
            citations = getattr(art, "citations", 0)
            references = getattr(art, "references", "")

        # Format EID
        if raw_id:
            eid = raw_id if raw_id.startswith("2-s2.0-") else f"2-s2.0-{raw_id}"
        else:
            eid = f"2-s2.0-{art_id}" if art_id else ""

        # Format Link
        if not link and doi:
            link = f"https://doi.org/{doi}"

        # Clean keywords with semicolon delimiter for VOSviewer
        kw_tokens = [k.strip() for k in re.split(r'[;|,\n]', keywords) if k.strip()]
        clean_keywords = "; ".join(kw_tokens)

        # Clean references with semicolon delimiter
        ref_tokens = [r.strip() for r in re.split(r'[\r\n;]+', references) if r.strip()]
        clean_references = "; ".join(ref_tokens)

        # Standard Scopus CSV record columns expected by VOSviewer
        records.append({
            "Authors": authors,
            "Author(s) ID": "",
            "Title": title,
            "Year": year if year is not None else "",
            "Source title": journal,
            "Volume": "",
            "Issue": "",
            "Art. No.": "",
            "Page start": "",
            "Page end": "",
            "Page count": "",
            "Cited by": citations,
            "DOI": doi,
            "Link": link,
            "Affiliations": affiliations,
            "Authors with affiliations": affiliations or authors,
            "Abstract": abstract,
            "Author Keywords": clean_keywords,
            "Index Keywords": clean_keywords,
            "Molecular Sequence Numbers": "",
            "Chemicals/CAS": "",
            "Tradenames": "",
            "Manufacturers": "",
            "Funding Details": "",
            "Funding Text 1": "",
            "References": clean_references,
            "Correspondence Address": "",
            "Editors": "",
            "Publisher": "",
            "ISSN": "",
            "ISBN": "",
            "CODEN": "",
            "PubMed ID": "",
            "Language of Original Document": "English",
            "Abbreviated Source Title": journal,
            "Document Type": "Article",
            "Publication Stage": "Final",
            "Open Access": "All Open Access" if link else "",
            "Source": "Scopus",
            "EID": eid,
        })

    df = pd.DataFrame(records)

    if output_path is not None:
        path_obj = Path(output_path)
        if path_obj.parent:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_obj, index=False, encoding="utf-8-sig")
        return output_path
    else:
        return df.to_csv(index=False, encoding="utf-8-sig")


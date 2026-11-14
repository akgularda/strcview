from collections import Counter
from typing import List, Dict, Any
import re
from src.models.article import Article

def compute_bibliometric_summary(articles: List[Article], dedup_stats: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Computes summary metrics, PRISMA numbers, year trends, top journals,
    and keyword distributions for the screened articles.
    """
    if not articles:
        return {
            "total_articles": 0,
            "prisma": {},
            "years": {},
            "top_journals": {},
            "top_keywords": {},
            "decisions": {}
        }
        
    # Decisions breakdown
    decisions = Counter(a.decision for a in articles)
    
    # PRISMA flow numbers
    wos_initial = dedup_stats.get("wos_imported", 0) if dedup_stats else sum(1 for a in articles if "WoS" in a.source_db)
    scopus_initial = dedup_stats.get("scopus_imported", 0) if dedup_stats else sum(1 for a in articles if "Scopus" in a.source_db)
    dups_removed = dedup_stats.get("duplicates_removed", 0) if dedup_stats else 0
    total_records = len(articles)
    
    included_count = decisions.get("Include", 0)
    excluded_count = decisions.get("Exclude", 0)
    borderline_count = decisions.get("Borderline", 0)
    unscreened_count = decisions.get("Unscreened", 0)
    
    identified_total = (dedup_stats or {}).get("total_imported", total_records + dups_removed)
    prisma = {
        "identified_wos": wos_initial,
        "identified_scopus": scopus_initial,
        "identified_total": identified_total,
        "identified_other": max(0, identified_total - wos_initial - scopus_initial),
        "duplicates_removed": dups_removed,
        "records_screened": total_records - unscreened_count,
        "records_excluded": excluded_count,
        "records_borderline": borderline_count,
        "records_included": included_count,
        "records_unscreened": unscreened_count
    }
    
    # Year distribution
    years = Counter(a.year for a in articles if a.year is not None)
    sorted_years = dict(sorted(years.items()))
    
    # Top Journals
    journals = Counter(a.journal.strip() for a in articles if a.journal and a.journal.strip())
    top_journals = dict(journals.most_common(15))
    
    # Top Keywords
    kw_counter = Counter()
    for a in articles:
        if a.keywords:
            # Split by common delimiters (; or |)
            parts = re.split(r'[;|]', a.keywords)
            for p in parts:
                clean_p = p.strip().capitalize()
                if clean_p and len(clean_p) > 2:
                    kw_counter[clean_p] += 1
    top_keywords = dict(kw_counter.most_common(20))
    
    # Source DB overlap
    sources = Counter(a.source_db for a in articles)
    
    return {
        "total_articles": total_records,
        "prisma": prisma,
        "decisions": dict(decisions),
        "years": sorted_years,
        "top_journals": top_journals,
        "top_keywords": top_keywords,
        "sources": dict(sources)
    }

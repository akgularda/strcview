import math
import re
import unicodedata
from typing import List, Tuple, Dict, Any, Set
from collections import defaultdict
from itertools import combinations
from src.models.article import Article

def normalize_doi(doi: str) -> str:
    """Normalizes a DOI by removing URL schemes, prefixes and whitespace."""
    if not doi:
        return ""
    clean = doi.strip().lower()
    clean = re.sub(r'^doi:\s*', '', clean)
    clean = re.sub(r'^(?:https?://)?(?:dx\.)?doi\.org/', '', clean)
    return clean.strip()

def normalize_title(title: str) -> str:
    """Normalizes a title by lowercasing and keeping only alphanumeric characters."""
    if not title:
        return ""
    # Remove HTML tags if present
    clean = re.sub(r'<[^>]+>', '', unicodedata.normalize('NFKC', title).casefold())
    # Keep only alphanumeric characters and single spaces
    clean = re.sub(r'[^\w\s]|_', ' ', clean)
    return re.sub(r'\s+', ' ', clean).strip()

def title_similarity_jaccard(t1: str, t2: str) -> float:
    """Calculates Jaccard similarity of word sets between two normalized titles."""
    if not t1 or not t2:
        return 0.0
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def merge_two_articles(a: Article, b: Article) -> Article:
    """Merges two duplicate articles, retaining the richest metadata from both."""
    # Source DB
    if a.source_db != b.source_db:
        if ("wos" in a.source_db.lower() and "scopus" in b.source_db.lower()) or ("scopus" in a.source_db.lower() and "wos" in b.source_db.lower()):
            new_source = "WoS & Scopus"
        else:
            new_source = f"{a.source_db} & {b.source_db}"
    else:
        new_source = a.source_db

    # Title: prefer longer/cleaner title
    new_title = a.title if len(a.title) >= len(b.title) else b.title

    # Abstract: prefer longest abstract
    new_abstract = a.abstract if len(a.abstract) >= len(b.abstract) else b.abstract

    # DOI: prefer valid DOI
    new_doi = a.doi if a.doi else b.doi

    # Authors: prefer non-empty
    new_authors = a.authors if a.authors else b.authors

    # Year: prefer non-zero year
    new_year = a.year if (a.year and a.year > 0) else b.year

    # Journal: prefer non-empty
    new_journal = a.journal if a.journal else b.journal

    # Keywords: combine unique
    kws = set()
    if a.keywords:
        kws.update([k.strip() for k in a.keywords.split("|") if k.strip()])
    if b.keywords:
        kws.update([k.strip() for k in b.keywords.split("|") if k.strip()])
    new_keywords = " | ".join(sorted(kws)) if kws else (a.keywords or b.keywords)

    # Citations: max
    cit_a = getattr(a, "citations", 0) or 0
    cit_b = getattr(b, "citations", 0) or 0
    new_citations = max(cit_a, cit_b)

    # Link: prefer non-empty
    new_link = a.link or b.link

    # References
    refs_a = a.references or ""
    refs_b = b.references or ""
    new_refs = refs_a if len(refs_a) >= len(refs_b) else refs_b

    # Raw IDs
    raw_ids = []
    if a.raw_id:
        raw_ids.append(a.raw_id)
    if b.raw_id and b.raw_id != a.raw_id:
        raw_ids.append(b.raw_id)
    new_raw_id = "; ".join(raw_ids)

    reviewed = max((a, b), key=lambda article: (article.manual_override, article.decision != "Unscreened"))
    model = reviewed if reviewed.model_used else (b if reviewed is a else a)

    merged = Article(
        id=a.id,
        title=new_title,
        abstract=new_abstract,
        authors=new_authors,
        year=new_year,
        journal=new_journal,
        doi=new_doi,
        source_db=new_source,
        keywords=new_keywords,
        citations=new_citations,
        affiliations=a.affiliations or b.affiliations,
        link=new_link,
        references=new_refs,
        raw_id=new_raw_id,
        source_file=" | ".join(dict.fromkeys(filter(None, (a.source_file, b.source_file)))),
        decision=reviewed.decision,
        relevance_score=model.relevance_score,
        confidence=model.confidence,
        reasoning=model.reasoning,
        model_used=model.model_used,
        manual_override=reviewed.manual_override,
        reviewer_notes=reviewed.reviewer_notes,
        model_decision=model.model_decision,
        reviewer_decision=reviewed.reviewer_decision,
        second_reviewer_decision=reviewed.second_reviewer_decision or (b if reviewed is a else a).second_reviewer_decision,
        consensus_decision=reviewed.consensus_decision,
        priority_score=reviewed.priority_score,
    )
    return merged


def find_duplicate_candidates(articles: List[Article], title_threshold: float = 0.88) -> List[Dict[str, Any]]:
    """
    Scans a list of articles and identifies candidate duplicate pairs (DOI match or Title similarity).
    Returns a list of duplicate candidate dictionaries for interactive user adjudication.
    """
    candidates = []
    seen_pairs = set()
    by_doi = defaultdict(list)
    by_title = defaultdict(list)
    words = [set(normalize_title(article.title).split()) for article in articles]

    def add(i, j, match_type, detail, similarity):
        pair = (i, j)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            a, b = articles[i], articles[j]
            candidates.append({
                "candidate_id": f"{min(a.id, b.id)}:{max(a.id, b.id)}",
                "match_type": match_type,
                "match_detail": detail,
                "similarity": similarity,
                "article_a": a,
                "article_b": b,
                "default_action": "merge",
            })

    for index, article in enumerate(articles):
        doi = normalize_doi(article.doi)
        title = normalize_title(article.title)
        if doi:
            by_doi[doi].append(index)
        if title:
            by_title[title].append(index)

    for doi, indexes in by_doi.items():
        for i, j in combinations(indexes, 2):
            add(i, j, "Tam DOI Eşleşmesi", f"DOI: {doi}", 1.0)
    for indexes in by_title.values():
        for i, j in combinations(indexes, 2):
            add(i, j, "Tam Başlık Eşleşmesi", "Normalized titles match", 1.0)

    # Any sufficiently similar pair shares at least one word in these prefixes.
    # Sort by document frequency to avoid comparing common words across the corpus.
    frequencies = defaultdict(int)
    for terms in words:
        for term in terms:
            frequencies[term] += 1
    prefix_index = defaultdict(list)
    for j, terms in enumerate(words):
        if len(terms) < 3:
            continue
        prefix_size = len(terms) - math.ceil(title_threshold * len(terms)) + 1
        prefix = sorted(terms, key=lambda term: (frequencies[term], term))[:prefix_size]
        neighbors = {i for term in prefix for i in prefix_index[term]}
        for i in sorted(neighbors):
            if (i, j) in seen_pairs:
                continue
            other = words[i]
            if min(len(terms), len(other)) < title_threshold * max(len(terms), len(other)):
                continue
            similarity = len(terms & other) / len(terms | other)
            if similarity >= title_threshold:
                add(i, j, "Başlık Benzerliği", f"Jaccard similarity: {similarity:.1%}", similarity)
        for term in prefix:
            prefix_index[term].append(j)
    return candidates


def apply_duplicate_resolutions(
    articles: List[Article],
    decisions: Dict[str, str],
    candidates: List[Dict[str, Any]]
) -> Tuple[List[Article], Dict[str, Any]]:
    """
    Applies user-selected decisions ('merge', 'keep_a', 'keep_b', 'keep_both')
    to candidate duplicate pairs and returns the resulting deduplicated articles list.
    """
    to_remove_ids: Set[str] = set()
    merged_replacements: Dict[str, Article] = {}

    art_dict = {a.id: a for a in articles}

    merged_count = 0
    discarded_count = 0

    for cand in candidates:
        cid = cand["candidate_id"]
        action = decisions.get(cid, cand.get("default_action", "merge"))

        art_a_id = cand["article_a"].id
        art_b_id = cand["article_b"].id

        # Skip if already handled by a previous cluster
        if art_a_id in to_remove_ids or art_b_id in to_remove_ids:
            continue

        curr_a = merged_replacements.get(art_a_id, art_dict.get(art_a_id))
        curr_b = merged_replacements.get(art_b_id, art_dict.get(art_b_id))

        if not curr_a or not curr_b:
            continue

        if action == "merge":
            merged = merge_two_articles(curr_a, curr_b)
            merged_replacements[art_a_id] = merged
            to_remove_ids.add(art_b_id)
            merged_count += 1
        elif action == "keep_a":
            to_remove_ids.add(art_b_id)
            discarded_count += 1
        elif action == "keep_b":
            to_remove_ids.add(art_a_id)
            discarded_count += 1
        elif action == "keep_both":
            pass  # Do nothing, both stay in the cohort

    # Build final list
    final_articles: List[Article] = []
    for art in articles:
        if art.id in to_remove_ids:
            continue
        if art.id in merged_replacements:
            final_articles.append(merged_replacements[art.id])
        else:
            final_articles.append(art)

    stats = {
        "total_initial": len(articles),
        "final_count": len(final_articles),
        "merged_count": merged_count,
        "discarded_count": discarded_count,
        "total_removed": len(to_remove_ids)
    }
    return final_articles, stats


def deduplicate_articles(articles: List[Article], title_threshold: float = 0.90) -> Tuple[List[Article], Dict[str, Any]]:
    """
    Automated batch deduplication using DOI matching and normalized Title matching.
    """
    doi_map: Dict[str, Article] = {}
    title_list: List[Tuple[str, Article]] = []
    unique_articles: List[Article] = []
    
    duplicates_count = 0
    merged_records = 0
    
    for art in articles:
        norm_doi = normalize_doi(art.doi)
        norm_tit = normalize_title(art.title)
        
        matched_article: Article = None
        
        if norm_doi and norm_doi in doi_map:
            matched_article = doi_map[norm_doi]
        elif norm_tit:
            for ex_title, ex_art in title_list:
                if norm_tit == ex_title:
                    matched_article = ex_art
                    break
                elif len(norm_tit.split()) > 3 and title_similarity_jaccard(norm_tit, ex_title) >= title_threshold:
                    matched_article = ex_art
                    break
                    
        if matched_article:
            duplicates_count += 1
            merged_records += 1
            
            # Combine source DB
            if matched_article.source_db != art.source_db:
                matched_article.source_db = "WoS & Scopus"
                
            if len(art.abstract) > len(matched_article.abstract):
                matched_article.abstract = art.abstract
                
            if not matched_article.doi and art.doi:
                matched_article.doi = art.doi
                doi_map[normalize_doi(art.doi)] = matched_article
                
            if not matched_article.link and art.link:
                matched_article.link = art.link
                
            if art.keywords and art.keywords not in matched_article.keywords:
                if matched_article.keywords:
                    matched_article.keywords = f"{matched_article.keywords} | {art.keywords}"
                else:
                    matched_article.keywords = art.keywords
                    
            matched_article.citations = max(matched_article.citations, art.citations)
            
            if getattr(art, 'references', None):
                if not getattr(matched_article, 'references', None) or len(art.references) > len(matched_article.references):
                    matched_article.references = art.references
            
            if art.raw_id and art.raw_id not in matched_article.raw_id:
                matched_article.raw_id = f"{matched_article.raw_id}; {art.raw_id}".strip("; ")
        else:
            unique_articles.append(art)
            if norm_doi:
                doi_map[norm_doi] = art
            if norm_tit:
                title_list.append((norm_tit, art))
                
    wos_count = sum(1 for a in unique_articles if a.source_db == "WoS")
    scopus_count = sum(1 for a in unique_articles if a.source_db == "Scopus")
    both_count = sum(1 for a in unique_articles if a.source_db == "WoS & Scopus")
    
    stats = {
        "total_imported": len(articles),
        "unique_count": len(unique_articles),
        "duplicates_removed": duplicates_count,
        "wos_only": wos_count,
        "scopus_only": scopus_count,
        "in_both_databases": both_count
    }
    
    return unique_articles, stats

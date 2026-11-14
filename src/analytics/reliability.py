"""
Inter-Rater Reliability and Screening Agreement Module for sTRCview.
Computes Cohen's Kappa, confusion matrices, and identifies screening conflicts
between independent human reviewers or human vs. AI evaluators.
"""

from typing import List, Dict, Any, Union, Iterable, Optional
from collections import defaultdict, Counter


def calculate_cohens_kappa(
    reviewer1_decisions: Iterable[Any],
    reviewer2_decisions: Iterable[Any]
) -> Dict[str, Any]:
    """
    Calculates Cohen's Kappa coefficient (κ) for inter-rater reliability.

    Parameters:
        reviewer1_decisions: List or iterable of decisions from Reviewer 1 (e.g. ['Include', 'Exclude', ...])
        reviewer2_decisions: List or iterable of decisions from Reviewer 2

    Returns:
        dict containing:
          - kappa: float (-1.0 to 1.0)
          - observed_agreement: float (0.0 to 1.0)
          - expected_agreement: float (0.0 to 1.0)
          - interpretation: str (Landis & Koch 1977 benchmark)
          - confusion_matrix: dict with categories, 2D matrix, and nested dict table
          - n_total: int
          - n_agreements: int
          - n_disagreements: int
    """
    r1 = list(reviewer1_decisions)
    r2 = list(reviewer2_decisions)

    if len(r1) != len(r2):
        raise ValueError(
            f"Reviewer decisions length mismatch: len(reviewer1)={len(r1)}, len(reviewer2)={len(r2)}"
        )

    n_total = len(r1)
    if n_total == 0:
        return {
            "kappa": 0.0,
            "observed_agreement": 0.0,
            "expected_agreement": 0.0,
            "interpretation": "No data",
            "confusion_matrix": {"categories": [], "matrix": [], "table": {}},
            "n_total": 0,
            "n_agreements": 0,
            "n_disagreements": 0,
        }

    # Normalize categories to string representation
    r1_norm = [str(x).strip() for x in r1]
    r2_norm = [str(x).strip() for x in r2]

    # Collect unique categories, preserving standard order if present
    preferred_order = ["Include", "Borderline", "Exclude", "Unscreened"]
    seen_cats = set(r1_norm).union(set(r2_norm))
    categories = [c for c in preferred_order if c in seen_cats]
    for c in sorted(seen_cats):
        if c not in categories:
            categories.append(c)

    # Count occurrences
    matrix_table: Dict[str, Dict[str, int]] = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    n_agreements = 0

    for d1, d2 in zip(r1_norm, r2_norm):
        matrix_table[d1][d2] += 1
        if d1 == d2:
            n_agreements += 1

    n_disagreements = n_total - n_agreements
    observed_agreement = n_agreements / n_total

    # Marginal probabilities
    r1_counts = Counter(r1_norm)
    r2_counts = Counter(r2_norm)

    expected_agreement = sum(
        (r1_counts[cat] / n_total) * (r2_counts[cat] / n_total)
        for cat in categories
    )

    # Calculate Cohen's Kappa
    if expected_agreement >= 1.0:
        # Both raters always assign the single same category
        kappa = 1.0 if observed_agreement >= 1.0 else 0.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    # Round kappa to 4 decimals
    kappa = round(float(kappa), 4)
    observed_agreement = round(float(observed_agreement), 4)
    expected_agreement = round(float(expected_agreement), 4)

    # Landis & Koch (1977) scale
    if kappa < 0.0:
        interpretation = "Poor agreement (< 0.00)"
    elif kappa <= 0.20:
        interpretation = "Slight agreement (0.00 - 0.20)"
    elif kappa <= 0.40:
        interpretation = "Fair agreement (0.21 - 0.40)"
    elif kappa <= 0.60:
        interpretation = "Moderate agreement (0.41 - 0.60)"
    elif kappa <= 0.80:
        interpretation = "Substantial agreement (0.61 - 0.80)"
    else:
        interpretation = "Almost Perfect agreement (0.81 - 1.00)"

    # Form 2D matrix
    matrix_2d = [[matrix_table[c1][c2] for c2 in categories] for c1 in categories]

    return {
        "kappa": kappa,
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "interpretation": interpretation,
        "confusion_matrix": {
            "categories": categories,
            "matrix": matrix_2d,
            "table": matrix_table,
        },
        "n_total": n_total,
        "n_agreements": n_agreements,
        "n_disagreements": n_disagreements,
    }


def identify_conflicts(articles_with_two_screeners: List[Any]) -> List[Dict[str, Any]]:
    """
    Identifies discrepancies/conflicts between two screeners.

    Parameters:
        articles_with_two_screeners: List of items representing dual-screened records.
        Accepted formats for each item:
          1. Dict with keys: 'id'/'article_id', 'title', and ('reviewer1'/'decision1' and 'reviewer2'/'decision2')
          2. Tuple: (id, title, decision1, decision2) or (article_obj, decision1, decision2)
          3. Article object with 'reviewer1_decision' and 'reviewer2_decision' attributes

    Returns:
        List of conflict dictionaries with keys:
          - id: str
          - title: str
          - reviewer1_decision: str
          - reviewer2_decision: str
          - conflict_type: str (e.g. 'Include vs Exclude')
          - requires_adjudication: bool (True)
          - article: original object or None
    """
    conflicts: List[Dict[str, Any]] = []

    for item in articles_with_two_screeners:
        art_id = ""
        title = ""
        d1 = ""
        d2 = ""
        raw_article = None

        if isinstance(item, dict):
            art_id = str(item.get("id") or item.get("article_id") or "")
            title = str(item.get("title") or "")
            d1 = str(item.get("reviewer1") or item.get("reviewer1_decision") or item.get("decision1") or item.get("r1") or "")
            d2 = str(item.get("reviewer2") or item.get("reviewer2_decision") or item.get("decision2") or item.get("r2") or "")
            raw_article = item.get("article")
        elif isinstance(item, (tuple, list)):
            if len(item) == 4:
                art_id = str(item[0])
                title = str(item[1])
                d1 = str(item[2])
                d2 = str(item[3])
            elif len(item) == 3:
                first = item[0]
                art_id = getattr(first, "id", str(first))
                title = getattr(first, "title", "")
                d1 = str(item[1])
                d2 = str(item[2])
                raw_article = first
        else:
            # Object with attributes
            art_id = getattr(item, "id", "")
            title = getattr(item, "title", "")
            d1 = getattr(item, "reviewer1_decision", getattr(item, "decision1", getattr(item, "decision", "")))
            d2 = getattr(item, "reviewer2_decision", getattr(item, "decision2", ""))
            raw_article = item

        d1_clean = d1.strip()
        d2_clean = d2.strip()

        # If both reviewers provided decisions and they disagree
        if d1_clean and d2_clean and d1_clean.lower() != d2_clean.lower():
            conflicts.append({
                "id": art_id,
                "title": title,
                "reviewer1_decision": d1_clean,
                "reviewer2_decision": d2_clean,
                "conflict_type": f"{d1_clean} vs {d2_clean}",
                "requires_adjudication": True,
                "article": raw_article,
            })

    return conflicts

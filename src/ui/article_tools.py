import html
import re
from urllib.parse import quote, urlencode, urlsplit

import requests


def resolve_unpaywall_doi(doi: str, email: str = "") -> dict:
    """Build a DOI link and optionally look up its open-access locations."""
    clean_doi = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", doi.strip(), flags=re.I)
    if not re.fullmatch(r"10\.\d{4,9}/\S+", clean_doi):
        return {"has_doi": False, "is_oa": False, "pdf_url": None, "doi_url": "", "api_url": ""}
    path = quote(clean_doi, safe="/")
    api_url = f"https://api.unpaywall.org/v2/{path}?{urlencode({'email': email})}"
    result = {
        "has_doi": True, "clean_doi": clean_doi, "doi_url": f"https://doi.org/{path}",
        "api_url": api_url, "is_oa": False, "pdf_url": None, "oa_status": "unknown",
    }
    if not email:
        return result
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        location = data.get("best_oa_location") or {}
        result.update(is_oa=bool(data.get("is_oa")), oa_status=data.get("oa_status", "unknown"))
        for key, url in (("pdf_url", location.get("url_for_pdf")), ("landing_url", location.get("url"))):
            if url and urlsplit(url).scheme in {"http", "https"}:
                result[key] = url
    except (requests.RequestException, ValueError, AttributeError, TypeError):
        result["error"] = True
    return result


def highlight_abstract_keywords(text: str, inclusion_kw: list[str], exclusion_kw: list[str]) -> str:
    """
    Highlights inclusion keywords in soft green and exclusion keywords in soft red.
    Preserves exact HTML escaping and non-overlapping tokenization.
    """
    if not text:
        return ""

    clean_inc = [k.strip() for k in inclusion_kw if k and len(k.strip()) >= 2]
    clean_exc = [k.strip() for k in exclusion_kw if k and len(k.strip()) >= 2]

    sorted_inc = sorted(list(set(clean_inc)), key=len, reverse=True)
    sorted_exc = sorted(list(set(clean_exc)), key=len, reverse=True)

    matches = []
    for kw in sorted_inc:
        pat = re.compile(r'\b' + re.escape(kw) + r'\b' if kw.replace(" ", "").isalnum() else re.escape(kw), re.IGNORECASE)
        for m in pat.finditer(text):
            matches.append((m.start(), m.end(), "inclusion"))

    for kw in sorted_exc:
        pat = re.compile(r'\b' + re.escape(kw) + r'\b' if kw.replace(" ", "").isalnum() else re.escape(kw), re.IGNORECASE)
        for m in pat.finditer(text):
            matches.append((m.start(), m.end(), "exclusion"))

    if not matches:
        return html.escape(text)

    # Sort matches by start position ascending, length descending
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Filter overlaps
    filtered = []
    curr_end = 0
    for s, e, t in matches:
        if s >= curr_end:
            filtered.append((s, e, t))
            curr_end = e

    out = []
    last_idx = 0
    for s, e, t in filtered:
        out.append(html.escape(text[last_idx:s]))
        matched_str = html.escape(text[s:e])
        if t == "inclusion":
            out.append(
                f"<mark style='background-color: #d1fae5; color: #065f46; padding: 2px 5px; border-radius: 4px; font-weight: 600;'>{matched_str}</mark>"
            )
        else:
            out.append(
                f"<mark style='background-color: #fee2e2; color: #991b1b; padding: 2px 5px; border-radius: 4px; font-weight: 600;'>{matched_str}</mark>"
            )
        last_idx = e
    out.append(html.escape(text[last_idx:]))
    return "".join(out)



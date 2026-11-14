"""
PRISMA 2020 Flow Diagram Generator for sTRCview.
Produces publication-ready vector SVG flowcharts representing the systematic review screening process.
"""

from typing import Dict, Any


def generate_prisma_2020_svg(prisma_data: Dict[str, Any]) -> str:
    """
    Generates a clean, publication-ready PRISMA 2020 SVG flowchart.

    Expected prisma_data keys (with sensible fallbacks):
      - identified_wos: int
      - identified_scopus: int
      - identified_total: int
      - duplicates_removed: int
      - records_screened: int
      - records_excluded: int
      - records_borderline: int (optional)
      - records_included: int
      - records_unscreened: int (optional)
    """
    def _val(*keys, default=0) -> int:
        for k in keys:
            if k in prisma_data and prisma_data[k] is not None:
                try:
                    return int(prisma_data[k])
                except (ValueError, TypeError):
                    continue
        return default

    wos = _val("identified_wos", "wos", "wos_imported")
    scopus = _val("identified_scopus", "scopus", "scopus_imported")
    other = _val("identified_other")
    total = _val("identified_total", "total", "total_imported")
    if total == 0 and (wos > 0 or scopus > 0):
        total = wos + scopus

    duplicates = _val("duplicates_removed", "duplicates", "duplicates_count")
    screened = _val("records_screened", "screened", "unique_count")
    if not any(key in prisma_data for key in ("records_screened", "screened", "unique_count")) and total > 0:
        screened = max(0, total - duplicates)

    excluded = _val("records_excluded", "excluded", "excluded_count")
    borderline = _val("records_borderline", "borderline", "borderline_count")
    included = _val("records_included", "included", "included_count")
    unscreened = _val("records_unscreened", "unscreened", "unscreened_count")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 870 510" width="870" height="510" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #ffffff;">
  <defs>
    <filter id="prisma-shadow" x="-3%" y="-5%" width="106%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.07" />
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 8 5 L 0 9 z" fill="#475569" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 8 5 L 0 9 z" fill="#dc2626" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 8 5 L 0 9 z" fill="#16a34a" />
    </marker>
  </defs>

  <!-- Title Header -->
  <text x="435" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#0f172a" letter-spacing="-0.3">PRISMA 2020 Flow Diagram · Title and Abstract Screening</text>
  <text x="435" y="44" text-anchor="middle" font-size="11" fill="#64748b">sTRCview · Screening-stage counts</text>

  <!-- ================= PHASE LABELS (LEFT PILLAR) ================= -->
  <!-- Identification Phase Badge -->
  <g transform="translate(20, 60)">
    <rect width="115" height="110" rx="8" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.5" />
    <text x="57" y="60" text-anchor="middle" font-size="12" font-weight="700" fill="#0369a1" letter-spacing="0.5">IDENTIFICATION</text>
  </g>

  <!-- Screening Phase Badge -->
  <g transform="translate(20, 200)">
    <rect width="115" height="120" rx="8" fill="#fffbeb" stroke="#fde68a" stroke-width="1.5" />
    <text x="57" y="65" text-anchor="middle" font-size="12" font-weight="700" fill="#b45309" letter-spacing="0.5">SCREENING</text>
  </g>

  <!-- Included Phase Badge -->
  <g transform="translate(20, 350)">
    <rect width="115" height="115" rx="8" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1.5" />
    <text x="57" y="63" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d" letter-spacing="0.5">INCLUDED</text>
  </g>

  <!-- ================= FLOW BOXES ================= -->

  <!-- Box 1: Identification (Databases) -->
  <g transform="translate(160, 60)" filter="url(#prisma-shadow)">
    <rect width="330" height="110" rx="8" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5" />
    <rect width="330" height="28" rx="8" fill="#f1f5f9" />
    <path d="M 0 20 h 330 v 8 h -330 z" fill="#f1f5f9" />
    <text x="165" y="19" text-anchor="middle" font-size="12" font-weight="700" fill="#1e293b">Records identified</text>
    <text x="20" y="43" font-size="12" fill="#334155">Web of Science: <tspan font-weight="600">n = {wos:,}</tspan></text>
    <text x="20" y="59" font-size="12" fill="#334155">Scopus: <tspan font-weight="600">n = {scopus:,}</tspan></text>
    <text x="20" y="75" font-size="12" fill="#334155">Other sources: <tspan font-weight="600">n = {other:,}</tspan></text>
    <text x="20" y="97" font-size="12" font-weight="700" fill="#0f172a">Total: <tspan fill="#0284c7">n = {total:,}</tspan></text>
  </g>

  <!-- Box 2: Deduplication (Removed before screening) -->
  <g transform="translate(530, 60)" filter="url(#prisma-shadow)">
    <rect width="310" height="110" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <rect width="310" height="28" rx="8" fill="#fff7ed" />
    <path d="M 0 20 h 310 v 8 h -310 z" fill="#fff7ed" />
    <text x="155" y="19" text-anchor="middle" font-size="12" font-weight="700" fill="#9a3412">Records removed before screening</text>
    <text x="20" y="55" font-size="12" fill="#475569">Duplicate records removed:</text>
    <text x="20" y="80" font-size="16" font-weight="700" fill="#ea580c">n = {duplicates:,}</text>
    <text x="20" y="98" font-size="10" fill="#94a3b8">Matched via normalized DOI &amp; Title</text>
  </g>

  <!-- Box 3: Screening -->
  <g transform="translate(160, 200)" filter="url(#prisma-shadow)">
    <rect width="330" height="120" rx="8" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5" />
    <rect width="330" height="28" rx="8" fill="#f8fafc" />
    <path d="M 0 20 h 330 v 8 h -330 z" fill="#f8fafc" />
    <text x="165" y="19" text-anchor="middle" font-size="12" font-weight="700" fill="#1e293b">Records screened (Title &amp; Abstract)</text>
    <text x="20" y="55" font-size="12" fill="#475569">Unique records evaluated:</text>
    <text x="20" y="80" font-size="18" font-weight="700" fill="#0284c7">n = {screened:,}</text>
    <text x="20" y="102" font-size="11" fill="#64748b">Awaiting screening: <tspan font-weight="600">n = {unscreened:,}</tspan></text>
  </g>

  <!-- Box 4: Excluded -->
  <g transform="translate(530, 200)" filter="url(#prisma-shadow)">
    <rect width="310" height="120" rx="8" fill="#ffffff" stroke="#fecaca" stroke-width="1.5" />
    <rect width="310" height="28" rx="8" fill="#fef2f2" />
    <path d="M 0 20 h 310 v 8 h -310 z" fill="#fef2f2" />
    <text x="155" y="19" text-anchor="middle" font-size="12" font-weight="700" fill="#991b1b">Records excluded</text>
    <text x="20" y="52" font-size="12" fill="#475569">Irrelevant or out of scope:</text>
    <text x="20" y="75" font-size="16" font-weight="700" fill="#dc2626">n = {excluded:,}</text>
  </g>

  <!-- Box 5: Included -->
  <g transform="translate(160, 350)" filter="url(#prisma-shadow)">
    <rect width="330" height="115" rx="8" fill="#ffffff" stroke="#86efac" stroke-width="2" />
    <rect width="330" height="28" rx="8" fill="#f0fdf4" />
    <path d="M 0 20 h 330 v 8 h -330 z" fill="#f0fdf4" />
    <text x="165" y="19" text-anchor="middle" font-size="12" font-weight="700" fill="#14532d">Included at screening stage</text>
    <text x="20" y="52" font-size="12" fill="#475569">Advance to full-text review:</text>
    <text x="20" y="78" font-size="20" font-weight="700" fill="#16a34a">n = {included:,}</text>
    <text x="20" y="100" font-size="11" fill="#64748b">Borderline decisions: <tspan font-weight="600">n = {borderline:,}</tspan></text>
  </g>

  <!-- ================= CONNECTING ARROWS ================= -->
  <!-- Box 1 -> Box 2 (Horizontal: Identification to Duplicates removed) -->
  <line x1="490" y1="115" x2="522" y2="115" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Box 1 -> Box 3 (Vertical: Identification to Screened) -->
  <line x1="325" y1="170" x2="325" y2="192" stroke="#475569" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Box 3 -> Box 4 (Horizontal: Screened to Excluded) -->
  <line x1="490" y1="260" x2="522" y2="260" stroke="#dc2626" stroke-width="2" marker-end="url(#arrow-red)" />

  <!-- Box 3 -> Box 5 (Vertical: Screened to Included) -->
  <line x1="325" y1="320" x2="325" y2="342" stroke="#16a34a" stroke-width="2.5" marker-end="url(#arrow-green)" />
</svg>"""
    return svg

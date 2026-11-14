"""
Keyword Co-occurrence Network Visualization Module for sTRCview.
Builds interactive Plotly network graphs (similar to VOSviewer) showing
top thematic keywords and their co-occurrence relationships.
"""

from typing import List, Dict, Any, Union, Tuple, Set
import re
from collections import Counter
import plotly.graph_objects as go

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    import math


def _extract_keywords_from_article(article: Any) -> List[str]:
    """Extracts, cleans, and normalizes keywords from an Article or dict."""
    raw_kw = ""
    if isinstance(article, dict):
        raw_kw = article.get("keywords") or article.get("author_keywords") or article.get("Author Keywords") or ""
    else:
        raw_kw = getattr(article, "keywords", "") or ""

    if not raw_kw:
        return []

    # Split by standard delimiters (; or | or comma)
    tokens = re.split(r'[;|,\n]', str(raw_kw))
    clean_kws = []
    seen = set()

    # Acronyms to keep uppercase
    known_acronyms = {
        "AI", "ML", "DL", "ANN", "CNN", "RNN", "LSTM", "SVM", "EEW", "GPS", "GNSS",
        "P-WAVE", "S-WAVE", "NLP", "LLM", "BERT", "PCA", "IoT", "PGA", "PGV", "GIS"
    }

    for t in tokens:
        item = t.strip()
        # Clean extra punctuation
        item = re.sub(r'^[\s"\'\-]+|[\s"\'\-]+$', '', item)
        if len(item) < 2:
            continue

        upper_item = item.upper()
        if upper_item in known_acronyms:
            normalized = upper_item
        else:
            normalized = item.title()

        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            clean_kws.append(normalized)

    return clean_kws


def build_keyword_cooccurrence_network(
    articles: List[Any],
    top_n: int = 30,
    min_cooccurrence: int = 2
) -> go.Figure:
    """
    Constructs an interactive Plotly co-occurrence network graph.

    Parameters:
        articles: List of Article objects or dictionaries
        top_n: Maximum number of high-frequency keywords to display (default: 30)
        min_cooccurrence: Minimum shared occurrences required to draw an edge (default: 2)

    Returns:
        go.Figure: Interactive network graph with sized nodes and weighted edges
    """
    fig = go.Figure()

    if not articles:
        fig.add_annotation(
            text="No articles provided for keyword network",
            showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            title="Keyword Co-occurrence Network"
        )
        return fig

    # 1. Extract keywords per article
    article_keywords: List[List[str]] = []
    global_kw_counter = Counter()

    for art in articles:
        kws = _extract_keywords_from_article(art)
        if kws:
            article_keywords.append(kws)
            for kw in kws:
                global_kw_counter[kw] += 1

    if not global_kw_counter:
        fig.add_annotation(
            text="No keywords found in the provided articles",
            showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            title="Keyword Co-occurrence Network"
        )
        return fig

    # 2. Select Top N keywords
    most_common = global_kw_counter.most_common(top_n)
    top_kws = [kw for kw, _ in most_common]
    top_kws_set = set(top_kws)
    kw_freq = {kw: count for kw, count in most_common}

    # 3. Calculate co-occurrences among top keywords
    edge_counts: Dict[Tuple[str, str], int] = Counter()
    for kws in article_keywords:
        filtered = sorted([kw for kw in kws if kw in top_kws_set])
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                pair = (filtered[i], filtered[j])
                edge_counts[pair] += 1

    # Filter edges by min_cooccurrence
    active_edges = {
        pair: weight for pair, weight in edge_counts.items()
        if weight >= min_cooccurrence
    }

    # If min_cooccurrence was too strict and eliminated all edges, fall back gracefully
    if not active_edges and edge_counts:
        max_seen = max(edge_counts.values())
        if max_seen < min_cooccurrence:
            active_edges = {pair: weight for pair, weight in edge_counts.items() if weight >= 1}

    # 4. Graph Layout calculation
    pos: Dict[str, Tuple[float, float]] = {}
    node_degrees: Dict[str, int] = {kw: 0 for kw in top_kws}

    for (u, v), w in active_edges.items():
        node_degrees[u] = node_degrees.get(u, 0) + 1
        node_degrees[v] = node_degrees.get(v, 0) + 1

    if HAS_NETWORKX:
        G = nx.Graph()
        for kw in top_kws:
            G.add_node(kw, weight=kw_freq.get(kw, 1))
        for (u, v), weight in active_edges.items():
            G.add_edge(u, v, weight=weight)

        # Spring layout with seed for stability
        pos = nx.spring_layout(G, k=0.55, seed=42, iterations=60)
    else:
        # Circular fallback if networkx is missing
        n_nodes = len(top_kws)
        for idx, kw in enumerate(top_kws):
            angle = 2 * math.pi * idx / max(1, n_nodes)
            pos[kw] = (math.cos(angle), math.sin(angle))

    # 5. Build Edge Traces (Grouped by weight for varied line thickness)
    max_weight = max(active_edges.values()) if active_edges else 1

    # We add edges with hover info and width proportional to weight
    for (u, v), weight in sorted(active_edges.items(), key=lambda x: x[1]):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Normalized line width (between 1.0 and 5.0)
        line_width = 1.0 + 4.0 * (weight / max(1, max_weight))
        alpha = 0.35 + 0.55 * (weight / max(1, max_weight))

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=line_width, color=f"rgba(100, 116, 139, {alpha:.2f})"),
            hoverinfo="text",
            hovertext=f"<b>Co-occurrence:</b> {u} — {v}<br><b>Shared Articles:</b> {weight}",
            showlegend=False
        ))

    # 6. Build Node Trace
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_size = []
    node_color = []

    min_freq = min(kw_freq.values()) if kw_freq else 1
    max_freq = max(kw_freq.values()) if kw_freq else 1
    freq_range = max(1, max_freq - min_freq)

    for kw in top_kws:
        if kw not in pos:
            continue
        x, y = pos[kw]
        node_x.append(x)
        node_y.append(y)
        node_text.append(kw)

        freq = kw_freq.get(kw, 1)
        deg = node_degrees.get(kw, 0)
        # Scaled marker size between 14 and 42
        scaled_size = 14 + 28 * ((freq - min_freq) / freq_range)
        node_size.append(scaled_size)
        node_color.append(deg)

        node_hover.append(
            f"<b>{kw}</b><br>"
            f"• Article Frequency: {freq}<br>"
            f"• Co-occurrence Degree: {deg} connections"
        )

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Tealgrn",
            showscale=True,
            colorbar=dict(
                title=dict(text="Network<br>Degree", font=dict(size=11, color="#334155")),
                thickness=12,
                len=0.7,
                x=1.02
            ),
            line=dict(color="#ffffff", width=2),
        ),
        textfont=dict(size=10, color="#1e293b", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
        showlegend=False
    ))

    fig.update_layout(
        title=dict(
            text=f"Thematic Keyword Co-occurrence Network (Top {len(top_kws)} Keywords, Min Co-occurrence = {min_cooccurrence})",
            font=dict(size=14, color="#0f172a"),
            x=0.03
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        margin=dict(l=20, r=40, t=50, b=20),
        height=580,
    )

    return fig

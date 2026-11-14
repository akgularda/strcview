"""Publication review, independent decisions and conflict resolution."""

import html
import os
import re
from collections.abc import Callable

import pandas as pd
import streamlit as st

from src.analytics.reliability import calculate_cohens_kappa
from src.models.article import Article
from src.parsers.deduplicator import normalize_doi
from src.ui.article_tools import highlight_abstract_keywords, resolve_unpaywall_doi
from src.ui.session import commit
from src.workspace import DECISIONS, Workspace


DECISION_LABELS = {"Include": "İlgili", "Borderline": "Kararsız", "Exclude": "İlgisiz", "Unscreened": "Taranmadı"}


def render_article(article: Article, *, show_highlighter: bool = True) -> None:
    meta = " · ".join(filter(None, (article.journal, str(article.year or ""), article.source_db)))
    if meta:
        st.caption(meta)
    st.markdown(f'<h3 class="publication-title">{html.escape(article.title)}</h3>', unsafe_allow_html=True)
    if article.authors:
        st.text(article.authors)
    doi = resolve_unpaywall_doi(article.doi)
    if doi["has_doi"]:
        st.link_button("DOI", doi["doi_url"], icon=":material/open_in_new:")
        email = os.getenv("STRCVIEW_UNPAYWALL_EMAIL", "")
        if email:
            if st.button("Look up open access"):
                st.session_state.oa_lookup = (article.doi, resolve_unpaywall_doi(article.doi, email))
            saved_doi, result = st.session_state.get("oa_lookup", (None, {}))
            if saved_doi == article.doi:
                if result.get("pdf_url"):
                    st.link_button("Open-access PDF", result["pdf_url"])
                elif result.get("landing_url"):
                    st.link_button("Open-access page", result["landing_url"])
                else:
                    st.caption("Lookup unavailable. Try again." if result.get("error") else "No open-access location returned.")
    terms_in = terms_out = ""
    if show_highlighter:
        with st.expander("Highlight terms"):
            left, right = st.columns(2)
            terms_in = left.text_input("Inclusion terms", help="Separate terms with commas.", key="highlight_include")
            terms_out = right.text_input("Exclusion terms", help="Separate terms with commas.", key="highlight_exclude")
    highlighted = highlight_abstract_keywords(
        article.abstract or "Özet bulunamadı.", re.split(r"[,;]", terms_in), re.split(r"[,;]", terms_out),
    )
    st.markdown(f'<div class="publication-abstract"><b>Özet</b><p>{highlighted}</p></div>', unsafe_allow_html=True)
    if article.keywords:
        st.caption(f"Anahtar kelimeler · {article.keywords}")


def render_review(workspace: Workspace, update_ranking: Callable[[Workspace], None]) -> None:
    st.subheader("Yayın ara ve karar düzenle")
    if not workspace.articles:
        st.info("İncelemeye başlamak için yayın ekle.")
        return
    blind = st.toggle("Kör inceleme", help="Model önerisini, notları ve mevcut kararı gizler.")
    search, status, sort = st.columns([2, 1, 1])
    query = search.text_input("Ara", placeholder="Başlık, özet, yazar veya DOI").strip().casefold()
    filters = ["All records", "Needs human review", "Reviewed", "Deferred"]
    sort_options = ["Import order", "Review priority"]
    if not blind:
        filters.extend(["Include", "Exclude", "Unscreened"])
        sort_options.append("Model score")
    filter_by = status.selectbox("Durum", filters, format_func=lambda value: {
        "All records": "Tüm yayınlar", "Needs human review": "İnceleme bekleyen",
        "Reviewed": "İncelenen", "Deferred": "Kararsız",
    }.get(value, DECISION_LABELS.get(value, value)))
    sort_labels = {
        "Import order": "Yükleme sırası", "Review priority": "İnceleme önceliği", "Model score": "Model puanı",
    }
    order_by = sort.selectbox("Sırala", sort_options, format_func=sort_labels.__getitem__)
    matches = [
        a for a in workspace.articles
        if query in " ".join((a.title, a.abstract, a.authors, a.doi)).casefold()
        and (filter_by == "All records" or filter_by == a.decision
             or (filter_by == "Needs human review" and not a.manual_override)
             or (filter_by == "Reviewed" and a.manual_override)
             or (filter_by == "Deferred" and a.manual_override and a.decision == "Borderline"))
    ]
    if order_by == "Review priority":
        matches.sort(key=lambda a: a.priority_score if a.priority_score is not None else -1, reverse=True)
    elif order_by == "Model score":
        matches.sort(key=lambda a: a.relevance_score if a.model_decision else -1, reverse=True)
    if not matches:
        st.info("Filtrelere uyan yayın bulunamadı.")
        return

    st.caption(f"{len(matches):,} eşleşen yayın")
    by_id = {article.id: article for article in matches}
    ids = list(by_id)
    active = st.session_state.get("review_record")
    if active not in by_id:
        active = ids[0]
    if active != st.session_state.get("review_record"):
        st.session_state.review_record = active
    selected = st.selectbox("Yayın", ids, format_func=lambda id: by_id[id].title, key="review_record")
    article = by_id[selected]

    with st.container(border=True, key="article_reader"):
        render_article(article)

    with st.container(border=True, key="review_decision_panel"):
        st.subheader("Hakem kararı")
        if not blind:
            if article.manual_override:
                st.caption(f"Kayıtlı hakem kararı · {DECISION_LABELS[article.decision]}")
            if article.model_decision:
                st.markdown(f"**Model önerisi** · {DECISION_LABELS.get(article.model_decision, article.model_decision)} · {article.relevance_score:.0%} · {article.model_used}")
                st.write(article.reasoning)
        with st.form(f"review_{article.id}_{blind}"):
            notes = st.text_area("İnceleme notu", value="" if blind else article.reviewer_notes)
            for column, decision in zip(st.columns(3), DECISIONS):
                label = DECISION_LABELS[decision]
                if column.form_submit_button(label, width="stretch", type="primary" if decision == "Include" else "secondary"):
                    workspace.review(article, decision, notes.strip())
                    update_ranking(workspace)
                    if not commit(workspace):
                        return
                    st.session_state.notice = f"Hakem kararı kaydedildi · {label}."
                    st.rerun()


def match_reviewer_file(upload, workspace: Workspace) -> dict[str, str]:
    frame = pd.read_csv(upload, dtype=str, keep_default_na=False)
    frame.columns = [str(col).strip().casefold() for col in frame.columns]
    decision_col = next((col for col in ("decision", "reviewer2", "karar") if col in frame), None)
    id_col = next((col for col in ("id", "article_id", "doi") if col in frame), None)
    if not id_col or not decision_col:
        raise ValueError("CSV requires a Decision column and an ID or DOI column.")
    by_id = {a.id.casefold(): a.id for a in workspace.articles}
    by_doi = {}
    for article in workspace.articles:
        doi = normalize_doi(article.doi)
        if doi:
            by_doi[doi] = None if doi in by_doi else article.id
    decisions = {}
    for key, value in frame[[id_col, decision_col]].itertuples(index=False, name=None):
        record_id = by_doi.get(normalize_doi(key)) if id_col == "doi" else by_id.get(key.strip().casefold())
        decision = value.strip().capitalize()
        if decision and decision not in DECISIONS:
            raise ValueError(f"Invalid reviewer decision: {value}. Use Include, Borderline or Exclude.")
        if record_id and decision:
            if record_id in decisions and decisions[record_id] != decision:
                raise ValueError("The reviewer CSV contains conflicting decisions for the same record.")
            decisions[record_id] = decision
    return decisions


def render_agreement(workspace: Workspace, update_ranking: Callable[[Workspace], None]) -> None:
    st.subheader("Reviewer agreement")
    mode = st.radio("Compare reviewer decisions with", ["Model proposals", "Second reviewer CSV"], horizontal=True)
    reviewed = [a for a in workspace.articles if a.reviewer_decision in DECISIONS]
    if mode == "Model proposals":
        other = {a.id: a.model_decision for a in reviewed if a.model_decision in DECISIONS}
    else:
        template = pd.DataFrame({
            "ID": [a.id for a in workspace.articles], "Title": [a.title for a in workspace.articles],
            "Abstract": [a.abstract for a in workspace.articles], "Decision": "",
        })
        st.download_button("Download reviewer template", template.to_csv(index=False), "reviewer-template.csv", "text/csv", on_click="ignore")
        upload = st.file_uploader("Second reviewer CSV", type=["csv"], help="Use the ID template or provide unique DOIs and decisions.")
        if upload:
            try:
                matched = match_reviewer_file(upload, workspace)
            except ValueError as error:
                st.error(str(error))
            else:
                st.caption(f"{len(matched):,} CSV decisions matched to the library.")
                if st.button("Save second reviewer decisions", disabled=not matched):
                    changed = 0
                    for article in workspace.articles:
                        if article.id in matched and article.second_reviewer_decision != matched[article.id]:
                            article.second_reviewer_decision = matched[article.id]
                            changed += 1
                    if changed:
                        workspace.log("Second reviewer imported", file=upload.name, records=changed)
                        if not commit(workspace):
                            return
                    st.success(f"{changed:,} second reviewer decisions updated.")
        other = {a.id: a.second_reviewer_decision for a in reviewed if a.second_reviewer_decision in DECISIONS}
    paired = [a for a in reviewed if a.id in other]
    if len(paired) < 2:
        st.info(f"{len(paired)} paired decisions. At least two are needed to display agreement.")
        return
    result = calculate_cohens_kappa([a.reviewer_decision for a in paired], [other[a.id] for a in paired])
    kappa, agreement, count = st.columns(3)
    kappa.metric("Cohen’s κ", "—" if result["expected_agreement"] == 1 else f"{result['kappa']:.3f}")
    agreement.metric("Observed agreement", f"{result['observed_agreement']:.1%}")
    count.metric("Paired decisions", len(paired))
    cm = result["confusion_matrix"]
    st.dataframe(pd.DataFrame(cm["matrix"], index=cm["categories"], columns=cm["categories"]), width="stretch")
    conflicts = [a for a in paired if a.reviewer_decision != other[a.id] and not a.consensus_decision]
    st.caption(f"{len(conflicts):,} unresolved conflicts")
    if not conflicts:
        return
    by_id = {a.id: a for a in conflicts}
    selected = st.selectbox("Conflicting publication", list(by_id), format_func=lambda id: by_id[id].title)
    article = by_id[selected]
    st.caption(f"Reviewer · {article.reviewer_decision}   |   {mode} · {other[article.id]}")
    st.write(article.abstract or "No abstract available.")
    with st.form(f"consensus_{article.id}"):
        decision = st.selectbox("Consensus decision", DECISIONS)
        notes = st.text_area("Consensus notes", value=article.reviewer_notes)
        if st.form_submit_button("Save consensus", type="primary"):
            workspace.review(article, decision, notes.strip(), consensus=True)
            update_ranking(workspace)
            if not commit(workspace):
                return
            st.session_state.notice = "Consensus decision saved."
            st.rerun()

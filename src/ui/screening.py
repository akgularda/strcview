"""One-record screening with background sampling and ranking."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Empty, SimpleQueue
from threading import Event

import streamlit as st

from src.evaluators.base import EvaluationCriteria
from src.evaluators.laya_ranking import LayaRanker, ranking_signature
from src.screening import (
    CANDIDATE_COUNT, SEED_COUNT, human_labels, laya_seeds, pending_records, source_sample,
)
from src.ui.review import render_article
from src.ui.session import commit
from src.workspace import Workspace, desktop_workspace_path


@st.cache_resource
def workers() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="screening")


def cancel_ranking() -> None:
    if ranking := st.session_state.pop("ranking_job", None):
        job, _, _, cancelled = ranking
        cancelled.set()
        job.cancel()


def start_ranking(workspace: Workspace, *, keep_previous: bool = True) -> None:
    labels = human_labels(workspace.articles)
    pending = pending_records(workspace.articles)
    criteria = EvaluationCriteria(workspace.research_question, workspace.inclusion_criteria,
                                  workspace.exclusion_criteria, workspace.threshold)
    signature = ranking_signature(criteria, labels, pending)
    if workspace.laya_ranking_signature == signature and all(article.priority_score is not None for article in pending):
        return
    cancel_ranking()
    st.session_state.pop("ranking_error", None)
    st.session_state.pop("screen_record", None)
    can_rank = (bool(criteria.research_question) and {article.decision for article in labels} == {"Include", "Exclude"}
                and bool(pending) and all(article.id not in workspace.seed_ids for article in pending))
    if not can_rank or not keep_previous or not workspace.laya_ranking_signature:
        workspace.laya_ranking_signature = ""
        for article in pending:
            article.priority_score = None
    for article in workspace.articles:
        if article.manual_override:
            article.priority_score = None
    if not can_rank:
        return
    if "laya_ranker" not in st.session_state:
        path = desktop_workspace_path()
        st.session_state.laya_ranker = LayaRanker(path.with_name("laya-features.npz") if path else None)
    progress, cancelled = SimpleQueue(), Event()
    job = workers().submit(
        st.session_state.laya_ranker.rank, [replace(article) for article in labels],
        [replace(article) for article in pending], criteria, lambda done, total: progress.put((done, total)), cancelled,
    )
    st.session_state.ranking_job = (job, signature, progress, cancelled)
    st.session_state.ranking_progress = None


def save_seeds(workspace: Workspace, ids: list[str], method: str) -> None:
    pending_ids = {article.id for article in pending_records(workspace.articles)}
    workspace.seed_ids = [record_id for record_id in ids if record_id in pending_ids]
    if len(workspace.seed_ids) < min(SEED_COUNT, len(pending_ids)):
        extra = [article for article in pending_records(workspace.articles) if article.id not in workspace.seed_ids]
        workspace.seed_ids.extend(article.id for article in source_sample(extra, SEED_COUNT - len(workspace.seed_ids)))
    workspace.seed_method = method
    workspace.log("Initial screening queue", method=method, ids=workspace.seed_ids)
    if commit(workspace):
        st.session_state.seed_error = False
        st.rerun()


@st.fragment(run_every=0.7)
def observe_jobs(workspace: Workspace) -> None:
    if job := st.session_state.get("seed_job"):
        if not job.done():
            st.info("Laya başlangıç örneklerini seçiyor…")
        else:
            st.session_state.pop("seed_job", None)
            try:
                ids = job.result()
            except Exception:
                st.session_state.seed_error = True
                st.rerun()
            method = "Laya" if st.session_state.pop("seed_scored", False) else "Sources"
            save_seeds(workspace, ids, method)
    if ranking := st.session_state.get("ranking_job"):
        job, signature, progress, _ = ranking
        try:
            while True:
                st.session_state.ranking_progress = progress.get_nowait()
        except Empty:
            pass
        if not job.done():
            if st.session_state.ranking_progress:
                done, total = st.session_state.ranking_progress
                st.progress(done / total, text=f"Laya {done:,} / {total:,} yayını işledi")
            else:
                st.info("Laya sıralamayı hazırlıyor…")
            return
        st.session_state.pop("ranking_job", None)
        try:
            scores = job.result().scores
        except Exception:
            st.session_state.ranking_error = True
        else:
            for article in pending_records(workspace.articles):
                article.priority_score = scores.get(article.id)
            workspace.laya_ranking_signature = signature
            if not commit(workspace):
                return
        st.rerun()


def render_screening(workspace: Workspace) -> None:
    st.header("Tarama")
    pending = pending_records(workspace.articles)
    if not workspace.articles:
        st.info("Başlamak için Kütüphane’ye yayın ekle.")
        return
    if not pending:
        st.success("Tarama kuyruğu tamamlandı. Kararsız kayıtları Kütüphane’den yeniden inceleyebilirsin.")
        return

    reviewed = len(workspace.articles) - len(pending)
    st.progress(reviewed / len(workspace.articles), text=f"{reviewed:,} / {len(workspace.articles):,} yayın incelendi")
    if st.session_state.get("seed_job"):
        observe_jobs(workspace)

    if not workspace.seed_method and not reviewed:
        if st.session_state.get("seed_job"):
            return
        if st.session_state.get("seed_error"):
            st.warning("Laya başlangıç örneklerini puanlayamadı. Yeniden dene veya kaynaklardan örnek seç.")
            if st.button("Kaynak çeşitliliğiyle devam et"):
                save_seeds(workspace, [article.id for article in source_sample(pending, SEED_COUNT)], "Sources")
                return
        if not workspace.research_question:
            st.info("Başlangıç örneklerini seçmek için Ayarlar’da araştırma sorusunu kaydet.")
            return
        if st.button("Başlangıç örneklerini seç", type="primary", icon=":material/play_arrow:"):
            candidates = source_sample(pending, CANDIDATE_COUNT)
            st.session_state.seed_scored = len(candidates) > SEED_COUNT
            criteria = EvaluationCriteria(workspace.research_question, workspace.inclusion_criteria,
                                          workspace.exclusion_criteria, workspace.threshold)
            st.session_state.seed_job = workers().submit(laya_seeds, [replace(article) for article in candidates], criteria)
            st.rerun()
        return

    pending_by_id = {article.id: article for article in pending}
    remaining_seeds = [pending_by_id[record_id] for record_id in workspace.seed_ids if record_id in pending_by_id]
    if remaining_seeds:
        article = remaining_seeds[0]
        position = len(workspace.seed_ids) - len(remaining_seeds) + 1
        st.caption(f"Başlangıç örneği {position} / {len(workspace.seed_ids)}")
    else:
        ranked = [article for article in pending if article.priority_score is not None]
        if st.session_state.get("ranking_job"):
            observe_jobs(workspace)
            if not ranked:
                return
        if st.session_state.get("ranking_error"):
            st.warning("Laya yayınları sıralayamadı. Yeniden deneyebilir veya Kütüphane’den incelemeye devam edebilirsin.")
            if st.button("Laya sıralamasını yeniden dene"):
                start_ranking(workspace)
                st.rerun()
            if not ranked:
                return
        active = pending_by_id.get(st.session_state.get("screen_record"))
        article = active or (max(ranked, key=lambda item: item.priority_score) if ranked else source_sample(pending, 1)[0])
        if not ranked and {item.decision for item in human_labels(workspace.articles)} != {"Include", "Exclude"}:
            st.caption("Sıralamanın öğrenmesi için bir ilgili ve bir ilgisiz karar gerekiyor.")

    st.session_state.screen_record = article.id
    with st.container(border=True, key="screening_card"):
        render_article(article, show_highlighter=False)
        with st.form(f"screen_{article.id}", border=False):
            with st.expander("Not ekle"):
                notes = st.text_area("İnceleme notu", label_visibility="collapsed")
            st.markdown("**Bu makale araştırma soruna uygun mu?**")
            for column, (label, decision, icon) in zip(st.columns(3), (
                ("İlgili", "Include", "check_circle"), ("İlgisiz", "Exclude", "cancel"),
                ("Kararsız", "Borderline", "help"),
            )):
                if column.form_submit_button(label, width="stretch", icon=f":material/{icon}:",
                                             type="primary" if decision == "Include" else "secondary"):
                    workspace.review(article, decision, notes.strip())
                    start_ranking(workspace)
                    if not commit(workspace):
                        return
                    st.session_state.pop("screen_record", None)
                    st.rerun()

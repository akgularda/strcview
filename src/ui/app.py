import os
import sys
from base64 import b64encode
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.environ.get("STRCVIEW_DATA_DIR"):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

import pandas as pd
import streamlit as st

from src.analytics.bibliometrics import compute_bibliometric_summary
from src.analytics.exporter import (
    export_to_csv, export_to_excel, export_vosviewer_scopus_format, generate_prisma_report,
)
from src.analytics.network import build_keyword_cooccurrence_network
from src.analytics.prisma_diagram import generate_prisma_2020_svg
from src.evaluators.base import EvaluationCriteria
from src.evaluators.laya_evaluator import LayaEvaluator
from src.evaluators.llama_evaluator import LlamaEvaluator
from src.evaluators.rule_evaluator import RuleEvaluator
from src.models.article import Article
from src.ui.article_tools import highlight_abstract_keywords, resolve_unpaywall_doi
from src.ui.charts import render_charts
from src.ui.ingestion import render_duplicates, render_import
from src.ui.review import render_agreement, render_review
from src.ui.screening import cancel_ranking, observe_jobs, render_screening, start_ranking
from src.ui.session import commit
from src.workspace import Workspace, desktop_workspace_path

NAVIGATION = {
    "Genel Bakış": "dashboard",
    "Tarama": "fact_check",
    "Kütüphane": "folder_open",
    "Raporlar": "bar_chart",
    "Ayarlar": "settings",
}
ENGINES = {"Laya": LayaEvaluator, "Ollama": LlamaEvaluator, "Keyword baseline": RuleEvaluator}


def navigate(page: str) -> None:
    st.session_state.page = page


def sidebar() -> None:
    with st.sidebar:
        st.html('<div class="app-brand">sTRC<span>view</span></div>')
        with st.container(key="workflow_nav"):
            for name, icon in NAVIGATION.items():
                st.button(
                    name, key=f"nav_{name}", icon=f":material/{icon}:", width="stretch",
                    type="primary" if st.session_state.page == name else "tertiary",
                    on_click=navigate, args=(name,),
                )


def render_home(workspace: Workspace) -> None:
    st.header("Genel Bakış")
    if workspace.research_question:
        st.write(workspace.research_question)
    reviewed = sum(article.manual_override for article in workspace.articles)
    for column, label, value in zip(st.columns(4), (
        "Kütüphanedeki yayın", "İncelenen", "İlgili", "Kararsız",
    ), (
        len(workspace.articles), reviewed,
        sum(a.manual_override and a.decision == "Include" for a in workspace.articles),
        sum(a.manual_override and a.decision == "Borderline" for a in workspace.articles),
    )):
        column.metric(label, f"{value:,}", border=True)
    if workspace.articles:
        st.progress(reviewed / len(workspace.articles), text=f"{reviewed:,} yayın incelendi")
        st.button("Taramaya git", type="primary", on_click=navigate, args=("Tarama",))
    else:
        st.button("Yayın ekle", type="primary", on_click=navigate, args=("Kütüphane",))
    if not workspace.articles:
        with st.container(border=True, key="example_panel"):
            left, right = st.columns([4, 1.5], vertical_alignment="center")
            with left:
                st.markdown("**Örnek yayınlarla dene**")
                st.caption("Biri kopya olan üç örnek kayıt.")
            if right.button("Örnekleri yükle", icon=":material/science:", width="stretch"):
                workspace.add_articles([
                    Article(title="Urban tree canopy and summer heat exposure", authors="Chen, A.; Rivera, M.", year=2024,
                            abstract="We measured the relationship between neighborhood tree canopy and summer land surface temperature.",
                            keywords="urban heat; tree canopy", doi="10.0000/strcview.example1", source_db="Example"),
                    Article(title="Urban tree canopy and summer heat exposure", authors="Chen, A.; Rivera, M.", year=2024,
                            abstract="A cross-sectional analysis compared summer surface temperature with local tree canopy across 120 neighborhoods.",
                            keywords="urban heat; tree canopy; remote sensing", doi="10.0000/strcview.example1", source_db="Example"),
                    Article(title="Transit access and neighborhood equity", authors="Okafor, S.", year=2022,
                            abstract="This survey examines travel behavior and transport access in five metropolitan areas.",
                            keywords="transit; equity", source_db="Example"),
                ], "strcview-example-v1", "Example records")
                if not commit(workspace):
                    return
                st.session_state.notice = "Üç örnek kayıt yüklendi."
                st.rerun()


def render_collection(workspace: Workspace) -> None:
    st.header("Kütüphane")
    section = st.segmented_control("Kütüphane bölümü", ["Yayınlar", "Dosya al", "Kopyalar"],
                                   default="Yayınlar" if workspace.articles else "Dosya al",
                                   key="library_section", label_visibility="collapsed", required=True)
    if section == "Dosya al":
        render_import(workspace)
    elif section == "Kopyalar":
        render_duplicates(workspace)
    else:
        render_review(workspace, start_ranking)


def render_protocol(workspace: Workspace) -> None:
    st.subheader("Araştırma protokolü")
    st.caption("Başlık ve özet taramasında kullanılacak araştırma sorusunu ve uygunluk ölçütlerini tanımla.")
    with st.form("protocol", border=True):
        question = st.text_area("Araştırma sorusu", workspace.research_question, height=100)
        left, right = st.columns(2)
        inclusion = left.text_area("Dahil etme ölçütleri", workspace.inclusion_criteria, height=180)
        exclusion = right.text_area("Hariç tutma ölçütleri", workspace.exclusion_criteria, height=180)
        if st.form_submit_button("Protokolü kaydet", type="primary"):
            workspace.research_question = question.strip()
            workspace.inclusion_criteria = inclusion.strip()
            workspace.exclusion_criteria = exclusion.strip()
            workspace.log("Protocol saved", question=workspace.research_question,
                          inclusion=workspace.inclusion_criteria, exclusion=workspace.exclusion_criteria)
            if job := st.session_state.pop("seed_job", None):
                job.cancel()
            st.session_state.seed_error = False
            st.session_state.pop("screen_record", None)
            start_ranking(workspace, keep_previous=False)
            if not commit(workspace):
                return
            st.session_state.notice = "Araştırma protokolü kaydedildi."
            st.rerun()


def render_model_screening(workspace: Workspace) -> None:
    st.subheader("Toplu model taraması")
    st.caption("Başlık ve özetlerden hakem incelemesi için model önerileri üret.")
    if not workspace.articles:
        st.info("Import publications to start screening.")
        return
    if not workspace.research_question:
        st.info("Başlamak için Araştırma protokolü bölümünde bir soru kaydet.")
    with st.container(border=True):
        left, right = st.columns(2)
        engine = left.selectbox("Evaluation model", list(ENGINES), index=list(ENGINES).index(workspace.model) if workspace.model in ENGINES else 0)
        threshold = right.slider("Relevance threshold", 0.0, 1.0, workspace.threshold, 0.05,
                                 disabled=engine == "Ollama", help="Used by Laya and the keyword baseline.")
        if (engine, threshold) != (workspace.model, workspace.threshold):
            workspace.model = engine
            workspace.threshold = threshold
            start_ranking(workspace, keep_previous=False)
            if not commit(workspace):
                return
        scope = st.radio("Records to screen", ["Without a model proposal", "All records"], horizontal=True,
                         help="A screening run preserves any saved reviewer or consensus decision.")
        pending = [a for a in workspace.articles if scope == "All records" or not a.model_decision]
        st.caption(f"{len(pending):,} records awaiting a model proposal." if scope != "All records" else f"{len(pending):,} records selected for screening.")
        if not pending:
            return
        batch = st.number_input("Batch size", 1, len(pending), len(pending))
        if not st.button("Run screening", type="primary", icon=":material/play_arrow:", disabled=not workspace.research_question):
            return
    criteria = EvaluationCriteria(workspace.research_question, workspace.inclusion_criteria, workspace.exclusion_criteria, threshold)
    progress = st.progress(0.0, text="Preparing evaluation model…")
    completed = 0
    fallback_count = 0
    try:
        evaluator = ENGINES[engine]()
        if engine == "Ollama" and not evaluator.check_connection():
            st.warning("Ollama is unavailable. Start it locally and try again.")
            return
        for article in pending[:batch]:
            result = evaluator.evaluate_article(article, criteria)
            if result.model_name == "LLaMA (Unavailable)":
                st.error(result.reasoning)
                break
            evaluator.apply_result(article, result)
            fallback_count += int("fallback" in result.model_name.casefold())
            completed += 1
            progress.progress(completed / batch, text=f"Screened {completed:,} of {batch:,} records")
    except Exception as error:
        st.error(f"Screening paused. {error}")
    if completed:
        workspace.log("Model screening", model=engine, records=completed, fallback=fallback_count,
                      threshold=threshold, question=criteria.research_question,
                      inclusion=criteria.inclusion_criteria, exclusion=criteria.exclusion_criteria)
        if not commit(workspace):
            return
        if fallback_count:
            st.warning(f"{fallback_count:,} records used Laya's rule fallback. The actual engine is listed per record.")
        st.success(f"{completed:,} model proposals saved.")


def render_reports(workspace: Workspace) -> None:
    st.header("Raporlar")
    if not workspace.articles:
        st.info("Import publications to generate reports.")
        return
    summary = compute_bibliometric_summary(workspace.articles, workspace.dedup_stats)
    view = st.segmented_control("Report view", ["Tarama akışı", "Bibliyometri", "Hakem uyumu", "Dışa aktar"], default="Tarama akışı", label_visibility="collapsed", required=True)
    if view == "Tarama akışı":
        p = summary["prisma"]
        for col, label, key in zip(st.columns(4), ("Identified", "Screened", "Pending", "Advance to full text"),
                                   ("identified_total", "records_screened", "records_unscreened", "records_included")):
            col.metric(label, f"{p[key]:,}", border=True)
        st.caption("Screening-stage counts based on current decisions. Full-text eligibility is a separate stage.")
        svg = generate_prisma_2020_svg(p)
        encoded = b64encode(svg.encode()).decode()
        st.html(f'<div class="report-figure"><img alt="Title and abstract screening flow" src="data:image/svg+xml;base64,{encoded}"></div>')
        st.download_button("Download screening flow", svg, "strcview-screening-flow.svg", "image/svg+xml", icon=":material/download:", on_click="ignore")
    elif view == "Hakem uyumu":
        render_agreement(workspace, start_ranking)
    elif view == "Dışa aktar":
        render_export(workspace)
    else:
        render_charts(summary)
        st.subheader("Keyword co-occurrence")
        left, right = st.columns(2)
        top = left.slider("Maximum terms", 10, 60, 30, 5)
        minimum = right.slider("Minimum co-occurrences", 1, 8, 2)
        st.plotly_chart(build_keyword_cooccurrence_network(workspace.articles, top_n=top, min_cooccurrence=minimum), width="stretch")


def render_export(workspace: Workspace) -> None:
    st.subheader("Sonuçları dışa aktar")
    if not workspace.articles:
        st.info("Import publications to export data.")
        return
    summary = compute_bibliometric_summary(workspace.articles, workspace.dedup_stats)
    st.caption(f"{len(workspace.articles):,} records available for export.")
    outputs = (
        ("Excel workbook", "Decisions, publication metadata and screening counts", lambda: export_to_excel(workspace.articles, summary), "strcview.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("Publication CSV", "Harmonized records and reviewer decisions", lambda: export_to_csv(workspace.articles), "strcview.csv", "text/csv"),
        ("Scopus-format CSV", "Publication metadata for VOSviewer and Bibliometrix", lambda: export_vosviewer_scopus_format(workspace.articles), "strcview-scopus.csv", "text/csv"),
        ("Screening report", "Research question and PRISMA screening counts", lambda: generate_prisma_report(summary, workspace.research_question), "strcview-screening.md", "text/markdown"),
    )
    columns = st.columns(2)
    for index, (title, detail, generate, filename, mime) in enumerate(outputs):
        with columns[index % 2], st.container(border=True):
            st.subheader(title)
            st.caption(detail)
            st.download_button("Download", generate, filename, mime, key=filename, icon=":material/download:", on_click="ignore")


def render_workspace(workspace: Workspace) -> None:
    st.subheader("Çalışma alanı dosyaları")
    if desktop_workspace_path():
        st.caption("Yayınlar, protokol ve kararlar data klasörüne otomatik kaydedilir.")
        if st.session_state.get("autosave_error") and st.button("Otomatik kaydı yeniden dene"):
            if commit(workspace):
                st.rerun()
    else:
        st.caption("Daha sonra devam etmek için çalışma alanını JSON olarak kaydet.")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Save workspace")
        st.download_button("Download workspace", workspace.to_json, "strcview-workspace.json", "application/json",
                           type="primary", icon=":material/download:", on_click="ignore")
    with right, st.container(border=True):
        st.subheader("Restore workspace")
        upload = st.file_uploader("sTRCview workspace file", type=["json"])
        if upload and st.button("Open workspace", help="This replaces the workspace in this tab. Download it first to keep a copy."):
            try:
                restored = Workspace.from_json(upload.getvalue())
            except (ValueError, TypeError, RecursionError) as error:
                st.error(str(error))
            else:
                if not commit(restored):
                    return
                if job := st.session_state.pop("seed_job", None):
                    job.cancel()
                cancel_ranking()
                for key in ("duplicate_cache", "review_record", "screen_record", "seed_error", "oa_lookup"):
                    st.session_state.pop(key, None)
                st.session_state.workspace = restored
                st.session_state.notice = f"Workspace restored · {len(restored.articles):,} records."
                start_ranking(restored)
                navigate("Genel Bakış")
                st.rerun()
    if workspace.audit:
        st.subheader("Activity")
        events = pd.DataFrame(reversed(workspace.audit))
        st.dataframe(events[["time", "action"]], hide_index=True, width="stretch")
        st.download_button("Download activity log", lambda: events.to_json(orient="records", indent=2, force_ascii=False),
                           "strcview-activity.json", "application/json", on_click="ignore")


def render_settings(workspace: Workspace) -> None:
    st.header("Ayarlar")
    section = st.segmented_control("Ayarlar bölümü", ["Araştırma protokolü", "Model", "Çalışma alanı"],
                                   default="Araştırma protokolü", label_visibility="collapsed", required=True)
    if section == "Araştırma protokolü":
        render_protocol(workspace)
    elif section == "Model":
        render_model_screening(workspace)
    else:
        render_workspace(workspace)


def main() -> None:
    st.set_page_config(page_title="sTRCview", page_icon=":material/library_books:", layout="wide",
                       initial_sidebar_state="expanded")
    st.html(Path(__file__).with_name("styles.css"))
    if "workspace" not in st.session_state:
        path = desktop_workspace_path()
        try:
            workspace = Workspace.from_json(path.read_bytes()) if path and path.exists() else Workspace()
        except (OSError, ValueError) as error:
            st.error(f"Çalışma alanı açılamadı. {error}")
            st.info(f"{path} dosyasını kontrol et ve uygulamayı yeniden aç.")
            return
        st.session_state.workspace = workspace
        start_ranking(workspace)
    if "page" not in st.session_state:
        st.session_state.page = "Genel Bakış"
    workspace = st.session_state.workspace
    sidebar()
    if st.session_state.get("autosave_error"):
        st.warning("Otomatik kayıt başarısız. Ayarlar → Çalışma alanı bölümünden yeniden dene veya JSON yedeği indir.")
    if notice := st.session_state.pop("notice", None):
        st.toast(notice, icon=":material/check_circle:")
    if st.session_state.page != "Tarama" and st.session_state.get("ranking_job"):
        observe_jobs(workspace)
    pages = {
        "Genel Bakış": render_home, "Tarama": render_screening, "Kütüphane": render_collection,
        "Raporlar": render_reports, "Ayarlar": render_settings,
    }
    pages.get(st.session_state.page, render_home)(workspace)


if __name__ == "__main__":
    main()

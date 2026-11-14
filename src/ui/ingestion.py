"""Import and duplicate-review controls."""

import csv
import hashlib
import re
from pathlib import Path

import streamlit as st

from src.parsers.custom_excel_parser import (
    detect_column_mapping, get_excel_sheet_names, load_tabular_file, parse_custom_dataframe,
)
from src.parsers.scopus_parser import parse_scopus_file
from src.parsers.wos_parser import parse_wos_file
from src.ui.screening import start_ranking
from src.ui.session import commit
from src.workspace import Workspace

FIELDS = {
    "title": "Title", "abstract": "Abstract", "authors": "Authors", "year": "Publication year",
    "journal": "Journal", "doi": "DOI", "keywords": "Keywords",
}


def identify_format(content: bytes) -> str:
    encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    text = content[:4096].decode(encoding, errors="replace")
    if re.search(r"(?m)^(FN .*Web of Science|PT [A-Z]\s*$)", text):
        return "WoS"
    if re.search(r"(?m)^TY\s{2}-\s", text):
        return "Scopus"
    header = text.partition("\n")[0]
    delimiter = "\t" if "\t" in header else ","
    columns = {col.strip().lower() for col in next(csv.reader([header], delimiter=delimiter))}
    if columns & {"accession number", "wos accession number", "ut", "keywords plus"}:
        return "WoS"
    if delimiter == "," and "title" in columns and columns & {"eid", "author(s) id", "index keywords"}:
        return "Scopus"
    return "Map columns"


def file_import(workspace: Workspace, name: str, content: bytes, key: str) -> None:
    extension = Path(name).suffix.lower()
    excel = extension in {".xlsx", ".xls"}
    digest = hashlib.sha256(content).hexdigest()
    with st.expander(name, expanded=True):
        try:
            if excel:
                sheets = get_excel_sheet_names(content)
                sheet = st.selectbox("Worksheet", sheets, key=f"sheet_{key}") if len(sheets) > 1 else sheets[0]
                source = "Map columns"
            else:
                sheet = None
                options = ["WoS", "Scopus", "Map columns"]
                source = st.selectbox("File format", options, index=options.index(identify_format(content)), key=f"format_{key}")

            import_key = hashlib.sha256(repr((digest, sheet)).encode()).hexdigest()
            if import_key in workspace.imports:
                st.success("Already imported into this workspace.")
                return

            if source == "Map columns":
                frame = load_tabular_file(content, name, sheet)
                if frame.empty:
                    st.warning("This table has no data rows.")
                    return
                columns = list(frame.columns)
                detected = detect_column_mapping(columns)
                st.caption("Confirm which columns contain the publication data. Title is required.")
                st.dataframe(frame.head(5), hide_index=True, width="stretch")
                with st.form(f"map_{key}_{sheet}"):
                    left, right = st.columns(2)
                    mapping = {}
                    suggested = set()
                    for index, (field, label) in enumerate(FIELDS.items()):
                        options = [None, *columns]
                        default = detected.get(field)
                        default = default if default in columns and default not in suggested else None
                        suggested.add(default)
                        mapping[field] = (left if index % 2 == 0 else right).selectbox(
                            label, options, index=options.index(default),
                            format_func=lambda option: "Select a column" if option is None else str(option),
                        )
                    submitted = st.form_submit_button("Import mapped records", type="primary")
                if not submitted:
                    return
                if mapping["title"] is None:
                    st.error("Select the title column to continue.")
                    return
                used_columns = [column for column in mapping.values() if column is not None]
                if len(used_columns) != len(set(used_columns)):
                    st.error("Map each field to a different column.")
                    return
                articles = parse_custom_dataframe(frame, mapping, "Custom Excel" if excel else "Custom table")
                skipped = len(frame) - len(articles)
            else:
                if not st.button(f"Import {'Web of Science' if source == 'WoS' else source} records", key=f"import_{key}", type="primary"):
                    return
                articles = (parse_wos_file if source == "WoS" else parse_scopus_file)(content)
                skipped = 0
        except ImportError as error:
            st.error(f"Unable to read {name}. Install the Excel dependencies listed in README.md. {error}")
            return
        except (ValueError, OSError, IndexError, SyntaxError) as error:
            st.error(f"Unable to read {name}. {error}")
            return
        except Exception:
            st.error(f"Unable to read {name}. Check that the file opens and matches the selected format.")
            return
        if not articles:
            st.warning("No publications with titles were found. Check the selected format or column mapping.")
            return
        workspace.add_articles(articles, import_key, name)
        if not commit(workspace):
            return
        start_ranking(workspace)
        st.session_state.pop("duplicate_cache", None)
        message = f"Imported {len(articles):,} records from {name}."
        if skipped:
            message += f" Skipped {skipped:,} rows without a title."
        st.session_state.notice = message
        st.rerun()


def render_import(workspace: Workspace) -> None:
    panel, guide = st.columns([2.15, 1], gap="medium")
    with panel, st.container(border=True, key="import_panel"):
        st.subheader("Import publications")
        st.caption("Upload database exports or map columns from your own spreadsheet.")
        files = st.file_uploader(
            "Publication files", type=["csv", "txt", "tsv", "ris", "xlsx", "xls"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        for index, upload in enumerate(files):
            file_import(workspace, upload.name, upload.getvalue(), f"{upload.file_id}_{index}")
    with guide, st.container(border=True, key="sources_panel"):
        st.subheader("Supported sources")
        st.markdown("**Web of Science**  \nTagged text, TSV and CSV  \n\n**Scopus**  \nCSV and RIS  \n\n**Custom table**  \nExcel, CSV and TSV with column mapping")


def render_duplicates(workspace: Workspace) -> None:
    st.subheader("Duplicate review")
    st.caption("Compare DOI and title matches before deciding which records to retain.")
    if st.button("Scan for duplicates", type="primary"):
        with st.spinner("Comparing DOIs and titles…"):
            st.session_state.duplicate_cache = workspace.duplicate_candidates()
    candidates = st.session_state.get("duplicate_cache")
    if candidates is None:
        return
    if not candidates:
        st.success("No unresolved duplicate pairs found.")
        return
    st.caption(f"{len(candidates):,} unresolved pairs")
    page = st.number_input("Pair number", 1, len(candidates), 1, key=f"duplicate_{len(candidates)}")
    candidate = candidates[page - 1]
    st.info(f"{candidate['match_type']} · {candidate['match_detail']}")
    for column, label, article in zip(st.columns(2), ("Record A", "Record B"), (candidate["article_a"], candidate["article_b"])):
        with column, st.container(border=True):
            st.caption(f"{label} · {article.source_db}")
            st.markdown(article.title.replace("\n", " "))
            st.write(" · ".join(filter(None, (article.journal, str(article.year or ""), article.authors))))
            st.text(article.doi or "No DOI")
            if article.source_file:
                st.caption(article.source_file)
            if article.manual_override:
                st.caption(f"Reviewer decision · {article.decision}")
            st.write(article.abstract or "No abstract available")
    actions = [("Merge records", "merge"), ("Keep A", "keep_a"), ("Keep B", "keep_b"), ("Keep both", "keep_both")]
    for column, (label, action) in zip(st.columns(4), actions):
        if column.button(label, key=f"{action}_{candidate['candidate_id']}", width="stretch", type="primary" if action == "merge" else "secondary"):
            removed = workspace.resolve_duplicates([candidate], action)
            if not commit(workspace):
                return
            start_ranking(workspace)
            st.session_state.duplicate_cache = workspace.duplicate_candidates()
            st.session_state.notice = f"Duplicate decision saved. {removed} record removed." if removed else "Both records retained."
            st.rerun()

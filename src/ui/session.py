"""Save desktop workspace mutations before acknowledging them in the UI."""

import streamlit as st

from src.workspace import Workspace, desktop_workspace_path, save_workspace


def commit(workspace: Workspace) -> bool:
    path = desktop_workspace_path()
    if path:
        try:
            save_workspace(workspace, path)
        except (OSError, ValueError) as error:
            st.session_state.autosave_error = str(error)
            st.error(f"Otomatik kayıt başarısız. Çalışma alanını Ayarlar’dan JSON olarak indir. {error}")
            return False
    st.session_state.pop("autosave_error", None)
    return True

"""Run sTRCview in a native desktop window."""

import ctypes
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from contextlib import contextmanager
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
WINDOWS = sys.platform == "win32"


@contextmanager
def desktop_lock(data: Path):
    with (data / "desktop.lock").open("a+b") as lock:
        if WINDOWS:
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def local_server(port: str) -> None:
    from streamlit.web import bootstrap

    options = {
        "global_developmentMode": False, "server_address": "127.0.0.1", "server_port": int(port),
        "server_headless": True, "server_fileWatcherType": "none", "client_toolbarMode": "minimal",
        "runner_magicEnabled": False,
        "browser_gatherUsageStats": False, "theme_base": "light", "theme_primaryColor": "#785600",
        "theme_backgroundColor": "#fbf9f4", "theme_secondaryBackgroundColor": "#f1ede4",
        "theme_textColor": "#292620",
    }
    bootstrap.load_config_options(flag_options=options)
    bootstrap.run(str(ROOT / "src/ui/app.py"), False, [], options)


def show_error(message: str) -> None:
    if WINDOWS:
        ctypes.windll.user32.MessageBoxW(None, message, "sTRCview", 0x10)
    else:
        subprocess.run(
            ["/usr/bin/osascript", "-e", "on run argv", "-e",
             'display alert "sTRCview" message (item 1 of argv) as critical', "-e", "end run", message],
            check=False,
        )


def start_desktop(data: Path) -> None:
    import webview

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    command = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, str(Path(__file__).resolve())]
    environment = {
        **os.environ,
        "STRCVIEW_DATA_DIR": str(data),
        "HF_HOME": str(data / "models"),
        "HF_HUB_CACHE": str(data / "models" / "hub"),
    }
    environment.pop("TRANSFORMERS_CACHE", None)
    with (data / "server.log").open("wb") as log:
        server = subprocess.Popen(
            [*command, "--serve", str(port)], env=environment, cwd=APP_DIR,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if WINDOWS else 0,
        )
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True
            webview.settings["ALLOW_FILE_URLS"] = False
            window = webview.create_window(
                "sTRCview", width=1220, height=850, min_size=(760, 560), background_color="#fbf9f4",
                html='<body style="background:#fbf9f4;font:18px Segoe UI;padding:36px;color:#292620">sTRCview açılıyor…</body>',
            )

            def open_when_ready() -> None:
                opener = build_opener(ProxyHandler({}))
                deadline = time.monotonic() + 50
                while time.monotonic() < deadline and server.poll() is None:
                    try:
                        with opener.open(f"{origin}/_stcore/health", timeout=0.5) as response:
                            ready = response.read() == b"ok"
                    except (URLError, OSError):
                        ready = False
                    if ready:
                        try:
                            window.load_url(origin)
                        except Exception:
                            logging.exception("Could not navigate to the local app")
                        return
                    time.sleep(0.2)
                if server.poll() is not None or time.monotonic() >= deadline:
                    try:
                        window.load_html('<body style="background:#fbf9f4;color:#292620;font:18px system-ui;padding:36px">sTRCview başlatılamadı. Çalışma alanındaki server.log dosyasını kontrol edin.</body>')
                    except Exception:
                        logging.exception("Could not display the startup error")

            webview.start(open_when_ready, gui="edgechromium" if WINDOWS else "cocoa",
                          private_mode=True, storage_path=str(data / "webview"))
            if WINDOWS and webview.renderer != "edgechromium":
                raise RuntimeError("Microsoft Edge WebView2 Runtime gerekli.")
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=10)


def main() -> None:
    multiprocessing.freeze_support()
    if sys.argv[1:] == ["--check-laya"]:
        from src.evaluators.base import EvaluationCriteria
        from src.evaluators.laya_ranking import LayaRanker
        from src.models.article import Article

        labels = [Article(title="Deep learning on earthquake P waves", decision="Include"),
                  Article(title="Population response to disaster alerts", decision="Exclude")]
        pending = [Article(title="Real-time seismic phase detection with neural networks")]
        result = LayaRanker().rank(labels, pending, EvaluationCriteria("Machine learning for earthquake early warning"))
        if len(result.scores) != 1 or result.scores == result.base_scores:
            raise RuntimeError("Laya ranking smoke test failed")
        print("Laya ranked the test publication")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        if sys.stdout is None:
            sys.stdout = open(Path(os.environ["STRCVIEW_DATA_DIR"]) / "server.log", "a", encoding="utf-8", buffering=1)
        if sys.stderr is None:
            sys.stderr = sys.stdout
        local_server(sys.argv[2])
        return

    data = APP_DIR / "data" if WINDOWS else Path.home() / "Library/Application Support/sTRCview"
    try:
        data.mkdir(parents=True, exist_ok=True)
        with desktop_lock(data):
            start_desktop(data)
    except BlockingIOError:
        show_error("sTRCview zaten açık.")
    except Exception as error:
        show_error(f"sTRCview başlatılamadı.\n\n{error}")


if __name__ == "__main__":
    main()

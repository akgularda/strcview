from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


root = Path(SPECPATH)
windows = sys.platform == "win32"
packages = ["streamlit", "laya", "webview", "vegaml"]
if windows:
    packages.append("clr_loader")
data = [(str(root / "src/ui/app.py"), "src/ui"), (str(root / "src/ui/styles.css"), "src/ui")]
for package in packages:
    data += collect_data_files(package)
for distribution in ("streamlit", "laya", "transformers", "torch", "huggingface-hub", "vegaml", "pywebview", "scikit-learn", "numpy", "pandas", "pyarrow"):
    data += copy_metadata(distribution)

hidden = [
    "src.ui.app", "streamlit.web.server.starlette.starlette_server", "openpyxl", "xlrd",
    "laya", "vegaml",
    "transformers.models.modernbert.modeling_modernbert", "transformers.models.modernbert.configuration_modernbert",
    "transformers.models.xlm_roberta.tokenization_xlm_roberta",
]
if windows:
    hidden += ["webview.platforms.winforms", "webview.platforms.edgechromium", "clr"]
else:
    hidden += ["webview.platforms.cocoa", "AppKit", "Foundation", "WebKit", "objc", "PyObjCTools.AppHelper"]
hidden += collect_submodules("laya", filter=lambda name: not name.startswith(("laya.serve", "laya.integrations", "laya.mcp", "laya.train", "laya.evals_")))

a = Analysis(
    [str(root / "src/desktop.py")], pathex=[str(root)], binaries=[], datas=data, hiddenimports=hidden,
    excludes=["IPython", "jupyter", "notebook", "matplotlib", "tkinter", "PySide6", "PyQt6", "cv2",
              "tensorflow", "torchvision", "torchaudio", "pytest", "numba"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="sTRCview", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="sTRCview", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(coll, name="sTRCview.app", bundle_identifier="com.akgularda.strcview",
                 version="0.3.0", info_plist={"NSHighResolutionCapable": True})

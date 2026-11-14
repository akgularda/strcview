@echo off
setlocal
title sTRCview - Research workspace
cd /d "%~dp0" || exit /b 1

set "STRCVIEW_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "STRCVIEW_PYTHON=%~dp0.venv\Scripts\python.exe"

"%STRCVIEW_PYTHON%" -c "import streamlit, pandas, openpyxl, sklearn, networkx, plotly, requests, pyarrow" >nul 2>&1
if errorlevel 1 (
    echo sTRCview dependencies are missing. See README.md for setup.
    pause
    exit /b 1
)

echo Starting sTRCview at http://localhost:8501
"%STRCVIEW_PYTHON%" -m streamlit run src\ui\app.py --server.headless false --server.address localhost --browser.gatherUsageStats false --theme.base light --theme.primaryColor "#0b4263"
if errorlevel 1 pause

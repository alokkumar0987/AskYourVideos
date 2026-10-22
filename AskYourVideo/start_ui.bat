@echo off
rem Double-click to start the AskYourVideo UI (opens http://127.0.0.1:7860).
rem Works from any folder or terminal; keeps all caches on this drive.
cd /d "%~dp0"
set "HF_HOME=%~dp0.cache\hf"
set "TORCH_HOME=%~dp0.cache\torch"
set "TEMP=%~dp0.cache\tmp"
set "TMP=%~dp0.cache\tmp"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "GRADIO_ANALYTICS_ENABLED=False"
set "PYTHONWARNINGS=ignore"
echo Starting AskYourVideo... the browser will open in a few seconds.
echo Keep this window open while using the UI. Close it to stop.
"%~dp0.venv\Scripts\askvideo-ui.exe"
echo.
echo The UI stopped. If you see an error above, copy it and share it.
pause

@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run py -3.12 setup_lab.py --device cuda first.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0run_lab.py" %*

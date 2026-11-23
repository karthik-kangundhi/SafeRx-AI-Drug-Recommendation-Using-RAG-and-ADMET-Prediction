@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  python -m venv .venv || exit /b 1
)
call ".venv\Scripts\activate.bat"
if not exist ".venv\installed.ok" (
  python -m pip install --upgrade pip || exit /b 1
  pip install -r backend\requirements.txt || exit /b 1
  type nul > ".venv\installed.ok"
)
cd /d "%~dp0backend"
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000

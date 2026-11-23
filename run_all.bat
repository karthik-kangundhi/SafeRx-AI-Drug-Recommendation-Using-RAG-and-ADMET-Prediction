@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
start "SafeRx AI Backend" cmd /k "call "%~dp0START_BACKEND.bat""
timeout /t 3 /nobreak >nul
start "SafeRx AI Frontend" cmd /k "call "%~dp0START_FRONTEND.bat""

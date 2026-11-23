@echo off
cd /d "%~dp0"
echo Starting SafeRx AI React frontend...
call npm install
npm run dev

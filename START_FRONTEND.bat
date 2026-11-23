@echo off
cd /d "%~dp0frontend"
if not exist "node_modules" npm install || exit /b 1
npm run dev

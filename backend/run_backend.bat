@echo off
REM Set SAFERX_LIVE_ENRICH=1 before starting if live PubChem/openFDA/PubMed enrichment is desired.
@echo off
cd /d "%~dp0"
call ..\.venv\Scripts\activate
uvicorn api:app --reload --host 127.0.0.1 --port 8000

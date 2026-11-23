@echo off
cd /d "%~dp0"
if exist runtime\drug_catalog.csv del /q runtime\drug_catalog.csv
if exist runtime\drug_catalog_complete.csv del /q runtime\drug_catalog_complete.csv
if exist runtime\admet_predictions_complete.csv del /q runtime\admet_predictions_complete.csv
if exist runtime\rag_index.pkl del /q runtime\rag_index.pkl
if exist runtime\rag_docs.json del /q runtime\rag_docs.json
echo SafeRx V5 runtime cache cleared.
pause

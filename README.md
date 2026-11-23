# SafeRx AI — Final V7

SafeRx AI is an educational/research decision-support prototype combining molecular analysis, pretrained ADMET inference, biomedical evidence retrieval, patient-context extraction, disease-aware ranking, explainability, and PDF reporting.

## Final V7 includes
- React + Vite + Tailwind CSS + Framer Motion + Lucide + Recharts frontend
- FastAPI + Uvicorn backend
- RDKit molecular parsing, descriptors and structure rendering
- ADMET-AI pretrained inference when available
- RAG using BioLinkBERT + FAISS when available, with deterministic TF-IDF fallback
- PubChem / PubMed / openFDA live enrichment when enabled
- Local patient-document extraction in **Drug Recommendation**
- PDF, JPG, JPEG, PNG and WEBP upload
- Tesseract OCR for image/scanned-document extraction
- Editable patient fields before ranking
- Allergy, comorbidity and medication safety checks
- Disease/condition-aware ranking with transparent weights
- Drug-specific deduplicated RAG evidence (up to 3 items)
- Large molecular structures and detailed ADMET information
- Explainability tied to the selected recommendation
- Exactly **two report types** in the UI:
  1. Patient + Drug Recommendation Report
  2. Molecule / ADMET Report
- Persistent dark/light theme
- Controlled errors and offline fallbacks

## Windows setup
Open PowerShell 1:

```powershell
cd "C:\Users\Chinmay Ranjan\Downloads\SafeRx_AI\SafeRx_AI_FINAL_V7"
if (!(Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
python -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Open PowerShell 2:

```powershell
cd "C:\Users\Chinmay Ranjan\Downloads\SafeRx_AI\SafeRx_AI_FINAL_V7\frontend"
npm install
npm run dev
```

Open the Vite URL, normally `http://127.0.0.1:5173/`.

## JPG / JPEG / PNG patient extraction
Patient upload is inside **Drug Recommendation**. Supported formats are PDF, PNG, JPG, JPEG and WEBP. The backend first extracts searchable PDF text and automatically falls back to local OCR for scanned PDFs. Images use local Tesseract OCR.

Install Tesseract on Windows if needed:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

Or run `install_ocr_windows.bat` to check whether Tesseract is available.

The application attempts standard Windows Tesseract locations automatically. You can set `TESSERACT_CMD` if it is installed elsewhere.

## Live evidence enrichment
The default mode is offline/fallback-friendly. Set:

```powershell
$env:SAFERX_LIVE_ENRICH="1"
```

before starting the backend to enable optional PubChem/openFDA/PubMed enrichment. External service failure must not prevent local startup.

## Important scientific limitations
- ADMET outputs are computational model predictions, not laboratory measurements.
- RAG evidence confidence is not clinical certainty.
- Synthetic patient data is clearly labeled as synthetic.
- The ranking is a research/educational ranking, not a prescription.
- Missing information is not invented.
- Pretrained models are not described as trained by this project.

## Main API endpoints
- `GET /api/health`
- `POST /api/initialize`
- `POST /api/molecule/analyze`
- `POST /api/recommend`
- `POST /api/evidence`
- `POST /api/report/analyze` — patient PDF/image extraction
- `GET /api/dashboard`
- `GET /api/evaluation`
- `GET /api/explain/{drug}`
- `POST /api/report/pdf`

## Verification performed before packaging
- Python syntax/AST checks passed for backend Python modules.
- Duplicate FastAPI route check found no duplicate route paths.
- Project excludes virtual environments, node_modules and generated caches.

For the full requirements and acceptance criteria, see `PROJECT_SPECIFICATION.txt`.

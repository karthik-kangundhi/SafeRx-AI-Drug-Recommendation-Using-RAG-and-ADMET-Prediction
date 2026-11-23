# SafeRx AI — V6

## V6 changes
- Drug Recommendation is the main patient workflow: upload PDF/image report here, verify fields, analyze the patient problem, then rank condition-aware candidates.
- Ranking list has NO dropdown/chevron arrows. Click any drug row to switch the full candidate analysis panel.
- Candidate analysis includes a large 2D molecular structure, formula, molecular weight, SMILES, why-ranked explanation, four ranking metrics, patient safety flags, molecular descriptors, detailed ADMET endpoint table, structured drug evidence and retrieved RAG chunks.
- Recommendation ranking uses condition-aware profiles plus semantic retrieval, ADMET safety, bioavailability, evidence, and patient-specific penalties.
- Report Center has three explicit choices: Patient Report, Molecule / ADMET Report, Drug Recommendation Report.
- Theme button visibly toggles Light/Dark and persists the selection.
- Explainability includes structure, basic drug details, molecular descriptors, ADMET endpoints and ranking contributions.

## Run
### Backend
```powershell
cd "<EXTRACTED_V6>\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

If an older SafeRx runtime cache is present, V6 validates/rebuilds it. For a completely clean run you may remove the `runtime` directory before starting.

### Frontend
```powershell
cd "<EXTRACTED_V6>\frontend"
npm install
npm run dev
```

Open the Vite URL shown by the terminal, normally http://127.0.0.1:5173/.

## Demo patient
Use the synthetic patient report supplied in the project conversation for the allergic-rhinitis workflow. After extraction, verify the fields and click **Analyze & rank drugs**.

## Important
This is a research/educational decision-support demonstration. ADMET values are computational predictions and are not laboratory measurements or clinical thresholds. The system does not diagnose, prescribe, or replace qualified clinical judgment.

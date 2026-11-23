# SafeRx AI V6

V6 fixes the remaining research-demo issues: readable typography, visible Recharts bars in dark/light themes, drug purpose and common uses, drug-specific deduplicated RAG evidence chunks, patient-context explanation, molecular structure in reports, and richer Molecule/ADMET PDF content.

## Workflow
Drug Recommendation is the main patient workflow: upload report -> verify extracted fields -> patient problem interpretation -> condition-aware ranking -> candidate molecular structure -> purpose/common uses -> why ranked -> safety checks -> ADMET/lab-style details -> RAG evidence.

## Evidence note
The project can operate offline with structured catalog evidence. When `SAFERX_LIVE_ENRICH=1` is enabled, PubChem/openFDA/PubMed enrichment is used when available. The UI distinguishes retrieval mode and does not claim fallback text is independent primary literature.

## Reports
Report Center provides three choices: Patient Report, Molecule / ADMET Report, and Drug Recommendation Report. The Molecule / ADMET report includes a rendered molecular structure, drug purpose, common uses, formula, molecular weight, scores and descriptors.

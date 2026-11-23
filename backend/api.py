import io
import os
import re
import base64
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
RUNTIME.mkdir(exist_ok=True)
os.environ.setdefault("SAFERX_ROOT", str(RUNTIME))
os.environ.setdefault("SAFERX_PATIENT_CSV", str(ROOT / "synthetic_medical_reports_1000 (1).csv"))

try:
    from backend import saferx_complete as sx
except ModuleNotFoundError:
    # Supports launching either as `backend.api` from the project root or
    # as `api` from inside the backend directory.
    import saferx_complete as sx

app = FastAPI(title="SafeRx AI API", version="3.0.0", description="Research/educational drug intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE: dict[str, Any] = {"ready": False, "loading": False, "error": None}
ENGINE_LOCK = threading.Lock()


class PatientRequest(BaseModel):
    age: int = Field(45, ge=0, le=120)
    sex: str = "Other"
    diagnosis: str = ""
    symptom: str = ""
    allergies: str = ""
    comorbidities: str = ""
    current_medications: str = ""
    top_k: int = Field(5, ge=1, le=10)


class MoleculeRequest(BaseModel):
    molecule: str


class EvidenceRequest(BaseModel):
    question: str
    top_k: int = Field(5, ge=1, le=10)


class ReportPdfRequest(BaseModel):
    title: str = "SafeRx AI Research Report"
    report_type: str = "complete"
    patient: dict[str, Any] = {}
    molecule: Optional[dict[str, Any]] = None
    recommendations: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    interpretation: str = ""


def ensure_engine():
    if ENGINE["ready"]:
        return ENGINE
    with ENGINE_LOCK:
        if ENGINE["ready"]:
            return ENGINE
        ENGINE["loading"] = True
        ENGINE["error"] = None
        try:
            patients, catalog, retriever, admet_engine, preds = sx.initialize_system(refresh_catalog=False)
            ENGINE.update({
                "patients": patients,
                "catalog": catalog,
                "retriever": retriever,
                "admet": admet_engine,
                "preds": preds,
                "ready": True,
                "loading": False,
            })
            return ENGINE
        except Exception as ex:
            ENGINE["loading"] = False
            ENGINE["error"] = str(ex)
            raise


def clean(value: Any):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value


def molecule_rag_evidence(e, drug_name: str, smiles: str = "", limit: int = 3):
    """Return up to 3 distinct drug-specific RAG evidence chunks for a molecule report.
    Evidence is never fabricated; for an unindexed molecule the result is empty with a clear note.
    """
    retriever = e.get("retriever")
    if retriever is None or not hasattr(retriever, "search"):
        return [], "RAG retriever unavailable for this molecule."
    query = f"Drug: {drug_name}. Indications, common uses, contraindications, warnings, adverse reactions, drug interactions."
    try:
        hits = retriever.search(query, top_k=max(10, limit * 4)) or []
    except Exception:
        return [], "RAG retrieval failed for this molecule."
    wanted = str(drug_name or "").strip().casefold()
    out, seen = [], set()
    for h in hits:
        name = str(h.get("drug_name") or "").strip().casefold()
        if wanted and name != wanted:
            continue
        text = re.sub(r"\s+", " ", str(h.get("text") or "")).strip()
        key = (name, str(h.get("evidence_type") or "").strip().casefold(), text.casefold())
        if not text or key in seen:
            continue
        seen.add(key)
        out.append({
            "drug_name": h.get("drug_name") or drug_name,
            "score": float(h.get("score", 0.0) or 0.0),
            "text": h.get("text") or "",
            "source": h.get("source") or "SafeRx evidence profile",
            "evidence_type": h.get("evidence_type") or "Biomedical evidence"
        })
        if len(out) >= limit:
            break
    note = "Drug-specific RAG evidence retrieved and deduplicated." if out else "No drug-specific RAG evidence was available in the current indexed corpus."
    return out, note


def score_payload(pred):
    rc = sx.risk_components(pred)
    toxicity = max(0.0, min(1.0, rc["Risk_Score_100"] / 100.0))
    safety = max(0.0, min(1.0, 1.0 - toxicity))
    bio = max(0.0, min(1.0, sx.bioavailability_score(pred)))
    if toxicity < 0.15:
        level, tone = "Low", "low"
    elif toxicity < 0.30:
        level, tone = "Moderate", "moderate"
    elif toxicity < 0.60:
        level, tone = "High", "high"
    else:
        level, tone = "Very High", "very-high"
    return {
        "toxicity": toxicity,
        "bioavailability": bio,
        "safety": safety,
        "riskScore100": rc["Risk_Score_100"],
        "riskLevel": level,
        "riskTone": tone,
        "components": rc,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "engineReady": ENGINE["ready"], "engineLoading": ENGINE["loading"], "error": ENGINE["error"]}


@app.post("/api/initialize")
def initialize():
    try:
        e = ensure_engine()
        return {"ready": True, "drugCount": int(len(e["catalog"]))}
    except Exception as ex:
        raise HTTPException(500, f"AI engine initialization failed: {ex}")


def _structure_svg(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        drawer = Draw.MolDraw2DSVG(620, 360)
        drawer.drawOptions().addAtomIndices = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def _structure_png(smiles: str) -> bytes:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return b""
        return Draw.MolToPNG(mol, size=(760, 460), options=None)
    except Exception:
        return b""


@app.post("/api/molecule/analyze")
def analyze(req: MoleculeRequest):
    query = req.molecule.strip()
    if not query:
        raise HTTPException(400, "Enter a molecule name or SMILES.")
    e = ensure_engine()
    info = sx.pubchem_lookup(query)
    smiles = info.get("smiles") or (query if sx.canonicalize_smiles(query) else None)
    if not smiles:
        raise HTTPException(400, "Molecule was not recognized as a name or valid SMILES.")
    try:
        pred = e["admet"].predict([smiles]).loc[smiles]
        scores = score_payload(pred)
        desc = sx.molecular_descriptors(smiles)
        svg = _structure_svg(smiles)
    except Exception as ex:
        raise HTTPException(500, f"ADMET prediction failed: {ex}")
    drug_name = str(info.get("title") or query).strip()
    info_profile = next((v for name, v in sx.DRUG_INFO.items() if name.casefold() == drug_name.casefold() or name.casefold() == query.casefold()), {})
    purpose = info_profile.get("purpose") or "Drug purpose / indication information should be verified against authoritative labeling."
    common_uses = info_profile.get("common_uses", [])
    drug_class = info_profile.get("class", "")
    return clean({
        "molecule": {**info, "smiles": smiles, "descriptors": desc, "structure_svg": svg, "purpose": purpose, "common_uses": common_uses, "drug_class": drug_class},
        "scores": scores,
        "admet": pred.to_dict(),
        "engine": {"name": "ADMET-AI", "connected": True},
    })


def _patient_assessment(patient: dict[str, Any]) -> dict[str, Any]:
    diagnosis = str(patient.get("diagnosis", "")).strip()
    symptom = str(patient.get("symptom", "")).strip()
    allergies = str(patient.get("allergies", "")).strip()
    comorbids = str(patient.get("comorbidities", "")).strip()
    meds = str(patient.get("current_medications", "")).strip()
    d = diagnosis.casefold()
    s = symptom.casefold()
    problem = diagnosis or "Symptom pattern requiring review"
    why = "The explanation is based only on the details entered in this research workflow; it is not an independent diagnosis."
    factors = []
    if "allergic rhinitis" in d or ("sneez" in s and ("runny nose" in s or "nasal" in s)):
        problem = diagnosis or "Allergic upper-airway symptom pattern"
        why = "The entered sneezing, runny/blocked nose and itchy or watery eyes are consistent with an allergic upper-airway symptom pattern. Seasonal or environmental exposure can be a contributing trigger when reported."
    elif "hypertension" in d or "high blood pressure" in d:
        why = "The entered diagnosis indicates persistently elevated blood pressure; the record should be interpreted together with measured blood-pressure values and cardiovascular risk factors."
    elif "diabetes" in d:
        why = "The entered diagnosis indicates abnormal glucose regulation; symptoms and laboratory glucose/HbA1c values should be reviewed to understand current control."
    elif "hypothy" in d or "thyroid" in d:
        why = "The entered diagnosis indicates reduced thyroid function; thyroid laboratory results such as TSH and free T4 are the key context for understanding the condition."
    elif "gerd" in d or "reflux" in d:
        why = "The entered diagnosis/symptoms indicate reflux-related irritation; meal timing, reflux triggers and symptom pattern may contribute."
    elif "cholesterol" in d or "hyperlip" in d:
        why = "The entered diagnosis indicates elevated lipid levels; lipid-panel results and cardiovascular risk factors are important context."
    elif "depress" in d:
        why = "The entered diagnosis indicates a depressive symptom condition; severity, duration, functional impact and other medicines are important context."
    if allergies: factors.append(f"Known allergy information: {allergies}")
    if comorbids: factors.append(f"Comorbidity information: {comorbids}")
    if meds: factors.append(f"Current medication information: {meds}")
    if not factors: factors.append("No additional allergy, comorbidity or current-medication context was entered.")
    return {"identified_problem": problem, "reported_symptoms": symptom or "Not provided", "why_it_may_be_happening": why, "patient_context_factors": factors, "limitation": "This is a research-oriented interpretation of entered text, not a clinical diagnosis."}

@app.post("/api/recommend")
def recommend(req: PatientRequest):
    e = ensure_engine()
    patient = sx.patient_dict(req.age, req.sex, req.diagnosis, req.symptom, req.allergies, req.comorbidities, req.current_medications)
    assessment = _patient_assessment(patient)
    try:
        rec = sx.recommend(patient, e["catalog"], e["preds"], e["retriever"], top_k=req.top_k)
    except Exception as ex:
        raise HTTPException(500, f"Recommendation engine failed: {ex}")
    rows = []
    for _, r in rec.iterrows():
        d = clean(r.to_dict())
        d["Toxicity_Score"] = max(0.0, min(1.0, float(r["Risk_Score_100"]) / 100.0))
        d["Safety_Score"] = float(r["Safety_Score"])
        d["Structure_SVG"] = _structure_svg(str(d.get("SMILES", "")))
        condition = patient.get('diagnosis') or 'the entered symptom profile'
        rel = float(r.get('Condition_Relevance', 0))
        sem = float(r.get('Semantic_Relevance', 0))
        safety = float(r.get('Safety_Score', 0))
        bio = float(r.get('Bioavailability_Score', 0))
        evidence_score = float(r.get('Evidence_Score', 0))
        why_parts = [f"Condition match for {condition}: {rel:.2f}", f"semantic evidence match: {sem:.2f}", f"computational safety: {safety:.2f}", f"bioavailability signal: {bio:.2f}", f"evidence support: {evidence_score:.2f}"]
        if r.get('Patient_Penalty', 0):
            why_parts.append(f"patient-specific penalty: {float(r.get('Patient_Penalty',0)):.2f}")
        d["why_recommended"] = "Ranked using " + "; ".join(why_parts) + "."
        d["evidence"] = {
            "indications": d.get("Indications", ""), "contraindications": d.get("Contraindications", ""),
            "warnings": d.get("Warnings", ""), "adverse_reactions": d.get("Adverse_Reactions", ""),
            "drug_interactions": d.get("Drug_Interactions", ""), "source": d.get("Evidence_Source", ""),
        }
        profile = next((v for name,v in sx.DRUG_INFO.items() if name.casefold()==str(d.get("Drug","")).casefold()), {})
        d["drug_class"] = profile.get("class", "")
        d["drug_purpose"] = profile.get("purpose", d.get("Indications", ""))
        d["common_uses"] = profile.get("common_uses", [])
        try:
            d["molecular_descriptors"] = clean(sx.molecular_descriptors(str(d.get("SMILES", ""))))
        except Exception:
            d["molecular_descriptors"] = {}
        try:
            d["admet"] = clean(e["preds"].loc[str(d.get("SMILES", ""))].to_dict())
        except Exception:
            d["admet"] = {}
        try:
            rag_query = f"{d.get('Drug','')} {patient.get('diagnosis','')} {patient.get('symptom','')} indications contraindications warnings adverse reactions interactions"
            # Retrieve several chunks for the selected drug. The catalog is chunked by
            # evidence type (indications, contraindications, warnings, adverse reactions,
            # interactions), so five displayed items are genuinely distinct evidence chunks.
            raw_rag = e["retriever"].search(rag_query, top_k=max(25, len(e["catalog"])))
            same_drug = [x for x in raw_rag if str(x.get("drug_name","" )).strip().casefold() == str(d.get("Drug","" )).strip().casefold()]
            evidence_items=[]
            seen=set()
            for x in same_drug:
                text=str(x.get("text","")).strip()
                if text and text not in seen:
                    evidence_items.append(x); seen.add(text)
            field_chunks = [
                ("Indications / common uses", d["evidence"].get("indications")),
                ("Contraindications", d["evidence"].get("contraindications")),
                ("Warnings", d["evidence"].get("warnings")),
                ("Adverse reactions", d["evidence"].get("adverse_reactions")),
                ("Drug interactions", d["evidence"].get("drug_interactions")),
            ]
            for title, text in field_chunks:
                text=str(text or "").strip()
                if text and text not in seen:
                    evidence_items.append({"drug_name":d.get("Drug"),"score":float(d.get("Evidence_Score",0)),"text":f"{title}: {text}","source":d["evidence"].get("source") or "SafeRx project evidence profile","evidence_type":title})
                    seen.add(text)
            # If a live PubMed enrichment exists, expose distinct paper records before
            # falling back to structured catalog fields. Never duplicate an item just to
            # make the count look larger.
            d["rag_evidence"] = clean(evidence_items[:3])
            d["rag_metadata"] = {"retriever_mode": getattr(e["retriever"], "mode", e["retriever"].__class__.__name__), "distinct_chunks": len(evidence_items[:3]), "evidence_note": "Chunks are drug-specific and deduplicated; external literature appears when live enrichment is enabled."}
        except Exception:
            d["rag_evidence"] = []
            d["rag_metadata"] = {"retriever_mode": getattr(e["retriever"], "mode", e["retriever"].__class__.__name__), "distinct_chunks": 0, "evidence_note": "RAG retrieval was unavailable for this candidate."}
        rows.append(d)
    return {"patient": clean(patient), "patient_assessment": clean(assessment), "recommendations": rows, "ranking_method": {"condition_aware": True, "weights": {"relevance": 0.35, "bioavailability": 0.25, "safety": 0.25, "evidence": 0.15}, "patient_penalty_multiplier": 0.30}, "engine": {"name": "SafeRx ranking engine", "connected": True}}


# A small, transparent chemistry knowledge layer is intentionally kept separate
# from the biomedical RAG corpus. This prevents a chemistry question from being
# answered with an unrelated drug snippet simply because it is the nearest vector.
CHEMISTRY_KB = {
    "phosphorus pentachloride": {"formula":"PCl₅","name":"Phosphorus pentachloride","elements":"Phosphorus (P) + Chlorine (Cl)","composition":"1 phosphorus atom + 5 chlorine atoms","molecular_weight":"208.22 g/mol","geometry":"Trigonal bipyramidal (gas-phase molecular geometry)","explanation":"The prefix penta- means 5. Phosphorus is P and chlorine is Cl, so one phosphorus combines with five chlorine atoms: PCl₅."},
    "sulfur dioxide": {"formula":"SO₂","name":"Sulfur dioxide","elements":"Sulfur (S) + Oxygen (O)","composition":"1 sulfur atom + 2 oxygen atoms","molecular_weight":"64.06 g/mol","geometry":"Bent molecular geometry","explanation":"Sulfur is S and oxygen is O. The subscript 2 means two oxygen atoms, giving SO₂."},
    "sulfur trioxide": {"formula":"SO₃","name":"Sulfur trioxide","elements":"Sulfur (S) + Oxygen (O)","composition":"1 sulfur atom + 3 oxygen atoms","molecular_weight":"80.06 g/mol","geometry":"Trigonal planar around sulfur","explanation":"Sulfur is S and oxygen is O. The subscript 3 means three oxygen atoms, giving SO₃."},
    "sulfur hexafluoride": {"formula":"SF₆","name":"Sulfur hexafluoride","elements":"Sulfur (S) + Fluorine (F)","composition":"1 sulfur atom + 6 fluorine atoms","molecular_weight":"146.06 g/mol","geometry":"Octahedral molecular geometry","explanation":"The prefix hexa- means 6. Sulfur is S and fluorine is F, so SF₆ contains one sulfur and six fluorine atoms."},
    "water": {"formula":"H₂O","name":"Water","elements":"Hydrogen (H) + Oxygen (O)","composition":"2 hydrogen atoms + 1 oxygen atom","molecular_weight":"18.015 g/mol","geometry":"Bent molecular geometry","explanation":"H is hydrogen and O is oxygen. The subscript 2 means there are two hydrogen atoms."},
    "sodium chloride": {"formula":"NaCl","name":"Sodium chloride","elements":"Sodium (Na) + Chlorine (Cl)","composition":"1 sodium + 1 chlorine","molecular_weight":"58.44 g/mol","geometry":"Ionic crystal lattice","explanation":"Na represents sodium and Cl represents chlorine. A 1:1 ratio gives NaCl."},
    "carbon dioxide": {"formula":"CO₂","name":"Carbon dioxide","elements":"Carbon (C) + Oxygen (O)","composition":"1 carbon atom + 2 oxygen atoms","molecular_weight":"44.01 g/mol","geometry":"Linear molecular geometry","explanation":"C is carbon and O is oxygen. The subscript 2 means two oxygen atoms."},
    "ammonia": {"formula":"NH₃","name":"Ammonia","elements":"Nitrogen (N) + Hydrogen (H)","composition":"1 nitrogen atom + 3 hydrogen atoms","molecular_weight":"17.03 g/mol","geometry":"Trigonal pyramidal molecular geometry","explanation":"N is nitrogen and H is hydrogen. The subscript 3 means three hydrogen atoms."},
    "methane": {"formula":"CH₄","name":"Methane","elements":"Carbon (C) + Hydrogen (H)","composition":"1 carbon atom + 4 hydrogen atoms","molecular_weight":"16.04 g/mol","geometry":"Tetrahedral molecular geometry","explanation":"C is carbon and the prefix tetra- indicates four hydrogen atoms, giving CH₄."},
    "hydrogen chloride": {"formula":"HCl","name":"Hydrogen chloride","elements":"Hydrogen (H) + Chlorine (Cl)","composition":"1 hydrogen atom + 1 chlorine atom","molecular_weight":"36.46 g/mol","geometry":"Linear diatomic molecule","explanation":"Hydrogen is H and chlorine is Cl, so the 1:1 formula is HCl."},
    "nitric acid": {"formula":"HNO₃","name":"Nitric acid","elements":"Hydrogen (H) + Nitrogen (N) + Oxygen (O)","composition":"1 hydrogen + 1 nitrogen + 3 oxygen atoms","molecular_weight":"63.01 g/mol","geometry":"Trigonal planar around nitrogen","explanation":"The formula contains one H, one N and three O atoms: HNO₃."},
    "sulfuric acid": {"formula":"H₂SO₄","name":"Sulfuric acid","elements":"Hydrogen (H) + Sulfur (S) + Oxygen (O)","composition":"2 hydrogen + 1 sulfur + 4 oxygen atoms","molecular_weight":"98.08 g/mol","geometry":"Tetrahedral arrangement around sulfur","explanation":"The subscripts indicate two H atoms and four O atoms around one sulfur atom: H₂SO₄."},
    "aspirin": {"formula":"C₉H₈O₄","name":"Aspirin (acetylsalicylic acid)","elements":"Carbon (C) + Hydrogen (H) + Oxygen (O)","composition":"9 carbon + 8 hydrogen + 4 oxygen atoms","molecular_weight":"180.16 g/mol","geometry":"Aromatic planar ring system with ester and carboxylic acid groups","explanation":"Aspirin is acetylsalicylic acid. Its molecular formula is C₉H₈O₄ and its molar mass is about 180.16 g/mol."},
    "ibuprofen": {"formula":"C₁₃H₁₈O₂","name":"Ibuprofen","elements":"Carbon (C) + Hydrogen (H) + Oxygen (O)","composition":"13 carbon + 18 hydrogen + 2 oxygen atoms","molecular_weight":"206.28 g/mol","geometry":"Aromatic ring with flexible side chain","explanation":"Ibuprofen contains 13 carbon, 18 hydrogen and 2 oxygen atoms: C₁₃H₁₈O₂."},
    "acetaminophen": {"formula":"C₈H₉NO₂","name":"Acetaminophen (paracetamol)","elements":"Carbon (C) + Hydrogen (H) + Nitrogen (N) + Oxygen (O)","composition":"8 carbon + 9 hydrogen + 1 nitrogen + 2 oxygen atoms","molecular_weight":"151.16 g/mol","geometry":"Aromatic ring system","explanation":"Acetaminophen, also called paracetamol, has formula C₈H₉NO₂ and molecular weight about 151.16 g/mol."},
    "paracetamol": {"formula":"C₈H₉NO₂","name":"Paracetamol (acetaminophen)","elements":"Carbon (C) + Hydrogen (H) + Nitrogen (N) + Oxygen (O)","composition":"8 carbon + 9 hydrogen + 1 nitrogen + 2 oxygen atoms","molecular_weight":"151.16 g/mol","geometry":"Aromatic ring system","explanation":"Paracetamol, also called acetaminophen, has formula C₈H₉NO₂ and molecular weight about 151.16 g/mol."},
}
FORMULA_TO_NAME = {v["formula"].replace("₅","5").replace("₆","6").replace("₂","2").replace("₃","3").replace("₄","4"): v["name"] for v in CHEMISTRY_KB.values()}

def _chemistry_answer(question: str):
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    for name, item in CHEMISTRY_KB.items():
        if re.search(r"\b" + re.escape(name) + r"\b", q):
            wants_formula = any(x in q for x in ["formula", "chemical formula", "composition", "write", "symbol"])
            wants_weight = any(x in q for x in ["molecular weight", "molar mass", "molecular mass", "weight"])
            if wants_weight:
                answer = item["molecular_weight"]
            else:
                answer = item["formula"]
            return {"type":"chemistry", **item, "answer":answer, "question_kind":"molecular weight" if wants_weight else "formula"}
    # Generic chemistry fallback: resolve a named compound through PubChem for
    # formula / molecular-weight questions, so the Q&A is not limited to the
    # small local teaching knowledge base.
    entity_match = re.search(r"(?:following compound|compound|formula for|formula of|molecular weight of|molar mass of|name of)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 -]{1,80})", question or "", flags=re.I)
    if entity_match:
        entity = re.sub(r"\s+", " ", entity_match.group(1)).strip(" .?!")
        info = sx.pubchem_lookup(entity)
        if info.get("molecular_formula") or info.get("molecular_weight"):
            wants_weight = any(x in q for x in ["molecular weight", "molar mass", "molecular mass", "weight"])
            answer = info.get("molecular_weight") if wants_weight else info.get("molecular_formula")
            if answer:
                return {"type":"chemistry","answer":str(answer),"question_kind":"molecular weight" if wants_weight else "formula","name":info.get("title") or entity,"formula":info.get("molecular_formula") or "—","elements":"Resolved through PubChem","composition":"See molecular formula","molecular_weight":info.get("molecular_weight") or "—","geometry":"See PubChem structure information","explanation":f"PubChem identifies {info.get('title') or entity} with molecular formula {info.get('molecular_formula') or '—'} and molecular weight {info.get('molecular_weight') or '—'}."}

    m = re.search(r"\b([A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)+)\b", question or "")
    if m:
        raw = m.group(1)
        key = raw.replace("₅","5").replace("₆","6").replace("₂","2").replace("₃","3").replace("₄","4")
        if key in FORMULA_TO_NAME:
            item = next(v for v in CHEMISTRY_KB.values() if v["name"] == FORMULA_TO_NAME[key])
            return {"type":"chemistry","answer":item["name"],"question_kind":"name",**item}
    return None

GENERAL_DRUG_EVIDENCE = {
    "aspirin": {"name":"Aspirin","indications":"Analgesic and antiplatelet medicine with uses that depend on dose and clinical context.","contraindications":"Important contraindications include aspirin/NSAID hypersensitivity and situations with significant bleeding risk; verify current labeling.","warnings":"Bleeding and gastrointestinal adverse effects are important safety considerations. Verify current label and patient-specific factors.","adverse":"Gastrointestinal irritation, bleeding and hypersensitivity reactions are recognized safety topics.","interactions":"Interactions can occur with anticoagulants, antiplatelet medicines and other drugs; verify current authoritative labeling."},
    "ibuprofen": {"name":"Ibuprofen","indications":"NSAID used for pain, fever and inflammatory conditions.","contraindications":"Important contraindications include certain NSAID hypersensitivity and some high-risk gastrointestinal or cardiovascular situations; verify current labeling.","warnings":"Gastrointestinal bleeding, kidney effects and cardiovascular risks are important safety topics.","adverse":"Dyspepsia, gastrointestinal irritation and other NSAID-related adverse effects may occur.","interactions":"Review concurrent anticoagulants, antiplatelet drugs, other NSAIDs and medicines affecting kidney function."},
    "paracetamol": {"name":"Paracetamol / Acetaminophen","indications":"Analgesic and antipyretic medicine.","contraindications":"Do not exceed labeled dosing; severe liver disease requires particular clinical consideration.","warnings":"Excess total daily exposure can cause serious liver injury.","adverse":"Nausea and other reactions may occur; severe overdose can cause liver toxicity.","interactions":"Check combination products to avoid duplicate acetaminophen exposure and review relevant medicines."},
}

def _drug_match(catalog, query):
    q = re.sub(r"\s+", " ", str(query).strip().casefold())
    names = [(str(x).strip(), str(x).strip().casefold()) for x in catalog["drug_name"].dropna().tolist()]
    exact = [n for n,k in names if k == q]
    if exact: return exact[0]
    contained = [n for n,k in names if k and re.search(r"\b"+re.escape(k)+r"\b", q)]
    return max(contained, key=len) if contained else None

@app.post("/api/evidence")
def evidence(req: EvidenceRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Enter a question.")
    chemistry = _chemistry_answer(question)
    if chemistry:
        technical = {k: chemistry[k] for k in ["name","formula","elements","composition","molecular_weight","geometry"]}
        return {"question":question,"mode":"chemistry","matched":True,"answer":chemistry["answer"],"explanation":chemistry["explanation"],"technical":technical,"results":[],"retrieval":{"type":"deterministic chemistry knowledge layer","chunks":1},"message":"Direct chemistry answer found. Expand the research details for composition and molecular geometry."}

    e = ensure_engine()
    try:
        named = _drug_match(e["catalog"], question)
        fallback = GENERAL_DRUG_EVIDENCE.get(str(named or "").casefold()) if named else None
        if not named and not fallback:
            return {"question":question,"mode":"biomedical","matched":False,"answer":"I could not identify a supported medicine in this question.","explanation":"SafeRx first identifies the medicine, then retrieves evidence tied to that medicine. This avoids presenting an unrelated drug as an answer.","technical":{},"results":[],"sections":[],"retrieval":{"type":getattr(e.get("retriever"), "mode", "BioLinkBERT + FAISS"),"chunks":0},"message":"Try a supported medicine such as Atorvastatin, Metformin, Aspirin, Amlodipine, Cetirizine, Omeprazole, Ibuprofen or Paracetamol."}
        allowed = str(named or "").casefold()
        results = []
        row = None
        if named:
            raw = e["retriever"].search(question, top_k=max(25, req.top_k*5))
            results = [r for r in raw if str(r.get("drug_name","")).strip().casefold() == allowed]
            matches = e["catalog"][e["catalog"]["drug_name"].astype(str).str.strip().str.casefold() == allowed]
            if not matches.empty:
                row = matches.iloc[0]
                field_map=[("Indications / common uses","fda_indications"),("Contraindications","fda_contraindications"),("Warnings","fda_warnings"),("Adverse reactions","fda_adverse_reactions"),("Drug interactions","fda_drug_interactions")]
                seen={str(x.get("text","")).strip() for x in results}
                for label,col in field_map:
                    text=str(row.get(col,"") or "").strip()
                    chunk=f"{label}: {text}"
                    if text and chunk not in seen:
                        results.append({"drug_name":named,"score":0.0,"text":chunk,"source":str(row.get("fda_source") or "SafeRx project evidence profile"),"evidence_type":label})
                        seen.add(chunk)
                results=results[:req.top_k]
        if row is not None:
            sections = [
                {"title":"Indications / usage","text":str(row.get("fda_indications","")).strip()},
                {"title":"Contraindications","text":str(row.get("fda_contraindications","")).strip()},
                {"title":"Warnings","text":str(row.get("fda_warnings","")).strip()},
                {"title":"Adverse reactions","text":str(row.get("fda_adverse_reactions","")).strip()},
                {"title":"Drug interactions","text":str(row.get("fda_drug_interactions","")).strip()},
            ]
        else:
            sections = [
                {"title":"Indications / usage","text":fallback["indications"]},
                {"title":"Contraindications","text":fallback["contraindications"]},
                {"title":"Warnings","text":fallback["warnings"]},
                {"title":"Adverse reactions","text":fallback["adverse"]},
                {"title":"Drug interactions","text":fallback["interactions"]},
            ]
        sections = [x for x in sections if x["text"] and x["text"].lower() not in {"nan","none"}]
        pubmed_text = str(row.get("pubmed_text","")).strip() if row is not None else ""
        paper_count = int(row.get("pubmed_count",0) or 0) if row is not None else 0
        if pubmed_text:
            sections.append({"title":"PubMed research context","text":pubmed_text[:5000]})
        source_label = str(row.get("fda_source","")) if row is not None else "Educational drug reference fallback"
        summary = f"Evidence was assembled for {named or fallback.get('name')}. It includes structured safety topics and retrieval context where available."
        return {"question":question,"mode":"biomedical","matched":True,"answer":summary,"explanation":"The response is tied to the named medicine. Live regulatory/literature fields are shown when available; otherwise the UI clearly labels the educational fallback profile. Retrieval scores are similarity signals, not clinical certainty.","technical":{"drug":named or fallback.get("name"),"pubmed_count":paper_count,"source":source_label,"retrieval_model":getattr(e.get("retriever"), "mode", "BioLinkBERT + FAISS")},"sections":sections,"results":clean(results),"retrieval":{"type":getattr(e.get("retriever"), "mode", "BioLinkBERT + FAISS"),"chunks":len(results)},"message":f"Retrieved {len(results)} catalog evidence chunk(s) for {named or fallback.get('name')}."}
    except Exception as ex:
        raise HTTPException(500, f"Evidence retrieval failed: {ex}")


@app.get("/api/dashboard")
def dashboard():
    e = ensure_engine()
    try:
        cats = e["catalog"]
        risk_rows=[]
        for smi in cats["smiles"].dropna().astype(str):
            if smi in e["preds"].index:
                risk_rows.append(score_payload(e["preds"].loc[smi]))
        risks=[r["riskScore100"] for r in risk_rows]
        bands={"Lower":0,"Moderate":0,"Higher":0,"Very high":0}
        for x in risks:
            bands["Lower" if x<15 else "Moderate" if x<30 else "Higher" if x<60 else "Very high"]+=1
        return clean({"patientCount":len(e["patients"]),"drugCount":len(cats),"admetEndpoints":int(len(e["preds"].columns)),"riskDistribution":[{"category":k,"count":v} for k,v in bands.items()],"drugCandidates":cats[["drug_name","pubchem_cid"]].head(20).to_dict(orient="records")})
    except Exception as ex:
        raise HTTPException(500, f"Dashboard data failed: {ex}")

@app.get("/api/evaluation")
def evaluation():
    e = ensure_engine()
    try:
        metrics = sx.evaluate_recommendation_system(e["patients"], e["catalog"], e["preds"], e["retriever"], max_rows=200)
    except Exception as ex:
        raise HTTPException(500, f"Evaluation failed: {ex}")
    out = {}
    for k, v in metrics.items():
        if hasattr(v, "to_dict"):
            out[k] = v.to_dict(orient="records")
        else:
            out[k] = v
    return clean(out)


@app.get("/api/explain/{drug}")
def explain(drug: str):
    e = ensure_engine()
    query = drug.strip()
    if not query:
        raise HTTPException(400, "Enter a drug name or SMILES.")
    # First try the full project recommendation catalog, then fall back to live/local
    # molecule resolution so Explainability is not artificially limited to the demo catalog.
    matches = e["catalog"][e["catalog"]["drug_name"].astype(str).str.strip().str.casefold() == query.casefold()]
    if not matches.empty:
        c = matches.iloc[0]
        smi = str(c.get("smiles", ""))
        pred = e["preds"].loc[smi] if smi in e["preds"].index else None
        if pred is None:
            raise HTTPException(404, "ADMET prediction not found for this catalog drug.")
        rag_items, rag_note = molecule_rag_evidence(e, str(c.get("drug_name") or query), smi)
        return clean({
            "drug": str(c.get("drug_name") or query),
            "molecule": {"title": c.get("title") or query, "molecular_formula": c.get("molecular_formula", ""), "molecular_weight": c.get("molecular_weight", ""), "smiles": smi, "structure_svg": _structure_svg(smi), "descriptors": sx.molecular_descriptors(smi), "drug_class": c.get("drug_class", ""), "purpose": c.get("drug_purpose", ""), "common_uses": str(c.get("common_uses", "")).split(" | ") if str(c.get("common_uses", "")).strip() else []},
            "admet": pred.to_dict(), "scores": score_payload(pred),
            "contributions": [],
            "evidence": {"indications": c.get("fda_indications", ""), "contraindications": c.get("fda_contraindications", ""), "warnings": c.get("fda_warnings", ""), "adverseReactions": c.get("fda_adverse_reactions", ""), "interactions": c.get("fda_drug_interactions", "")},
            "rag_evidence": rag_items, "rag_metadata": {"retriever_mode": getattr(e.get("retriever"), "mode", "RAG"), "note": rag_note},
            "mode": "catalog_molecule"
        })
    info = sx.pubchem_lookup(query)
    smi = info.get("smiles") or (query if sx.canonicalize_smiles(query) else None)
    if not smi:
        raise HTTPException(404, "Drug or molecule could not be resolved. Enter a known drug name or valid SMILES.")
    try:
        pred = e["admet"].predict([smi]).loc[smi]
        scores = score_payload(pred)
        desc = sx.molecular_descriptors(smi)
    except Exception as ex:
        raise HTTPException(500, f"ADMET prediction failed: {ex}")
    resolved_name = str(info.get("title") or query)
    rag_items, rag_note = molecule_rag_evidence(e, resolved_name, smi)
    return clean({
        "drug": resolved_name,
        "molecule": {"title": resolved_name, "molecular_formula": info.get("molecular_formula", ""), "molecular_weight": info.get("molecular_weight", ""), "smiles": smi, "structure_svg": _structure_svg(smi), "descriptors": desc, "drug_class": "", "purpose": "Purpose/indication information should be verified against authoritative sources.", "common_uses": []},
        "admet": pred.to_dict(), "scores": scores, "contributions": [], "evidence": {},
        "rag_evidence": rag_items, "rag_metadata": {"retriever_mode": getattr(e.get("retriever"), "mode", "RAG"), "note": rag_note},
        "mode": "resolved_molecule"
    })


# ----------------------------- Patient report upload -------------------------
def _configure_tesseract():
    """Locate Tesseract on Windows without exposing or uploading document data."""
    import pytesseract
    configured = os.environ.get("TESSERACT_CMD", "").strip()
    candidates = [
        configured,
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return ""


def _preprocess_for_ocr(image):
    """Lightweight local preprocessing for photographed/scanned medical pages."""
    from PIL import ImageOps, ImageFilter
    img = image.convert("L")
    # Upscale small scans so character strokes are easier for OCR to detect.
    max_width = 2400
    if img.width < max_width:
        scale = min(2.0, max_width / max(1, img.width))
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _ocr_image(image) -> str:
    try:
        import pytesseract
        cmd = _configure_tesseract()
        if not cmd:
            return ""
        processed = _preprocess_for_ocr(image)
        # PSM 6 works well for structured single-page reports; PSM 3 is the
        # fallback for multi-block layouts. Keep all OCR local.
        text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
        if len(text.strip()) < 20:
            text = pytesseract.image_to_string(processed, config="--oem 3 --psm 3")
        return text
    except Exception:
        return ""


def _extract_pdf(data: bytes) -> str:
    """Extract searchable PDF text, then automatically OCR image-only PDFs.

    The previous implementation stopped after pypdf extraction, which returns
    an empty string for a PDF made from a scanned/photographed medical image.
    This fallback renders each page locally with PyMuPDF and OCRs the pixels.
    No document content is sent to an external service.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as ex:
        raise HTTPException(400, f"Could not read PDF. Details: {ex}")

    # Searchable PDFs are preferred. For image-only PDFs, OCR locally.
    if len(re.sub(r"\s+", "", text)) >= 40:
        return text

    try:
        import fitz  # PyMuPDF
        import PIL.Image
        pages = []
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
            image = PIL.Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            page_text = _ocr_image(image)
            if page_text.strip():
                pages.append(page_text)
        doc.close()
        ocr_text = "\n\n".join(pages).strip()
        return ocr_text or text
    except Exception:
        return text


def _extract_image(data: bytes) -> str:
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return _ocr_image(image)
    except Exception:
        return ""


def _first(patterns: list[str], text: str) -> str:
    for p in patterns:
        m = re.search(p, text, flags=re.I | re.M)
        if m:
            return m.group(1).strip(" :,-")
    return ""


def _label_value(text: str, labels: list[str]) -> str:
    """Extract a value whether the label and value are on one line or separate lines."""
    lines = [re.sub(r"\s+", " ", x).strip() for x in (text or "").splitlines() if x.strip()]
    canonical = [x.casefold().rstrip(":") for x in labels]
    stop_labels = set(canonical + [
        "patient name", "age", "gender", "sex", "date", "weight", "blood pressure",
        "chief complaint", "diagnosis", "known allergies", "allergies", "comorbidities",
        "current medications", "current medication", "previous medication", "clinical notes",
        "investigation", "system test candidate", "notice"
    ])
    for i, line in enumerate(lines):
        low = line.casefold().rstrip(":")
        for lab in canonical:
            if low == lab:
                vals=[]
                for nxt in lines[i+1:]:
                    nl=nxt.casefold().rstrip(":")
                    if nl in stop_labels:
                        break
                    vals.append(nxt)
                    if len(vals)>=3: break
                if vals: return " ".join(vals).strip()
            if low.startswith(lab + ":"):
                return line.split(":",1)[1].strip()
    return ""


def parse_patient_report(text: str) -> dict[str, Any]:
    raw = text or ""
    normalized = re.sub(r"[ \t]+", " ", raw)
    # First pass: explicit labels. Second pass: common clinical sentence forms.
    age_label = _label_value(raw, ["Age", "Patient Age"])
    age_match = re.search(r"\b(?:age|aged)\s*[:=-]?\s*(\d{1,3})\b", normalized, re.I)
    if not age_match:
        age_match = re.search(r"\b(\d{1,3})\s*(?:years? old|y/o)\b", normalized, re.I)
    age_fallback = re.search(r"\b(\d{1,3})\b", age_label or "")
    age_candidate = age_match.group(1) if age_match else (age_fallback.group(1) if age_fallback else "")
    age_val = int(age_candidate) if str(age_candidate).isdigit() and 0 < int(age_candidate) <= 120 else 45

    sex = _label_value(raw, ["Sex", "Gender"]) or _first([r"\b(?:sex|gender)\s*[:=-]\s*(male|female|other)\b"], normalized) or "Other"
    diagnosis = _label_value(raw, ["Diagnosis", "Diagnosed With", "Condition", "Primary Diagnosis"])
    symptom = _label_value(raw, ["Symptoms", "Symptom", "Chief Complaint", "Main Symptom", "Reported Symptom", "Presenting Complaint"])
    allergies = _label_value(raw, ["Known Allergies", "Allergies", "Allergy", "Drug Allergies"])
    comorbidities = _label_value(raw, ["Comorbidities", "Comorbidity", "Past Medical History", "PMH"])
    meds = _label_value(raw, ["Current Medications", "Current Medication", "Medications", "Medicines", "Medication History"])

    # Sentence-style fallbacks catch PDFs whose extractor joins labels into prose.
    diagnosis = diagnosis or _first([r"\bDiagnosis\s*[:=-]\s*([^\.\n]+)", r"\bDiagnosed with\s*[:=-]?\s*([^\.\n]+)"], normalized)
    symptom = symptom or _first([r"\b(?:Reported|Main|Presenting) symptom\s*[:=-]\s*([^\.\n]+)", r"\bChief complaint\s*[:=-]\s*([^\.\n]+)", r"\bSymptoms?\s*[:=-]\s*([^\.\n]+)"], normalized)
    allergies = allergies or _first([r"\b(?:Known )?Allerg(?:y|ies)\s*[:=-]\s*([^\.\n]+)"], normalized)
    comorbidities = comorbidities or _first([r"\bComorbidit(?:y|ies)\s*[:=-]\s*([^\.\n]+)"], normalized)
    meds = meds or _first([r"\bCurrent medications?\s*[:=-]\s*([^\.\n]+)", r"\bMedication(?:s)?\s*[:=-]\s*([^\.\n]+)"], normalized)

    def tidy(x):
        v = re.sub(r"\s+", " ", str(x or "")).strip(" :,-")
        return "" if v.casefold() in {"none", "n/a", "na", "not reported", "unknown"} else v
    values = {"diagnosis":tidy(diagnosis),"symptom":tidy(symptom),"allergies":tidy(allergies),"comorbidities":tidy(comorbidities),"current_medications":tidy(meds)}
    detected = sum(bool(v) for v in [age_match or age_label, sex and sex != "Other", *values.values()])
    return {"age":age_val,"sex":tidy(sex).title(),**values,"raw_text":raw[:16000],"fields_detected":int(detected),"extraction_note":"Fields are extracted automatically and should be reviewed before research ranking."}


@app.post("/api/report/analyze")
async def analyze_report(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Choose a PDF or image.")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "Supported formats: PDF, PNG, JPG, JPEG and WEBP.")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "File is too large. Maximum size is 15 MB.")
    text = _extract_pdf(data) if ext == ".pdf" else _extract_image(data)
    if not text.strip():
        tesseract_hint = " Set TESSERACT_CMD if Tesseract is installed outside the standard Windows locations."
        raise HTTPException(422, "No readable text was detected. This file appears to require OCR. Install Tesseract OCR and retry." + tesseract_hint)
    parsed = parse_patient_report(text)
    return {
        "filename": file.filename,
        "fileType": ext[1:].upper(),
        "patient": clean(parsed),
        "extraction": {
            "method": "PDF text extraction + local OCR fallback" if ext == ".pdf" else "local Tesseract OCR",
            "privacy": "Document processing is performed locally by this application; no OCR upload service is used.",
        },
    }


# ----------------------------- PDF report generation -------------------------
def _pdf_text(canvas, text, x, y, width=500, size=10, leading=14):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = str(text or "").split()
    line = ""
    lines = []
    for word in words:
        candidate = f"{line} {word}".strip()
        if stringWidth(candidate, "Helvetica", size) <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    canvas.setFont("Helvetica", size)
    for line in lines:
        if y < 55:
            canvas.showPage(); y = 760; canvas.setFont("Helvetica", size)
        canvas.drawString(x, y, line[:120])
        y -= leading
    return y


@app.post("/api/report/pdf")
def report_pdf(req: ReportPdfRequest):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
    except Exception as ex:
        raise HTTPException(500, f"PDF generator dependency missing: {ex}")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 55
    c.setTitle(req.title)
    c.setFillColor(colors.HexColor("#172554")); c.setFont("Helvetica-Bold", 22)
    c.drawString(42, y, "SafeRx AI")
    y -= 28
    c.setFillColor(colors.HexColor("#475569")); c.setFont("Helvetica", 10)
    c.drawString(42, y, req.title)
    y -= 28
    c.setStrokeColor(colors.HexColor("#cbd5e1")); c.line(42, y, width - 42, y); y -= 24

    report_type = (req.report_type or "complete").strip().lower()
    c.setFillColor(colors.HexColor("#0f172a")); c.setFont("Helvetica-Bold", 13)
    heading = {"patient":"Patient Report", "molecule":"Molecule / ADMET Report", "recommendation":"Drug Recommendation Report", "complete":"Patient / analysis summary"}.get(report_type, "Patient / analysis summary")
    c.drawString(42, y, heading); y -= 20
    patient = req.patient or {}
    if report_type in {"patient", "recommendation", "complete"}:
        for key, label in [("age", "Age"), ("sex", "Sex"), ("diagnosis", "Diagnosis"), ("symptom", "Main symptom"), ("allergies", "Allergies"), ("comorbidities", "Comorbidities"), ("current_medications", "Current medications")]:
            if patient.get(key) not in (None, ""):
                c.setFont("Helvetica-Bold", 9); c.drawString(48, y, f"{label}:")
                c.setFont("Helvetica", 9); y = _pdf_text(c, patient.get(key), 130, y, width - 175, 9, 12); y -= 2
        y -= 8

    assessment = patient.get("assessment") or patient.get("patient_assessment")
    if assessment and report_type in {"patient", "recommendation", "complete"}:
        c.setFont("Helvetica-Bold", 11); c.drawString(48, y, "Patient problem interpretation"); y -= 16
        for label, key in [("Identified problem", "identified_problem"), ("Reported symptoms", "reported_symptoms"), ("Why it may be happening", "why_it_may_be_happening")]:
            if assessment.get(key):
                c.setFont("Helvetica-Bold", 8); c.drawString(52, y, label + ":")
                c.setFont("Helvetica", 8); y = _pdf_text(c, assessment.get(key), 155, y, width-205, 8, 11); y -= 2
        y -= 5

    if req.molecule and report_type in {"molecule", "recommendation", "complete"}:
        c.setFont("Helvetica-Bold", 13); c.drawString(42, y, "Molecule / ADMET result"); y -= 20
        m = req.molecule; scores = m.get("scores", {}); mol = m.get("molecule", {})
        c.setFont("Helvetica-Bold", 11); c.drawString(48, y, str(mol.get("title") or mol.get("drug_name") or "Molecule")); y -= 17
        formula = str(mol.get("molecular_formula") or "")
        mw = str(mol.get("molecular_weight") or "")
        if formula or mw:
            c.setFont("Helvetica", 9); c.drawString(48, y, f"Formula: {formula or '—'}    Molecular weight: {mw or '—'} g/mol"); y -= 15
        smiles = str(mol.get("smiles") or "")
        png = _structure_png(smiles)
        if png and report_type == "molecule":
            from reportlab.lib.utils import ImageReader
            y -= 5
            c.drawImage(ImageReader(io.BytesIO(png)), 48, y-145, width=300, height=145, preserveAspectRatio=True, mask='auto')
            y -= 158
        c.setFont("Helvetica", 9)
        c.drawString(48, y, f"Toxicity: {float(scores.get('toxicity',0)):.2f} | Bioavailability: {float(scores.get('bioavailability',0)):.2f} | Safety: {float(scores.get('safety',0)):.2f}"); y -= 15
        c.drawString(48, y, f"Risk level: {scores.get('riskLevel','—')} | Risk score: {float(scores.get('riskScore100',0)):.2f}/100"); y -= 15
        purpose = mol.get("purpose") or "Purpose/indication information should be verified against authoritative sources."
        c.setFont("Helvetica-Bold", 9); c.drawString(48, y, "What is this drug / purpose:"); y -= 12
        c.setFont("Helvetica", 9); y = _pdf_text(c, purpose, 48, y, width-95, 9, 12); y -= 5
        uses = mol.get("common_uses") or []
        if uses:
            c.setFont("Helvetica-Bold", 9); c.drawString(48, y, "Common uses:"); y -= 12
            for use in uses:
                y = _pdf_text(c, "• " + str(use), 58, y, width-105, 9, 12); y -= 1
        drug_class = mol.get("drug_class") or ""
        if drug_class:
            c.setFont("Helvetica-Bold", 9); c.drawString(48, y, "Drug class:"); c.setFont("Helvetica", 9); c.drawString(105, y, str(drug_class)); y -= 15
        descriptors = mol.get("descriptors") or {}
        if descriptors:
            y -= 5
            c.setFont("Helvetica-Bold", 11); c.drawString(48, y, "Molecular descriptors"); y -= 15
            items = list(descriptors.items())
            for i in range(0, len(items), 3):
                chunk = items[i:i+3]
                x = 52
                for k, v in chunk:
                    try: val = f"{float(v):.3f}"
                    except Exception: val = str(v)
                    c.setFont("Helvetica-Bold", 7.5); c.drawString(x, y, str(k)[:18])
                    c.setFont("Helvetica", 8); c.drawString(x, y-10, val)
                    x += 165
                y -= 24
            # Lipinski-style checks where the required descriptors are available.
            c.setFont("Helvetica-Bold", 11); c.drawString(48, y, "Drug-likeness / Lipinski-style checks"); y -= 14
            checks=[]
            limits={"MW":500,"LogP":5,"HBD":5,"HBA":10}
            aliases={"MW":["MW","MolWt"],"LogP":["LogP"],"HBD":["HBD"],"HBA":["HBA"]}
            for label, lim in limits.items():
                val=None
                for key in aliases[label]:
                    if key in descriptors:
                        try: val=float(descriptors[key]); break
                        except Exception: pass
                if val is not None: checks.append(f"{label}={val:.2f} {'PASS' if val<=lim else 'FAIL'} (rule reference <= {lim:g})")
            if checks:
                y = _pdf_text(c, " | ".join(checks), 52, y, width-105, 8, 11) - 6

        admet = m.get("admet") or {}
        if admet:
            if y < 260: c.showPage(); y = height - 55
            c.setFont("Helvetica-Bold", 11); c.drawString(48, y, "ADMET model predictions"); y -= 15
            c.setFont("Helvetica-Bold", 7.5); c.drawString(50, y, "Endpoint"); c.drawString(190, y, "Prediction"); c.drawString(275, y, "Interpretation")
            y -= 12
            for k,v in list(admet.items())[:40]:
                if y < 55: c.showPage(); y = height - 55
                try: val=f"{float(v):.5f}"; interp="Higher model signal" if float(v)>=.5 else "Lower model signal"
                except Exception: val=str(v); interp="Model output"
                c.setFont("Helvetica", 7.5); c.drawString(50, y, str(k)[:25]); c.drawString(190, y, val); c.drawString(275, y, interp)
                y -= 10
            y -= 8
            c.setFont("Helvetica-Oblique", 7.5); y = _pdf_text(c, "ADMET values are computational model predictions, not laboratory measurements or clinical test results.", 48, y, width-95, 7.5, 10) - 8

        # Drug-specific RAG evidence is part of the standalone Molecule / ADMET report.
        rag_items = m.get("rag_evidence") or []
        rag_meta = m.get("rag_metadata") or {}
        if report_type == "molecule":
            if y < 220: c.showPage(); y = height - 55
            c.setFont("Helvetica-Bold", 11); c.drawString(48, y, "RAG / Biomedical evidence"); y -= 15
            c.setFont("Helvetica", 7.5)
            y = _pdf_text(c, str(rag_meta.get("note") or "Evidence retrieved from the SafeRx biomedical evidence layer."), 48, y, width-95, 7.5, 10) - 4
            if rag_items:
                for i, evx in enumerate(rag_items[:3], 1):
                    if y < 95: c.showPage(); y = height - 55
                    c.setFont("Helvetica-Bold", 8);
                    label = f"Evidence {i}: {evx.get('evidence_type','Biomedical evidence')} | {evx.get('source','SafeRx evidence')} | score {float(evx.get('score',0) or 0):.3f}"
                    y = _pdf_text(c, label, 52, y, width-105, 7.5, 9) - 1
                    c.setFont("Helvetica", 7.5)
                    y = _pdf_text(c, str(evx.get('text') or ''), 60, y, width-115, 7.5, 9) - 3
            else:
                y = _pdf_text(c, "No drug-specific RAG evidence was available in the current indexed corpus. No evidence was fabricated.", 52, y, width-105, 7.5, 10) - 4

    if req.recommendations and report_type in {"recommendation", "complete"}:
        c.setFont("Helvetica-Bold", 13); c.drawString(42, y, "Ranked candidates"); y -= 20
        for r in req.recommendations[:10]:
            c.setFont("Helvetica-Bold", 10); c.drawString(48, y, f"#{r.get('Rank','—')}  {r.get('Drug','Candidate')}"); y -= 14
            c.setFont("Helvetica", 9)
            line = f"Recommendation {float(r.get('Final_Recommendation_Score',0)):.1f}/100 | Safety {float(r.get('Safety_Score',0)):.2f} | Bioavailability {float(r.get('Bioavailability_Score',0)):.2f} | Risk {float(r.get('Risk_Score_100',0)):.1f}/100"
            y = _pdf_text(c, line, 60, y, width - 110, 9, 12); y -= 2
            why = r.get("why_recommended") or ""
            if why:
                y = _pdf_text(c, "Why ranked: " + why, 60, y, width - 110, 8, 11); y -= 2
            ev = r.get("evidence") or {}
            indication = ev.get("indications") or ""
            if indication:
                y = _pdf_text(c, "Evidence / indication: " + indication, 60, y, width - 110, 8, 11); y -= 5
            rag = r.get("rag_evidence") or []
            if rag:
                c.setFont("Helvetica-Bold", 8); c.drawString(60, y, "RAG evidence (up to 3 distinct evidence items):"); y -= 11
                for evx in rag[:5]:
                    label = f"[{evx.get('evidence_type','Evidence')}] {evx.get('source','SafeRx evidence')}"
                    y = _pdf_text(c, label, 68, y, width-120, 7.5, 9); y -= 1
                    y = _pdf_text(c, str(evx.get('text','')), 76, y, width-130, 7.5, 9); y -= 2

    if req.interpretation:
        y -= 5; c.setFont("Helvetica-Bold", 13); c.drawString(42, y, "Plain-language interpretation"); y -= 20
        y = _pdf_text(c, req.interpretation, 48, y, width - 95, 9, 13)

    y -= 25
    c.setFillColor(colors.HexColor("#64748b")); c.setFont("Helvetica-Oblique", 8)
    _pdf_text(c, "Research/educational decision-support only. Model outputs are not clinical thresholds, diagnosis, treatment advice, or a prescription. Verify against authoritative clinical sources and qualified professionals.", 42, y, width - 84, 8, 11)
    c.save(); buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="SafeRx_AI_Report.pdf"'})

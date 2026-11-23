# SafeRx AI - Complete Implementation
# Adds the missing recommendation, personalization, live evidence, safety validation,
# explainability, and evaluation layers while preserving ADMET-AI as the core ADMET engine.

import os, re, json, time, math, hashlib, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, AllChem

# Optional packages are loaded with clear messages so the notebook can still boot.
try:
    import faiss
except Exception:
    faiss = None

try:
    from admet_ai import ADMETModel
except Exception:
    ADMETModel = None

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except Exception:
    AutoTokenizer = AutoModel = torch = None

try:
    import shap
except Exception:
    shap = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import gradio as gr
except Exception:
    gr = None

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ROOT = Path(os.environ.get("SAFERX_ROOT", "/content/saferx_complete"))
ROOT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

PATIENT_CSV = os.environ.get(
    "SAFERX_PATIENT_CSV",
    "/content/drive/MyDrive/Colab Notebooks/major project/synthetic_medical_reports_1000 (1).csv",
)

BIOLINKBERT_NAME = "michiyasunaga/BioLinkBERT-base"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # optional; not required for deterministic reports

# Paper-aligned multi-criteria ranking weights. These are configurable research weights,
# not clinically validated prescribing weights.
W_RELEVANCE = 0.35
W_BIOAVAIL = 0.25
W_SAFETY = 0.25
W_EVIDENCE = 0.15

# Existing project toxicity groups.
MAJOR_SAFETY_COLS = ["DILI", "hERG", "ClinTox", "Skin_Reaction", "Carcinogens_Lagunin"]
GENOTOXICITY_COLS = ["AMES", "SR-ATAD5", "SR-p53"]
RECEPTOR_COLS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma"]
CELLULAR_STRESS_COLS = ["SR-ARE", "SR-HSE", "SR-MMP"]

# -----------------------------------------------------------------------------
# HTTP/CACHE HELPERS
# -----------------------------------------------------------------------------
session = requests.Session()
session.headers.update({"User-Agent": "SafeRx-AI/1.0 research-project"})


def _cache_path(prefix: str, key: str, suffix: str = ".json") -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return CACHE / f"{prefix}_{digest}{suffix}"


def get_json(url: str, params: Optional[dict] = None, timeout: int = 30, cache_prefix: Optional[str] = None):
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    path = _cache_path(cache_prefix or "http", key)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
    except Exception as e:
        warnings.warn(f"Request failed: {url} | {e}")
    return None

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
def load_patient_data(path: str = PATIENT_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Patient CSV not found: {path}")
    df = pd.read_csv(path)
    required = ["patient_id", "age", "sex", "diagnosis", "drug_name", "dosage", "symptom", "report_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df

# -----------------------------------------------------------------------------
# CHEMINFORMATICS
# -----------------------------------------------------------------------------
def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def molecular_descriptors(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {}
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RotB": float(Lipinski.NumRotatableBonds(mol)),
        "Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "HeavyAtoms": float(Lipinski.HeavyAtomCount(mol)),
    }


def morgan_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr

# -----------------------------------------------------------------------------
# PUBCHEM / OPENFDA / PUBMED
# -----------------------------------------------------------------------------
def pubchem_lookup(drug_name: str) -> Dict:
    data = get_json(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + requests.utils.quote(str(drug_name)) + "/property/Title,IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,ConnectivitySMILES,InChIKey/JSON",
        cache_prefix="pubchem_prop",
    )
    try:
        p = data["PropertyTable"]["Properties"][0]
        return {
            "drug_name": drug_name,
            "pubchem_cid": p.get("CID"),
            "title": p.get("Title"),
            "iupac_name": p.get("IUPACName"),
            "molecular_formula": p.get("MolecularFormula"),
            "molecular_weight": p.get("MolecularWeight"),
            "smiles": p.get("ConnectivitySMILES") or p.get("CanonicalSMILES") or p.get("IsomericSMILES"),
            "inchikey": p.get("InChIKey"),
        }
    except Exception:
        return {"drug_name": drug_name}


def pubchem_annotation_text(cid) -> str:
    if not cid:
        return ""
    data = get_json(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{int(cid)}/JSON",
        cache_prefix="pubchem_view",
    )
    if not data:
        return ""
    texts = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "String" and isinstance(v, str):
                    texts.append(v)
                elif k == "StringWithMarkup" and isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and isinstance(item.get("String"), str):
                            texts.append(item["String"])
                else:
                    walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)
    walk(data)
    # de-duplicate while preserving order and cap extremely large annotations
    seen, out = set(), []
    for t in texts:
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) >= 20 and t not in seen:
            seen.add(t); out.append(t)
    return "\n".join(out[:250])


def openfda_label(drug_name: str) -> Dict[str, str]:
    q = f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"'
    data = get_json(
        "https://api.fda.gov/drug/label.json",
        params={"search": q, "limit": 1},
        cache_prefix="openfda",
    )
    if not data or not data.get("results"):
        # second attempt by exact name without openfda field
        data = get_json(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'"{drug_name}"', "limit": 1},
            cache_prefix="openfda2",
        )
    if not data or not data.get("results"):
        return {}
    r = data["results"][0]
    def join_field(k):
        v = r.get(k, [])
        return "\n".join(v) if isinstance(v, list) else str(v)
    return {
        "indications": join_field("indications_and_usage"),
        "contraindications": join_field("contraindications"),
        "warnings": join_field("warnings_and_cautions"),
        "adverse_reactions": join_field("adverse_reactions"),
        "drug_interactions": join_field("drug_interactions"),
        "boxed_warning": join_field("boxed_warning"),
        "dosage": join_field("dosage_and_administration"),
        "source": "openFDA Drug Label",
    }


def pubmed_search(drug_name: str, topics=None, per_topic: int = 5) -> List[Dict]:
    topics = topics or ["indications", "pharmacokinetics", "metabolism", "toxicity", "drug interactions", "adverse effects"]
    papers = []
    for topic in topics:
        data = get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": f'"{drug_name}" AND {topic}', "retmax": per_topic, "retmode": "json"},
            cache_prefix="pubmed_search",
        )
        ids = (data or {}).get("esearchresult", {}).get("idlist", [])
        if not ids:
            continue
        # efetch is XML, so use direct request/cache rather than get_json
        key = "pubmed_fetch:" + ",".join(ids)
        path = _cache_path("pubmed_fetch", key)
        xml = None
        if path.exists():
            xml = path.read_text(encoding="utf-8")
        else:
            try:
                r = session.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, timeout=30)
                if r.status_code == 200:
                    xml = r.text
                    path.write_text(xml, encoding="utf-8")
            except Exception:
                xml = None
        if not xml:
            continue
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml)
            for art in root.findall(".//PubmedArticle"):
                pmid = art.findtext(".//PMID")
                title = art.findtext(".//ArticleTitle") or ""
                journal = art.findtext(".//Journal/Title") or ""
                year = art.findtext(".//PubDate/Year") or ""
                abs_parts = [x.text for x in art.findall(".//AbstractText") if x.text]
                papers.append({"pmid": pmid, "title": title, "journal": journal, "year": year, "topic": topic, "abstract": " ".join(abs_parts)})
        except Exception:
            continue
    # de-duplicate by PMID
    unique = {}
    for p in papers:
        if p.get("pmid"):
            unique[p["pmid"]] = p
    return list(unique.values())


# Offline-safe structures for the drugs actually present in the synthetic project dataset.
# These are parent-drug structures; live PubChem enrichment is still attempted first.
DRUG_FALLBACKS = {
    "Atorvastatin": {"smiles":"CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4","cid":60823,"formula":"C33H35FN2O5"},
    "Amoxicillin": {"smiles":"CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C","cid":33613,"formula":"C16H19N3O5S"},
    "Levothyroxine": {"smiles":"C1=C(C=C(C(=C1I)OC2=CC(=C(C(=C2)I)O)I)I)C[C@@H](C(=O)O)N","cid":5819,"formula":"C15H11I4NO4"},
    "Omeprazole": {"smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC","cid":4594,"formula":"C17H19N3O3S"},
    "Amlodipine": {"smiles":"CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN","cid":2162,"formula":"C20H25ClN2O5"},
    "Lisinopril": {"smiles":"C1C[C@H](N(C1)C(=O)[C@H](CCCCN)N[C@@H](CCC2=CC=CC=C2)C(=O)O)C(=O)O","cid":5362119,"formula":"C21H31N3O5"},
    "Sertraline": {"smiles":"CN[C@H]1CC[C@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl","cid":68617,"formula":"C17H17Cl2N"},
    "Cetirizine": {"smiles":"C1CN(CCN1CCOCC(=O)O)C(C2=CC=CC=C2)C3=CC=C(C=C3)Cl","cid":2678,"formula":"C21H25ClN2O3"},
    "Metformin": {"smiles":"CN(C)C(=N)N=C(N)N","cid":4091,"formula":"C4H11N5"},
    "Losartan": {"smiles":"CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl","cid":3961,"formula":"C22H23ClN6O"},
}

DRUG_INFO = {
    "Atorvastatin": {"class":"Statin (HMG-CoA reductase inhibitor)","purpose":"A cholesterol-lowering medicine that reduces LDL cholesterol by inhibiting HMG-CoA reductase.","common_uses":["High LDL cholesterol / hyperlipidemia","Reduction of cardiovascular risk"]},
    "Amoxicillin": {"class":"Penicillin-class antibiotic","purpose":"An antibiotic used to treat susceptible bacterial infections.","common_uses":["Certain bacterial respiratory infections","Ear, nose and throat bacterial infections","Certain urinary tract and skin infections"]},
    "Levothyroxine": {"class":"Thyroid hormone replacement","purpose":"A synthetic form of thyroxine (T4) used to replace thyroid hormone when the body does not make enough.","common_uses":["Hypothyroidism","Thyroid hormone replacement after some thyroid conditions"]},
    "Omeprazole": {"class":"Proton-pump inhibitor (PPI)","purpose":"A medicine that reduces stomach acid by inhibiting the gastric proton pump.","common_uses":["Gastroesophageal reflux disease (GERD)","Acid-related dyspepsia","Peptic ulcer disease and related acid-control indications"]},
    "Amlodipine": {"class":"Calcium-channel blocker","purpose":"A calcium-channel blocker that relaxes blood vessels and lowers blood pressure.","common_uses":["Hypertension","Certain types of angina"]},
    "Lisinopril": {"class":"ACE inhibitor","purpose":"An ACE inhibitor that lowers blood pressure and reduces the formation of angiotensin II.","common_uses":["Hypertension","Heart failure","Reduction of cardiovascular risk in selected patients"]},
    "Sertraline": {"class":"Selective serotonin reuptake inhibitor (SSRI)","purpose":"An antidepressant that increases serotonin signaling in the brain.","common_uses":["Major depressive disorder","Obsessive-compulsive disorder","Panic disorder","Post-traumatic stress disorder"]},
    "Cetirizine": {"class":"Second-generation antihistamine","purpose":"An antihistamine that blocks histamine H1 receptors and reduces common allergy symptoms.","common_uses":["Allergic rhinitis","Sneezing, runny or itchy nose and watery eyes","Hives (urticaria)"]},
    "Metformin": {"class":"Biguanide antihyperglycemic","purpose":"An oral glucose-lowering medicine that improves insulin sensitivity and reduces hepatic glucose production.","common_uses":["Type 2 diabetes","Improving glycemic control alongside diet and exercise"]},
    "Losartan": {"class":"Angiotensin II receptor blocker (ARB)","purpose":"An ARB that blocks angiotensin II receptors and helps lower blood pressure.","common_uses":["Hypertension","Reduction of stroke risk in selected patients with hypertension and left ventricular hypertrophy","Diabetic kidney disease in selected patients"]},
}

DRUG_FALLBACK_EVIDENCE = {
    "Atorvastatin": ("Statin lipid-lowering medicine. Research profile covers cholesterol lowering and cardiovascular risk reduction.", "Common safety topics include liver enzyme abnormalities and muscle-related adverse effects; verify current labeling before interpretation."),
    "Amoxicillin": ("Penicillin-class antibacterial used for susceptible bacterial infections.", "Important safety topic: beta-lactam hypersensitivity. Allergy history should be reviewed before any clinical use."),
    "Levothyroxine": ("Synthetic thyroid hormone used for thyroid hormone replacement.", "Safety depends strongly on dose and thyroid status; excessive exposure can produce hyperthyroid-type effects."),
    "Omeprazole": ("Proton-pump inhibitor used for acid-related gastrointestinal conditions.", "Longer-term therapy has important monitoring considerations; consult current prescribing information for warnings and interactions."),
    "Amlodipine": ("Calcium-channel blocker used for hypertension and certain cardiovascular conditions.", "Common tolerability topics include peripheral edema, dizziness and blood-pressure lowering."),
    "Lisinopril": ("ACE inhibitor used for hypertension and cardiovascular indications.", "Important safety topics include hypotension, kidney function changes, potassium elevation and angioedema."),
    "Sertraline": ("SSRI antidepressant used for several depressive and anxiety-related disorders.", "Safety review should include serotonergic interactions, bleeding risk, activation and other labeled warnings."),
    "Cetirizine": ("Second-generation H1 antihistamine used for allergic symptoms and urticaria.", "Sedation can occur in some people; review concomitant sedating substances and current labeling."),
    "Metformin": ("Biguanide antihyperglycemic medicine commonly used in type 2 diabetes management.", "Kidney function and gastrointestinal tolerability are important safety considerations; use current labeling for contraindications."),
    "Losartan": ("Angiotensin II receptor blocker used for hypertension and related cardiovascular indications.", "Safety topics include blood pressure, kidney function and potassium; pregnancy-related warnings are important."),
}

def _catalog_is_valid(df: pd.DataFrame) -> bool:
    required = {"drug_name", "smiles"}
    return required.issubset(df.columns) and len(df) > 0 and df["smiles"].notna().any()

def build_drug_catalog(patient_df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    drugs = sorted(patient_df["drug_name"].dropna().astype(str).unique())
    cache_csv = ROOT / "drug_catalog.csv"
    if cache_csv.exists() and not refresh:
        try:
            cached = pd.read_csv(cache_csv)
            if _catalog_is_valid(cached):
                # Repair older caches that contain missing structures row-by-row.
                repaired = False
                for drug, fb in DRUG_FALLBACKS.items():
                    m = cached["drug_name"].astype(str).str.strip().str.casefold() == drug.casefold()
                    if m.any() and cached.loc[m, "smiles"].isna().all():
                        cached.loc[m, "smiles"] = fb["smiles"]; repaired = True
                if repaired:
                    cached.to_csv(cache_csv, index=False)
                if cached["smiles"].notna().all():
                    return cached
        except Exception:
            pass
    rows = []
    live_enrich = os.environ.get("SAFERX_LIVE_ENRICH", "0").strip().lower() in {"1", "true", "yes", "on"}
    for i, drug in enumerate(drugs, 1):
        print(f"[{i}/{len(drugs)}] Building catalog entry: {drug}")
        fb = DRUG_FALLBACKS.get(drug, {})
        p = pubchem_lookup(drug) if live_enrich else {}
        p = p or {}
        if not p.get("smiles") and fb.get("smiles"):
            p.update({"smiles": fb["smiles"], "pubchem_cid": p.get("pubchem_cid") or fb.get("cid"), "molecular_formula": p.get("molecular_formula") or fb.get("formula")})
        fda = openfda_label(drug) if live_enrich else {}
        fda = fda or {}
        papers = pubmed_search(drug) if live_enrich else []
        papers = papers or []
        pubtext = pubchem_annotation_text(p.get("pubchem_cid")) if live_enrich else ""
        papertext = "\n\n".join([f"PMID {x['pmid']} | {x['topic']} | {x['title']}\n{x['abstract']}" for x in papers])
        fallback_summary, fallback_safety = DRUG_FALLBACK_EVIDENCE.get(drug, ("Project catalog medicine profile.", "Review authoritative labeling and primary literature for safety interpretation."))
        row = {
            "drug_name": drug, "smiles": p.get("smiles", ""), "pubchem_cid": p.get("pubchem_cid") or fb.get("cid"),
            "title": p.get("title") or drug, "iupac_name": p.get("iupac_name", ""),
            "molecular_formula": p.get("molecular_formula") or fb.get("formula", ""), "molecular_weight": p.get("molecular_weight", ""),
            "inchikey": p.get("inchikey", ""),
            "fda_indications": fda.get("indications", fallback_summary),
            "fda_contraindications": fda.get("contraindications", "See current authoritative label for contraindications."),
            "fda_warnings": fda.get("warnings", fallback_safety),
            "fda_adverse_reactions": fda.get("adverse_reactions", "See current authoritative label for adverse reactions."),
            "fda_drug_interactions": fda.get("drug_interactions", "Check current authoritative labeling for drug interactions."),
            "fda_boxed_warning": fda.get("boxed_warning", ""), "fda_dosage": fda.get("dosage", ""),
            "fda_source": fda.get("source", "Project fallback profile"), "pubchem_text": pubtext,
            "pubmed_text": papertext, "pubmed_count": len(papers),
            "drug_class": DRUG_INFO.get(drug, {}).get("class", ""),
            "drug_purpose": DRUG_INFO.get(drug, {}).get("purpose", fallback_summary),
            "common_uses": " | ".join(DRUG_INFO.get(drug, {}).get("common_uses", [])),
        }
        rows.append(row)
        time.sleep(0.05)
    catalog = pd.DataFrame(rows)
    # Guarantee schema even when all live services are unavailable.
    for c in ["drug_name","smiles","fda_indications","fda_contraindications","fda_warnings","fda_adverse_reactions","fda_drug_interactions","fda_source","pubchem_text","pubmed_text"]:
        if c not in catalog.columns: catalog[c] = ""
    catalog.to_csv(cache_csv, index=False)
    return catalog

# -----------------------------------------------------------------------------
# BIOLINKBERT RAG
# -----------------------------------------------------------------------------
class SimpleFallbackRetriever:
    """Offline-safe lexical retriever over drug-specific evidence chunks."""
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix = None
    def build(self, catalog):
        docs=[]
        fields=[
            ("Indications / common uses","fda_indications"),
            ("Contraindications","fda_contraindications"),
            ("Warnings","fda_warnings"),
            ("Adverse reactions","fda_adverse_reactions"),
            ("Drug interactions","fda_drug_interactions"),
        ]
        for _, r in catalog.iterrows():
            for label,col in fields:
                text=str(r.get(col,"") or "").strip()
                if text:
                    docs.append({"drug_name":r.get("drug_name"),"smiles":r.get("smiles"),"text":f"{label}: {text}","source":str(r.get("fda_source") or "Project evidence profile"),"evidence_type":label})
        self.documents=docs
        texts=[d["text"] for d in docs]
        self.matrix=self.vectorizer.fit_transform(texts) if texts else None
        return self
    def search(self, query, top_k=5):
        if not self.documents or self.matrix is None: return []
        q=self.vectorizer.transform([query])
        scores=cosine_similarity(q,self.matrix)[0]
        idxs=np.argsort(-scores)[:min(top_k,len(self.documents))]
        return [{**self.documents[int(i)],"score":float(scores[int(i)]),"rank":j+1} for j,i in enumerate(idxs)]

class BioLinkBERTRetriever:
    def __init__(self, model_name=BIOLINKBERT_NAME):
        if AutoTokenizer is None:
            raise ImportError("transformers/torch are required for BioLinkBERT RAG")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.index = None
        self.documents = []
        self.embeddings = None

    @staticmethod
    def mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

    def encode(self, texts: List[str], batch_size=8, max_length=256) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                y = self.model(**tok)
                emb = self.mean_pool(y.last_hidden_state, tok["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype("float32") if out else np.empty((0, 768), dtype="float32")

    def build(self, catalog: pd.DataFrame):
        docs = []
        fields=[
            ("Indications / common uses","fda_indications"),
            ("Contraindications","fda_contraindications"),
            ("Warnings","fda_warnings"),
            ("Adverse reactions","fda_adverse_reactions"),
            ("Drug interactions","fda_drug_interactions"),
        ]
        for _, r in catalog.iterrows():
            for label,col in fields:
                text=str(r.get(col,"") or "").strip()
                if text:
                    docs.append({"drug_name":r.get("drug_name"),"smiles":r.get("smiles"),"text":f"Drug: {r.get('drug_name','')}\n{label}: {text}","source":str(r.get("fda_source") or "Project evidence profile"),"evidence_type":label})
        self.documents = docs
        self.embeddings = self.encode([d["text"] for d in docs])
        if faiss is not None and len(self.embeddings):
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        return self

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.documents or self.embeddings is None:
            return []
        q = self.encode([query])
        k = min(top_k, len(self.documents))
        if self.index is not None:
            scores, idxs = self.index.search(q, k)
            scores, idxs = scores[0], idxs[0]
        else:
            scores_all = cosine_similarity(q, self.embeddings)[0]
            idxs = np.argsort(-scores_all)[:k]
            scores = scores_all[idxs]
        return [{**self.documents[int(i)], "score": float(s), "rank": j+1} for j, (i, s) in enumerate(zip(idxs, scores))]

# -----------------------------------------------------------------------------
# ADMET ENGINE
# -----------------------------------------------------------------------------
class ADMETEngine:
    def __init__(self):
        if ADMETModel is None:
            raise ImportError("admet-ai is required. Run the install cell first.")
        self.model = ADMETModel()
        self.cache = {}

    def predict(self, smiles_list):
        smiles_list = [smiles_list] if isinstance(smiles_list, str) else list(smiles_list)
        todo = [s for s in smiles_list if s not in self.cache]
        if todo:
            preds = self.model.predict(smiles=todo)
            if isinstance(preds, pd.DataFrame):
                for smi, row in preds.iterrows():
                    self.cache[str(smi)] = row.to_dict()
            else:
                self.cache[todo[0]] = preds
        return pd.DataFrame([self.cache.get(s, {}) for s in smiles_list], index=smiles_list)

# -----------------------------------------------------------------------------
# TOXICITY / BIOAVAILABILITY / PERSONALIZED SAFETY
# -----------------------------------------------------------------------------
def safe_mean(row: pd.Series, cols: List[str]) -> float:
    vals = []
    for c in cols:
        if c in row.index:
            try:
                v = float(row[c])
                if np.isfinite(v): vals.append(v)
            except Exception: pass
    return float(np.mean(vals)) if vals else 0.0


def risk_components(row: pd.Series) -> Dict[str, float]:
    major = safe_mean(row, MAJOR_SAFETY_COLS)
    geno = safe_mean(row, GENOTOXICITY_COLS)
    receptor = safe_mean(row, RECEPTOR_COLS)
    cellular = safe_mean(row, CELLULAR_STRESS_COLS)
    risk = 100.0 * (0.40*major + 0.25*geno + 0.20*receptor + 0.15*cellular)
    return {"Major_Safety_Risk": major, "Genotoxicity_Risk": geno, "Receptor_Risk": receptor, "Cellular_Stress_Risk": cellular, "Risk_Score_100": risk}


def risk_category(score: float) -> str:
    if score < 15: return "Lower model-estimated risk"
    if score < 30: return "Moderate model-estimated risk"
    if score < 60: return "Higher model-estimated risk"
    return "Very high model-estimated risk"


def bioavailability_score(row: pd.Series) -> float:
    candidates = [
        "Bioavailability_Ma", "HIA_Hou", "HIA", "Caco2_Wang", "Solubility_AqSolDB",
        "Lipophilicity_AqSolDB", "LogP"
    ]
    vals = []
    for c in candidates:
        if c in row.index:
            try:
                v = float(row[c])
                if np.isfinite(v): vals.append(v)
            except Exception: pass
    if not vals:
        return 0.5
    # Logistic-like normalization for heterogeneous ADMET endpoints.
    x = float(np.mean(vals))
    if 0 <= x <= 1:
        return x
    return float(1.0 / (1.0 + math.exp(-x/5.0)))


def text_match_penalty(text: str, terms: List[str]) -> Tuple[float, List[str]]:
    text = (text or "").lower()
    hits = []
    for term in terms:
        t = str(term).strip().lower()
        if not t: continue
        if len(t) >= 3 and t in text:
            hits.append(term)
    return min(1.0, 0.25 * len(hits)), hits


def validate_patient_constraints(candidate: pd.Series, patient: Dict) -> Dict:
    drug = str(candidate.get("drug_name", ""))
    contraind = str(candidate.get("fda_contraindications", ""))
    warnings_text = str(candidate.get("fda_warnings", ""))
    interactions = str(candidate.get("fda_drug_interactions", ""))
    adverse = str(candidate.get("fda_adverse_reactions", ""))
    combined = " ".join([contraind, warnings_text, interactions, adverse]).lower()

    allergies = [x.strip() for x in re.split(r"[,;|]", str(patient.get("allergies", ""))) if x.strip()]
    comorbids = [x.strip() for x in re.split(r"[,;|]", str(patient.get("comorbidities", ""))) if x.strip()]
    current_meds = [x.strip() for x in re.split(r"[,;|]", str(patient.get("current_medications", ""))) if x.strip()]

    allergy_pen, allergy_hits = text_match_penalty(combined, allergies)
    comorb_pen, comorb_hits = text_match_penalty(combined, comorbids)
    ddi_pen, ddi_hits = text_match_penalty(interactions, current_meds)

    # Also flag direct candidate-name overlap with the allergy list.
    direct_allergy = any(a.lower() == drug.lower() for a in allergies)
    if direct_allergy:
        allergy_pen = 1.0
        allergy_hits.append(drug)

    penalty = min(1.0, allergy_pen + comorb_pen + ddi_pen)
    return {
        "penalty": penalty,
        "allergy_hits": allergy_hits,
        "comorbidity_hits": comorb_hits,
        "interaction_hits": ddi_hits,
        "contraindications": contraind,
        "warnings": warnings_text,
        "interactions": interactions,
    }

# -----------------------------------------------------------------------------
# PERSONALIZED RECOMMENDER
# -----------------------------------------------------------------------------
# Transparent condition-to-therapy profiles keep the research demo clinically
# coherent instead of allowing a generic semantic retriever to return the same
# medicines for every diagnosis. These are broad educational mappings, not
# prescribing rules or clinical guidelines.
CONDITION_PROFILES = {
    # Broad indication profiles are candidate-generation knowledge, not prescribing rules.
    # The final rank still depends on patient constraints, molecular/ADMET signals and evidence.
    "hypertension": {"drugs": {"Amlodipine":1.0,"Lisinopril":0.96,"Losartan":0.94,"Valsartan":0.91,"Hydrochlorothiazide":0.88}, "keywords":["high blood pressure","elevated blood pressure"]},
    "high blood pressure": {"drugs": {"Amlodipine":1.0,"Lisinopril":0.96,"Losartan":0.94,"Valsartan":0.91,"Hydrochlorothiazide":0.88}, "keywords":["hypertension","blood pressure"]},
    "type 2 diabetes": {"drugs": {"Metformin":1.0,"Empagliflozin":0.93,"Sitagliptin":0.88,"Glipizide":0.82}, "keywords":["high blood sugar","glucose","hyperglycemia"]},
    "diabetes": {"drugs": {"Metformin":1.0,"Empagliflozin":0.93,"Sitagliptin":0.88,"Glipizide":0.82}, "keywords":["blood sugar","glucose","hyperglycemia"]},
    "allergic rhinitis": {"drugs": {"Cetirizine":1.0,"Loratadine":0.94,"Fexofenadine":0.92,"Fluticasone":0.88}, "keywords":["sneezing","runny nose","nasal congestion","itchy eyes","watery eyes"]},
    "allergy": {"drugs": {"Cetirizine":1.0,"Loratadine":0.94,"Fexofenadine":0.92,"Fluticasone":0.88}, "keywords":["sneezing","runny nose","itchy","watery eyes"]},
    "hypothyroidism": {"drugs": {"Levothyroxine":1.0}, "keywords":["low thyroid","thyroid hormone"]},
    "hypothyroid": {"drugs": {"Levothyroxine":1.0}, "keywords":["low thyroid","thyroid hormone"]},
    "gerd": {"drugs": {"Omeprazole":1.0,"Esomeprazole":0.93,"Famotidine":0.86}, "keywords":["acid reflux","heartburn","reflux"]},
    "acid reflux": {"drugs": {"Omeprazole":1.0,"Esomeprazole":0.93,"Famotidine":0.86}, "keywords":["heartburn","reflux"]},
    "hyperlipidemia": {"drugs": {"Atorvastatin":1.0,"Rosuvastatin":0.96,"Simvastatin":0.86}, "keywords":["high cholesterol","cholesterol","lipid"]},
    "high cholesterol": {"drugs": {"Atorvastatin":1.0,"Rosuvastatin":0.96,"Simvastatin":0.86}, "keywords":["cholesterol","lipid"]},
    "depression": {"drugs": {"Sertraline":1.0,"Escitalopram":0.95,"Fluoxetine":0.90,"Duloxetine":0.84}, "keywords":["low mood","depressed mood"]},
    "anxiety": {"drugs": {"Sertraline":1.0,"Escitalopram":0.95,"Duloxetine":0.84}, "keywords":["panic","anxious","generalized anxiety"]},
    "bacterial infection": {"drugs": {"Amoxicillin":1.0,"Azithromycin":0.90,"Doxycycline":0.86,"Cephalexin":0.82}, "keywords":["infection","bacterial"]},
    "pneumonia": {"drugs": {"Amoxicillin":0.94,"Azithromycin":0.90,"Doxycycline":0.86}, "keywords":["lung infection","respiratory infection"]},
    "asthma": {"drugs": {"Albuterol":1.0,"Budesonide":0.93,"Montelukast":0.88}, "keywords":["wheezing","bronchospasm","shortness of breath"]},
    "migraine": {"drugs": {"Sumatriptan":1.0,"Propranolol":0.82,"Topiramate":0.80}, "keywords":["migraine","severe headache","photophobia"]},
    "pain": {"drugs": {"Acetaminophen":1.0,"Ibuprofen":0.90,"Naproxen":0.86}, "keywords":["pain","ache"]},
    "fever": {"drugs": {"Acetaminophen":1.0,"Ibuprofen":0.90}, "keywords":["temperature","febrile"]},
}

# Additional candidate names are resolved through PubChem at recommendation time when
# they are not already present in the local catalog. This makes candidate generation
# open-ended without pretending that the original synthetic 10-drug dataset is complete.
def _candidate_drug_names(patient: Dict) -> Dict[str,float]:
    profile_scores, _ = _condition_profile(patient)
    return profile_scores

def expand_catalog_for_patient(patient: Dict, catalog: pd.DataFrame) -> pd.DataFrame:
    names = _candidate_drug_names(patient)
    if not names:
        return catalog
    existing = {str(x).casefold() for x in catalog.get("drug_name", pd.Series(dtype=str)).tolist()}
    additions = []
    for name in names:
        if name.casefold() in existing:
            continue
        try:
            info = pubchem_lookup(name)
            smi = str(info.get("smiles") or "").strip()
            if not smi:
                continue
            row = {"drug_name": name, "smiles": smi}
            row.update({
                "fda_indications": info.get("indications", ""),
                "fda_contraindications": info.get("contraindications", ""),
                "fda_warnings": info.get("warnings", ""),
                "fda_adverse_reactions": info.get("adverse_reactions", ""),
                "fda_drug_interactions": info.get("interactions", ""),
                "evidence_source": "PubChem resolution"
            })
            additions.append(row)
            existing.add(name.casefold())
        except Exception:
            continue
    if not additions:
        return catalog
    return pd.concat([catalog, pd.DataFrame(additions)], ignore_index=True, sort=False)

def _condition_profile(patient: Dict) -> Tuple[Dict[str, float], List[str]]:
    diagnosis = str(patient.get("diagnosis", "")).strip().casefold()
    symptom = str(patient.get("symptom", "")).strip().casefold()
    text = f"{diagnosis} {symptom}"
    direct: Dict[str, float] = {}
    matched_terms: List[str] = []
    for key, profile in CONDITION_PROFILES.items():
        if key in diagnosis or key in text:
            direct.update(profile["drugs"])
            matched_terms.append(key)
            continue
        hits = sum(1 for kw in profile["keywords"] if kw in text)
        if hits >= 2:
            for drug, score in profile["drugs"].items():
                direct[drug] = max(direct.get(drug, 0.0), min(1.0, 0.55 + 0.15*hits))
            matched_terms.append(f"symptoms: {', '.join(profile['keywords'][:3])}")
    return direct, matched_terms

def recommend(patient: Dict, catalog: pd.DataFrame, admet_df: pd.DataFrame, retriever: BioLinkBERTRetriever, top_k=5) -> pd.DataFrame:
    # Build a disease-aware candidate pool first. New candidates are resolved from
    # PubChem and predicted lazily; the local synthetic catalog is never treated as
    # the universe of possible medicines.
    catalog = expand_catalog_for_patient(patient, catalog)
    missing = [str(s) for s in catalog.get("smiles", pd.Series(dtype=str)).tolist() if str(s) and str(s) not in admet_df.index]
    if missing:
        try:
            extra = ADMETEngine().predict(missing)
            admet_df = pd.concat([admet_df, extra[~extra.index.isin(admet_df.index)]])
        except Exception:
            pass
    diagnosis = str(patient.get("diagnosis", ""))
    symptom = str(patient.get("symptom", ""))
    query = f"Disease/diagnosis: {diagnosis}. Symptom: {symptom}. Find appropriate therapeutic drugs and evidence, including indications, contraindications, adverse effects and drug interactions."
    retrieved = retriever.search(query, top_k=len(catalog))
    semantic = {r["drug_name"]: max(0.0, min(1.0, (r["score"] + 1)/2)) for r in retrieved}
    profile_scores, _ = _condition_profile(patient)
    # If the condition can be mapped to a therapeutic profile, rank only the
    # indication-relevant candidate pool. This prevents unrelated drugs from
    # reaching the top merely because their ADMET/evidence scores look good.
    if profile_scores:
        wanted = {k.casefold() for k in profile_scores}
        filtered = catalog[catalog["drug_name"].astype(str).str.casefold().isin(wanted)].copy()
        if not filtered.empty:
            catalog = filtered

    rows = []
    for _, c in catalog.iterrows():
        drug = str(c.get("drug_name", ""))
        smi = str(c.get("smiles", ""))
        if not smi or smi == "nan" or smi not in admet_df.index:
            continue
        pred = admet_df.loc[smi]
        rc = risk_components(pred)
        safety = max(0.0, 1.0 - rc["Risk_Score_100"]/100.0)
        bio = bioavailability_score(pred)
        safety_check = validate_patient_constraints(c, patient)
        penalty = safety_check["penalty"]
        # Condition-aware relevance is dominant; semantic retrieval provides
        # secondary evidence so different diagnoses do not collapse to one list.
        rel_sem = semantic.get(drug, 0.0)
        rel_profile = profile_scores.get(drug, 0.0)
        rel = max(rel_profile, 0.30*rel_sem)
        evidence_count = sum(1 for key in ["fda_indications","fda_contraindications","fda_warnings","fda_adverse_reactions","fda_drug_interactions"] if str(c.get(key, "")).strip())
        evidence_score = min(1.0, evidence_count/5.0)
        final = 100.0 * (W_RELEVANCE*rel + W_BIOAVAIL*bio + W_SAFETY*safety + W_EVIDENCE*evidence_score - penalty*0.30)
        rows.append({
            "Drug": drug, "SMILES": smi,
            "Clinical_Relevance": rel, "Semantic_Relevance": rel_sem, "Condition_Relevance": rel_profile,
            "Bioavailability_Score": bio, "Safety_Score": safety, "Evidence_Score": evidence_score,
            "Patient_Penalty": penalty, "Final_Recommendation_Score": final,
            "Risk_Score_100": rc["Risk_Score_100"], "Risk_Category": risk_category(rc["Risk_Score_100"]),
            "Major_Safety_Risk": rc["Major_Safety_Risk"], "Genotoxicity_Risk": rc["Genotoxicity_Risk"],
            "Receptor_Risk": rc["Receptor_Risk"], "Cellular_Stress_Risk": rc["Cellular_Stress_Risk"],
            "Allergy_Hits": "; ".join(safety_check["allergy_hits"]),
            "Comorbidity_Hits": "; ".join(safety_check["comorbidity_hits"]),
            "Interaction_Hits": "; ".join(safety_check["interaction_hits"]),
            "Indications": str(c.get("fda_indications", "")),
            "Contraindications": str(c.get("fda_contraindications", "")),
            "Warnings": str(c.get("fda_warnings", "")),
            "Adverse_Reactions": str(c.get("fda_adverse_reactions", "")),
            "Drug_Interactions": str(c.get("fda_drug_interactions", "")),
            "Evidence_Source": str(c.get("fda_source", "")),
            "Molecular_Formula": str(c.get("molecular_formula", "")),
            "Molecular_Weight": str(c.get("molecular_weight", "")),
        })
    out = pd.DataFrame(rows).sort_values("Final_Recommendation_Score", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out)+1))
    return out.head(top_k)

# -----------------------------------------------------------------------------
# EXPLAINABILITY
# -----------------------------------------------------------------------------
def explain_recommendation(row: pd.Series) -> pd.DataFrame:
    contributions = [
        ("Clinical relevance", W_RELEVANCE*float(row["Clinical_Relevance"])),
        ("Bioavailability", W_BIOAVAIL*float(row["Bioavailability_Score"])),
        ("Safety", W_SAFETY*float(row["Safety_Score"])),
        ("Evidence support", W_EVIDENCE*float(row["Evidence_Score"])),
        ("Patient penalty", -0.30*float(row["Patient_Penalty"])),
    ]
    return pd.DataFrame(contributions, columns=["Feature", "Score_Contribution"])


def shap_recommendation_explanation(recommendation_df: pd.DataFrame):
    """SHAP explanation of the explicit multi-criteria recommendation function.
    This explains the ranking score, not ADMET-AI's internal neural model."""
    if shap is None or recommendation_df.empty:
        return None
    cols = ["Clinical_Relevance", "Bioavailability_Score", "Safety_Score", "Evidence_Score", "Patient_Penalty"]
    X = recommendation_df[cols].astype(float).values
    def f(Xv):
        return 100.0*(W_RELEVANCE*Xv[:,0] + W_BIOAVAIL*Xv[:,1] + W_SAFETY*Xv[:,2] + W_EVIDENCE*Xv[:,3] - 0.30*Xv[:,4])
    try:
        explainer = shap.Explainer(f, X)
        values = explainer(X)
        out = recommendation_df[["Rank","Drug"]].copy()
        for j, c in enumerate(cols):
            out["SHAP_"+c] = values.values[:,j]
        return out
    except Exception as e:
        warnings.warn(f"SHAP explanation failed: {e}")
        return None

# -----------------------------------------------------------------------------
# EVALUATION ON THE SYNTHETIC DATASET
# -----------------------------------------------------------------------------
def evaluate_recommendation_system(patient_df: pd.DataFrame, catalog: pd.DataFrame, admet_df: pd.DataFrame, retriever: BioLinkBERTRetriever, k_values=(1,3,5), max_rows=200) -> Dict:
    # Evaluation treats the medication recorded in the synthetic dataset as the reference label.
    # It is NOT a clinical efficacy benchmark.
    sample = patient_df.dropna(subset=["diagnosis", "drug_name"]).sample(min(max_rows, len(patient_df)), random_state=42)
    hits = {k: 0 for k in k_values}
    reciprocal = []
    evaluated = 0
    rows = []
    for _, p in sample.iterrows():
        patient = p.to_dict()
        try:
            # Build the full ranking for evaluation. Truncating to top-5 before
            # calculating Hit@1/3/5 can make the metrics misleading or identical.
            rec = recommend(patient, catalog, admet_df, retriever, top_k=len(catalog))
        except Exception:
            continue
        if rec.empty: continue
        evaluated += 1
        drugs = rec["Drug"].tolist()
        target = str(p["drug_name"])
        rank = next((i+1 for i,d in enumerate(drugs) if d.lower() == target.lower()), None)
        for k in k_values:
            if target.lower() in [d.lower() for d in drugs[:k]]:
                hits[k] += 1
        reciprocal.append(1.0/rank if rank else 0.0)
        rows.append({"patient_id":p["patient_id"],"target_drug":target,"predicted_rank":rank})
    metrics = {f"Hit@{k}": hits[k]/evaluated if evaluated else 0.0 for k in k_values}
    metrics["MRR"] = float(np.mean(reciprocal)) if reciprocal else 0.0
    metrics["evaluated_patients"] = evaluated
    details_df = pd.DataFrame(rows)
    if not details_df.empty:
        details_df["hit_at_1"] = details_df["predicted_rank"].eq(1)
        details_df["hit_at_3"] = details_df["predicted_rank"].fillna(999).le(3)
        details_df["hit_at_5"] = details_df["predicted_rank"].fillna(999).le(5)
    return {"metrics": metrics, "details": details_df}

# -----------------------------------------------------------------------------
# OPTIONAL RF/XGBOOST BENCHMARK IMPLEMENTATION
# -----------------------------------------------------------------------------
def train_morgan_baseline(smiles: List[str], labels: List[int], model_type="rf"):
    X = np.vstack([morgan_fp(s) for s in smiles])
    y = np.asarray(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if model_type == "xgb":
        if xgb is None: raise ImportError("xgboost not installed")
        model = xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    result = {"Accuracy":accuracy_score(y_test,pred),"Precision":precision,"Recall":recall,"F1":f1,"Confusion_Matrix":confusion_matrix(y_test,pred).tolist()}
    if prob is not None and len(np.unique(y_test)) == 2:
        result["ROC_AUC"] = roc_auc_score(y_test, prob)
    return model, result


# -----------------------------------------------------------------------------
# TDC + CHEMPROP BENCHMARK HELPERS
# -----------------------------------------------------------------------------
def load_tdc_benchmark(dataset_name: str = "HIA_Hou", task_group: str = "ADME"):
    """Load a real labelled TDC benchmark for RF/XGBoost/Chemprop evaluation.
    TDC currently exposes ADME and Tox single-prediction datasets including
    HIA_Hou, BBB_Martins, CYP3A4_Veith, hERG, AMES and DILI.
    """
    try:
        if task_group.upper() == "TOX":
            from tdc.single_pred import Tox
            data = Tox(name=dataset_name)
        else:
            from tdc.single_pred import ADME
            data = ADME(name=dataset_name)
        return data.get_data()
    except Exception as e:
        raise RuntimeError(
            f"Could not load TDC dataset {dataset_name}. Run `pip install PyTDC`. Error: {e}"
        )


def prepare_chemprop_csv(tdc_df: pd.DataFrame, smiles_col: str = "Drug", target_col: str = "Y", out_path: str = None) -> str:
    """Convert a TDC dataframe into Chemprop v2's CSV format."""
    if smiles_col not in tdc_df.columns or target_col not in tdc_df.columns:
        # TDC commonly uses Drug/Y; fall back to the first SMILES-like and target columns.
        candidates = [c for c in tdc_df.columns if str(c).lower() in {"drug", "smiles", "compound"}]
        if candidates: smiles_col = candidates[0]
        target_candidates = [c for c in tdc_df.columns if c != smiles_col]
        if target_candidates: target_col = target_candidates[-1]
    out = tdc_df[[smiles_col, target_col]].copy()
    out.columns = ["smiles", "target"]
    out = out.dropna()
    path = out_path or str(ROOT / "chemprop_benchmark.csv")
    out.to_csv(path, index=False)
    return path


def run_chemprop_train(data_csv: str, output_dir: str = None, task_type: str = "classification", epochs: int = 30, ensemble_size: int = 3) -> str:
    """Train a Chemprop v2 MPNN ensemble from a real labelled dataset.
    Uses the current Chemprop CLI, with scaffold-balanced splitting for a stricter molecular split.
    """
    import subprocess, sys
    output_dir = output_dir or str(ROOT / "chemprop_training")
    cmd = [
        sys.executable, "-m", "chemprop", "train",
        "--data-path", data_csv,
        "--task-type", task_type,
        "--output-dir", output_dir,
        "--split-type", "scaffold_balanced",
        "--epochs", str(epochs),
        "--ensemble-size", str(ensemble_size),
        "--smiles-columns", "smiles",
        "--target-columns", "target",
    ]
    subprocess.run(cmd, check=True)
    return output_dir

# -----------------------------------------------------------------------------
# COMPLETE PIPELINE
# -----------------------------------------------------------------------------
def initialize_system(refresh_catalog=False):
    patients = load_patient_data()
    catalog = build_drug_catalog(patients, refresh=refresh_catalog)
    # Never let an incomplete enrichment cache crash the API.
    for c in ["drug_name", "smiles"]:
        if c not in catalog.columns: catalog[c] = ""
    catalog = catalog.dropna(subset=["drug_name", "smiles"]).copy()
    catalog = catalog[catalog["smiles"].astype(str).str.strip().ne("")].reset_index(drop=True)
    if catalog.empty:
        raise RuntimeError("No valid drug structures are available. The project fallback catalog could not be built.")
    try:
        retriever = BioLinkBERTRetriever().build(catalog)
        rag_mode = "BioLinkBERT + FAISS"
    except Exception as exc:
        warnings.warn(f"BioLinkBERT unavailable; using offline TF-IDF retrieval: {exc}")
        retriever = SimpleFallbackRetriever().build(catalog)
        rag_mode = "TF-IDF offline fallback"
    admet = ADMETEngine()
    preds = admet.predict(catalog["smiles"].tolist())
    preds.to_csv(ROOT / "admet_predictions_complete.csv", index=True, index_label="smiles")
    catalog.to_csv(ROOT / "drug_catalog_complete.csv", index=False)
    # Keep mode discoverable by the API/UI without changing the retriever contract.
    retriever.mode = rag_mode
    return patients, catalog, retriever, admet, preds

# -----------------------------------------------------------------------------
# DETERMINISTIC REPORT / UI HELPERS
# -----------------------------------------------------------------------------
def patient_dict(age, sex, diagnosis, symptom, allergies="", comorbidities="", current_medications=""):
    return {"age": age, "sex": sex, "diagnosis": diagnosis, "symptom": symptom, "allergies": allergies, "comorbidities": comorbidities, "current_medications": current_medications}


def recommendation_report(patient: Dict, rec: pd.DataFrame, catalog: pd.DataFrame) -> str:
    lines = ["# SafeRx AI — Personalized Computational Drug Ranking", "", "> Research/educational decision-support only. Not a prescription.", ""]
    lines += [f"**Diagnosis:** {patient.get('diagnosis','')}", f"**Symptom:** {patient.get('symptom','')}", f"**Age/Sex:** {patient.get('age','')}/{patient.get('sex','')}", ""]
    for _, r in rec.iterrows():
        lines += [f"## Rank {int(r['Rank'])}: {r['Drug']}", f"**Recommendation score:** {r['Final_Recommendation_Score']:.2f}/100", f"**ADMET risk:** {r['Risk_Score_100']:.2f}/100 ({r['Risk_Category']})"]
        hits = [x for x in [r.get('Allergy_Hits',''), r.get('Comorbidity_Hits',''), r.get('Interaction_Hits','')] if str(x).strip()]
        if hits:
            lines.append("**Safety validator flags:** " + " | ".join(hits))
        lines += ["", "**Score contributions:**", f"- Clinical relevance: {r['Clinical_Relevance']:.3f}", f"- Bioavailability: {r['Bioavailability_Score']:.3f}", f"- Safety: {r['Safety_Score']:.3f}", f"- Evidence support: {r['Evidence_Score']:.3f}", f"- Patient penalty: {r['Patient_Penalty']:.3f}", ""]
    lines += ["## Interpretation", "The ranking combines semantic relevance, computational ADMET safety, a bioavailability proxy, evidence availability, and patient-specific text-based safety constraints. The weights are research design choices and require clinical validation before any real prescribing use."]
    return "\n".join(lines)


def launch_app(patients, catalog, retriever, admet_engine, admet_preds):
    if gr is None:
        raise ImportError("gradio is required")
    drug_names = catalog["drug_name"].tolist()
    def run(age, sex, diagnosis, symptom, allergies, comorbidities, current_meds):
        p = patient_dict(age, sex, diagnosis, symptom, allergies, comorbidities, current_meds)
        rec = recommend(p, catalog, admet_preds, retriever, top_k=min(5,len(catalog)))
        report = recommendation_report(p, rec, catalog)
        shap_df = shap_recommendation_explanation(rec)
        return rec, report, shap_df if shap_df is not None else pd.DataFrame()
    with gr.Blocks(title="SafeRx AI — Complete") as demo:
        gr.Markdown("# 🧪 SafeRx AI\n### RAG + ADMET + Personalized Safety Ranking\nResearch mode — not a clinical prescription system.")
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age", value=30)
                sex = gr.Dropdown(["Male","Female","Other"], label="Sex", value="Female")
                diagnosis = gr.Textbox(label="Diagnosis / condition", value="Hypertension")
                symptom = gr.Textbox(label="Symptoms", value="Dizziness")
                allergies = gr.Textbox(label="Allergies (comma separated)")
                comorbidities = gr.Textbox(label="Comorbidities (comma separated)")
                current_meds = gr.Textbox(label="Current medications (comma separated)")
                btn = gr.Button("Analyze & Rank")
            with gr.Column():
                result_table = gr.Dataframe(label="Personalized Candidate Ranking", interactive=False)
                report = gr.Markdown()
                shap_table = gr.Dataframe(label="SHAP — Recommendation Score Explanation", interactive=False)
        btn.click(run, [age,sex,diagnosis,symptom,allergies,comorbidities,current_meds], [result_table,report,shap_table])
    return demo

if __name__ == "__main__":
    print("SafeRx AI complete implementation loaded. Run initialize_system() in Colab, then launch_app(...).")

import re
import os
import joblib
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from docx import Document
from PyPDF2 import PdfReader


# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FOLDER = "biobert_model"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

MODEL_PATH = os.path.join(MODEL_FOLDER, "best_model.pt")

ICD_LABELS_PATH = os.path.join(MODEL_FOLDER, "icd_labels.pkl")
CPT_LABELS_PATH = os.path.join(MODEL_FOLDER, "cpt_labels.pkl")
CHAPTER_LABELS_PATH = os.path.join(MODEL_FOLDER, "chapter_labels.pkl")

MAX_LEN = 512

MIN_PROB_ICD = 0.20
MIN_PROB_CPT = 0.50


MAX_CODES = 5


ABBREVIATIONS = {
    r"\bHTN\b": "hypertension",
    r"\bDM\b": "diabetes mellitus",
    r"\bT2DM\b": "type 2 diabetes mellitus",
    r"\bCAD\b": "coronary artery disease",
    r"\bCHF\b": "congestive heart failure",
    r"\bSOB\b": "shortness of breath",
    r"\bCP\b": "chest pain",
}

def clean_text(text: str) -> str:
    text = text.upper()
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(abbr, full.upper(), text)
    return text.lower()


def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_text(path):
    if path.endswith(".txt"):
        return read_txt(path)
    if path.endswith(".pdf"):
        return read_pdf(path)
    if path.endswith(".docx"):
        return read_docx(path)
    raise ValueError("Unsupported file format")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

icd_mlb = joblib.load(ICD_LABELS_PATH)
cpt_mlb = joblib.load(CPT_LABELS_PATH)
chapter_mlb = joblib.load(CHAPTER_LABELS_PATH)


class MedicalCodingModel(nn.Module):
    def __init__(self, num_icd, num_cpt, num_chapter):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size

        self.icd_head = nn.Linear(hidden, num_icd)
        self.cpt_head = nn.Linear(hidden, num_cpt)
        self.chapter_head = nn.Linear(hidden, num_chapter)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)

        return {
            "icd": self.icd_head(pooled),
            "cpt": self.cpt_head(pooled),
            "chapter": self.chapter_head(pooled),
        }


# =====================================================
# LOAD MODEL
# =====================================================
model = MedicalCodingModel(
    num_icd=len(icd_mlb.classes_),
    num_cpt=len(cpt_mlb.classes_),
    num_chapter=len(chapter_mlb.classes_)
).to(DEVICE)

state = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True  
)

model.load_state_dict(state)
model.eval()



def predict_codes_from_file(file_path):
    raw_text = load_text(file_path)
    text = clean_text(raw_text)

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)
        )

        icd_probs = torch.sigmoid(outputs["icd"]).cpu().numpy()[0]
        cpt_probs = torch.sigmoid(outputs["cpt"]).cpu().numpy()[0]
        chapter_probs = torch.sigmoid(outputs["chapter"]).cpu().numpy()[0]

    icd_preds = [
        (icd_mlb.classes_[i], icd_probs[i])
        for i in np.argsort(icd_probs)[::-1]
        if icd_probs[i] >= MIN_PROB_ICD
    ][:MAX_CODES]

    cpt_preds = [
        (cpt_mlb.classes_[i], cpt_probs[i])
        for i in np.argsort(cpt_probs)[::-1]
        if cpt_probs[i] >= MIN_PROB_CPT
    ][:MAX_CODES]


    print(f"\nFile: {os.path.basename(file_path)}")

    print("\nSuggested ICD Codes:")
    if not icd_preds:
        print("  None above threshold")
    else:
        for code, score in icd_preds:
            print(f"  {code:<10} | {score:.3f}")

    print("\nSuggested CPT Codes:")
    if not cpt_preds:
        print("  None above threshold")
    else:
        for code, score in cpt_preds:
            print(f"  {code:<10} | {score:.3f}")

    



INPUT_PATH = "CASE – 8689439.txt"   # file OR folder

if os.path.isdir(INPUT_PATH):
    for fname in os.listdir(INPUT_PATH):
        if fname.endswith((".txt", ".pdf", ".docx")):
            predict_codes_from_file(os.path.join(INPUT_PATH, fname))
else:
    predict_codes_from_file(INPUT_PATH)

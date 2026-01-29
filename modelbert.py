import re
import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
EPOCHS = 15               
BATCH_SIZE = 4
LR = 2e-5
THRESHOLD = 0.1
SAVE_FOLDER = "biobert_model"
os.makedirs(SAVE_FOLDER, exist_ok=True)


ABBREVIATIONS = {
    r"\bHTN\b": "hypertension",
    r"\bDM\b": "diabetes mellitus",
    r"\bT2DM\b": "type 2 diabetes mellitus",
    r"\bCAD\b": "coronary artery disease",
    r"\bCHF\b": "congestive heart failure",
    r"\bSOB\b": "shortness of breath",
    r"\bCP\b": "chest pain",
}


def clean_text(text):
    text = text.upper()
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(abbr, full.upper(), text)
    return text.lower()


df = pd.read_csv("trainingfile_1.csv")

df["text"] = (
    df["diagnoses"].fillna("") + " " +
    df["assessment"].fillna("") + " " +
    df["plan"].fillna("")
).apply(clean_text)

df["icd_list"] = df["icd_codes"].fillna("").apply(
    lambda x: [c.strip() for c in re.split(r"[,\s]+", x) if c.strip()]
)
df["cpt_list"] = df["cpt_codes"].fillna("").apply(
    lambda x: [c.strip() for c in re.split(r"[,\s]+", x) if c.strip()]
)


def icd_chapter(code):
    if not code:
        return "OTHER"
    if code[0] == "E":
        return "ENDOCRINE"
    if code[0] == "I":
        return "CARDIO"
    if code[0] == "J":
        return "RESPIRATORY"
    if code[0] == "K":
        return "DIGESTIVE"
    return "OTHER"


df["icd_chapter"] = df["icd_list"].apply(
    lambda codes: list(set(icd_chapter(c) for c in codes))
)


icd_mlb = MultiLabelBinarizer()
cpt_mlb = MultiLabelBinarizer()
chapter_mlb = MultiLabelBinarizer()

y_icd = icd_mlb.fit_transform(df["icd_list"])
y_cpt = cpt_mlb.fit_transform(df["cpt_list"])
y_chapter = chapter_mlb.fit_transform(df["icd_chapter"])

NUM_ICD = len(icd_mlb.classes_)
NUM_CPT = len(cpt_mlb.classes_)
NUM_CHAPTER = len(chapter_mlb.classes_)

joblib.dump(icd_mlb, os.path.join(SAVE_FOLDER, "icd_labels.pkl"))
joblib.dump(cpt_mlb, os.path.join(SAVE_FOLDER, "cpt_labels.pkl"))
joblib.dump(chapter_mlb, os.path.join(SAVE_FOLDER, "chapter_labels.pkl"))

X_train, X_val, icd_train, icd_val, cpt_train, cpt_val, chap_train, chap_val = train_test_split(
    df["text"].tolist(),
    y_icd,
    y_cpt,
    y_chapter,
    test_size=0.1,
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class ClinicalDataset(Dataset):
    def __init__(self, texts, icd, cpt, chapter):
        self.texts = texts
        self.icd = icd
        self.cpt = cpt
        self.chapter = chapter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "icd_labels": torch.tensor(self.icd[idx], dtype=torch.float),
            "cpt_labels": torch.tensor(self.cpt[idx], dtype=torch.float),
            "chapter_labels": torch.tensor(self.chapter[idx], dtype=torch.float),
        }


train_loader = DataLoader(
    ClinicalDataset(X_train, icd_train, cpt_train, chap_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    ClinicalDataset(X_val, icd_val, cpt_val, chap_val),
    batch_size=BATCH_SIZE
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


class MedicalCodingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size

        self.icd_head = nn.Linear(hidden, NUM_ICD)
        self.cpt_head = nn.Linear(hidden, NUM_CPT)
        self.chapter_head = nn.Linear(hidden, NUM_CHAPTER)

        self.loss_fn = FocalLoss()

    def forward(self, input_ids, attention_mask,
                icd_labels=None, cpt_labels=None, chapter_labels=None):

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)

        icd_logits = self.icd_head(pooled)
        cpt_logits = self.cpt_head(pooled)
        chapter_logits = self.chapter_head(pooled)

        loss = None
        if icd_labels is not None:
            loss = (
                self.loss_fn(icd_logits, icd_labels)
                + self.loss_fn(cpt_logits, cpt_labels)
                + 0.5 * self.loss_fn(chapter_logits, chapter_labels)
            )

        return loss, icd_logits, cpt_logits, chapter_logits


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            _, icd_logits, _, _ = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE)
            )
            probs = torch.sigmoid(icd_logits).cpu().numpy()
            preds.append((probs > THRESHOLD).astype(int))
            trues.append(batch["icd_labels"].numpy())

    return f1_score(np.vstack(trues), np.vstack(preds), average="micro")


model = MedicalCodingModel().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()

        loss, _, _, _ = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["icd_labels"].to(DEVICE),
            batch["cpt_labels"].to(DEVICE),
            batch["chapter_labels"].to(DEVICE)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_f1 = evaluate(model, val_loader)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(
            SAVE_FOLDER, "best_model.pt"))

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val Micro-F1: {val_f1:.4f} | "
        f"Best: {best_f1:.4f}"
    )


# -------------------------
# Save final artifacts
# -------------------------
torch.save(model.state_dict(), os.path.join(SAVE_FOLDER, "final_model.pt"))
tokenizer.save_pretrained(os.path.join(SAVE_FOLDER, "tokenizer"))

print(f"\n All training artifacts saved in: {SAVE_FOLDER}")

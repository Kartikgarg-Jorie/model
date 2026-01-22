
import re
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizerFast, LongformerModel
from torch.optim import AdamW
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

df["icd_list"] = df["icd_codes"].fillna("").apply(lambda x: x.split(",") if x != "" else [])
df["cpt_list"] = df["cpt_codes"].fillna("").apply(lambda x: x.split(",") if x != "" else [])



def icd_chapter(code):
    letter = code[0]
    if letter == "E":
        return "ENDOCRINE"
    if letter == "I":
        return "CARDIO"
    if letter == "J":
        return "RESPIRATORY"
    if letter == "K":
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

joblib.dump(icd_mlb, "icd_labels.pkl")
joblib.dump(cpt_mlb, "cpt_labels.pkl")
joblib.dump(chapter_mlb, "chapter_labels.pkl")

NUM_ICD = len(icd_mlb.classes_)
NUM_CPT = len(cpt_mlb.classes_)
NUM_CHAPTER = len(chapter_mlb.classes_)


X_train, X_val, icd_train, icd_val, cpt_train, cpt_val, chap_train, chap_val = train_test_split(
    df["text"].tolist(),
    y_icd,
    y_cpt,
    y_chapter,
    test_size=0.1,
    random_state=42
)


MODEL_NAME = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_NAME)
MAX_LEN = 4096


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()


class MedicalCodingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LongformerModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size

        self.icd_head = nn.Linear(hidden, NUM_ICD)
        self.cpt_head = nn.Linear(hidden, NUM_CPT)
        self.chapter_head = nn.Linear(hidden, NUM_CHAPTER)

        self.loss_fn = FocalLoss()

    def forward(self, input_ids, attention_mask,
                icd_labels=None, cpt_labels=None, chapter_labels=None):

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]

        icd_logits = self.icd_head(pooled)
        cpt_logits = self.cpt_head(pooled)
        chapter_logits = self.chapter_head(pooled)

        loss = None
        if icd_labels is not None:
            loss = (
                self.loss_fn(icd_logits, icd_labels) +
                self.loss_fn(cpt_logits, cpt_labels) +
                self.loss_fn(chapter_logits, chapter_labels)
            )

        return loss, icd_logits, cpt_logits


train_ds = ClinicalDataset(X_train, icd_train, cpt_train, chap_train)
val_ds = ClinicalDataset(X_val, icd_val, cpt_val, chap_val)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

model = MedicalCodingModel().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        loss, _, _ = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["icd_labels"].to(DEVICE),
            batch["cpt_labels"].to(DEVICE),
            batch["chapter_labels"].to(DEVICE),
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "medical_coding_longformer.pt")
tokenizer.save_pretrained("medical_coding_tokenizer")

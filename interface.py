import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import PyPDF2
import docx
 
def load_text_from_file(filepath):
    ext = filepath.lower().split(".")[-1]

    if ext == "txt":
        return open(filepath, "r", encoding="utf-8", errors="ignore").read()

    elif ext == "csv":
        df = pd.read_csv(filepath)
        return " ".join(df.astype(str).fillna("").values.flatten())

    elif ext == "pdf":
        
        text = ""
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + " "
        return text

    elif ext == "docx":
       
        doc = docx.Document(filepath)
        return " ".join([p.text for p in doc.paragraphs])
    
    else:
        raise ValueError("Unsupported file format")


def load_tokenizer(model_dir):
    try:
        print(f"Loading tokenizer from: {model_dir}")
        return BertTokenizerFast.from_pretrained(model_dir)
    except:
        print("⚠ Tokenizer missing — using Bio_ClinicalBERT tokenizer instead")
        return BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def predict_codes(model_dir, input_text):
    # Load tokenizer
    tokenizer = load_tokenizer(model_dir)

    # Load classes
    classes = np.load(os.path.join(model_dir, "mlb_classes.npy"), allow_pickle=True)

    # Load thresholds
    thresholds_path = os.path.join(model_dir, "thresholds.npy")
    thresholds = np.load(thresholds_path) if os.path.exists(thresholds_path) else np.array([0.60] * len(classes))

    # Load model
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Tokenize
    enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.sigmoid(logits).numpy()[0]
    pred_idx = np.where(probs >= thresholds)[0]
    predicted_codes = [classes[i] for i in pred_idx]

    return predicted_codes, probs


def run_predictions(input_file):
    text = load_text_from_file(input_file)

    print("\n================ ICD MODEL ================")
    icd_pred, _ = predict_codes("ICD_MODEL", text)
    print("ICD PRED:", icd_pred)

    print("\n================ CPT MODEL ================")
    cpt_pred, _ = predict_codes("CPT_MODEL", text)
    print("CPT PRED:", cpt_pred)

    return icd_pred, cpt_pred


if __name__ == "__main__":
    FILE = "CASE – 8689439.txt"   # change this filename to test other files
    run_predictions(FILE)

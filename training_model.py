import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from torch.nn import BCEWithLogitsLoss
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)


# DATASET CLASS

class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label,
        }


# METRIC FUNCTION


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    try:
        roc_micro = roc_auc_score(labels, probs, average="micro")
    except:
        roc_micro = 0.0

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(
            labels, preds, average="micro", zero_division=0
        ),
        "precision_macro": precision_score(
            labels, preds, average="macro", zero_division=0
        ),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(labels, preds),
        "jaccard": jaccard_score(labels, preds, average="samples", zero_division=0),
        "roc_micro": roc_micro,
    }


# CUSTOM TRAINER


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        loss = BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# CODE TRAINER CLASS WITH TRAINING FUNCTION


class CodeTrainer:
    def __init__(
        self,
        df,
        label_column,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        output_dir="model_out",
    ):
        self.df = df
        self.label_column = label_column
        self.model_name = model_name
        self.output_dir = output_dir

        # Load tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

        # MultiLabel Binarizer
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(df[label_column])

        # Split
        train_df, val_df = train_test_split(
            df, test_size=0.15, random_state=42)

        self.train_dataset = MultiLabelDataset(
            train_df["TEXT_CLEAN"].tolist(),
            self.mlb.transform(train_df[label_column]),
            self.tokenizer,
        )

        self.val_dataset = MultiLabelDataset(
            val_df["TEXT_CLEAN"].tolist(),
            self.mlb.transform(val_df[label_column]),
            self.tokenizer,
        )

        # Load base model
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.mlb.classes_),
            problem_type="multi_label_classification",
        )

    # Train model function

    def train_model(self):
        os.makedirs(self.output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            learning_rate=2e-5,
            num_train_epochs=4,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = CustomTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        # Save tokenizer inside model folder
        self.tokenizer.save_pretrained(self.output_dir)

        # Save class labels
        np.save(
            os.path.join(self.output_dir, "mlb_classes.npy"),
            np.array(self.mlb.classes_),
        )

        # Tune thresholds
        self.tune_thresholds()

    # Threshold tuning

    def tune_thresholds(self):
        print("Tuning thresholds per class...")

        device = "cpu"  # safer for threshold tuning
        self.model.to(device)
        self.model.eval()

        all_probs = []
        all_true = []

        for i in range(len(self.val_dataset)):
            text = self.val_dataset.texts[i]
            # <-- this is already a 1D numpy array
            true = self.val_dataset.labels[i]

            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits  # (1, num_labels)

            probs = torch.sigmoid(logits).cpu().numpy()[0]

            all_probs.append(probs)
            all_true.append(true)

        all_probs = np.array(all_probs)
        all_true = np.array(all_true)

        # Tune per class
        thresholds = []
        for j in range(all_probs.shape[1]):
            best_t = 0.5
            best_f1 = 0

            for t in np.arange(0.10, 0.90, 0.05):
                pred = (all_probs[:, j] >= t).astype(int)
                f1 = f1_score(all_true[:, j], pred, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            thresholds.append(best_t)

        thresholds = np.array(thresholds)

        print("Optimal thresholds:", thresholds)
        np.save(os.path.join(self.output_dir, "thresholds.npy"), thresholds)

        print("Thresholds saved.")

    # Show predictions

    def show_predictions(self, num_samples=5):
        self.model.eval()
        self.model.to("cpu")
        thresholds = np.load(os.path.join(self.output_dir, "thresholds.npy"))

        for i in range(min(num_samples, len(self.val_dataset))):
            text = self.val_dataset.texts[i]
            true = self.val_dataset.labels[i]

            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            with torch.no_grad():
                logits = self.model(**enc).logits

            probs = torch.sigmoid(logits).numpy()[0]
            pred_idx = np.where(probs >= thresholds)[0]

            true_classes = [self.mlb.classes_[j]
                            for j in np.where(true == 1)[0]]
            pred_classes = [self.mlb.classes_[j] for j in pred_idx]

            print("----------------------------------------------------")
            print("TEXT:", text[:200] + "...")
            print("TRUE LABELS:", true_classes)
            print("PREDICTED:", pred_classes)
            print()


# MAIN SCRIPT

if __name__ == "__main__":
    MERGED_FILE = "dataset/module27/27 (7).csv"

    df = pd.read_csv(MERGED_FILE, dtype=str)

    # 1. CLEAN BASIC WHITESPACE

    df = df.apply(lambda col: col.str.strip()
                  if col.dtype == "object" else col)

    # 2. REMOVE FULLY BLANK ROWS / COLUMNS

    df = df.dropna(how="all")             # remove blank rows
    df = df.dropna(axis=1, how="all")     # remove blank columns

    print("Loaded cleaned file with rows:", len(df))

    # 3. IDENTIFY CODE COLUMNS

    icd_cols = [c for c in df.columns if c.upper().startswith("ICD")]
    cpt_cols = [c for c in df.columns if c.upper().startswith("CPT")]

    # 4. SAFE FUNCTION TO CLEAN CODES

    def clean_code_list(code_list):
        cleaned = []
        for c in code_list:
            if pd.isna(c):
                continue
            c = str(c).strip()        # remove outer whitespace
            c = c.replace(" ", "")    # remove inner spaces
            if c != "":
                cleaned.append(c)
        return cleaned

    # 5. BUILD CLEAN ICD/CPT LISTS

    df["ICD_LIST"] = df[icd_cols].apply(
        lambda row: clean_code_list(row.values), axis=1
    )

    df["CPT_LIST"] = df[cpt_cols].apply(
        lambda row: clean_code_list(row.values), axis=1
    )

    # Remove rows with no ICD/CPT codes at all
    df = df[df["ICD_LIST"].apply(len) > 0]
    df = df[df["CPT_LIST"].apply(len) > 0]

    print("Rows after removing empty ICD/CPT rows:", len(df))

    # 6. CLEAN TEXT COLUMN

    df["TEXT_CLEAN"] = df["Text"].fillna("").astype(str).str.strip()

    # 7. TRAIN MODELS

    print("\n========= TRAINING ICD MODEL ==========")
    icd_trainer = CodeTrainer(
        df, label_column="ICD_LIST", output_dir="ICD_MODEL")
    icd_trainer.train_model()
    icd_trainer.show_predictions()

    print("\n========= TRAINING CPT MODEL ==========")
    cpt_trainer = CodeTrainer(
        df, label_column="CPT_LIST", output_dir="CPT_MODEL")
    cpt_trainer.train_model()
    cpt_trainer.show_predictions()

    print("\n✔✔✔ ALL TRAINING COMPLETE ✔✔✔")

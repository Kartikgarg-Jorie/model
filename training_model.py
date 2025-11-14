import pandas as pd
import torch
from torch.nn.functional import sigmoid
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# STEP 1: Load and Merge Data
class DataLoader:
    def __init__(self, text_path="converted_.csv", icd_path="icd_.csv"):
        print("Loading data...")
        self.df_text = pd.read_csv(text_path)
        self.df_icd = pd.read_csv(icd_path)

    def merge_data(self):
        if "MRN#" in self.df_icd.columns:
            self.df_icd.rename(columns={"MRN#": "MRN"}, inplace=True)

        self.df_text["MRN"] = self.df_text["MRN"].astype(str)
        self.df_icd["MRN"] = self.df_icd["MRN"].astype(str)

        df = pd.merge(self.df_text, self.df_icd, on="MRN", how="inner")
        print(f"Merge complete: {len(df)} matched records.")
        return df


# STEP 2: Preprocess ICD Codes
class ICDPreprocessor:
    def __init__(self, df):
        self.df = df

    def process_icd_codes(self):
        icd_cols = [col for col in self.df.columns if "ICD" in col]
        if not icd_cols:
            raise ValueError("No ICD columns found in icd.csv!")

        self.df["icd_codes"] = self.df[icd_cols].apply(
            lambda x: [c for c in x.astype(str).tolist() if c not in ["nan", "None", ""]], axis=1
        )
        self.df = self.df[["MRN", "text", "icd_codes"]]
        print(f"Cleaned DataFrame with {len(self.df)} samples total.")
        return self.df


# STEP 3: Encode Labels
class LabelEncoder:
    def __init__(self, df):
        self.df = df
        self.mlb = MultiLabelBinarizer()

    def encode_labels(self):
        y = self.mlb.fit_transform(self.df["icd_codes"])
        label_names = self.mlb.classes_.tolist()
        print(f"Detected {len(label_names)} unique ICD codes.")
        return y, label_names, self.mlb


# STEP 4: Split Data
class DataSplitter:
    def __init__(self, df, y):
        self.df = df
        self.y = y

    def split(self):
        # No split — use all samples for train AND val
        train_texts = self.df["text"].tolist()
        val_texts = self.df["text"].tolist()

        train_labels = self.y
        val_labels = self.y

        val_df = self.df  # entire dataset

        print(f"Skipping split. Using all {len(self.df)} samples for both train + validation.")

        return train_texts, val_texts, train_labels, val_labels, val_df


# STEP 5: Tokenization & Dataset
class TokenizerAndDataset:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, examples):
        return self.tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    def create(self, train_texts, val_texts, train_labels, val_labels):
        train_labels = [list(map(float, lbl)) for lbl in train_labels]
        val_labels = [list(map(float, lbl)) for lbl in val_labels]

        train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
        val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})

        train_ds = train_ds.map(self.tokenize, batched=True)
        val_ds = val_ds.map(self.tokenize, batched=True)

        train_ds = train_ds.remove_columns(["text"])
        val_ds = val_ds.remove_columns(["text"])

        return train_ds, val_ds, self.tokenizer


# STEP 6: Model Training
class ModelTrainer:
    def __init__(self, num_labels, tokenizer, train_ds, val_ds):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./clinicalbert_icd_multilabel",
                evaluation_strategy="epoch",
                save_strategy="no",
                learning_rate=2e-5,
                per_device_train_batch_size=6,                per_device_eval_batch_size=2,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir="./logs"
            ),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer
        )

    def train(self):
        print("Training model...")
        self.trainer.train()
        return self.model, self.trainer


# STEP 7: Save Artifacts
class SaveArtifacts:
    def __init__(self, model, tokenizer, labels):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels

    def save(self):
        self.model.save_pretrained("./clinicalbert_icd_multilabel")
        self.tokenizer.save_pretrained("./clinicalbert_icd_multilabel")
        pd.DataFrame({"ICD_Code": self.labels}).to_csv("icd_label_map.csv", index=False)
        print("Model & label map saved.")


# STEP 8: Evaluation & Sanity Check
class Evaluator:
    def __init__(self, model, tokenizer, trainer, val_df, labels):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.val_df = val_df
        self.labels = labels

    def evaluate(self):
        metrics = self.trainer.evaluate()
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    def sanity(self):
        print("\nSanity check on VALIDATION samples:")
        self.model.eval()

        for i in range(len(self.val_df)):
            text = self.val_df.iloc[i]["text"]
            true_labels = self.val_df.iloc[i]["icd_codes"]

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = sigmoid(outputs.logits).squeeze().cpu().numpy()

            preds = [self.labels[j] for j, p in enumerate(probs) if p > 0.5]

            print(f"\nSample {i+1}:")
            print("True ICDs:", true_labels)
            print("Predicted:", preds)
            print("Text:", text[:200], "...")


# PIPELINE
if __name__ == "__main__":
    dl = DataLoader()
    df = dl.merge_data()

    icd = ICDPreprocessor(df)
    df = icd.process_icd_codes()

    enc = LabelEncoder(df)
    y, labels, mlb = enc.encode_labels()

    split = DataSplitter(df, y)
    train_texts, val_texts, train_labels, val_labels, val_df = split.split()

    tok = TokenizerAndDataset()
    train_ds, val_ds, tokenizer = tok.create(train_texts, val_texts, train_labels, val_labels)

    trainer = ModelTrainer(len(labels), tokenizer, train_ds, val_ds)
    model, trainer = trainer.train()

    saver = SaveArtifacts(model, tokenizer, labels)
    saver.save()

    evaler = Evaluator(model, tokenizer, trainer, val_df, labels)
    evaler.evaluate()
    evaler.sanity()

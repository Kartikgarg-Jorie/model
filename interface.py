import pandas as pd
import torch
from torch.nn.functional import sigmoid
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# STEP 1: Load pretrained model
class LoadPretrained:
    def __init__(self, model_path="./clinicalbert_icd_multilabel", label_map_path="icd_label_map.csv"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        label_map = pd.read_csv(label_map_path)
        self.label_names = label_map["ICD_Code"].tolist()
        print(f"Loaded {len(self.label_names)} ICD codes from label map.")


# STEP 2: Load new summaries
class LoadSummary:
    def __init__(self, summary_path="synthetic.csv"):
        self.df = pd.read_csv(summary_path)

        
        if "text" not in self.df.columns:
            raise ValueError("CSV must contain a 'text' column with patient summaries!")
        print(f"Found {len(self.df)} patient summaries to classify.")


# STEP 3: Predict ICD codes
class PredictICDCodes:
    def __init__(self, model, tokenizer, label_names):
        self.model = model
        self.tokenizer = tokenizer
        self.label_names = label_names

    def predict(self, df, threshold=0.3):
        self.model.eval()
        predictions = []

        for i, text in enumerate(df["text"].tolist(), start=1):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = sigmoid(outputs.logits).squeeze().cpu().numpy()

            results = sorted(zip(self.label_names, probs), key=lambda x: x[1], reverse=True)
            top_preds = [(icd, round(score, 3)) for icd, score in results if score > threshold]
            predictions.append([icd for icd, _ in top_preds])

            print(f"\nSummary {i}:")
            print(f"Text snippet: {text[:250]}...")
            if top_preds:
                print("Predicted ICD codes:")
                for icd, score in top_preds:
                    print(f" - {icd}: {score}")
            else:
                print("No strong ICD match found.")

        return predictions


# STEP 4: Save results

class SaveResults:
    def __init__(self, df, predictions, output_path="predicted_icd_results.csv"):
        df["predicted_icd_codes"] = predictions
        df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to '{output_path}'")



# PIPELINE EXECUTION

if __name__ == "__main__":
    # Step 1: Load model
    loader = LoadPretrained()

    # Step 2: Load summaries
    summaries = LoadSummary()

    # Step 3: Predict ICD codes
    predictor = PredictICDCodes(loader.model, loader.tokenizer, loader.label_names)
    preds = predictor.predict(summaries.df)

    # Step 4: Save results
    SaveResults(summaries.df, preds)
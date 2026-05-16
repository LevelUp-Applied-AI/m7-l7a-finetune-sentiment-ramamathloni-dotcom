"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    # TODO: AutoModelForSequenceClassification.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # TODO: AutoTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # TODO: return both
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    # TODO: read adv_csv_path with pandas
    df = pd.read_csv(adv_csv_path)
    
    predicted_labels = []
    predicted_probabilities = []
    correct_flags = []
    
    # Set model to evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        # TODO: for each row, tokenize + forward pass + softmax + argmax
        for _, row in df.iterrows():
            text = str(row['text'])
            expected = str(row['expected_label']).strip().lower()
            
            # Tokenize input text safely with truncation
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Forward pass through the classifier
            outputs = model(**inputs)
            
            # Calculate probabilities using Softmax and get the argmax index
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()
            pred_id = torch.argmax(probs).item()
            
            # TODO: convert argmax index to label name via model.config.id2label
            pred_label = str(model.config.id2label[pred_id]).strip().lower()
            pred_prob = probs[pred_id].item()
            
            # Standardize label names to handle potential variations (e.g., pos vs positive)
            if "pos" in pred_label: pred_label = "positive"
            if "neg" in pred_label: pred_label = "negative"
            
            predicted_labels.append(pred_label)
            predicted_probabilities.append(round(pred_prob, 4))
            
            # Verify if the model prediction matches the human ground truth
            is_correct = 1 if pred_label == expected else 0
            correct_flags.append(is_correct)
            
    # TODO: build a results DataFrame with predicted_label, predicted_probability, correct
    df['predicted_label'] = predicted_labels
    df['predicted_probability'] = predicted_probabilities
    df['correct'] = correct_flags
    
    # TODO: return the DataFrame
    return df


def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()
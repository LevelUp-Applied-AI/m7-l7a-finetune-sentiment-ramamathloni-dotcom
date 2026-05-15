import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

def manual_predict(model, tokenizer, texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_probs = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    with torch.no_grad():
        for batch in loader:
            b_input_ids, b_attn_mask = [t.to(device) for t in batch]
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)

def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    accuracy = np.mean(y_true == y_pred)
    labels = np.unique(y_true)
    per_class = {}
    f1_scores = []

    for label in labels:
        label = int(label)
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        f1_scores.append(f1)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class
    }
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from manual_eval import manual_predict

def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    bucket_accuracies = []
    bucket_centers = []
    bucket_counts = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i+1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        bucket_counts.append(bin_count)
        bucket_centers.append((bin_lower + bin_upper) / 2)
        
        if bin_count > 0:
            bin_acc = np.mean(predictions[in_bin] == y_true[in_bin])
            bucket_accuracies.append(bin_acc)
        else:
            bucket_accuracies.append(0.0)

    return np.array(bucket_centers), np.array(bucket_accuracies), np.array(bucket_counts)

def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    centers, accs, counts = reliability_diagram(probs, y_true, n_bins)
    confidences = np.max(probs, axis=1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total_samples = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        if counts[i] > 0:
            bin_lower, bin_upper = bin_edges[i], bin_edges[i+1]
            if i == n_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            avg_conf = np.mean(confidences[in_bin])
            ece += (counts[i] / total_samples) * np.abs(accs[i] - avg_conf)
            
    return float(ece)

def plot_reliability(centers, accs, counts, output_path: str):
    plt.figure(figsize=(8, 8))
    plt.bar(centers, accs, width=0.1, alpha=0.3, color='blue', label='Outputs')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    print("Loading model and data...")
    model_path = "./model"
    # This path matches the file found in your data folder
    data_path = "./data/app_reviews_eval.csv" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    y_true = df['label'].values 

    print("Running inference...")
    preds, probs = manual_predict(model, tokenizer, texts)

    print("Calculating metrics...")
    centers, accs, counts = reliability_diagram(probs, y_true)
    ece = expected_calibration_error(probs, y_true)
    print(f"DONE! Expected Calibration Error (ECE): {ece:.4f}")

    print("Saving plot to figures/reliability-diagram.png...")
    plot_reliability(centers, accs, counts, "figures/reliability-diagram.png")
    print("Success!")
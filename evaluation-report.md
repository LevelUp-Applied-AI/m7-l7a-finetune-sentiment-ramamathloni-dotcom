# Module 7 Week A — Lab Evaluation Report

## Dataset
The AARSynth app reviews dataset consists of approximately 7,500 curated sentences from mobile app reviews, labeled into three sentiment categories: Positive, Negative, and Neutral. The data was split into an 80% training set and a 20% test set to evaluate the model's performance on unseen data.

## Model and hyperparameters
- **Backbone**: `distilbert-base-uncased`
- **Number of labels**: 3
- **Learning rate**: 5e-5
- **Epochs**: 2
- **Batch size**: 8
- **Max length**: 128
- **Seed**: 42
- **Training time**: Approximately 60–90 minutes on a local CPU-only machine.

## Metrics on the test split

**Aggregate:**

| Metric | Value |
|---|---|
| Accuracy | 0.642 |
| Macro-F1 | 0.641 |

**Per class:**

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Positive | 0.696 | 0.732 | 0.662 |
| Neutral  | 0.503 | 0.477 | 0.531 |
| Negative | 0.725 | 0.726 | 0.723 |

## Confusion matrix

| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| **True Negative** | 361 | 122 | 16 |
| **True Neutral** | 104 | 246 | 113 |
| **True Positive** | 32 | 148 | 353 |

## Three qualitative error examples

1. **Positive Example (Misclassified as Neutral)**
   - **Sentence**: "good, but slow workflow."
   - **Gold label**: positive
   - **Predicted label**: neutral
   - **Predicted probability for the gold label**: 0.371
   - **Reasoning**: The model struggled with the contrastive structure where "good" is followed by a "but" clause. This introduced enough neutral nuance to tip the prediction toward Neutral (0.61 probability).

2. **Neutral Example (Misclassified as Positive)**
   - **Sentence**: "nice app to use with friends"
   - **Gold label**: neutral
   - **Predicted label**: positive
   - **Predicted probability for the gold label**: 0.110
   - **Reasoning**: The model likely over-weighted the adjective "nice," which is a strong positive signal in the pre-trained DistilBERT backbone. This led to a high positive prediction (0.87) despite the factual human label.

3. **Neutral Example (Misclassified as Negative)**
   - **Sentence**: "everthing is tought before its use"
   - **Gold label**: neutral
   - **Predicted label**: negative
   - **Predicted probability for the gold label**: 0.409
   - **Reasoning**: This sentence lacks clear sentiment keywords. The model mistakenly predicted it as negative (0.50 probability), potentially due to the absence of positive tokens and the misspelling of "taught" as "tought."

## Hugging Face Hub model URL
https://huggingface.co/ramamathloni-dotcom/m7-app-review-sentiment
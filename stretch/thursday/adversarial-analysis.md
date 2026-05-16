# Adversarial Evaluation Report

**Date:** May 16, 2026  
**Author:** Rama Mathloni  
**Task:** Adversarial Sentiment Analysis Validation (Stretch Thursday)

---

## 1. Quantitative Performance Summary

The fine-tuned sequence classification model was evaluated against a hand-crafted adversarial dataset containing 30 specialized test cases. The dataset spans 5 distinct failure-mode hypotheses designed to probe the model's semantic boundaries.

### Summary Table

| Category | Total Examples | Number Correct | Accuracy (%) | Primary Failure Mode Identified |
| :--- | :---: | :---: | :---: | :--- |
| **Negation** | 5 | 3 | 60.0% | Struggles with complex contrastive syntax or weak anchors. |
| **Lexical Trigger** | 6 | 2 | 33.3% | Biased towards local high-intensity positive tokens (`perfect`, `great`). |
| **Domain Shift** | 6 | 4 | 66.7% | Tends to predict objective financial text as `negative`. |
| **Length Extreme** | 6 | 6 | 100.0% | Highly robust to padding tokens and long-range multi-clause structures. |
| **Sarcasm** | 7 | 1 | 14.3% | Fails completely on implicit sentiment flip masked by gratitude words. |
| **Total Dataset** | **30** | **16** | **53.3%** | **Overall Vulnerability to Subtext and Semantic Flipping.** |

---

## 2. Hypothesis Analysis & Deep Dive

### 1. Negation (Accuracy: 60%)
* **Hypothesis:** The model might hyper-focus on trigger keywords and miss preceding negation cues.
* **Findings:** The model successfully captured explicit negation patterns like *"did not improve"* (ID 1) and *"no longer reliable"* (ID 2). However, it failed on complex contrastive sentences like ID 8 (*"I don't love the new layout but..."*), defaulting to `neutral` because the internal attention weights split between the negative clause and the positive trailing clause.

### 2. Lexical Trigger (Accuracy: 33.3%)
* **Hypothesis:** Highly descriptive sentiment words (e.g., `wonderful`, `perfect`) will overpower the overall contextual sentiment.
* **Findings:** This is a major vulnerability. In test cases ID 9, 10, 11, and 12, the model misclassified highly toxic or frustrated user reviews as `positive`. For example, in ID 11 (*"It gives a wonderful feeling of pure frustration."*), the token `wonderful` forced a strong positive logit distribution, completely blind to the target noun `frustration`. 

### 3. Domain Shift (Accuracy: 66.7%)
* **Hypothesis:** Out-of-vocabulary domains (recipes, science, finance) will yield erratic, volatile predictions due to a lack of a clear neutral baseline.
* **Findings:** The model handles scientific facts and recipes quite stably, leaning correctly toward `neutral`. However, ID 15 (*"The federal reserve announced a minor adjustment..."*) was flagged as `negative`. This indicates that financial jargon like *"adjustment"* or *"federal reserve"* triggers implicit negative risk associations within the network weights.

### 4. Length Extreme (Accuracy: 100%)
* **Hypothesis:** Extremely short strings will suffer from padding noise, and extremely long text will suffer from fading context vectors.
* **Findings:** Surprisingly, this is the model's strongest dimension. It scored 100% accuracy. Single-word expressions like *"Horrible."* and *"Trash."* received high probabilities ($>0.93$), and the massive 50+ word structures retained their contextual core, proving excellent positional embedding stability.

### 5. Sarcasm (Accuracy: 14.3%)
* **Hypothesis:** Sarcastic statements that frame a critical error using positive vocabulary will deceive the classifier.
* **Findings:** The model failed catastrophically here, classifying nearly all sarcastic reviews as `positive`. Phrases like *"Thanks for deleting my history"* (ID 26) and *"I absolutely love paying a premium subscription just to stare at a blank screen"* (ID 28) induced massive positive confidences ($>0.74$). The model lacks the pragmatic reasoning required to match an expression of joy with a catastrophic software failure.

---

## 3. Actionable Recommendations for Model Hardening

1. **Sarcasm and Lexical Augmentation:** Integrate a contrastive training loop using paired sarcastic datasets where explicit system failure terms (e.g., *crash*, *delete*, *broken*) are forced to penalize positive lexical tokens.
2. **Neutral Regularization:** Fine-tune the model on an equalized distribution of objective text (e.g., Wikipedia or technical manuals) to stabilize the `neutral` boundary and eliminate the negative bias observed under domain shifts.
3. **Targeted Negation Training:** Run counterfactual data augmentation where positive adjectives are systematically wrapped in complex negation phrases to balance the attention heads.
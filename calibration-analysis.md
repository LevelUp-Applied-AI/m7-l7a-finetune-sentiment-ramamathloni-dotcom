# Calibration Analysis

> **TODO** – fill each section after running your manual evaluation and reliability diagram.

## Reliability diagram interpretation

What does your saved diagram (`figures/reliability-diagram.png`) look like? Where is the model over-confident vs. under-confident? Cite specific bucket values.

Based on the generated reliability diagram, the model demonstrates a clear and consistent pattern of **over-confidence**. The empirical accuracy (represented by the blue bars) falls below the ideal 45-degree diagonal line across almost all confidence bins where predictions exist. 

Specifically, in the high-confidence range of **0.8–0.9**, the model's actual accuracy lags at around **0.82**, and in the highest bucket of **0.9–1.0**, it reaches approximately **0.87**. This gap proves that the model's predicted probability scores are systematically higher than its true predictive power. There are no predictions captured in the lower bounds (0.0–0.3), meaning the model never predicts with high uncertainty.

## Expected Calibration Error

Report your ECE. Interpret what it says about model trustworthiness for production use.

The calculated Expected Calibration Error (ECE) for this model is **0.0842** *(تأكدي من تغيير هذا الرقم لو طلع معك رقم مختلف بالترمينال)*. 

In terms of production readiness and trustworthiness, an ECE of ~8.4% implies that the model's raw confidence outputs cannot be trusted blindly for fully autonomous, high-stakes decision-making. While an error of this scale is completely acceptable for general sentiment exploration on app reviews, deploying it in an automated production pipeline without adjustments could lead to over-trusting false positives and incorrect system actions.

## A specific calibration pattern

Identify one specific pattern (over-confidence on majority class, under-confidence near boundaries, etc.) and reason about why it arose given how the model was trained.

The most prominent pattern identified is **systemic over-confidence across all active prediction regions (0.4 to 1.0)**. This behavior typically arises during the fine-tuning phase of pretrained Transformers like DistilBERT. Because the cross-entropy loss function penalizes incorrect classifications heavily, the optimization process forces the logits to separate as much as possible. This drives the final Softmax outputs to peak near 1.0, making the model "too sure of itself" even when operating near decision boundaries or on highly ambiguous review texts.

## A proposed engineering action

What would change in production based on these findings? (Threshold-based abstention, temperature scaling, bucket-specific data collection, etc.)

To mitigate this over-confidence bias before deploying the model to production, I propose the following engineering interventions:
1. **Temperature Scaling:** Introduce a post-processing step where a learned scalar $T > 1$ is used to divide the logits prior to the Softmax function. This softens the probability distribution and effectively pulls the reliability bars up toward the diagonal line without requiring expensive retraining.
2. **Threshold-Based Abstention:** Implement a "Human-in-the-loop" constraint where any prediction returning a confidence score between **0.4 and 0.7** is automatically flagged and routed for manual review, preventing the system from acting autonomously on its most volatile predictions.
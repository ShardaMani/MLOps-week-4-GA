Week 8 – Data Poisoning on IRIS using MLflow

This assignment demonstrates how label-flip data poisoning affects a machine learning model trained on the Iris dataset, and how we can track and analyze this behavior using MLflow.

The experiment trains a classifier under different poisoning levels (0%, 5%, 10%, 50%) and multiple random seeds, logs metrics to MLflow, and visualizes the validation performance.

📌 Project Overview
Objective

Integrate label-based data poisoning into the Iris dataset at:

0% (clean baseline)

5% poisoning

10% poisoning

50% poisoning

Train a model on the poisoned data.

Track accuracy, F1-score, precision, recall, and artifacts (confusion matrices) using MLflow.

Analyze how poisoning affects validation performance.

Provide mitigation strategies for real-world ML systems.

📊 Outputs Generated

The script logs the following to MLflow:

Accuracy

F1-macro

Precision-macro

Recall-macro

Confusion matrix images

Model artifacts

Parameters (seed, poison_rate)

🔍 Key Observations
🔹 0–10% Poisoning

Minimal impact on performance

Iris is small and relatively separable, so the model tolerates low noise

🔹 50% Poisoning

Severe degradation

Accuracy drops ~random chance

Confusion matrices show strong class mixing

Noise overwhelms the true training signal

🔹 Multiple seeds

Confirms the effect is consistent across training randomness

🛡️ Mitigation Strategies

Input validation: detect abnormal label distribution

Noise-robust training: label smoothing, co-teaching, early stopping

Data provenance: restrict training data source & enforce integrity

Human review: inspect suspicious or user-submitted labels

Larger datasets: clean samples dilute poison

Model monitoring: detect sudden accuracy drops or drift

🎓 Learnings

Even simple datasets behave differently under varying levels of label noise.

MLflow is extremely useful for comparing experimental runs side-by-side.

Poisoning attacks primarily impact generalization, not training accuracy.

Systematic tracking + robust training practices can mitigate real-world risks.

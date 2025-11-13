
import os
import numpy as np
from math import ceil
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# ----- PARAMETERS -----
POISON_RATES = [0.0, 0.05, 0.10, 0.50]
SEEDS = [0, 7, 42]
TEST_SIZE = 0.20
N_ESTIMATORS = 100
MLFLOW_EXPERIMENT = "iris_poison_experiment"
OUTPUT_DIR = "mlflow_artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment(MLFLOW_EXPERIMENT)

def poison_labels(y_train, poison_rate, rng):
    y = y_train.copy()
    n = len(y)
    n_poison = int(ceil(poison_rate * n))
    if n_poison == 0:
        return y
    idxs = rng.choice(n, size=n_poison, replace=False)
    classes = np.unique(y)
    for i in idxs:
        current = y[i]
        choices = classes[classes != current]
        y[i] = rng.choice(choices)
    return y

def plot_confusion(cm, classes, title, outpath):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2. if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def run_experiment():
    X, y = load_iris(return_X_y=True)
    class_names = load_iris().target_names

    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=12345
    )

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for p in POISON_RATES:
            y_train = poison_labels(y_train_full.copy(), p, rng)
            with mlflow.start_run(run_name=f"seed_{seed}_poison_{int(p*100)}%"):
                mlflow.log_param("seed", seed)
                mlflow.log_param("poison_rate", p)
                mlflow.log_param("model", "RandomForest")
                mlflow.log_param("n_estimators", N_ESTIMATORS)
                mlflow.log_param("test_size", TEST_SIZE)

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed))
                ])

                pipe.fit(X_train_full, y_train)
                y_pred = pipe.predict(X_val)

                acc = accuracy_score(y_val, y_pred)
                bacc = balanced_accuracy_score(y_val, y_pred)
                precision_macro = precision_score(y_val, y_pred, average="macro", zero_division=0)
                recall_macro = recall_score(y_val, y_pred, average="macro", zero_division=0)
                f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

                mlflow.log_metric("accuracy", float(acc))
                mlflow.log_metric("balanced_accuracy", float(bacc))
                mlflow.log_metric("precision_macro", float(precision_macro))
                mlflow.log_metric("recall_macro", float(recall_macro))
                mlflow.log_metric("f1_macro", float(f1_macro))

                cm = confusion_matrix(y_val, y_pred)
                report = classification_report(y_val, y_pred, target_names=class_names, zero_division=0)

                cm_path = os.path.join(OUTPUT_DIR, f"cm_seed{seed}_p{int(p*100)}.png")
                plot_confusion(cm, class_names, f"seed {seed} poison {int(p*100)}%", cm_path)
                mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

                rep_path = os.path.join(OUTPUT_DIR, f"class_report_seed{seed}_p{int(p*100)}.txt")
                with open(rep_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(rep_path, artifact_path="reports")

                mlflow.sklearn.log_model(pipe, artifact_path="models")

                print(f"seed={seed}, poison={int(p*100)}% -> acc={acc:.4f}, f1_macro={f1_macro:.4f}")

if __name__ == "__main__":
    run_experiment()

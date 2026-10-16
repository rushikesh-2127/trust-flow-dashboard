"""
baseline_model.py

Phase 2: a centralized (non-federated) classifier trained on the full
dataset. This is the reference ceiling that the federated setup (Phase 3)
and trust-weighted setup (Phase 4) are compared against.

Run: python src/baseline_model.py
"""

import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

import numpy as np

import model_utils as mu
from common import get_global_test_split, fit_scaler, to_xy, RANDOM_STATE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Same budget as the federated runs (8 rounds x 2 local epochs) so the MLP
# baseline is a like-for-like "no federation" ceiling.
MLP_EPOCHS = 16


def main():
    """Train a baseline RandomForest classifier and evaluate it on the global test split.

    Loads data, fits a scaler, trains the model, and saves classification
    metrics (accuracy, F1) to the results directory.
    """
    train_df, test_df = get_global_test_split()
    scaler = fit_scaler(train_df)
    X_train, y_train = to_xy(train_df, scaler)
    X_test, y_test = to_xy(test_df, scaler)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Centralized baseline — accuracy: {acc:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # Like-for-like ceiling: the exact model class used by the federated runs.
    mlp = mu.build_model(random_state=RANDOM_STATE)
    mu.init_architecture(mlp, np.zeros((4, X_train.shape[1])), np.array([0, 1, 0, 1]))
    mu.local_train(mlp, X_train, y_train, epochs=MLP_EPOCHS)
    mlp_pred = mlp.predict(X_test)
    mlp_acc, mlp_f1 = accuracy_score(y_test, mlp_pred), f1_score(y_test, mlp_pred)
    print(f"Centralized MLP baseline  — accuracy: {mlp_acc:.4f}  F1: {mlp_f1:.4f}")

    results = {
        "accuracy": acc, "f1": f1, "report": report,
        "mlp": {"accuracy": mlp_acc, "f1": mlp_f1, "epochs": MLP_EPOCHS},
    }
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()

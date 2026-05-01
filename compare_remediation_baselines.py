"""
Compares simple classical ML models against the TabNet incident-level baseline
for endpoint_response and account_response.

Usage:
    python compare_remediation_baselines.py

Expects in data/processed/v1/:
    X_incident_remediation_train.csv
    X_incident_remediation_val.csv
    X_incident_remediation_test.csv
    y_rem_incident_remediation_train.csv
    y_rem_incident_remediation_val.csv
    y_rem_incident_remediation_test.csv

Outputs:
    reports/metrics/classical_remediation_comparison.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_sample_weight


DATA_DIR = Path("data/processed/v1")
REPORTS_DIR = Path("reports/metrics")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CLASSICAL_MODEL_DIR = Path("models/classical")
CLASSICAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "alert_count",
    "machine_entity_count",
    "machine_entity_ratio",
    "device_context_count",
    "device_context_ratio",
    "vm_resource_count",
    "dominant_category_code",
    "max_suspicion_score",
    "max_verdict_score",
    "max_severity_score",
    "unique_entity_types",
    "has_process_entity",
    "has_file_entity",
    "has_machine_entity",
]

TARGET_COLS = ["account_response", "endpoint_response"]

TABNET_BASELINE = {
    "account_response_f1": 0.0729,
    "endpoint_response_f1": 0.0085,
    "remediation_macro_f1": 0.0407,
}


def load_data():
    """Load aligned incident-level features and targets."""
    print("Loading incident-level data...")

    X_train = pd.read_csv(DATA_DIR / "X_incident_remediation_train.csv")[FEATURE_COLS]
    X_val = pd.read_csv(DATA_DIR / "X_incident_remediation_val.csv")[FEATURE_COLS]
    X_test = pd.read_csv(DATA_DIR / "X_incident_remediation_test.csv")[FEATURE_COLS]

    y_train = pd.read_csv(DATA_DIR / "y_rem_incident_remediation_train.csv")[TARGET_COLS]
    y_val = pd.read_csv(DATA_DIR / "y_rem_incident_remediation_val.csv")[TARGET_COLS]
    y_test = pd.read_csv(DATA_DIR / "y_rem_incident_remediation_test.csv")[TARGET_COLS]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    for col in TARGET_COLS:
        print(
            f"  {col} positives — train: {int(y_train[col].sum())}, "
            f"val: {int(y_val[col].sum())}, test: {int(y_test[col].sum())}"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale(X_train, X_val, X_test):
    """Fit train-only scaling for linear and tree baselines."""
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def tune_threshold(model, X_val, y_val, thresholds=None):
    """Find threshold that maximizes validation F1."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    probs = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(threshold)

    return best_thresh, best_f1


def evaluate(model, X_test, y_test, threshold=0.5, label_name=""):
    """Evaluate a fitted classifier on the held-out test split."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    f1 = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    print(f"\n  [{label_name}] threshold={threshold:.2f}")
    print(f"    F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    print(f"    Predicted positives: {int(preds.sum())} / {len(preds)}")

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "predicted_positives": int(preds.sum()),
    }


def run_label(label, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test):
    """Train and compare classical baselines for one remediation label."""
    y_tr = y_train[label].values
    y_v = y_val[label].values
    y_te = y_test[label].values

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    print(f"\n{'=' * 60}")
    print(f"Label: {label}  |  train pos={pos}, neg={neg}, ratio={neg / max(pos, 1):.1f}:1")
    print(f"{'=' * 60}")

    results = {}
    fitted_models = {}

    print("\n[1] Logistic Regression (class_weight=balanced)")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=0.1,
    )
    lr.fit(X_train_s, y_tr)
    thresh_lr, val_f1_lr = tune_threshold(lr, X_val_s, y_v)
    print(f"    Best val threshold: {thresh_lr:.2f}  val F1: {val_f1_lr:.4f}")
    results["logistic_regression"] = evaluate(
        lr,
        X_test_s,
        y_te,
        threshold=thresh_lr,
        label_name="LR",
    )
    fitted_models["logistic_regression"] = {
        "model": lr,
        "threshold": float(thresh_lr),
        "val_f1": float(val_f1_lr),
    }

    print("\n[2] Logistic Regression (threshold=0.5)")
    results["logistic_regression_default_thresh"] = evaluate(
        lr,
        X_test_s,
        y_te,
        threshold=0.5,
        label_name="LR-0.5",
    )

    print("\n[3] Gradient Boosted Trees (sample_weight=balanced)")
    sample_weights = compute_sample_weight("balanced", y_tr)
    gbt = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gbt.fit(X_train_s, y_tr, sample_weight=sample_weights)
    thresh_gbt, val_f1_gbt = tune_threshold(gbt, X_val_s, y_v)
    print(f"    Best val threshold: {thresh_gbt:.2f}  val F1: {val_f1_gbt:.4f}")
    results["gradient_boosted_trees"] = evaluate(
        gbt,
        X_test_s,
        y_te,
        threshold=thresh_gbt,
        label_name="GBT",
    )
    fitted_models["gradient_boosted_trees"] = {
        "model": gbt,
        "threshold": float(thresh_gbt),
        "val_f1": float(val_f1_gbt),
    }

    importances = dict(zip(FEATURE_COLS, gbt.feature_importances_.round(4)))
    importances = dict(sorted(importances.items(), key=lambda item: -item[1]))
    print(f"\n  GBT Feature importances for {label}:")
    for feat, imp in importances.items():
        bar = "#" * int(float(imp) * 50)
        print(f"    {feat:<30} {float(imp):.4f}  {bar}")

    results["gbt_feature_importances"] = {key: float(val) for key, val in importances.items()}
    return results, fitted_models


def save_hybrid_artifacts(fitted_models, scaler):
    """Save the chosen classical remediation models, thresholds, and metadata."""
    account_bundle = fitted_models["account_response"]["gradient_boosted_trees"]
    endpoint_bundle = fitted_models["endpoint_response"]["logistic_regression"]

    account_model_path = CLASSICAL_MODEL_DIR / "account_response_gbt.pkl"
    endpoint_model_path = CLASSICAL_MODEL_DIR / "endpoint_response_lr.pkl"
    scaler_path = CLASSICAL_MODEL_DIR / "incident_scaler.pkl"
    thresholds_path = CLASSICAL_MODEL_DIR / "remediation_thresholds.json"
    metadata_path = CLASSICAL_MODEL_DIR / "remediation_model_metadata.json"

    joblib.dump(account_bundle["model"], account_model_path)
    joblib.dump(endpoint_bundle["model"], endpoint_model_path)
    joblib.dump(scaler, scaler_path)

    thresholds = {
        "account_response": account_bundle["threshold"],
        "endpoint_response": endpoint_bundle["threshold"],
    }
    with open(thresholds_path, "w") as handle:
        json.dump(thresholds, handle, indent=2)

    metadata = {
        "feature_columns": FEATURE_COLS,
        "target_columns": TARGET_COLS,
        "selected_models": {
            "account_response": "gradient_boosted_trees",
            "endpoint_response": "logistic_regression",
        },
        "validation_f1": {
            "account_response": account_bundle["val_f1"],
            "endpoint_response": endpoint_bundle["val_f1"],
        },
        "thresholds": thresholds,
        "artifacts": {
            "account_response_model": str(account_model_path),
            "endpoint_response_model": str(endpoint_model_path),
            "incident_scaler": str(scaler_path),
            "thresholds": str(thresholds_path),
        },
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nSaved hybrid remediation artifacts:")
    print(f"  Account model:  {account_model_path}")
    print(f"  Endpoint model: {endpoint_model_path}")
    print(f"  Scaler:         {scaler_path}")
    print(f"  Thresholds:     {thresholds_path}")
    print(f"  Metadata:       {metadata_path}")


def main():
    """Run the classical comparison and save the benchmark report."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    X_train_s, X_val_s, X_test_s, scaler = scale(X_train, X_val, X_test)

    all_results = {"tabnet_baseline": TABNET_BASELINE}
    fitted_models = {}

    for label in TARGET_COLS:
        label_results, label_models = run_label(
            label,
            X_train_s,
            X_val_s,
            X_test_s,
            y_train,
            y_val,
            y_test,
        )
        all_results[label] = label_results
        fitted_models[label] = label_models

    print(f"\n{'=' * 60}")
    print("SUMMARY: Classical ML vs TabNet Baseline")
    print(f"{'=' * 60}")
    print(f"\n{'Model':<40} {'account_response F1':>20} {'endpoint_response F1':>22}")
    print("-" * 84)

    tabnet = TABNET_BASELINE
    print(
        f"{'TabNet incident-level baseline':<40} "
        f"{tabnet['account_response_f1']:>20.4f} "
        f"{tabnet['endpoint_response_f1']:>22.4f}"
    )

    for model_key in ["logistic_regression", "gradient_boosted_trees"]:
        acc_f1 = all_results["account_response"][model_key]["f1"]
        end_f1 = all_results["endpoint_response"][model_key]["f1"]
        macro = (acc_f1 + end_f1) / 2.0
        all_results.setdefault("summary", {})[model_key] = {
            "account_response_f1": float(acc_f1),
            "endpoint_response_f1": float(end_f1),
            "remediation_macro_f1": float(macro),
        }
        print(f"{model_key:<40} {acc_f1:>20.4f} {end_f1:>22.4f}")

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")

    best_end_f1 = max(
        all_results["endpoint_response"]["logistic_regression"]["f1"],
        all_results["endpoint_response"]["gradient_boosted_trees"]["f1"],
    )
    best_acc_f1 = max(
        all_results["account_response"]["logistic_regression"]["f1"],
        all_results["account_response"]["gradient_boosted_trees"]["f1"],
    )

    if best_end_f1 > tabnet["endpoint_response_f1"]:
        verdict_end = (
            "Classical ML BEATS TabNet on endpoint_response "
            f"({best_end_f1:.4f} vs {tabnet['endpoint_response_f1']:.4f})"
        )
    else:
        verdict_end = (
            "TabNet still wins on endpoint_response. "
            f"Classical ML best: {best_end_f1:.4f}"
        )

    if best_acc_f1 > tabnet["account_response_f1"]:
        verdict_acc = (
            "Classical ML BEATS TabNet on account_response "
            f"({best_acc_f1:.4f} vs {tabnet['account_response_f1']:.4f})"
        )
    else:
        verdict_acc = (
            "TabNet still wins on account_response. "
            f"Classical ML best: {best_acc_f1:.4f}"
        )

    print(f"\n  endpoint_response: {verdict_end}")
    print(f"  account_response:  {verdict_acc}")

    if best_end_f1 <= tabnet["endpoint_response_f1"]:
        print("\n  CONCLUSION: endpoint_response is non-viable at current data volumes.")
        print("  Neither deep learning nor classical ML can learn a reliable boundary.")
        print("  Recommended action: document as non-viable, require 300-500+ positives.")
    else:
        print("\n  CONCLUSION: Classical ML provides a better remediation model.")
        print("  Recommended action: replace TabNet remediation head with classical model.")

    save_hybrid_artifacts(fitted_models, scaler)

    out_path = REPORTS_DIR / "classical_remediation_comparison.json"
    with open(out_path, "w") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

"""
LightGBM training entry point.
"""

import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import TriageEvaluator
from models.lightgbm.predict import load_model, predict
from models.lightgbm.train import train_lightgbm_triage_model


def train_and_evaluate():
    print("=" * 80)
    print("LIGHTGBM TRIAGE MODEL - TRAINING PIPELINE")
    print("=" * 80)

    print("\nSTEP 1: TRAINING MODEL")
    model, train_metrics = train_lightgbm_triage_model(
        version="v1",
        model_dir="models/lightgbm",
        reports_dir="reports/metrics",
        verbose=True,
    )

    print("\nSTEP 2: LOADING DATA")
    X_val = pd.read_csv("data/processed/v1/X_val.csv")
    X_test = pd.read_csv("data/processed/v1/X_test.csv")
    y_val = pd.read_csv("data/processed/v1/y_val.csv").iloc[:, 0]
    y_test = pd.read_csv("data/processed/v1/y_test.csv").iloc[:, 0]

    print("\nSTEP 3: VALIDATION EVALUATION")
    loaded_model, _ = load_model(verbose=False)
    evaluator = TriageEvaluator(n_classes=3)

    val_pred, val_proba = predict(loaded_model, X_val, return_proba=True)
    val_metrics = evaluator.compute_metrics(y_true=y_val.values, y_pred=val_pred, y_proba=val_proba)
    print(evaluator.format_results(val_metrics))

    os.makedirs("reports/metrics", exist_ok=True)
    with open("reports/metrics/lightgbm_val_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(val_metrics, handle, indent=2)

    print("\nSTEP 4: TEST EVALUATION")
    test_pred, test_proba = predict(loaded_model, X_test, return_proba=True)
    test_metrics = evaluator.compute_metrics(y_true=y_test.values, y_pred=test_pred, y_proba=test_proba)
    print(evaluator.format_results(test_metrics))

    with open("reports/metrics/lightgbm_test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print("\nTRAINING SUMMARY")
    print(f"Training/Test Macro F1: {train_metrics['macro_f1']:.4f}")
    print(f"Validation Macro F1:    {val_metrics['macro_f1']:.4f}")
    print(f"Test Macro F1:          {test_metrics['macro_f1']:.4f}")

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate()

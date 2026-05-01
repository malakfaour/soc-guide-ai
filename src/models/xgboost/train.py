"""
XGBoost triage baseline training utilities.

Supports both the reusable triage artifact pipeline under ``models/xgboost/``
and the existing training helpers used by ``src/training/train.py``.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

from evaluation.metrics import TriageEvaluator, save_triage_metrics
from utils.versioning import load_dataset_by_version


def save_triage_comparison(
    xgboost_metrics: Dict[str, Any],
    reports_dir: str = "reports",
) -> Path:
    """Save an XGBoost vs TabNet triage comparison using existing saved metrics."""
    reports_root = PROJECT_ROOT / reports_dir
    comparisons_dir = reports_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    tabnet_metrics_path = reports_root / "metrics" / "triage_metrics.json"
    comparison_path = comparisons_dir / "tabnet_vs_xgboost_triage.json"

    comparison = {
        "xgboost": {
            "macro_f1": xgboost_metrics["macro_f1"],
            "overall_accuracy": xgboost_metrics["overall_accuracy"],
        }
    }
    if tabnet_metrics_path.exists():
        with open(tabnet_metrics_path, "r", encoding="utf-8") as handle:
            tabnet_metrics = json.load(handle)
        comparison["tabnet"] = {
            "macro_f1": tabnet_metrics["macro_f1"],
            "overall_accuracy": tabnet_metrics["overall_accuracy"],
        }
        comparison["winner_by_macro_f1"] = (
            "xgboost"
            if xgboost_metrics["macro_f1"] > tabnet_metrics["macro_f1"]
            else "tabnet"
        )
    else:
        comparison["tabnet"] = "missing triage_metrics.json"

    with open(comparison_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    return comparison_path


def load_xgboost_data(
    version: str = "v1",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the existing processed triage dataset for XGBoost."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version(
        version=version,
        verbose=verbose,
    )
    feature_names = X_train.columns.tolist()
    return (
        X_train.to_numpy(dtype=np.float32),
        X_val.to_numpy(dtype=np.float32),
        X_test.to_numpy(dtype=np.float32),
        y_train.to_numpy(dtype=np.int64),
        y_val.to_numpy(dtype=np.int64),
        y_test.to_numpy(dtype=np.int64),
        feature_names,
    )


def compute_sample_weights(y) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute balanced sample weights for class-imbalanced training."""
    y_array = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()
    classes = np.unique(y_array)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_array,
    )

    class_weight_dict = {
        int(label): float(weight)
        for label, weight in zip(classes, class_weights)
    }
    sample_weights = np.array([class_weight_dict[int(label)] for label in y_array])

    return sample_weights, class_weight_dict


def train_xgboost_model(
    X_train,
    y_train,
    X_val,
    y_val,
    sample_weights=None,
) -> XGBClassifier:
    """Train the legacy JSON-export XGBoost model used by the FastAPI app."""
    X_train_array = X_train.values if hasattr(X_train, "values") else X_train
    X_val_array = X_val.values if hasattr(X_val, "values") else X_val
    y_train_array = y_train.values.ravel() if hasattr(y_train, "values") else np.asarray(y_train).ravel()
    y_val_array = y_val.values.ravel() if hasattr(y_val, "values") else np.asarray(y_val).ravel()

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train_array)),
        eval_metric="mlogloss",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        early_stopping_rounds=50,
        tree_method="hist",
        device="cuda",
    )

    fit_kwargs = {
        "eval_set": [(X_val_array, y_val_array)],
        "verbose": 100,
    }
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    model.fit(X_train_array, y_train_array, **fit_kwargs)
    return model


def predict_with_threshold(model: XGBClassifier, X, high_threshold: float = 0.35):
    """Bias predictions toward the highest-severity class when confidence is sufficient."""
    X_array = X.values if hasattr(X, "values") else X
    probabilities = model.predict_proba(X_array)
    predictions = np.where(
        probabilities[:, 2] > high_threshold,
        2,
        np.argmax(probabilities, axis=1),
    )
    return predictions, probabilities


def train_xgboost_triage_model(
    version: str = "v1",
    model_dir: str = "models/xgboost",
    reports_dir: str = "reports/metrics",
    verbose: bool = True,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train the reusable triage baseline and save model, config, and metrics."""
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    ) = load_xgboost_data(version=version, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("Training XGBoost Triage Baseline")
        print("=" * 60)
        print(f"Train: {X_train.shape}")
        print(f"Val:   {X_val.shape}")
        print(f"Test:  {X_test.shape}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_proba_test = model.predict_proba(X_test)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    evaluator = TriageEvaluator(n_classes=len(np.unique(y_train)))
    metrics = evaluator.compute_metrics(y_test, y_pred_test, y_proba_test)

    model_output_dir = PROJECT_ROOT / model_dir
    reports_output_dir = PROJECT_ROOT / reports_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_output_dir / "triage_model.pkl"
    config_path = model_output_dir / "triage_model_config.json"
    metrics_path = save_triage_metrics(
        metrics,
        output_dir=str(reports_output_dir),
        filename="xgboost_triage_metrics.json",
    )
    comparison_path = save_triage_comparison(metrics, reports_dir="reports")

    joblib.dump(model, model_path)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": "triage_model",
                "model_type": "XGBClassifier",
                "dataset_version": version,
                "n_classes": int(len(np.unique(y_train))),
                "classes": sorted(int(value) for value in np.unique(y_train)),
                "feature_names": feature_names,
                "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
                "params": model.get_params(),
                "train_shape": list(X_train.shape),
                "val_shape": list(X_val.shape),
                "test_shape": list(X_test.shape),
                "metrics_file": str(metrics_path),
                "comparison_file": str(comparison_path),
            },
            handle,
            indent=2,
        )

    if verbose:
        print("\n[RESULTS]")
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Model:    {model_path}")
        print(f"  Config:   {config_path}")
        print(f"  Metrics:  {metrics_path}")
        print(f"  Compare:  {comparison_path}")

    return model, metrics


if __name__ == "__main__":
    _, metrics = train_xgboost_triage_model(verbose=True)
    print(
        json.dumps(
            {
                "macro_f1": metrics["macro_f1"],
                "overall_accuracy": metrics["overall_accuracy"],
            },
            indent=2,
        )
    )

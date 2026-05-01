"""
LightGBM triage baseline training.

Uses the existing processed v1 row-level triage dataset without introducing
new preprocessing or feature scaling.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np


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
    lightgbm_metrics: Dict[str, Any],
    reports_dir: str = "reports",
) -> Path:
    """Save a LightGBM vs TabNet triage comparison using existing saved metrics."""
    reports_root = PROJECT_ROOT / reports_dir
    comparisons_dir = reports_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    tabnet_metrics_path = reports_root / "metrics" / "triage_metrics.json"
    comparison_path = comparisons_dir / "tabnet_vs_lightgbm_triage.json"

    comparison = {
        "lightgbm": {
            "macro_f1": lightgbm_metrics["macro_f1"],
            "overall_accuracy": lightgbm_metrics["overall_accuracy"],
        }
    }
    if tabnet_metrics_path.exists():
        with open(tabnet_metrics_path, "r") as handle:
            tabnet_metrics = json.load(handle)
        comparison["tabnet"] = {
            "macro_f1": tabnet_metrics["macro_f1"],
            "overall_accuracy": tabnet_metrics["overall_accuracy"],
        }
        comparison["winner_by_macro_f1"] = (
            "lightgbm"
            if lightgbm_metrics["macro_f1"] > tabnet_metrics["macro_f1"]
            else "tabnet"
        )
    else:
        comparison["tabnet"] = "missing triage_metrics.json"

    with open(comparison_path, "w") as handle:
        json.dump(comparison, handle, indent=2)

    return comparison_path


def load_lightgbm_data(
    version: str = "v1",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the existing processed triage dataset for LightGBM."""
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


def train_lightgbm_triage_model(
    version: str = "v1",
    model_dir: str = "models/lightgbm",
    reports_dir: str = "reports/metrics",
    verbose: bool = True,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """Train the LightGBM triage baseline and save its artifacts."""
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    ) = load_lightgbm_data(version=version, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("Training LightGBM Triage Baseline")
        print("=" * 60)
        print(f"Train: {X_train.shape}")
        print(f"Val:   {X_val.shape}")
        print(f"Test:  {X_test.shape}")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        class_weight="balanced",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

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
        filename="lightgbm_triage_metrics.json",
    )
    comparison_path = save_triage_comparison(metrics, reports_dir="reports")

    joblib.dump(model, model_path)
    with open(config_path, "w") as handle:
        json.dump(
            {
                "model_name": "triage_model",
                "model_type": "LGBMClassifier",
                "dataset_version": version,
                "n_classes": int(len(np.unique(y_train))),
                "classes": sorted(int(value) for value in np.unique(y_train)),
                "feature_names": feature_names,
                "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
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
    _, metrics = train_lightgbm_triage_model(verbose=True)
    print(json.dumps(
        {
            "macro_f1": metrics["macro_f1"],
            "overall_accuracy": metrics["overall_accuracy"],
        },
        indent=2,
    ))

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception as e:
    print("[ERROR] Failed to import TabNetClassifier from pytorch_tabnet.tab_model")
    print(f"  Root cause: {type(e).__name__}: {e}")
    print("  Verify that both pytorch-tabnet and torch import cleanly.")
    raise

from src.models.tabnet.utils import (
    class_weights_to_sample_weights,
    compute_tabnet_class_weights,
    save_tabnet_model,
    scale_tabnet_features,
)
from src.training.train_tabnet import load_tabnet_data


DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "n_independent": 2,
    "n_shared": 2,
    "lambda_sparse": 1e-3,
    "momentum": 0.02,
    "epsilon": 1e-15,
    "seed": 42,
    "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
    "max_epochs": 200,
    "patience": 20,
    "batch_size": 256,
    "virtual_batch_size": 128,
}


def train_tabnet_triage_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[TabNetClassifier, Dict[str, Any]]:
    """Train the triage TabNet model and return predictions plus artifacts."""

    if X_train.shape[0] == 0:
        raise ValueError("Training data is empty")
    if X_val.shape[0] == 0:
        raise ValueError("Validation data is empty")
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError("Training data/labels size mismatch")
    if y_val.shape[0] != X_val.shape[0]:
        raise ValueError("Validation data/labels size mismatch")

    if verbose:
        print("=" * 60)
        print("TabNet Triage Model Training")
        print("=" * 60)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_tabnet_features(
        X_train,
        X_val,
        X_test,
        verbose=verbose,
    )

    class_weights = compute_tabnet_class_weights(y_train, verbose=verbose)
    sample_weights = class_weights_to_sample_weights(class_weights, y_train)

    tabnet_params = dict(DEFAULT_HYPERPARAMS)
    if model_params:
        tabnet_params.update(model_params)

    fit_max_epochs = int(tabnet_params.pop("max_epochs"))
    fit_patience = int(tabnet_params.pop("patience"))
    fit_batch_size = int(tabnet_params.pop("batch_size"))
    fit_virtual_batch_size = int(tabnet_params.pop("virtual_batch_size"))

    tabnet_params["optimizer_fn"] = torch.optim.Adam
    tabnet_params["verbose"] = 1 if verbose else 0
    tabnet_params["device_name"] = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"  [OK] Device: {tabnet_params['device_name']}")

    model = TabNetClassifier(**tabnet_params)
    model.fit(
        X_train=X_train_scaled,
        y_train=y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric=["balanced_accuracy"],
        max_epochs=fit_max_epochs,
        patience=fit_patience,
        batch_size=fit_batch_size,
        virtual_batch_size=fit_virtual_batch_size,
        num_workers=0,
        weights=sample_weights,
    )

    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_train = model.predict_proba(X_train_scaled)
    y_proba_val = model.predict_proba(X_val_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)

    results = {
        "model": model,
        "scaler": scaler,
        "class_weights": class_weights,
        "predictions": {
            "train": y_pred_train,
            "val": y_pred_val,
            "test": y_pred_test,
        },
        "probabilities": {
            "train": y_proba_train,
            "val": y_proba_val,
            "test": y_proba_test,
        },
        "accuracy": {
            "train": float(np.mean(y_pred_train == y_train)),
            "val": float(np.mean(y_pred_val == y_val)),
            "test": float(np.mean(y_pred_test == y_test)),
        },
        "metrics": {
            "n_features": int(X_train.shape[1]),
            "n_classes": int(len(np.unique(y_train))),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
        },
        "hyperparams": {
            **tabnet_params,
            "max_epochs": fit_max_epochs,
            "patience": fit_patience,
            "batch_size": fit_batch_size,
            "virtual_batch_size": fit_virtual_batch_size,
        },
    }

    if verbose:
        print(f"  [OK] Best epoch: {getattr(model, 'best_epoch', 'n/a')}")
        print(f"  [OK] Best validation score: {getattr(model, 'best_cost', 'n/a')}")
        print(f"  [OK] Test accuracy: {results['accuracy']['test']:.4f}")

    return model, results


def train(verbose: bool = True) -> Tuple[TabNetClassifier, Dict[str, Any]]:
    """Convenience entrypoint for local TabNet training."""

    X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()
    model, results = train_tabnet_triage_model(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        verbose=verbose,
    )

    save_tabnet_model(
        model=model,
        scaler=results["scaler"],
        class_weights=results["class_weights"],
        hyperparams=results["hyperparams"],
        model_dir="models/tabnet",
        model_name="triage_model",
        verbose=verbose,
    )
    return model, results


if __name__ == "__main__":
    train(verbose=True)

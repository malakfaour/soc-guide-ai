"""
Scaling and artifact utilities for the TabNet triage model.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None


class TabNetScaler:
    """Reusable QuantileTransformer wrapper for TabNet features."""

    def __init__(self, n_quantiles: int = 1000, output_distribution: str = "normal"):
        self.scaler = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=100000,
            random_state=42,
        )
        self.is_fitted = False

    def fit_transform_train(self, X_train: np.ndarray) -> np.ndarray:
        if X_train.size == 0:
            raise ValueError("X_train is empty")
        if np.isnan(X_train).any():
            raise ValueError("X_train contains NaN values")

        X_train = X_train.astype(np.float32)
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        self.is_fitted = True

        print(f"  [OK] Scaler fitted on {X_train.shape[0]} training samples")
        print(f"  [OK] Training data scaled shape: {X_train_scaled.shape}")

        return X_train_scaled

    def transform(self, X: np.ndarray, split_name: str = "Data") -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transformation")
        if X.size == 0:
            raise ValueError(f"{split_name} is empty")
        if np.isnan(X).any():
            raise ValueError(f"{split_name} contains NaN values")

        X = X.astype(np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        print(f"  [OK] {split_name} data scaled: {X.shape} -> {X_scaled.shape}")
        return X_scaled

    def __call__(self, X: np.ndarray, split_name: str = "Data") -> np.ndarray:
        return self.transform(X, split_name)


def scale_tabnet_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TabNetScaler]:
    """Fit a scaler on train and transform train/val/test without leakage."""

    n_features = X_train.shape[1]
    if X_val.shape[1] != n_features:
        raise ValueError(
            f"Feature mismatch: X_train has {n_features} features but X_val has {X_val.shape[1]}"
        )
    if X_test.shape[1] != n_features:
        raise ValueError(
            f"Feature mismatch: X_train has {n_features} features but X_test has {X_test.shape[1]}"
        )

    scaler = TabNetScaler(n_quantiles=min(1000, len(X_train)), output_distribution="normal")

    if verbose:
        print("=" * 60)
        print("Scaling TabNet Features")
        print("=" * 60)

    X_train_scaled = scaler.fit_transform_train(X_train)
    X_val_scaled = scaler.transform(X_val, split_name="Validation")
    X_test_scaled = scaler.transform(X_test, split_name="Test")

    if verbose:
        print("  [OK] Scaler fitted only on training data (no data leakage)")
        print("  [OK] All splits have same feature dimension")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def compute_tabnet_class_weights(
    y: np.ndarray,
    verbose: bool = True,
) -> Dict[int, float]:
    """Compute inverse-frequency class weights for TabNet."""

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"compute_tabnet_class_weights requires at least 2 classes, found: {classes}"
        )

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    if verbose:
        print("[CLASS WEIGHTS]")
        for class_label, weight in sorted(class_weights.items()):
            print(f"  [OK] Class {class_label}: {weight:.4f}")

    return class_weights


def class_weights_to_sample_weights(class_weights: Dict[int, float], y: np.ndarray) -> np.ndarray:
    """Expand class weights to per-sample weights."""
    return np.array([class_weights[int(label)] for label in y], dtype=np.float32)


def save_tabnet_model(
    model: Any,
    scaler: TabNetScaler,
    class_weights: Dict[int, float],
    hyperparams: Optional[Dict[str, Any]] = None,
    model_dir: str = "models/tabnet",
    model_name: str = "triage_model",
    verbose: bool = True,
) -> Dict[str, str]:
    """Save the TabNet model, fitted scaler, and config."""

    if model is None:
        raise ValueError("Model is None")
    if scaler is None or not scaler.is_fitted:
        raise ValueError("Scaler is None or not fitted")
    if not class_weights:
        raise ValueError("Class weights are empty")

    os.makedirs(model_dir, exist_ok=True)

    model_base_path = os.path.join(model_dir, model_name)
    model_path = f"{model_base_path}.zip"
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    config_path = os.path.join(model_dir, f"{model_name}_config.json")

    model.save_model(model_base_path)
    joblib.dump(scaler.scaler, scaler_path)

    config: Dict[str, Any] = {
        "model_name": model_name,
        "model_type": "TabNetClassifier",
        "class_weights": {str(int(k)): float(v) for k, v in class_weights.items()},
        "n_classes": len(class_weights),
        "classes": sorted(int(k) for k in class_weights.keys()),
        "scaler_type": "QuantileTransformer",
        "scaler_config": {
            "n_quantiles": scaler.scaler.n_quantiles,
            "output_distribution": scaler.scaler.output_distribution,
        },
    }
    if hyperparams:
        config["hyperparams"] = hyperparams

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    if verbose:
        print(f"  [OK] Model saved: {model_path}")
        print(f"  [OK] Scaler saved: {scaler_path}")
        print(f"  [OK] Config saved: {config_path}")

    return {
        "model": model_path,
        "scaler": scaler_path,
        "config": config_path,
    }


def load_tabnet_model(
    model_dir: str = "models/tabnet",
    model_name: str = "triage_model",
    verbose: bool = True,
) -> Tuple[Any, TabNetScaler, Dict[str, Any]]:
    """Load a saved TabNet model, scaler wrapper, and config."""

    model_path = os.path.join(model_dir, f"{model_name}.zip")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    config_path = os.path.join(model_dir, f"{model_name}_config.json")

    for path in (model_path, scaler_path, config_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    scaler_obj = joblib.load(scaler_path)
    scaler = TabNetScaler(
        n_quantiles=int(config.get("scaler_config", {}).get("n_quantiles", 1000)),
        output_distribution=config.get("scaler_config", {}).get("output_distribution", "normal"),
    )
    scaler.scaler = scaler_obj
    scaler.is_fitted = True

    if TabNetClassifier is None:
        raise OSError("Failed to load model: pytorch_tabnet is required to load a saved TabNetClassifier")

    model = TabNetClassifier()
    model.load_model(model_path)

    class_weights = {
        int(k): float(v) for k, v in config.get("class_weights", {}).items()
    }
    merged_config = {**config, "class_weights": class_weights}

    if verbose:
        print(f"  [OK] Model loaded: {model_path}")
        print(f"  [OK] Scaler loaded: {scaler_path}")
        print(f"  [OK] Config loaded: {config_path}")

    return model, scaler, merged_config

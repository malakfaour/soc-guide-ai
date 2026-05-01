"""
XGBoost triage baseline inference.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_model(
    model_dir: str = "models/xgboost",
    model_name: str = "triage_model",
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a saved XGBoost model, preferring the packaged triage artifact."""
    model_path = PROJECT_ROOT / model_dir / f"{model_name}.pkl"
    config_path = PROJECT_ROOT / model_dir / f"{model_name}_config.json"

    if model_path.exists() and config_path.exists():
        model = joblib.load(model_path)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        if verbose:
            print("=" * 60)
            print("Loading XGBoost Triage Model")
            print("=" * 60)
            print(f"  Model:   {model_path}")
            print(f"  Config:  {config_path}")
            print(f"  Classes: {config['classes']}")

        return model, config

    legacy_model_path = PROJECT_ROOT / "models" / "xgboost_model.json"
    if not legacy_model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found. Checked {model_path} and {legacy_model_path}."
        )

    model = XGBClassifier()
    model.load_model(str(legacy_model_path))
    config = {
        "model_name": "xgboost_model",
        "model_type": "XGBClassifier",
        "classes": [0, 1, 2],
        "artifact_format": "json",
    }

    if verbose:
        print("=" * 60)
        print("Loading Legacy XGBoost Model")
        print("=" * 60)
        print(f"  Model:   {legacy_model_path}")

    return model, config


def predict(
    model: Any,
    data: Union[np.ndarray, pd.DataFrame],
    return_proba: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generate triage predictions using the saved XGBoost model."""
    if isinstance(data, pd.DataFrame):
        array = data.to_numpy(dtype=np.float32)
    else:
        array = np.asarray(data, dtype=np.float32)

    if array.size == 0:
        raise ValueError("Input data is empty")
    if np.isnan(array).any():
        raise ValueError("Input data contains NaN values")

    probabilities = model.predict_proba(array)
    predictions = np.argmax(probabilities, axis=1)

    if return_proba:
        return predictions, probabilities
    return predictions


if __name__ == "__main__":
    model, config = load_model(verbose=True)
    sample = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "v1" / "X_test.csv").head(5)
    preds, probs = predict(model, sample, return_proba=True)
    print(
        json.dumps(
            {
                "predictions": preds.tolist(),
                "probabilities_shape": list(probs.shape),
                "classes": config["classes"],
            },
            indent=2,
        )
    )

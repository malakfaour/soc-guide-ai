from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd

from src.models.tabnet.utils import load_tabnet_model


def load_model(
    model_dir: str = "models/tabnet",
    model_name: str = "triage_model",
    verbose: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the saved TabNet triage model, scaler wrapper, and config."""
    return load_tabnet_model(
        model_dir=model_dir,
        model_name=model_name,
        verbose=verbose,
    )


def predict(
    model: Any,
    scaler: Any,
    data: Union[np.ndarray, pd.DataFrame],
    return_proba: bool = True,
    verbose: bool = False,
):
    """Run predictions on already prepared feature data."""

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    array = np.asarray(data, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Input data is empty")
    if np.isnan(array).any():
        raise ValueError("Input data contains NaN values")

    data_scaled = scaler.transform(array, split_name="Inference")
    probabilities = model.predict_proba(data_scaled)
    predictions = np.argmax(probabilities, axis=1)

    if verbose:
        print(f"  [OK] Input shape: {array.shape}")
        print(f"  [OK] Scaled shape: {data_scaled.shape}")
        print(f"  [OK] Predictions shape: {predictions.shape}")
        print(f"  [OK] Probabilities shape: {probabilities.shape}")

    if return_proba:
        return predictions, probabilities
    return predictions


def predict_csv(
    csv_path: str,
    model_dir: str = "models/tabnet",
    model_name: str = "triage_model",
    output_path: str = "predictions.csv",
) -> pd.DataFrame:
    """Load a CSV file, run TabNet predictions, and save them."""

    model, scaler, _ = load_model(
        model_dir=model_dir,
        model_name=model_name,
        verbose=True,
    )
    df = pd.read_csv(csv_path)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    predictions, probabilities = predict(
        model=model,
        scaler=scaler,
        data=df,
        return_proba=True,
        verbose=True,
    )

    output = pd.DataFrame({
        "prediction": predictions,
        "confidence": probabilities.max(axis=1),
    })
    output.to_csv(output_path, index=False)
    print(f"[OK] Saved predictions to {output_path}")
    return output


if __name__ == "__main__":
    predict_csv("data/processed/v1/X_test.csv")

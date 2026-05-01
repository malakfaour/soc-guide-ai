"""
TabNet prediction module for SOC alert triage model.

Loads a trained TabNet model with its fitted scaler and configuration,
applies feature scaling to input data, and generates predictions with
class probabilities.
"""

import numpy as np
import pandas as pd
import sys
import os
import joblib
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Union

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception as e:
    print("[ERROR] Failed to import TabNetClassifier from pytorch_tabnet.tab_model")
    print(f"  Root cause: {type(e).__name__}: {e}")
    sys.exit(1)


def load_model(
    model_dir: str = "models/tabnet",
    model_name: str = "triage_model",
    verbose: bool = True
) -> Tuple[TabNetClassifier, Any, Dict[str, Any]]:
    """
    Load trained TabNet model with scaler and configuration.

    Parameters
    ----------
    model_dir : str, default="models/tabnet"
        Directory containing saved model artifacts
    model_name : str, default="triage_model"
        Base name of saved model files
    verbose : bool, default=True
        Print loading information

    Returns
    -------
    Tuple[TabNetClassifier, scaler, config]
        Loaded model, QuantileTransformer scaler, and configuration dictionary

    Raises
    ------
    FileNotFoundError
        If required artifact files are missing
    OSError
        If loading fails
    """

    if verbose:
        print("=" * 60)
        print("Loading TabNet Prediction Model")
        print("=" * 60)

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    config_path = os.path.join(model_dir, f"{model_name}_config.json")

    # Check file existence
    if verbose:
        print(f"\n[LOAD] Checking artifact files...")

    for path in [model_path, scaler_path, config_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")
        if verbose:
            print(f"  ✓ Found: {os.path.basename(path)}")

    # Load configuration
    if verbose:
        print(f"\n[LOAD] Loading configuration...")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if verbose:
        print(f"  ✓ Model type: {config['model_type']}")
        print(f"  ✓ Classes: {config['classes']}")
        print(f"  ✓ Number of classes: {config['n_classes']}")

    # Load scaler
    if verbose:
        print(f"\n[LOAD] Loading feature scaler...")

    try:
        scaler = joblib.load(scaler_path)
        if verbose:
            print(f"  ✓ Scaler loaded: QuantileTransformer")
            print(f"    - n_quantiles: {scaler.n_quantiles}")
            print(f"    - output_distribution: {scaler.output_distribution}")
    except Exception as e:
        raise OSError(f"Failed to load scaler: {str(e)}")

    # Load model
    if verbose:
        print(f"\n[LOAD] Loading TabNet model...")

    try:
        # Try loading with TabNetClassifier.load_model() if available
        try:
            model = TabNetClassifier()
            model.load_model(model_path)
        except Exception:
            # Fall back to joblib
            model = joblib.load(model_path)

        if verbose:
            print(f"  ✓ Model loaded successfully")
    except Exception as e:
        raise OSError(f"Failed to load model: {str(e)}")

    if verbose:
        print("\n" + "=" * 60)
        print("✓ Model loaded successfully!")
        print("=" * 60)

    return model, scaler, config


def predict(
    model: TabNetClassifier,
    scaler: Any,
    data: Union[np.ndarray, pd.DataFrame],
    return_proba: bool = True,
    verbose: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Generate predictions on input data using loaded TabNet model.

    Parameters
    ----------
    model : TabNetClassifier
        Loaded TabNet model
    scaler : QuantileTransformer
        Fitted feature scaler from training
    data : np.ndarray or pd.DataFrame
        Input features for prediction (n_samples, n_features)
    return_proba : bool, default=True
        If True, return both predictions and probabilities.
        If False, return only class predictions.
    verbose : bool, default=False
        Print prediction information

    Returns
    -------
    If return_proba=True:
        Tuple[np.ndarray, np.ndarray]
            (class_predictions, class_probabilities)
            - class_predictions: shape (n_samples,), integer class indices
            - class_probabilities: shape (n_samples, n_classes), probability for each class
    If return_proba=False:
        np.ndarray
            Class predictions only, shape (n_samples,)

    Raises
    ------
    ValueError
        If input data is invalid or has wrong dimensions
    """

    if verbose:
        print("\n[PREDICT] Input validation...")

    # Convert DataFrame to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Validate input
    if data.size == 0:
        raise ValueError("Input data is empty")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")

    if verbose:
        print(f"  ✓ Input shape: {data.shape}")

    # Scale features using fitted scaler
    if verbose:
        print(f"\n[PREDICT] Applying feature scaling...")

    try:
        data_scaled = scaler.transform(data)
        if verbose:
            print(f"  ✓ Data scaled: {data.shape} → {data_scaled.shape}")
    except Exception as e:
        raise ValueError(f"Failed to scale data: {str(e)}")

    # Make predictions
    if verbose:
        print(f"\n[PREDICT] Generating predictions...")

    try:
        # Get probability predictions
        proba = model.predict_proba(data_scaled)

        # Get class predictions (argmax of probabilities)
        pred = np.argmax(proba, axis=1)

        if verbose:
            print(f"  ✓ Predictions: {pred.shape}")
            print(f"  ✓ Probabilities: {proba.shape}")
            print(f"    - Unique classes predicted: {np.unique(pred)}")
            print(f"    - Average confidence: {proba.max(axis=1).mean():.4f}")

    except Exception as e:
        raise ValueError(f"Failed to make predictions: {str(e)}")

    if return_proba:
        return pred, proba
    else:
        return pred


if __name__ == "__main__":
    """
    Example usage: Load model and make predictions on test data.
    """

    print("\n" + "=" * 60)
    print("TabNet Prediction Example")
    print("=" * 60)

    # Load model
    try:
        model, scaler, config = load_model(
            model_dir="models/tabnet",
            model_name="triage_model",
            verbose=True
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Load example test data
    try:
        print("\n[DATA] Loading example test data...")
        sample_data = pd.read_csv("data/processed/v1/X_test.csv").iloc[:10]
        print(f"  ✓ Loaded {sample_data.shape[0]} samples with {sample_data.shape[1]} features")
    except Exception as e:
        print(f"\n[ERROR] Failed to load test data: {e}")
        print("[INFO] Using synthetic data instead...")
        # Generate synthetic data if test data not available
        sample_data = np.random.randn(5, 20)  # Adjust n_features based on your model

    # Make predictions
    try:
        preds, probs = predict(
            model=model,
            scaler=scaler,
            data=sample_data,
            return_proba=True,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("Prediction Results")
        print("=" * 60)
        print(f"\nClass Predictions:")
        print(preds)
        print(f"\nClass Probabilities (first 5 samples):")
        print(probs[:5])

        # Show prediction distribution
        print(f"\nPrediction Distribution:")
        unique, counts = np.unique(preds, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} predictions ({count/len(preds)*100:.1f}%)")

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Prediction complete!")
    print("=" * 60)

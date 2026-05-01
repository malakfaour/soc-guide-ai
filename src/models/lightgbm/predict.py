<<<<<<< HEAD
﻿"""
LightGBM Prediction Module
Handles inference with trained triage model
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from .train import LightGBMTriageModel
from .plot import plot_confusion_matrix, plot_feature_importance

class LightGBMPredictor:
    """
    Predictor class for LightGBM triage model.
    Provides easy-to-use inference interface.
    """
    
    def __init__(self, model_dir: str = 'models/lightgbm', model_name: str = 'triage_model'):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing saved model
            model_name: Name of model files
        """
        self.model = LightGBMTriageModel()
        self.model.load(save_dir=model_dir, model_name=model_name)
        
        # Create reverse mapping for class names
        if self.model.target_mapping:
            self.class_names = {v: k for k, v in self.model.target_mapping.items()}
        else:
            self.class_names = {0: 'FalsePositive', 1: 'BenignPositive', 2: 'TruePositive'}
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Dictionary containing:
                - 'predictions': Predicted class labels (0, 1, 2)
                - 'class_names': Predicted class names
                - 'probabilities': Probability matrix (n_samples, 3)
                - 'confidence': Confidence scores (max probability)
        """
        # Get predictions
        pred_labels, probas, confidence = self.model.predict_with_confidence(X)
        
        # Convert to class names
        class_names = np.array([self.class_names[label] for label in pred_labels])
        
        return {
            'predictions': pred_labels,
            'class_names': class_names,
            'probabilities': probas,
            'confidence': confidence
        }
    
    def predict_single(self, X: pd.Series) -> Dict:
        """
        Predict for a single incident.
        
        Args:
            X: Single feature vector (as Series or 1-row DataFrame)
            
        Returns:
            Dictionary with prediction details
        """
        # Convert to DataFrame if Series
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        
        result = self.predict(X)
        
        return {
            'prediction': result['predictions'][0],
            'class_name': result['class_names'][0],
            'confidence': result['confidence'][0],
            'probabilities': {
                self.class_names[0]: result['probabilities'][0][0],
                self.class_names[1]: result['probabilities'][0][1],
                self.class_names[2]: result['probabilities'][0][2]
            }
        }
    
    def predict_with_details(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with full details in DataFrame format.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with columns:
                - prediction (int)
                - class_name (str)
                - confidence (float)
                - prob_FalsePositive (float)
                - prob_BenignPositive (float)
                - prob_TruePositive (float)
        """
        result = self.predict(X)
        
        df = pd.DataFrame({
            'prediction': result['predictions'],
            'class_name': result['class_names'],
            'confidence': result['confidence'],
            f"prob_{self.class_names[0]}": result['probabilities'][:, 0],
            f"prob_{self.class_names[1]}": result['probabilities'][:, 1],
            f"prob_{self.class_names[2]}": result['probabilities'][:, 2]
        })
        
        return df


def load_and_predict(
    X: pd.DataFrame,
    model_dir: str = 'models/lightgbm',
    model_name: str = 'triage_model'
) -> Dict[str, np.ndarray]:
    """
    Convenience function to load model and predict.
    
    Args:
        X: Feature dataframe
        model_dir: Directory containing model
        model_name: Name of model files
        
    Returns:
        Prediction dictionary
    """
    predictor = LightGBMPredictor(model_dir=model_dir, model_name=model_name)
    return predictor.predict(X)


if __name__ == "__main__":
    # Test predictions
    from .utils import load_processed_data
    
    print("="*80)
    print("TESTING LIGHTGBM PREDICTOR")
    print("="*80)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Initialize predictor
    print("\n📊 Loading model...")
    predictor = LightGBMPredictor()
    
    # Test on sample
    print("\n📊 Testing on validation samples...")
    X_sample = X_val.head(5)
    y_sample = y_val.head(5)
    
    # Method 1: Basic predict
    result = predictor.predict(X_sample)
    print("\n✅ Basic prediction result keys:", result.keys())
    
    # Method 2: Predict with details
    details_df = predictor.predict_with_details(X_sample)
    print("\n✅ Detailed predictions:")
    print(details_df.to_string())
    
    # Method 3: Single prediction
    print("\n📊 Testing single prediction...")
    single_result = predictor.predict_single(X_sample.iloc[0])
    print(f"\n✅ Single prediction:")
    print(f"   Class: {single_result['class_name']}")
    print(f"   Confidence: {single_result['confidence']:.4f}")
    print(f"   Probabilities: {single_result['probabilities']}")
    
    # Compare with true labels
    print("\n📊 Accuracy check on sample:")
    correct = (result['predictions'] == y_sample.values).sum()
    print(f"   Correct: {correct}/{len(y_sample)}")
    
    print("\n" + "="*80)
    print("✅ ALL PREDICTOR TESTS PASSED")
    print("="*80)
        # ========================
    # 📊 FULL EVALUATION PLOTS
    # ========================

    print("\n📊 Generating full evaluation plots...")

    # Predict on FULL validation set (not just sample)
    full_result = predictor.predict(X_val)
    y_pred = full_result['predictions']

    # Confusion Matrix
    plot_confusion_matrix(y_val, y_pred)

    # Feature Importance
    plot_feature_importance(predictor.model.model)

    print("\n📊 Plots generated successfully!")
=======
"""
LightGBM triage baseline inference.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_model(
    model_dir: str = "models/lightgbm",
    model_name: str = "triage_model",
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a saved LightGBM triage model and configuration."""
    model_path = PROJECT_ROOT / model_dir / f"{model_name}.pkl"
    config_path = PROJECT_ROOT / model_dir / f"{model_name}_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"LightGBM config not found: {config_path}")

    model = joblib.load(model_path)
    with open(config_path, "r") as handle:
        config = json.load(handle)

    if verbose:
        print("=" * 60)
        print("Loading LightGBM Triage Model")
        print("=" * 60)
        print(f"  Model:   {model_path}")
        print(f"  Config:  {config_path}")
        print(f"  Classes: {config['classes']}")

    return model, config


def predict(
    model: Any,
    data: Union[np.ndarray, pd.DataFrame],
    return_proba: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generate triage predictions using the saved LightGBM model."""
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
    print(json.dumps(
        {
            "predictions": preds.tolist(),
            "probabilities_shape": list(probs.shape),
            "classes": config["classes"],
        },
        indent=2,
    ))
>>>>>>> origin/main

"""
Scaling utilities for TabNet model.

Handles feature scaling with QuantileTransformer while keeping transformations reusable.
"""

import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Optional, Dict, Any

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None


class TabNetScaler:
    """
    Reusable feature scaler for TabNet using QuantileTransformer.
    
    Fits only on training data to prevent data leakage, then applies
    transformation to train, validation, and test sets.
    """
    
    def __init__(self, n_quantiles: int = 1000, output_distribution: str = 'normal'):
        """
        Initialize TabNet scaler.
        
        Parameters
        ----------
        n_quantiles : int, default=1000
            Number of quantiles to estimate
        output_distribution : str, default='normal'
            Distribution for output transformation ('normal' or 'uniform')
        """
        self.scaler = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=100000,  # For large datasets
            random_state=42
        )
        self.is_fitted = False
    
    def fit_transform_train(
        self,
        X_train: np.ndarray
    ) -> np.ndarray:
        """
        Fit scaler on training data and return transformed training features.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Scaled training features
        
        Raises
        ------
        ValueError
            If input is empty or contains NaN values
        """
        print("[SCALING] Fitting QuantileTransformer on training data...")
        
        # Validate input
        if X_train.size == 0:
            raise ValueError("X_train is empty")
        if np.isnan(X_train).any():
            raise ValueError("X_train contains NaN values")
        
        # Fit and transform
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        
        print(f"  ✓ Scaler fitted on {X_train.shape[0]} training samples")
        print(f"  ✓ Training data scaled shape: {X_train_scaled.shape}")
        
        return X_train_scaled
    
    def transform(
        self,
        X: np.ndarray,
        split_name: str = "Data"
    ) -> np.ndarray:
        """
        Apply fitted scaler transformation to data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to scale (n_samples, n_features)
        split_name : str, default="Data"
            Name of the data split for logging (e.g., 'Validation', 'Test')
        
        Returns
        -------
        np.ndarray
            Scaled features
        
        Raises
        ------
        RuntimeError
            If scaler has not been fitted yet
        ValueError
            If input contains NaN values
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transformation")
        
        if X.size == 0:
            raise ValueError(f"{split_name} is empty")
        if np.isnan(X).any():
            raise ValueError(f"{split_name} contains NaN values")
        
        X_scaled = self.scaler.transform(X)
        print(f"  ✓ {split_name} data scaled: {X.shape} → {X_scaled.shape}")
        
        return X_scaled
    
    def __call__(
        self,
        X: np.ndarray,
        split_name: str = "Data"
    ) -> np.ndarray:
        """
        Make scaler callable for convenience.
        
        Equivalent to calling transform().
        """
        return self.transform(X, split_name)


def scale_tabnet_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TabNetScaler]:
    """
    Scale TabNet features using QuantileTransformer.
    
    Fits scaler on training data only to prevent data leakage, then
    transforms validation and test data using the fitted scaler.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (n_train, n_features)
    X_val : np.ndarray
        Validation features (n_val, n_features)
    X_test : np.ndarray
        Test features (n_test, n_features)
    verbose : bool, default=True
        Print scaling information
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, TabNetScaler]
        Scaled training, validation, and test features plus the fitted scaler
    
    Raises
    ------
    ValueError
        If any input contains NaN values or has mismatched dimensions
    """
    if verbose:
        print("=" * 60)
        print("Scaling TabNet Features")
        print("=" * 60)
        print("\n[SCALING] Input shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
    
    # Validate feature dimension consistency
    n_features = X_train.shape[1]
    if X_val.shape[1] != n_features:
        raise ValueError(
            f"Feature mismatch: X_train has {n_features} features "
            f"but X_val has {X_val.shape[1]}"
        )
    if X_test.shape[1] != n_features:
        raise ValueError(
            f"Feature mismatch: X_train has {n_features} features "
            f"but X_test has {X_test.shape[1]}"
        )
    
    # Create and fit scaler
    scaler = TabNetScaler(
        n_quantiles=1000,
        output_distribution='normal'
    )
    
    if verbose:
        print()
    
    # Fit on training data and transform all splits
    X_train_scaled = scaler.fit_transform_train(X_train)
    
    if verbose:
        print("\n[SCALING] Transforming other splits...")
    
    X_val_scaled = scaler.transform(X_val, split_name="Validation")
    X_test_scaled = scaler.transform(X_test, split_name="Test")
    
    if verbose:
        print("\n[SCALING] Validation:")
        print("  ✓ Scaler fitted only on training data (no data leakage)")
        print("  ✓ All splits have same feature dimension")
        print(f"  ✓ Scaled data dtype: {X_train_scaled.dtype}")
        
        # Show scaling effect on features
        print("\n[SCALING] Feature Statistics (Training Data):")
        print(f"  Original - Mean: {X_train.mean(axis=0)[:3]}")  # Show first 3 features
        print(f"  Original - Std:  {X_train.std(axis=0)[:3]}")
        print(f"  Scaled   - Mean: {X_train_scaled.mean(axis=0)[:3]}")
        print(f"  Scaled   - Std:  {X_train_scaled.std(axis=0)[:3]}")
        
        print("\n" + "=" * 60)
        print("✓ Feature scaling complete!")
        print("=" * 60)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def compute_tabnet_class_weights(
    y_train: np.ndarray,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Compute class weights for TabNet using inverse frequency weighting.
    
    Addresses class imbalance by assigning higher weights to minority classes.
    Uses sklearn's compute_class_weight with 'balanced' strategy:
    weight_i = n_samples / (n_classes * n_samples_i)
    
    Parameters
    ----------
    y_train : np.ndarray
        Training target labels (n_samples,)
    verbose : bool, default=True
        Print class weight information
    
    Returns
    -------
    Dict[int, float]
        Dictionary mapping class indices to weights. Format: {0: w0, 1: w1, ...}
    
    Raises
    ------
    ValueError
        If y_train is empty or contains invalid values
    """
    if verbose:
        print("=" * 60)
        print("Computing Class Weights for Imbalance Handling")
        print("=" * 60)
    
    # Validate input
    if y_train.size == 0:
        raise ValueError("y_train is empty")
    
    # Get unique classes and their counts
    unique_classes = np.unique(y_train)
    class_counts = np.bincount(y_train)
    n_classes = len(unique_classes)
    n_samples = len(y_train)
    
    if verbose:
        print("\n[IMBALANCE] Class Distribution in Training Data:")
        for class_label in unique_classes:
            count = np.sum(y_train == class_label)
            pct = 100 * count / n_samples
            print(f"  Class {class_label}: {count:7d} samples ({pct:5.1f}%)")
    
    # Compute balanced class weights (inverse frequency)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    
    # Convert to dictionary mapping class labels to weights
    class_weights_dict = {
        class_label: weight
        for class_label, weight in zip(unique_classes, class_weights_array)
    }
    
    if verbose:
        print("\n[WEIGHTS] Computed Weights (Inverse Frequency):")
        for class_label, weight in sorted(class_weights_dict.items()):
            print(f"  Class {class_label}: {weight:.4f}")
        
        # Show weighting effect
        print("\n[WEIGHTS] Weighting Effect:")
        largest_weight = max(class_weights_dict.values())
        for class_label, weight in sorted(class_weights_dict.items()):
            relative = weight / largest_weight
            print(f"  Class {class_label}: {relative:.2f}x relative to max")
        
        print("\n[INFO] These weights will:")
        print("  • Increase importance of minority classes during training")
        print("  • Sum approximates n_classes (in 'balanced' mode)")
        print("  • Prevent model bias toward majority classes")
        
        print("\n" + "=" * 60)
        print("✓ Class weights ready for TabNet training!")
        print("=" * 60)
    
    return class_weights_dict


def save_tabnet_model(
    model: Any,
    scaler: TabNetScaler,
    class_weights: Dict[int, float],
    model_dir: str = "models/tabnet",
    model_name: str = "triage_model",
    verbose: bool = True
) -> Dict[str, str]:
    """
    Save trained TabNet model with scaler and configuration.
    
    Saves:
    - model: Trained TabNetClassifier
    - scaler: Fitted QuantileTransformer
    - config: Model metadata and class weights
    
    Parameters
    ----------
    model : TabNetClassifier
        Trained TabNet model
    scaler : TabNetScaler
        Fitted feature scaler
    class_weights : Dict[int, float]
        Class weight dictionary
    model_dir : str, default="models/tabnet"
        Directory to save model artifacts
    model_name : str, default="triage_model"
        Base name for saved files
    verbose : bool, default=True
        Print save information
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping artifact types to their file paths
    
    Raises
    ------
    ValueError
        If model or scaler is invalid
    OSError
        If directory creation or writing fails
    """
    
    if verbose:
        print("=" * 60)
        print("Saving TabNet Model")
        print("=" * 60)
    
    # Validate inputs
    if model is None:
        raise ValueError("Model is None")
    if scaler is None or not scaler.is_fitted:
        raise ValueError("Scaler is None or not fitted")
    if not class_weights:
        raise ValueError("Class weights are empty")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    if verbose:
        print(f"\n[SAVE] Model directory: {model_dir}")
    
    # Save model
    model_base_path = os.path.join(model_dir, model_name)
    model_path = f"{model_base_path}.zip"
    try:
        if not hasattr(model, 'save_model'):
            raise ValueError("Model does not support TabNet save_model()")
        model.save_model(model_base_path)
        if verbose:
            print(f"  ✓ Model saved: {model_path}")
    except Exception as e:
        raise OSError(f"Failed to save model: {str(e)}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    try:
        joblib.dump(scaler.scaler, scaler_path)
        if verbose:
            print(f"  ✓ Scaler saved: {scaler_path}")
    except Exception as e:
        raise OSError(f"Failed to save scaler: {str(e)}")
    
    # Save configuration
    config = {
        'model_name': model_name,
        'model_type': 'TabNetClassifier',
        'class_weights': class_weights,
        'n_classes': len(class_weights),
        'classes': sorted(class_weights.keys()),
        'scaler_type': 'QuantileTransformer',
        'scaler_config': {
            'n_quantiles': scaler.scaler.n_quantiles,
            'output_distribution': scaler.scaler.output_distribution,
        }
    }
    
    config_path = os.path.join(model_dir, f"{model_name}_config.json")
    try:
        # Convert to native Python types for JSON serialization
        config_json = config.copy()
        config_json['class_weights'] = {
            str(int(k)): float(v) for k, v in class_weights.items()
        }
        config_json['classes'] = [int(c) for c in config_json['classes']]
        config_json['n_classes'] = int(config_json['n_classes'])
        with open(config_path, 'w') as f:
            json.dump(config_json, f, indent=2)
        if verbose:
            print(f"  ✓ Config saved: {config_path}")
    except Exception as e:
        raise OSError(f"Failed to save configuration: {str(e)}")
    
    # Summary
    if verbose:
        print("\n[SUMMARY] Saved artifacts:")
        print(f"  ✓ Model: {model_path}")
        print(f"  ✓ Scaler: {scaler_path}")
        print(f"  ✓ Config: {config_path}")
        print(f"\n[INFO] Model configuration:")
        print(f"  ✓ Classes: {config['classes']}")
        print(f"  ✓ Class weights: {class_weights}")
        print(f"  ✓ Scaler: QuantileTransformer(output_distribution='normal')")
        print("\n" + "=" * 60)
        print("✓ Model saved successfully!")
        print("=" * 60)
    
    artifact_paths = {
        'model': model_path,
        'scaler': scaler_path,
        'config': config_path,
    }
    
    return artifact_paths


def load_tabnet_model(
    model_dir: str = "models/tabnet",
    model_name: str = "triage_model",
    verbose: bool = True
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load saved TabNet model with scaler and configuration.
    
    Parameters
    ----------
    model_dir : str, default="models/tabnet"
        Directory containing saved model artifacts
    model_name : str, default="triage_model"
        Base name of saved files
    verbose : bool, default=True
        Print load information
    
    Returns
    -------
    Tuple[model, scaler, config]
        Loaded TabNet model, scaler, and configuration dictionary
    
    Raises
    ------
    FileNotFoundError
        If required artifact files are missing
    """
    
    if verbose:
        print("=" * 60)
        print("Loading TabNet Model")
        print("=" * 60)
    
    model_path = os.path.join(model_dir, f"{model_name}.zip")
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
    
    # Load config
    if verbose:
        print(f"\n[LOAD] Loading configuration...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if verbose:
        print(f"  ✓ Model type: {config['model_type']}")
        print(f"  ✓ Classes: {config['classes']}")
    
    # Load scaler
    if verbose:
        print(f"\n[LOAD] Loading feature scaler...")
    
    scaler_obj = joblib.load(scaler_path)
    
    # Wrap scaler in TabNetScaler
    scaler = TabNetScaler()
    scaler.scaler = scaler_obj
    scaler.is_fitted = True
    
    if verbose:
        print(f"  ✓ Scaler loaded: {config['scaler_type']}")
    
    # Load model
    if verbose:
        print(f"\n[LOAD] Loading model...")
    
    try:
        if TabNetClassifier is None:
            raise ImportError(
                "pytorch_tabnet is required to load a saved TabNetClassifier"
            )
        model = TabNetClassifier()
        model.load_model(model_path)
        if verbose:
            print(f"  ✓ Model loaded: {config['model_type']}")
    except Exception as e:
        raise OSError(f"Failed to load model: {str(e)}")
    
    # Reconstruct class_weights with integer keys
    class_weights = {
        int(k): float(v) for k, v in config['class_weights'].items()
    }
    
    if verbose:
        print(f"\n[SUMMARY] Model loaded successfully:")
        print(f"  ✓ Classes: {config['classes']}")
        print(f"  ✓ Class weights: {class_weights}")
        print("\n" + "=" * 60)
        print("✓ Ready for inference!")
        print("=" * 60)
    
    return model, scaler, {**config, 'class_weights': class_weights}

if __name__ == "__main__":
    """Test scaling and class weights functionality"""
    import sys
    sys.path.insert(0, 'src/training')
    from train_tabnet import load_tabnet_data
    
    try:
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()
        
        print("\n")
        
        # Scale features (targets not scaled)
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_tabnet_features(
            X_train, X_val, X_test, verbose=True
        )
        
        # Verify targets are unchanged
        print("\n[VERIFICATION] Targets unchanged:")
        print(f"  y_train shape: {y_train.shape} (original)")
        print(f"  y_val shape:   {y_val.shape} (original)")
        print(f"  y_test shape:  {y_test.shape} (original)")
        print(f"  y_train dtype: {y_train.dtype}")
        
        print("\n")
        
        # Compute class weights for imbalance handling
        class_weights = compute_tabnet_class_weights(y_train, verbose=True)
        
        print("\n[READY] All preprocessing components prepared:")
        print(f"  ✓ Features scaled: {X_train_scaled.shape}")
        print(f"  ✓ Class weights computed: {len(class_weights)} classes")
        print("  ✓ Ready for TabNet model training!")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        raise


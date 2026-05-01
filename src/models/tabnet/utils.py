import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None


class TabNetScaler:
    def __init__(self, n_quantiles=1000):
        self.scaler = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution='normal',
            random_state=42
        )
        self.is_fitted = False

    def fit_transform_train(self, X):
        X = X.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        self.is_fitted = True
        return X_scaled

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("TabNetScaler must be fitted before calling transform().")
        return self.scaler.transform(X.astype(np.float32)).astype(np.float32)


def scale_tabnet_features(X_train, X_val, X_test):
    scaler = TabNetScaler(n_quantiles=min(1000, len(X_train)))
    X_train = scaler.fit_transform_train(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def compute_tabnet_class_weights(y):
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"compute_tabnet_class_weights requires at least 2 classes, found: {classes}"
        )
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def class_weights_to_sample_weights(class_weights: dict, y: np.ndarray) -> np.ndarray:
    """
    Convert a {class_index: weight} dict into a per-sample weight array.

    TabNet's `weights` parameter must be a 1D array of length == n_train_samples,
    where each value is the weight for that sample's class.  Passing a dict, a
    per-class array, or anything of the wrong length raises:
        "Custom weights should match number of train samples."
    """
    return np.array([class_weights[int(label)] for label in y], dtype=np.float32)


<<<<<<< HEAD
def save_tabnet_model(model, scaler, class_weights, hyperparams=None, model_dir="models/tabnet"):
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model")
    model.save_model(model_path)

    joblib.dump(scaler.scaler, os.path.join(model_dir, "scaler.pkl"))

    config = {"class_weights": {str(k): v for k, v in class_weights.items()}}
    if hyperparams:
        config["hyperparams"] = hyperparams

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {model_dir}")
    return model_path


def load_tabnet_model(model_dir="models/tabnet"):
    from pytorch_tabnet.tab_model import TabNetClassifier

    model_zip    = os.path.join(model_dir, "model.zip")
    scaler_path  = os.path.join(model_dir, "scaler.pkl")
    config_path  = os.path.join(model_dir, "config.json")

    for path in [model_zip, scaler_path, config_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")

    model = TabNetClassifier()
    model.load_model(model_zip)
    scaler = joblib.load(scaler_path)

    with open(config_path) as f:
        config = json.load(f)

    if "class_weights" in config:
        class_weights = {int(k): v for k, v in config["class_weights"].items()}
        hyperparams   = config.get("hyperparams", {})
    else:
        # backwards-compatible with old flat format
        class_weights = {int(k): v for k, v in config.items()}
        hyperparams   = {}

    return model, scaler, class_weights, hyperparams
=======
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

>>>>>>> origin/main

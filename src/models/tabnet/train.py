"""
TabNet triage model training script.

Trains a TabNet classifier for SOC alert triage with early stopping,
class weights for imbalance handling, and probability outputs.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# Add parent paths to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from train_tabnet import load_tabnet_data
from utils import (
    scale_tabnet_features,
    compute_tabnet_class_weights,
    TabNetScaler,
    save_tabnet_model,
)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception as e:
    print("[ERROR] Failed to import TabNetClassifier from pytorch_tabnet.tab_model")
    print(f"  Root cause: {type(e).__name__}: {e}")
    print("  Verify that both pytorch-tabnet and torch import cleanly.")
    sys.exit(1)


def train_tabnet_triage_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any] = None,
    verbose: bool = True
) -> Tuple[TabNetClassifier, Dict[str, Any]]:
    """
    Train TabNet classifier for SOC alert triage.
    
    Includes:
    - Feature scaling (QuantileTransformer)
    - Class weight balancing
    - Early stopping validation
    - Probability and class predictions
    
    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Feature matrices
    y_train, y_val, y_test : np.ndarray
        Target labels
    model_params : Dict[str, Any], optional
        Custom TabNetClassifier hyperparameters
    verbose : bool, default=True
        Print training information
    
    Returns
    -------
    Tuple[TabNetClassifier, Dict[str, Any]]
        Trained model and training results dictionary
    
    Raises
    ------
    ValueError
        If data validation fails
    """
    
    if verbose:
        print("=" * 60)
        print("TabNet Triage Model Training")
        print("=" * 60)
    
    # ===== STEP 1: LOAD AND VALIDATE DATA =====
    if verbose:
        print("\n[STEP 1] Data Loading")
    
    if X_train.shape[0] == 0:
        raise ValueError("Training data is empty")
    if X_val.shape[0] == 0:
        raise ValueError("Validation data is empty")
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError("Training data/labels size mismatch")
    if y_val.shape[0] != X_val.shape[0]:
        raise ValueError("Validation data/labels size mismatch")
    
    if verbose:
        print(f"  ✓ Training samples: {X_train.shape[0]}")
        print(f"  ✓ Validation samples: {X_val.shape[0]}")
        print(f"  ✓ Test samples: {X_test.shape[0]}")
        print(f"  ✓ Features: {X_train.shape[1]}")
    
    # ===== STEP 2: SCALE FEATURES =====
    if verbose:
        print("\n[STEP 2] Feature Scaling (QuantileTransformer)")
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_tabnet_features(
        X_train, X_val, X_test, verbose=False
    )
    
    if verbose:
        print(f"  ✓ Training scaled: {X_train_scaled.shape}")
        print(f"  ✓ Validation scaled: {X_val_scaled.shape}")
        print(f"  ✓ Test scaled: {X_test_scaled.shape}")
    
    # ===== STEP 3: COMPUTE CLASS WEIGHTS =====
    if verbose:
        print("\n[STEP 3] Class Weight Computation (Inverse Frequency)")
    
    class_weights = compute_tabnet_class_weights(y_train, verbose=False)
    
    if verbose:
        for class_label, weight in sorted(class_weights.items()):
            print(f"  ✓ Class {class_label}: {weight:.4f}")
    
    n_classes = len(np.unique(y_train))
    
    # ===== STEP 4: CONFIGURE TABNET CLASSIFIER =====
    if verbose:
        print("\n[STEP 4] TabNetClassifier Configuration")
    
    # Default hyperparameters (can be overridden)
    default_params = {
        'n_d': 64,                      # Width of decision step features
        'n_a': 64,                      # Width of attention features
        'n_steps': 5,                   # Number of decision steps
        'gamma': 1.5,                   # Feature reuse coefficient
        'n_independent': 2,             # Independent components
        'n_shared': 2,                  # Shared components
        'lambda_sparse': 1e-3,          # Feature sparsity regularization
        'momentum': 0.02,               # Momentum for batch norm
        'epsilon': 1e-15,               # Batch norm epsilon
        'seed': 42,
        'optimizer_params': {'lr': 2e-2, 'weight_decay': 1e-5},
        'verbose': 0,  # 0 for silent, 1 for epoch-level, 2 for detailed
    }
    
    # Merge with custom params if provided
    if model_params:
        default_params.update(model_params)
    
    if verbose:
        print("  Hyperparameters:")
        print(f"    n_d (decision features): {default_params['n_d']}")
        print(f"    n_a (attention features): {default_params['n_a']}")
        print(f"    n_steps (decision steps): {default_params['n_steps']}")
        print(f"    gamma (feature reuse): {default_params['gamma']}")
        print(f"    lambda_sparse (sparsity): {default_params['lambda_sparse']}")
    
    # Create model
    model = TabNetClassifier(**default_params)
    
    # ===== STEP 5: TRAIN WITH EARLY STOPPING =====
    if verbose:
        print("\n[STEP 5] Training with Early Stopping")
        print("  Training on training set with validation monitoring...")
    
    # Train with early stopping on validation metric
    model.fit(
        X_train=X_train_scaled,
        y_train=y_train,
        eval_set=[(X_val_scaled, y_val)],  # Validation for early stopping
        eval_metric=['accuracy'],         # Multiclass-safe validation metric
        max_epochs=200,
        patience=20,                      # Stop if no improvement for 20 epochs
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        weights=class_weights,  # Handle class imbalance
    )
    
    if verbose:
        print(f"  ✓ Training completed (stopped at epoch)")
    
    # ===== STEP 6: MAKE PREDICTIONS =====
    if verbose:
        print("\n[STEP 6] Generating Predictions")
    
    # Predicted classes
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Prediction probabilities
    y_proba_train = model.predict_proba(X_train_scaled)
    y_proba_val = model.predict_proba(X_val_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)
    
    if verbose:
        print(f"  ✓ Training predictions: {y_pred_train.shape}")
        print(f"  ✓ Training probabilities: {y_proba_train.shape}")
        print(f"  ✓ Validation predictions: {y_pred_val.shape}")
        print(f"  ✓ Validation probabilities: {y_proba_val.shape}")
        print(f"  ✓ Test predictions: {y_pred_test.shape}")
        print(f"  ✓ Test probabilities: {y_proba_test.shape}")
    
    # ===== STEP 7: EVALUATE PREDICTIONS =====
    if verbose:
        print("\n[STEP 7] Prediction Analysis")
        
        # Calculate accuracy
        train_accuracy = np.mean(y_pred_train == y_train)
        val_accuracy = np.mean(y_pred_val == y_val)
        test_accuracy = np.mean(y_pred_test == y_test)
        
        print(f"  Training accuracy: {train_accuracy:.4f}")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        
        # Confidence analysis
        max_proba_train = np.max(y_proba_train, axis=1)
        max_proba_val = np.max(y_proba_val, axis=1)
        max_proba_test = np.max(y_proba_test, axis=1)
        
        print(f"\n  Average prediction confidence:")
        print(f"    Training: {np.mean(max_proba_train):.4f}")
        print(f"    Validation: {np.mean(max_proba_val):.4f}")
        print(f"    Test: {np.mean(max_proba_test):.4f}")
        
        # Output sample predictions
        print(f"\n  Sample predictions (first 5 test samples):")
        for i in range(min(5, len(y_test))):
            pred_class = y_pred_test[i]
            confidence = np.max(y_proba_test[i])
            actual = y_test[i]
            match = "✓" if pred_class == actual else "✗"
            print(f"    [{match}] Sample {i}: Pred={pred_class}, "
                  f"Confidence={confidence:.4f}, Actual={actual}")
    
    # ===== PREPARE RESULTS =====
    results = {
        'model': model,
        'scaler': scaler,
        'class_weights': class_weights,
        'predictions': {
            'train': y_pred_train,
            'val': y_pred_val,
            'test': y_pred_test,
        },
        'probabilities': {
            'train': y_proba_train,
            'val': y_proba_val,
            'test': y_proba_test,
        },
        'accuracy': {
            'train': np.mean(y_pred_train == y_train),
            'val': np.mean(y_pred_val == y_val),
            'test': np.mean(y_pred_test == y_test),
        },
        'metrics': {
            'n_features': X_train.shape[1],
            'n_classes': n_classes,
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
        }
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ TabNet Triage Model Training Complete!")
        print("=" * 60)
    
    return model, results


def main():
    """Main execution - load data and train model"""
    try:
        # Load data
        print()
        X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()
        
        # Train model
        print("\n")
        model, results = train_tabnet_triage_model(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            verbose=True
        )
        
        # Summary
        print("\n[SUCCESS] All training components ready:")
        print(f"  ✓ Model trained: {type(model).__name__}")
        print(f"  ✓ Predictions available: class + probabilities")
        print(f"  ✓ Class weights applied: {len(results['class_weights'])} classes")
        print(f"  ✓ Scaler saved for inference")
        
        # Save model
        print("\n")
        artifact_paths = save_tabnet_model(
            model=model,
            scaler=results['scaler'],
            class_weights=results['class_weights'],
            model_dir="models/tabnet",
            model_name="triage_model",
            verbose=True
        )
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Artifact management for preprocessing pipeline.
Saves and loads preprocessing objects for reproducibility and inference.
"""
import os
import joblib
import pickle
from typing import Dict, Any, Optional


ARTIFACTS_DIR = "models/artifacts"


def ensure_artifacts_dir():
    """Ensure artifacts directory exists."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def save_artifacts(
    encoders: Dict[str, Dict[str, float]],
    target_mapping: Dict[str, int],
    scaler=None,
    verbose: bool = True
) -> None:
    """
    Save preprocessing artifacts for reproducibility.
    
    Args:
        encoders: Dictionary of feature encoders (freq encoding maps)
        target_mapping: Mapping for target variable encoding
        scaler: Optional QuantileTransformer scaler object
        verbose: Print confirmation messages
    
    Saves:
        - models/artifacts/encoders.pkl
        - models/artifacts/target_mapping.pkl
        - models/artifacts/scaler.pkl (if scaler provided)
    """
    ensure_artifacts_dir()
    
    try:
        # Save encoders (frequency encoding dictionaries)
        encoders_path = os.path.join(ARTIFACTS_DIR, "encoders.pkl")
        joblib.dump(encoders, encoders_path)
        if verbose:
            print(f"[ARTIFACTS] Saved encoders to {encoders_path}")
            print(f"[ARTIFACTS]   Contains {len(encoders)} feature encoders")
        
        # Save target mapping
        target_path = os.path.join(ARTIFACTS_DIR, "target_mapping.pkl")
        joblib.dump(target_mapping, target_path)
        if verbose:
            print(f"[ARTIFACTS] Saved target mapping to {target_path}")
            print(f"[ARTIFACTS]   Mapping: {target_mapping}")
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            if verbose:
                print(f"[ARTIFACTS] Saved scaler to {scaler_path}")
                print(f"[ARTIFACTS]   Scaler type: {type(scaler).__name__}")
        
        if verbose:
            print(f"\n[ARTIFACTS] Artifacts saved successfully to {ARTIFACTS_DIR}/")
    
    except Exception as e:
        print(f"[ERROR] Failed to save artifacts: {str(e)}")
        raise


def load_artifacts(verbose: bool = True) -> Dict[str, Any]:
    """
    Load preprocessing artifacts for inference.
    
    Args:
        verbose: Print confirmation messages
    
    Returns:
        Dictionary containing:
            - 'encoders': Feature encoders
            - 'target_mapping': Target variable mapping
            - 'scaler': Scaler object (None if not available)
    
    Raises:
        FileNotFoundError: If required artifact files are missing
    """
    artifacts = {}
    
    try:
        # Load encoders
        encoders_path = os.path.join(ARTIFACTS_DIR, "encoders.pkl")
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders not found: {encoders_path}")
        
        encoders = joblib.load(encoders_path)
        artifacts['encoders'] = encoders
        
        if verbose:
            print(f"[ARTIFACTS] Loaded encoders from {encoders_path}")
            print(f"[ARTIFACTS]   Contains {len(encoders)} feature encoders")
        
        # Load target mapping
        target_path = os.path.join(ARTIFACTS_DIR, "target_mapping.pkl")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target mapping not found: {target_path}")
        
        target_mapping = joblib.load(target_path)
        artifacts['target_mapping'] = target_mapping
        
        if verbose:
            print(f"[ARTIFACTS] Loaded target mapping from {target_path}")
            print(f"[ARTIFACTS]   Mapping: {target_mapping}")
        
        # Load scaler (optional)
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            artifacts['scaler'] = scaler
            
            if verbose:
                print(f"[ARTIFACTS] Loaded scaler from {scaler_path}")
                print(f"[ARTIFACTS]   Scaler type: {type(scaler).__name__}")
        else:
            artifacts['scaler'] = None
            if verbose:
                print(f"[ARTIFACTS] No scaler found (optional)")
        
        if verbose:
            print(f"\n[ARTIFACTS] Artifacts loaded successfully!")
        
        return artifacts
    
    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {str(e)}")
        raise


def get_artifact_info(verbose: bool = True) -> Dict[str, Any]:
    """
    Get information about saved artifacts without loading them.
    
    Args:
        verbose: Print information
    
    Returns:
        Dictionary with artifact metadata
    """
    info = {'exists': False, 'files': {}}
    
    encoders_path = os.path.join(ARTIFACTS_DIR, "encoders.pkl")
    target_path = os.path.join(ARTIFACTS_DIR, "target_mapping.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    
    if os.path.exists(encoders_path):
        size = os.path.getsize(encoders_path)
        info['files']['encoders'] = {'path': encoders_path, 'size': size}
        info['exists'] = True
    
    if os.path.exists(target_path):
        size = os.path.getsize(target_path)
        info['files']['target_mapping'] = {'path': target_path, 'size': size}
    
    if os.path.exists(scaler_path):
        size = os.path.getsize(scaler_path)
        info['files']['scaler'] = {'path': scaler_path, 'size': size}
    
    if verbose:
        if info['exists']:
            print(f"\n[ARTIFACTS] Saved artifacts found in {ARTIFACTS_DIR}:")
            for name, details in info['files'].items():
                size_kb = details['size'] / 1024
                print(f"  {name}: {size_kb:.1f} KB")
        else:
            print(f"\n[ARTIFACTS] No artifacts found in {ARTIFACTS_DIR}/")
    
    return info


if __name__ == "__main__":
    # Test artifact saving and loading
    print("Testing artifact management...")
    
    # Sample encoders
    sample_encoders = {
        'Category': {'Security': 0.25, 'Network': 0.35, 'Unknown': 0.40},
        'ResourceType': {'File': 0.50, 'Registry': 0.50}
    }
    
    # Sample target mapping
    sample_target_mapping = {
        'FalsePositive': 0,
        'BenignPositive': 1,
        'TruePositive': 2
    }
    
    # Save
    print("\nSaving artifacts...")
    save_artifacts(sample_encoders, sample_target_mapping, scaler=None)
    
    # Get info
    print("\nChecking artifact info...")
    get_artifact_info()
    
    # Load
    print("\nLoading artifacts...")
    artifacts = load_artifacts()
    
    print("\nTest completed successfully!")

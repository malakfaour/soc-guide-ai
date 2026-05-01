"""
LightGBM Utilities
Handles data loading, validation, and helper functions
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Dict
from sklearn.utils.class_weight import compute_class_weight


def load_processed_data(
    data_dir: str = 'data/processed/v1'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load preprocessed train, validation, and test data.
    
    Args:
        data_dir: Directory containing processed CSV files
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print(f"Loading data from {data_dir}...")
    
    # Load features
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    
    # Load targets
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['IncidentGrade']
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))['IncidentGrade']
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['IncidentGrade']
    
    print(f"✅ Loaded X_train: {X_train.shape}")
    print(f"✅ Loaded X_val: {X_val.shape}")
    print(f"✅ Loaded X_test: {X_test.shape}")
    print(f"✅ Loaded y_train: {y_train.shape}")
    print(f"✅ Loaded y_val: {y_val.shape}")
    print(f"✅ Loaded y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_target_mapping(artifact_dir: str = 'models/artifacts') -> Dict[str, int]:
    """
    Load target mapping from preprocessing artifacts.
    
    Args:
        artifact_dir: Directory containing preprocessing artifacts
        
    Returns:
        Dictionary mapping class names to encoded values
    """
    mapping_path = os.path.join(artifact_dir, 'target_mapping.pkl')
    
    with open(mapping_path, 'rb') as f:
        target_mapping = pickle.load(f)
    
    print(f"✅ Loaded target mapping: {target_mapping}")
    return target_mapping


def get_class_weights(y_train: pd.Series) -> Dict[int, float]:
    """
    Compute class weights for handling imbalance.
    
    Args:
        y_train: Training target labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    print(f"\n📊 Class weights computed:")
    for cls, weight in class_weights.items():
        print(f"   Class {cls}: {weight:.4f}")
    
    return class_weights


def get_sample_weights(y_train: pd.Series, class_weights: Dict[int, float]) -> np.ndarray:
    """
    Convert class weights to sample weights.
    
    Args:
        y_train: Training target labels
        class_weights: Dictionary of class weights
        
    Returns:
        Array of sample weights
    """
    sample_weights = np.array([class_weights[y] for y in y_train])
    return sample_weights


def validate_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Validate that data is ready for training.
    
    Args:
        X_train, X_val, X_test: Feature dataframes
        y_train, y_val, y_test: Target series
        
    Raises:
        ValueError: If data validation fails
    """
    print("\n🔍 Validating data...")
    
    # Check shapes match
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
    assert len(X_val) == len(y_val), "X_val and y_val length mismatch"
    assert len(X_test) == len(y_test), "X_test and y_test length mismatch"
    
    # Check no missing values
    assert X_train.isnull().sum().sum() == 0, "Missing values in X_train"
    assert y_train.isnull().sum() == 0, "Missing values in y_train"
    
    # Check same features
    assert list(X_train.columns) == list(X_val.columns), "Feature mismatch between train and val"
    assert list(X_train.columns) == list(X_test.columns), "Feature mismatch between train and test"
    
    # Check all numeric
    assert X_train.select_dtypes(include=[np.number]).shape[1] == X_train.shape[1], \
        "Non-numeric features found"
    
    print("✅ Data validation passed!")


if __name__ == "__main__":
    # Test the functions
    print("=" * 80)
    print("TESTING LIGHTGBM UTILS")
    print("=" * 80)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Load target mapping
    target_mapping = load_target_mapping()
    
    # Validate data
    validate_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Compute class weights
    class_weights = get_class_weights(y_train)
    
    # Get sample weights
    sample_weights = get_sample_weights(y_train, class_weights)
    print(f"\n✅ Sample weights shape: {sample_weights.shape}")
    print(f"✅ Sample weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    
    print("\n" + "=" * 80)
    print("✅ ALL UTILS TESTS PASSED")
    print("=" * 80)
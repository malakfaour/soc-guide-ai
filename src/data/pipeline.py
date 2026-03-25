"""
Data pipeline orchestration.

Coordinates data loading, validation, and export workflows.
"""

import pandas as pd
import os
from typing import Tuple, Optional
from src.data.loader import load_train_test_data, load_train_data_only


def validate_data(df: pd.DataFrame, name: str = "DataFrame") -> bool:
    """
    Validate loaded data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    name : str
        Name for logging
    
    Returns
    -------
    bool
        True if valid, raises exception otherwise
    """
    print(f"Validating {name}...")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    if df.isnull().all().any():
        raise ValueError(f"{name} has fully null columns")
    
    print(f"  ✓ Shape: {df.shape}")
    print(f"  ✓ Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return True


def export_preprocessed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data/processed"
) -> None:
    """
    Export preprocessed datasets to disk.
    
    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature datasets
    y_train, y_val, y_test : pd.Series
        Target datasets
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    X_train.to_parquet(f"{output_dir}/X_train.parquet", index=True)
    X_val.to_parquet(f"{output_dir}/X_val.parquet", index=True)
    X_test.to_parquet(f"{output_dir}/X_test.parquet", index=True)
    
    # Save targets
    y_train.to_frame().to_parquet(f"{output_dir}/y_train.parquet", index=True)
    y_val.to_frame().to_parquet(f"{output_dir}/y_val.parquet", index=True)
    y_test.to_frame().to_parquet(f"{output_dir}/y_test.parquet", index=True)
    
    print(f"✓ Exported to {output_dir}/")


def load_preprocessed_data(
    input_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load preprocessed datasets from disk.
    
    Parameters
    ----------
    input_dir : str
        Directory containing parquet files
    
    Returns
    -------
    Tuple[6 DataFrames/Series]
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train = pd.read_parquet(f"{input_dir}/X_train.parquet")
    X_val = pd.read_parquet(f"{input_dir}/X_val.parquet")
    X_test = pd.read_parquet(f"{input_dir}/X_test.parquet")
    
    y_train = pd.read_parquet(f"{input_dir}/y_train.parquet").iloc[:, 0]
    y_val = pd.read_parquet(f"{input_dir}/y_val.parquet").iloc[:, 0]
    y_test = pd.read_parquet(f"{input_dir}/y_test.parquet").iloc[:, 0]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

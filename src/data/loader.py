"""
Data loading module for GUIDE dataset.

Handles efficient loading of training and test datasets.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional


def load_train_test_data(
    train_path: str = "data/raw/GUIDE_Train.csv",
    test_path: str = "data/raw/GUIDE_Test.csv",
    nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets efficiently.
    
    Parameters
    ----------
    train_path : str
        Path to training CSV file (relative to project root)
    test_path : str
        Path to test CSV file (relative to project root)
    nrows : int, optional
        Limit number of rows to load (useful for testing)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and test DataFrames
    
    Raises
    ------
    FileNotFoundError
        If CSV files do not exist
    ValueError
        If DataFrames are empty or incompatible
    """
    # Ensure paths exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    # Load datasets with dtype optimization for large files
    df_train = pd.read_csv(
        train_path,
        nrows=nrows,
        low_memory=True,
        dtype_backend='numpy'
    )
    
    df_test = pd.read_csv(
        test_path,
        nrows=nrows,
        low_memory=True,
        dtype_backend='numpy'
    )
    
    # Validate datasets
    if df_train.empty:
        raise ValueError("Training dataset is empty")
    if df_test.empty:
        raise ValueError("Test dataset is empty")
    
    print(f"✓ Loaded train shape: {df_train.shape}")
    print(f"✓ Loaded test shape: {df_test.shape}")
    
    return df_train, df_test


def load_train_data_only(
    train_path: str = "data/raw/GUIDE_Train.csv",
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load only training dataset.
    
    Useful for when test data is accessed separately or for CV splits.
    
    Parameters
    ----------
    train_path : str
        Path to training CSV file
    nrows : int, optional
        Limit number of rows
    
    Returns
    -------
    pd.DataFrame
        Training DataFrame
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    df_train = pd.read_csv(
        train_path,
        nrows=nrows,
        low_memory=True,
        dtype_backend='numpy'
    )
    
    if df_train.empty:
        raise ValueError("Training dataset is empty")
    
    print(f"✓ Loaded train shape: {df_train.shape}")
    return df_train

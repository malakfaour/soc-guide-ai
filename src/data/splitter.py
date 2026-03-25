"""
Data splitting module for GUIDE dataset.

Handles stratified train/validation/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import os


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_col: str = None
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series
]:
    """
    Perform stratified train/validation/test split.
    
    Ensures class distribution is maintained across all splits.
    
    Split ratios:
    - Train: ~0.70 (70%)
    - Val: ~0.15 (15%)
    - Test: ~0.15 (15%)
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proportion of data for test (default 0.2, overall becomes ~15%)
    val_size : float
        Proportion of train+val data to go to validation (default 0.15)
    random_state : int
        Random state for reproducibility
    stratify_col : str, optional
        Column to stratify by. If None, uses y.
    
    Returns
    -------
    Tuple[6 DataFrames/Series]
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Combine X and y for joint splitting
    data = pd.concat([X, y], axis=1)
    
    if stratify_col is None:
        stratify_col = y.name
    
    # First split: train+val vs test
    # If test_size=0.2, then train+val = 0.8
    data_train_val, data_test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[stratify_col]
    )
    
    # Second split: train vs val
    # val_size is proportion of remaining data (0.15 of 0.8 ≈ 0.12 of total)
    # So train gets remaining 0.85 of 0.8 = 0.68 of total
    data_train, data_val = train_test_split(
        data_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=data_train_val[stratify_col]
    )
    
    # Separate features and target
    X_train = data_train.drop(columns=[stratify_col])
    X_val = data_val.drop(columns=[stratify_col])
    X_test = data_test.drop(columns=[stratify_col])
    
    y_train = data_train[stratify_col]
    y_val = data_val[stratify_col]
    y_test = data_test[stratify_col]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    Dict
]:
    """
    Main data splitting pipeline.
    
    Performs stratified split to maintain class distribution.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proportion for test set (default 0.2)
    val_size : float
        Proportion of train+val for validation (default 0.15)
    random_state : int
        Random state for reproducibility
    stratify : bool
        Whether to use stratified split (default True)
    
    Returns
    -------
    Tuple[6 DataFrames/Series + Dict]
        (X_train, X_val, X_test, y_train, y_val, y_test, split_info)
    """
    print("\n=== SPLITTING DATA ===")
    print(f"Total samples: {len(X)}")
    print(f"Test size: {test_size * 100:.1f}%")
    print(f"Val size of train+val: {val_size * 100:.1f}%")
    
    if stratify:
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
            X, y,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify_col=y.name if hasattr(y, 'name') else None
        )
        print(f"✓ Stratified split applied (by target class)")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
            X, y,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
    
    # Print split statistics
    print(f"\nSplit results:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Print class distribution
    print(f"\nClass distribution:")
    for split_name, y_split in [
        ('Train', y_train),
        ('Val', y_val),
        ('Test', y_test)
    ]:
        dist = y_split.value_counts(normalize=True).to_dict()
        print(f"  {split_name}:")
        for class_val, proportion in sorted(dist.items()):
            print(f"    {class_val}: {proportion*100:.1f}%")
    
    split_info = {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_pct': len(X_train) / len(X) * 100,
        'val_pct': len(X_val) / len(X) * 100,
        'test_pct': len(X_test) / len(X) * 100,
        'train_indices': X_train.index.tolist(),
        'val_indices': X_val.index.tolist(),
        'test_indices': X_test.index.tolist(),
        'stratified': stratify,
        'random_state': random_state
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


def save_split_indices(
    split_info: Dict,
    output_dir: str = "data/splits"
) -> None:
    """
    Save split indices to disk for reproducibility.
    
    Parameters
    ----------
    split_info : Dict
        Split information from split_data()
    output_dir : str
        Directory to save indices
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV files
    pd.DataFrame({'index': split_info['train_indices']}).to_csv(
        f"{output_dir}/train_indices.csv", index=False
    )
    pd.DataFrame({'index': split_info['val_indices']}).to_csv(
        f"{output_dir}/val_indices.csv", index=False
    )
    pd.DataFrame({'index': split_info['test_indices']}).to_csv(
        f"{output_dir}/test_indices.csv", index=False
    )
    
    # Save split metadata
    metadata = {
        k: v for k, v in split_info.items()
        if k not in ['train_indices', 'val_indices', 'test_indices']
    }
    
    import json
    with open(f"{output_dir}/split_config.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Split indices saved to {output_dir}/")


def load_split_indices(
    split_dir: str = "data/splits"
) -> Dict:
    """
    Load split indices from disk.
    
    Parameters
    ----------
    split_dir : str
        Directory containing split files
    
    Returns
    -------
    Dict
        Split information
    """
    import json
    
    train_indices = pd.read_csv(f"{split_dir}/train_indices.csv")['index'].tolist()
    val_indices = pd.read_csv(f"{split_dir}/val_indices.csv")['index'].tolist()
    test_indices = pd.read_csv(f"{split_dir}/test_indices.csv")['index'].tolist()
    
    with open(f"{split_dir}/split_config.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        **metadata
    }

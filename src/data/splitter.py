"""
<<<<<<< HEAD
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
    
"""
Data splitting module.

Stratified split on target after encoding.
Create train/test split first.
Then create validation split from training portion.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(df, target_col='IncidentGrade', train_ratio=0.7, val_ratio=0.15, 
               test_ratio=0.15, random_state=42, verbose=True):
    """
    Split data into train/validation/test using stratified split.
    
    Args:
        df: Encoded DataFrame
        target_col: Name of target column
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation from train (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
        random_state: Random seed
        verbose: Print debug information
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if verbose:
        print(f"\n[SPLIT] Starting data split...")
        print(f"[SPLIT] Input shape: {df.shape}")
    
    # Separate features and target
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    if verbose:
        print(f"[SPLIT] Features shape: {X.shape}")
        print(f"[SPLIT] Target shape: {y.shape}")
        print(f"[SPLIT] Target distribution before split:")
        unique, counts = np.unique(y, return_counts=True)
        for val, count in zip(unique, counts):
            pct = 100 * count / len(y)
            print(f"  Class {int(val)}: {count:,} ({pct:.2f}%)")
    
    # First split: train+val vs test
    test_size_ratio = test_ratio / (1.0 - 0)  # We want test to be test_ratio of total
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: train vs val from temp
    val_size_in_temp = val_ratio / (1.0 - test_ratio)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_in_temp,
        stratify=y_temp,
        random_state=random_state
    )
    
    if verbose:
        print(f"\n[SPLIT] Split ratios:")
        print(f"  Train: {len(X_train):,} ({100*len(X_train)/len(X):.2f}%)")
        print(f"  Val:   {len(X_val):,} ({100*len(X_val)/len(X):.2f}%)")
        print(f"  Test:  {len(X_test):,} ({100*len(X_test)/len(X):.2f}%)")
        
        print(f"\n[SPLIT] Class distribution by split:")
        for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            print(f"  {split_name}:")
            unique, counts = np.unique(y_split, return_counts=True)
            for val, count in zip(unique, counts):
                pct = 100 * count / len(y_split)
                print(f"    Class {int(val)}: {count:,} ({pct:.2f}%)")
    
    # Convert back to dataframes
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    y_train_df = pd.Series(y_train, name=target_col)
    y_val_df = pd.Series(y_val, name=target_col)
    y_test_df = pd.Series(y_test, name=target_col)
    
    return X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df


if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'feat1': np.random.randn(1000),
        'feat2': np.random.randn(1000),
        'target': np.random.choice([0, 1, 2], 1000)
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col='target')
    print("Splits created successfully!")
>>>>>>> f80310cf3a2ef8383f0d05dcca483e9bcc64aa12

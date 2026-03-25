"""
Data splitting module.
- Stratified split on target after encoding
- Create train/test split first
- Then create validation split from training portion
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

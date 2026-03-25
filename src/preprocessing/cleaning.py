"""
Minimal data cleaning module.
- Fill numerical missing values with median
- Fill categorical missing values with 'unknown'
- NO encoding, NO scaling
"""
import pandas as pd
import numpy as np


def clean_data(df, target_col='IncidentGrade', verbose=True):
    """
    Minimal data cleaning.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column (will not be cleaned)
        verbose: Print debug information
    
    Returns:
        Cleaned DataFrame
    """
    if verbose:
        print(f"\n[CLEANING] Starting data cleaning...")
        print(f"[CLEANING] Input shape: {df.shape}")
        print(f"[CLEANING] Missing values before cleaning:")
        missing_before = df.isnull().sum()
        print(f"  Total NaN cells: {missing_before.sum()}")
        print(f"  Columns with missing: {(missing_before > 0).sum()}")
    
    df_clean = df.copy()
    
    # Separate target and features
    if target_col in df_clean.columns:
        target = df_clean[target_col]
        features = df_clean.drop(columns=[target_col])
    else:
        features = df_clean
        target = None
    
    # Identify numeric and categorical columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    
    if verbose:
        print(f"\n[CLEANING] Column types:")
        print(f"  Numeric columns: {len(numeric_cols)}")
        print(f"  Categorical columns: {len(categorical_cols)}")
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if features[col].isnull().sum() > 0:
            median_val = features[col].median()
            if verbose:
                print(f"[CLEANING] Filling {col} with median: {median_val}")
            features[col] = features[col].fillna(median_val)
    
    # Fill categorical missing values with 'unknown'
    for col in categorical_cols:
        if features[col].isnull().sum() > 0:
            if verbose:
                print(f"[CLEANING] Filling {col} with 'unknown'")
            features[col] = features[col].fillna('unknown')
    
    # Handle target column missing values
    if target is not None:
        target_missing = target.isnull().sum()
        if target_missing > 0:
            if verbose:
                print(f"\n[CLEANING] Target column has {target_missing} missing values")
                print(f"[CLEANING] Removing rows with missing target...")
            # Remove rows where target is NaN
            valid_idx = target.notna()
            features = features[valid_idx]
            target = target[valid_idx]
            if verbose:
                print(f"[CLEANING] New shape: {features.shape}")
    
    # Reconstruct dataframe
    if target is not None:
        df_clean = features.copy()
        df_clean[target_col] = target
    else:
        df_clean = features
    
    if verbose:
        print(f"\n[CLEANING] Output shape: {df_clean.shape}")
        print(f"[CLEANING] Missing values after cleaning:")
        missing_after = df_clean.isnull().sum()
        print(f"  Total NaN cells: {missing_after.sum()}")
        if missing_after.sum() == 0:
            print(f"  ✓ All missing values handled!")
        else:
            print(f"  ✗ Still missing values in: {missing_after[missing_after > 0].index.tolist()}")
    
    return df_clean


if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'num1': [1, 2, np.nan, 4, 5],
        'num2': [1.5, np.nan, 3.5, 4.5, 5.5],
        'cat1': ['a', 'b', 'a', np.nan, 'c'],
        'target': ['TP', 'BP', 'FP', 'TP', 'BP']
    })
    print("Input:\n", df)
    df_clean = clean_data(df, target_col='target')
    print("\nOutput:\n", df_clean)

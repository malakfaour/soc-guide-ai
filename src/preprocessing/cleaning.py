"""
Data cleaning module for GUIDE dataset.
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
        missing_before = df.isnull().sum()
        print(f"[CLEANING] Total NaN cells: {missing_before.sum()}")

    df_clean = df.copy()

    if target_col in df_clean.columns:
        target = df_clean[target_col]
        features = df_clean.drop(columns=[target_col])
    else:
        features = df_clean
        target = None

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()

    if verbose:
        print(f"[CLEANING] Numeric columns: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

    for col in numeric_cols:
        if features[col].isnull().sum() > 0:
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val)
            if verbose:
                print(f"[CLEANING] Filled {col} with median: {median_val:.4f}")

    for col in categorical_cols:
        if features[col].isnull().sum() > 0:
            features[col] = features[col].fillna('unknown')
            if verbose:
                print(f"[CLEANING] Filled {col} with 'unknown'")

    if target is not None:
        target_missing = target.isnull().sum()
        if target_missing > 0:
            valid_idx = target.notna()
            features = features[valid_idx]
            target = target[valid_idx]
            if verbose:
                print(f"[CLEANING] Dropped {target_missing} rows with missing target")

    if target is not None:
        df_clean = features.copy()
        df_clean[target_col] = target
    else:
        df_clean = features

    if verbose:
        print(f"[CLEANING] Output shape: {df_clean.shape}, NaN remaining: {df_clean.isnull().sum().sum()}")

    return df_clean
"""
Data cleaning module for GUIDE dataset.

Handles missing values, irrelevant column removal, and data validation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# Columns to drop as they are identifiers or not useful for prediction
IDENTIFIER_COLUMNS = [
    'Id',
    'AlertId',
    'HashId',
    'UUID',
    'RecordId',
    'Timestamp',
    'Date',
    'Time',
    'SessionId',
    'EventId'
]

# Target column
TARGET_COLUMN = 'IncidentGrade'


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    """
    Identify numerical, categorical, and target columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    Tuple[List[str], List[str], str]
        (numerical_cols, categorical_cols, target_col)
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from feature columns if present
    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN)
    if TARGET_COLUMN in categorical_cols:
        categorical_cols.remove(TARGET_COLUMN)
    
    return numerical_cols, categorical_cols, TARGET_COLUMN


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier and irrelevant columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with irrelevant columns removed
    """
    cols_to_drop = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    
    if cols_to_drop:
        print(f"✓ Dropping {len(cols_to_drop)} identifier columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    numerical_strategy: str = 'median',
    categorical_fill_value: str = 'unknown'
) -> pd.DataFrame:
    """
    Fill missing values for numerical and categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numerical_cols : List[str]
        List of numerical column names
    categorical_cols : List[str]
        List of categorical column names
    numerical_strategy : str
        Strategy for filling numerical missing values ('mean' or 'median')
    categorical_fill_value : str
        Value to fill categorical missing values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print(f"✓ Found {missing_info.sum()} missing values")
        print(f"  {missing_info[missing_info > 0].to_dict()}")
    
    # Fill numerical columns
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            if numerical_strategy == 'mean':
                fill_value = df[col].mean()
            else:  # median
                fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            print(f"  → Filled {col} (numeric) with {numerical_strategy}: {fill_value:.2f}")
    
    # Fill categorical columns
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(categorical_fill_value, inplace=True)
            print(f"  → Filled {col} (categorical) with '{categorical_fill_value}'")
    
    return df


def clean_data(
    df: pd.DataFrame,
    drop_identifiers: bool = True,
    numerical_strategy: str = 'median',
    categorical_fill_value: str = 'unknown'
) -> pd.DataFrame:
    """
    Main cleaning pipeline.
    
    Applies all cleaning steps in sequence:
    1. Drop irrelevant/identifier columns
    2. Handle missing values
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame
    drop_identifiers : bool
        Whether to drop identifier columns
    numerical_strategy : str
        Strategy for filling numerical missing values
    categorical_fill_value : str
        Value for filling categorical missing values
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    print("\n=== CLEANING DATA ===")
    
    # Step 1: Drop identifier columns
    if drop_identifiers:
        df = drop_irrelevant_columns(df)
    
    # Step 2: Identify column types
    numerical_cols, categorical_cols, target_col = identify_column_types(df)
    print(f"\nColumn types identified:")
    print(f"  - Numerical: {len(numerical_cols)}")
    print(f"  - Categorical: {len(categorical_cols)}")
    print(f"  - Target: {target_col}")
    
    # Step 3: Handle missing values
    df = handle_missing_values(
        df,
        numerical_cols,
        categorical_cols,
        numerical_strategy=numerical_strategy,
        categorical_fill_value=categorical_fill_value
    )
    
    print(f"\n✓ Cleaning complete. Final shape: {df.shape}")
    return df

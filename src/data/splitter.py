"""
Dataset splitting utilities.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df_or_X: pd.DataFrame,
    y: pd.Series | None = None,
    target_col: str = "IncidentGrade",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create stratified train/validation/test splits.

    Supports either:
    - split_data(df, target_col="IncidentGrade")
    - split_data(X, y, target_col="IncidentGrade")
    """
    if y is None:
        if target_col not in df_or_X.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        X = df_or_X.drop(columns=[target_col])
        y = df_or_X[target_col]
    else:
        X = df_or_X

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    if verbose:
        total = len(X)
        print("\n[SPLIT] Stratified split complete")
        print(f"[SPLIT] Train: {len(X_train)} ({len(X_train) / total:.1%})")
        print(f"[SPLIT] Val:   {len(X_val)} ({len(X_val) / total:.1%})")
        print(f"[SPLIT] Test:  {len(X_test)} ({len(X_test) / total:.1%})")

    return X_train, X_val, X_test, y_train, y_val, y_test

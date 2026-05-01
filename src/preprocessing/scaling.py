"""
Feature scaling helpers for preprocessing.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler


def _build_scaler(scale_method: str):
    method = scale_method.lower()
    if method == "quantile":
        return QuantileTransformer(
            output_distribution="normal",
            random_state=42,
            n_quantiles=1000,
            subsample=100000,
        )
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaling method: {scale_method}")


def scale_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scale_method: str = "quantile",
    verbose: bool = True,
    return_scaler: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit a scaler on training data and transform train/val/test splits.
    """
    scaler = _build_scaler(scale_method)

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    if verbose:
        print(f"\n[SCALING] Applied {scale_method} scaling")
        print(f"[SCALING] Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    if return_scaler:
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    return X_train_scaled, X_val_scaled, X_test_scaled

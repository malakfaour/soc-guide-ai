"""
Complete preprocessing pipeline orchestration.

Loads the training dataset, cleans features, applies frequency encoding,
creates stratified train/validation/test splits, optionally scales features,
and saves preprocessing artifacts for inference reproducibility.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.data.loader import load_train_data_only
from src.data.splitter import split_data
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import FrequencyEncoder, identify_categorical_columns
from src.preprocessing.scaling import scale_data
from src.utils.artifact_manager import save_artifacts


TARGET_COLUMN = "IncidentGrade"
TARGET_MAPPING = {
    "FalsePositive": 0,
    "BenignPositive": 1,
    "TruePositive": 2,
    "FP": 0,
    "BP": 1,
    "TP": 2,
}


def _encode_target(y: pd.Series) -> pd.Series:
    encoded = y.map(TARGET_MAPPING)
    if encoded.isna().any():
        unknown = sorted(y[encoded.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unknown target labels encountered: {unknown}")
    return encoded.astype(int)


def run_preprocessing(
    apply_scaling: bool = False,
    sample_size: int | None = 100000,
    verbose: bool = True,
    save_artifacts_flag: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Execute the end-to-end preprocessing pipeline for row-level triage data.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("PREPROCESSING PIPELINE: FULL EXECUTION")
        print("=" * 80)

    df_train = load_train_data_only(nrows=sample_size)
    df_clean = clean_data(df_train, target_col=TARGET_COLUMN, verbose=verbose)

    if TARGET_COLUMN not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found after cleaning.")

    y = _encode_target(df_clean[TARGET_COLUMN])
    X = df_clean.drop(columns=[TARGET_COLUMN]).copy()

    categorical_cols = identify_categorical_columns(X, exclude_columns=[])
    encoder = FrequencyEncoder()
    X_encoded = encoder.fit_transform(X, categorical_cols)
    encoders = encoder.encodings

    if verbose:
        print(f"\n[PIPELINE] Encoded {len(categorical_cols)} categorical columns")
        print(f"[PIPELINE] Feature matrix shape: {X_encoded.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_encoded,
        y,
        target_col=TARGET_COLUMN,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        verbose=verbose,
    )

    scaler = None
    if apply_scaling:
        X_train, X_val, X_test, scaler = scale_data(
            X_train,
            X_val,
            X_test,
            scale_method="quantile",
            verbose=verbose,
            return_scaler=True,
        )
    elif verbose:
        print("\n[PIPELINE] Skipping scaling (XGBoost/LightGBM mode)")

    if save_artifacts_flag:
        save_artifacts(
            encoders=encoders,
            target_mapping={k: v for k, v in TARGET_MAPPING.items() if len(k) > 2},
            scaler=scaler,
            verbose=verbose,
        )

    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    run_preprocessing(apply_scaling=False, sample_size=100000, verbose=True, save_artifacts_flag=True)

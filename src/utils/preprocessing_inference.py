"""
Preprocessing utilities for inference using saved artifacts.
Applies saved encoders, target mappings, and scalers to new data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

from utils.artifact_manager import load_artifacts
from preprocessing.cleaning import clean_data


def apply_preprocessing_artifacts(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    apply_scaling: bool = False,
    verbose: bool = True
):
    """
    Apply saved preprocessing artifacts to new data.

    Args:
        df: Raw DataFrame to preprocess
        target_col: If present in df, will be label-encoded using saved mapping
        apply_scaling: If True, apply saved scaler to features
        verbose: Print debug information

    Returns:
        (X_processed, y_encoded) if target_col provided, else X_processed
    """
    if verbose:
        print(f"\n[INFERENCE] Applying preprocessing artifacts to shape {df.shape}...")

    artifacts = load_artifacts(verbose=verbose)
    encoders = artifacts['encoders']
    target_mapping = artifacts['target_mapping']
    scaler = artifacts['scaler']

    df_processed = df.copy()

    # Step 1: Clean
    df_cleaned = clean_data(df_processed, target_col=target_col or 'IncidentGrade', verbose=verbose)

    # Step 2: Split features / target
    if target_col and target_col in df_cleaned.columns:
        features = df_cleaned.drop(columns=[target_col])
        target = df_cleaned[target_col]
    else:
        features = df_cleaned
        target = None

    # Step 3: Apply saved frequency encoders
    for col, freq_map in encoders.items():
        if col in features.columns:
            features[col] = features[col].map(freq_map)
            if features[col].isna().sum() > 0:
                median_freq = float(np.median(list(freq_map.values())))
                features[col] = features[col].fillna(median_freq)
                if verbose:
                    print(f"[INFERENCE] Unknown values in '{col}' filled with median freq {median_freq:.6f}")

    # Step 4: Encode target
    target_encoded = None
    if target is not None:
        target_encoded = target.map(target_mapping)
        if target_encoded.isna().sum() > 0 and verbose:
            print(f"[INFERENCE] WARNING: {target_encoded.isna().sum()} unknown target values")

    # Step 5: Optional scaling
    if apply_scaling and scaler is not None:
        features = pd.DataFrame(
            scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        if verbose:
            print(f"[INFERENCE] Scaling applied")

    if verbose:
        print(f"[INFERENCE] Output shape: {features.shape}")

    if target_encoded is not None:
        return features, target_encoded
    return features


def get_target_mapping(verbose: bool = False) -> Tuple[Dict, Dict]:
    """
    Returns (encoding_dict, decoding_dict) for the target variable.
    encoding_dict: label -> int
    decoding_dict: int -> label
    """
    artifacts = load_artifacts(verbose=verbose)
    target_mapping = artifacts['target_mapping']
    reverse_mapping = {v: k for k, v in target_mapping.items()}
    return target_mapping, reverse_mapping
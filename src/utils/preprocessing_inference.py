"""
Preprocessing utilities for inference using saved artifacts.
Applies saved encoders, target mappings, and scalers to new data.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
import numpy as np
from utils.artifact_manager import load_artifacts
from preprocessing.cleaning import clean_data


def apply_preprocessing_artifacts(df, target_col=None, apply_scaling=False, verbose=True):
    """
    Apply previously saved preprocessing artifacts to new data.
    
    Assumes the data has already been cleaned by the cleaning module.
    
    Args:
        df: Raw DataFrame to preprocess
        target_col: If present in df, will be label encoded using saved mapping
        apply_scaling: If True, apply saved scaler to features
        verbose: Print debug information
    
    Returns:
        Tuple of (X_processed, y_encoded) or (X_processed,) if no target_col
    
    Raises:
        FileNotFoundError: If required artifacts are missing
    """
    if verbose:
        print(f"\n[INFERENCE] Applying preprocessing artifacts...")
        print(f"[INFERENCE] Input shape: {df.shape}")
    
    # Load saved artifacts
    if verbose:
        print(f"[INFERENCE] Loading artifacts...")
    artifacts = load_artifacts(verbose=verbose)
    
    encoders = artifacts['encoders']
    target_mapping = artifacts['target_mapping']
    scaler = artifacts['scaler']
    
    # Create inverse mapping for target (numeric -> original label)
    # For inference, we might need both directions
    target_reverse_mapping = {v: k for k, v in target_mapping.items()}
    
    df_processed = df.copy()
    
    # Step 1: Clean data (handle missing values, etc.)
    if verbose:
        print(f"\n[INFERENCE] Step 1: Cleaning data...")
    df_cleaned = clean_data(df_processed, target_col=target_col, verbose=verbose)
    
    # Step 2: Apply frequency encoding to categorical features
    if verbose:
        print(f"\n[INFERENCE] Step 2: Applying saved encoders...")
    
    if target_col and target_col in df_cleaned.columns:
        features = df_cleaned.drop(columns=[target_col])
        target = df_cleaned[target_col]
    else:
        features = df_cleaned
        target = None
    
    # Apply saved encoders to categorical features
    for col, freq_map in encoders.items():
        if col in features.columns:
            if verbose:
                print(f"[INFERENCE]   Applying encoder for: {col}")
            
            # Map using saved frequency dictionary
            # For unknown values (not seen during training), use a default
            features[col] = features[col].map(freq_map)
            
            # Handle unknown values - fill with median frequency
            if features[col].isna().sum() > 0:
                median_freq = np.median(list(freq_map.values()))
                features[col].fillna(median_freq, inplace=True)
                if verbose:
                    print(f"[INFERENCE]     Unknown values filled with median frequency")
    
    # Step 3: Encode target if present
    if target is not None:
        if verbose:
            print(f"\n[INFERENCE] Step 3: Encoding target variable...")
        
        target_encoded = target.map(target_mapping)
        
        if target_encoded.isna().sum() > 0:
            if verbose:
                print(f"[INFERENCE]   WARNING: {target_encoded.isna().sum()} unknown target values")
        
        if verbose:
            print(f"[INFERENCE]   Target mapping applied")
    
    # Step 4: Optional scaling
    if apply_scaling and scaler is not None:
        if verbose:
            print(f"\n[INFERENCE] Step 4: Applying saved scaler...")
        
        X_scaled = pd.DataFrame(
            scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        features = X_scaled
        
        if verbose:
            print(f"[INFERENCE]   Scaling applied successfully")
    
    if verbose:
        print(f"\n[INFERENCE] Output shape: {features.shape}")
        if target is not None:
            print(f"[INFERENCE] Target shape: {target_encoded.shape}")
    
    # Return processed data
    if target is not None:
        return features, target_encoded
    else:
        return features


def get_target_mapping(verbose=True):
    """
    Load and return the saved target mapping.
    
    Useful for converting model predictions back to original labels.
    
    Returns:
        Tuple of (encoding_dict, decoding_dict)
        - encoding_dict: Maps string labels to numeric codes
        - decoding_dict: Maps numeric codes back to string labels
    """
    artifacts = load_artifacts(verbose=verbose)
    target_mapping = artifacts['target_mapping']
    reverse_mapping = {v: k for k, v in target_mapping.items()}
    
    return target_mapping, reverse_mapping


if __name__ == "__main__":
    # Test loading and using artifacts
    print("Testing artifact loading and inference preprocessing...")
    
    # Get target mappings for reference
    print("\nLoading target mappings...")
    encoding_map, decoding_map = get_target_mapping()
    
    print(f"\nEncoding mapping (label -> numeric):")
    for label, code in encoding_map.items():
        print(f"  {label}: {code}")
    
    print(f"\nDecoding mapping (numeric -> label):")
    for code, label in decoding_map.items():
        print(f"  {code}: {label}")
    
    print("\n✓ Artifact loading test completed successfully!")

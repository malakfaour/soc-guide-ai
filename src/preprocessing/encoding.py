"""
Minimal encoding module.
- Frequency encoding for categorical features ONLY
- Label mapping for target: TP->2, BP->1, FP->0
- NO target encoding, NO one-hot encoding
"""
import pandas as pd
import numpy as np


def encode_data(df, target_col='IncidentGrade', verbose=True, return_encoders=False):
    """
    Minimal categorical encoding.
    - Frequency encode categorical features
    - Label encode target
    - Exclude target from feature encoding
    
    Args:
        df: DataFrame with cleaning completed
        target_col: Name of target column
        verbose: Print debug information
        return_encoders: If True, return encoding dictionaries along with data
    
    Returns:
        If return_encoders=False:
            Encoded DataFrame with numeric features only
        If return_encoders=True:
            Tuple of (encoded_df, encoders_dict, target_mapping_dict)
    """
    if verbose:
        print(f"\n[ENCODING] Starting encoding...")
        print(f"[ENCODING] Input shape: {df.shape}")
    
    df_encoded = df.copy()
    encoders = {}  # Store encoding dictionaries for all categorical features
    
    # Separate target and features
    if target_col in df_encoded.columns:
        target = df_encoded[target_col]
        features = df_encoded.drop(columns=[target_col])
    else:
        features = df_encoded
        target = None
    
    # Identify categorical columns (excluding target)
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    
    if verbose:
        print(f"[ENCODING] Found {len(categorical_cols)} categorical columns:")
        if len(categorical_cols) > 0:
            print(f"  {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
    
    # Apply frequency encoding to categorical features
    for col in categorical_cols:
        if verbose:
            print(f"[ENCODING] Frequency encoding: {col}")
        
        # Get frequency of each value
        freq_map = features[col].value_counts(normalize=True).to_dict()
        
        # Store encoder for later use
        encoders[col] = freq_map
        
        # Map values to their frequencies
        features[col] = features[col].map(freq_map)
        
        if verbose:
            print(f"  Unique values: {len(freq_map)}, all converted to frequencies")
    
    # Encode target variable
    target_mapping = {
        'TruePositive': 2,
        'BenignPositive': 1,
        'FalsePositive': 0
    }
    
    if target is not None:
        if verbose:
            print(f"\n[ENCODING] Encoding target column: {target_col}")
        
        target_encoded = target.map(target_mapping)
        
        if verbose:
            print(f"  Target mapping: {target_mapping}")
            print(f"  Unique values after encoding: {sorted(target_encoded.unique())}")
    
    # Verify all features are numeric
    if verbose:
        print(f"\n[ENCODING] Verifying all features are numeric...")
    
    non_numeric = features.select_dtypes(exclude=['number']).columns.tolist()
    if len(non_numeric) > 0:
        print(f"[ENCODING] ✗ Non-numeric columns remain: {non_numeric}")
    else:
        if verbose:
            print(f"[ENCODING] ✓ All features are numeric!")
    
    # Reconstruct dataframe
    features_final = features.copy()
    if target is not None:
        features_final[target_col] = target_encoded
    
    if verbose:
        print(f"\n[ENCODING] Output shape: {features_final.shape}")
        print(f"[ENCODING] Output dtypes sample: {features_final.dtypes.value_counts().to_dict()}")
    
    if return_encoders:
        return features_final, encoders, target_mapping
    else:
        return features_final


if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'cat1': ['a', 'b', 'a', 'a', 'c'],
        'cat2': ['x', 'y', 'x', 'z', 'z'],
        'target': ['TP', 'BP', 'FP', 'TP', 'BP']
    })
    print("Input:\n", df)
    print("\nInput dtypes:\n", df.dtypes)
    
    df_encoded = encode_data(df, target_col='target')
    print("\nOutput:\n", df_encoded)
    print("\nOutput dtypes:\n", df_encoded.dtypes)

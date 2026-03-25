"""
Debug script: Test data encoding.
Validates that categorical features are encoded and all features are numeric.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
import numpy as np
from data.loader import load_train_data
from preprocessing.cleaning import clean_data
from preprocessing.encoding import encode_data


def debug_encoding():
    """Test data encoding."""
    print("\n" + "="*80)
    print("STEP 4: DEBUG DATA ENCODING")
    print("="*80)
    
    try:
        # Load and clean data
        print("\n[TEST] Loading and cleaning data...")
        df_train = load_train_data(nrows=50000)
        df_clean = clean_data(df_train, target_col='IncidentGrade', verbose=False)
        
        print(f"[TEST] After cleaning:")
        print(f"  Shape: {df_clean.shape}")
        numeric_before = df_clean.select_dtypes(include=['number']).shape[1]
        categorical_before = df_clean.select_dtypes(include=['object']).shape[1]
        print(f"  Numeric columns: {numeric_before}")
        print(f"  Categorical columns: {categorical_before}")
        
        # Encode data
        print(f"\n[TEST] Running encoding...")
        df_encoded = encode_data(df_clean, target_col='IncidentGrade', verbose=True)
        
        print(f"\n[TEST] After encoding:")
        print(f"  Shape: {df_encoded.shape}")
        numeric_after = df_encoded.select_dtypes(include=['number']).shape[1]
        categorical_after = df_encoded.select_dtypes(include=['object']).shape[1]
        print(f"  Numeric columns: {numeric_after}")
        print(f"  Categorical columns: {categorical_after}")
        
        # Verify all numeric
        if categorical_after == 0:
            print(f"[TEST] ✓ All features are numeric!")
        else:
            print(f"[TEST] ✗ Still have categorical columns: {df_encoded.select_dtypes(include=['object']).columns.tolist()}")
        
        # Check for infinite values
        print(f"\n[TEST] Checking for infinite values...")
        inf_count = np.isinf(df_encoded.select_dtypes(include=['number'])).sum().sum()
        print(f"  Infinite values: {inf_count}")
        if inf_count == 0:
            print(f"[TEST] ✓ No infinite values!")
        
        # Check target encoding
        print(f"\n[TEST] Target encoding verification:")
        target = df_encoded['IncidentGrade']
        unique_targets = sorted(target.unique())
        target_counts = target.value_counts().sort_index()
        print(f"  Unique values: {unique_targets}")
        print(f"  Value counts:")
        for val, count in target_counts.items():
            print(f"    {val}: {count}")
        
        # Verify mapping is correct
        if set(unique_targets) == {0, 1, 2}:
            print(f"[TEST] ✓ Target correctly mapped (0=FP, 1=BP, 2=TP)")
        else:
            print(f"[TEST] ✗ Target values unexpected: {unique_targets}")
        
        print("\n" + "="*80)
        print("✓ STEP 4 COMPLETE: Data encoding validated")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\n✗ STEP 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_encoding()
    sys.exit(0 if success else 1)

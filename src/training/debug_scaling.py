"""
Debug script: Test data scaling.
Validates optional scaling (primarily for TabNet).
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
import numpy as np
from data.loader import load_train_data
from preprocessing.cleaning import clean_data
from preprocessing.encoding import encode_data
from data.splitter import split_data
from preprocessing.scaling import scale_data


def debug_scaling():
    """Test data scaling."""
    print("\n" + "="*80)
    print("STEP 6: DEBUG DATA SCALING")
    print("="*80)
    
    try:
        # Load, clean, encode, and split data
        print("\n[TEST] Preparing data (load, clean, encode, split)...")
        df_train = load_train_data(nrows=50000)
        df_clean = clean_data(df_train, target_col='IncidentGrade', verbose=False)
        df_encoded = encode_data(df_clean, target_col='IncidentGrade', verbose=False)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df_encoded, 
            target_col='IncidentGrade',
            verbose=False
        )
        
        print(f"[TEST] Data ready:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape:   {X_val.shape}")
        print(f"  X_test shape:  {X_test.shape}")
        
        # Test 1: No scaling (default for XGBoost/LightGBM)
        print(f"\n[TEST] Test 1: No scaling (XGBoost/LightGBM mode)")
        X_train_ns, X_val_ns, X_test_ns = scale_data(
            X_train, X_val, X_test,
            scale_method=None,
            verbose=True
        )
        
        print(f"[TEST] ✓ No scaling completed")
        print(f"  Shapes unchanged: {X_train_ns.shape == X_train.shape}")
        
        # Test 2: Quantile scaling (TabNet mode)
        print(f"\n[TEST] Test 2: Quantile scaling (TabNet mode)")
        X_train_qs, X_val_qs, X_test_qs = scale_data(
            X_train, X_val, X_test,
            scale_method='quantile',
            verbose=True
        )
        
        print(f"[TEST] ✓ Quantile scaling completed")
        print(f"  X_train shape: {X_train_qs.shape}")
        print(f"  X_train mean:  {X_train_qs.mean().mean():.6f}")
        print(f"  X_train std:   {X_train_qs.std().mean():.6f}")
        
        # Verify no NaN in scaled data
        print(f"\n[TEST] Verifying scaled data quality:")
        nan_count = X_train_qs.isnull().sum().sum()
        inf_count = np.isinf(X_train_qs.select_dtypes(include=['number'])).sum().sum()
        print(f"  NaN values: {nan_count}")
        print(f"  Infinite values: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print(f"[TEST] ✓ Scaled data is clean!")
        else:
            print(f"[TEST] ✗ Scaled data has issues!")
        
        print("\n" + "="*80)
        print("✓ STEP 6 COMPLETE: Data scaling validated")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\n✗ STEP 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_scaling()
    sys.exit(0 if success else 1)

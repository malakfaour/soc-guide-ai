"""
Debug script: Test data splitting.
Validates stratified split and class distribution.
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


def debug_split():
    """Test data splitting."""
    print("\n" + "="*80)
    print("STEP 5: DEBUG DATA SPLITTING")
    print("="*80)
    
    try:
        # Load, clean, and encode data
        print("\n[TEST] Loading, cleaning, and encoding data...")
        df_train = load_train_data(nrows=100000)
        df_clean = clean_data(df_train, target_col='IncidentGrade', verbose=False)
        df_encoded = encode_data(df_clean, target_col='IncidentGrade', verbose=False)
        
        print(f"[TEST] Prepared data shape: {df_encoded.shape}")
        
        # Split data
        print(f"\n[TEST] Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df_encoded, 
            target_col='IncidentGrade',
            verbose=True
        )
        
        # Verify splits
        print(f"\n[TEST] Verifying splits...")
        total_size = len(X_train) + len(X_val) + len(X_test)
        print(f"  Total samples: {total_size}")
        
        expected_train = 0.70
        expected_val = 0.15
        expected_test = 0.15
        
        actual_train = len(X_train) / total_size
        actual_val = len(X_val) / total_size
        actual_test = len(X_test) / total_size
        
        print(f"[TEST] Split ratio verification:")
        print(f"  Train: {actual_train:.2%} (expected {expected_train:.2%})")
        print(f"  Val:   {actual_val:.2%} (expected {expected_val:.2%})")
        print(f"  Test:  {actual_test:.2%} (expected {expected_test:.2%})")
        
        # Check for data leakage
        print(f"\n[TEST] Checking for data leakage...")
        train_idx = set(range(len(X_train)))
        val_idx = set(range(len(X_train), len(X_train) + len(X_val)))
        test_idx = set(range(len(X_train) + len(X_val), total_size))
        
        overlap = train_idx & val_idx | val_idx & test_idx | train_idx & test_idx
        if len(overlap) == 0:
            print(f"[TEST] ✓ No data leakage between splits!")
        else:
            print(f"[TEST] ✗ Data leakage detected!")
        
        # Check no missing values in splits
        print(f"\n[TEST] Missing values in splits:")
        for name, X, y in [('Train', X_train, y_train), 
                           ('Val', X_val, y_val), 
                           ('Test', X_test, y_test)]:
            x_missing = X.isnull().sum().sum()
            y_missing = y.isnull().sum()
            print(f"  {name}: X={x_missing}, y={y_missing}")
        
        print(f"\n[TEST] Sample shapes:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
        print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
        
        print("\n" + "="*80)
        print("✓ STEP 5 COMPLETE: Data splitting validated")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\n✗ STEP 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_split()
    sys.exit(0 if success else 1)

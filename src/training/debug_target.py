"""
Debug script: Validate target column.
Check that target exists, has expected values, and understand the class distribution.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
from data.loader import load_train_data


def debug_target():
    """Test target column validation."""
    print("\n" + "="*80)
    print("STEP 2: DEBUG TARGET COLUMN")
    print("="*80)
    
    try:
        # Load training data
        print("\n[TEST] Loading training data...")
        df_train = load_train_data(nrows=50000)
        
        # Look for target column
        target_col = 'IncidentGrade'
        
        if target_col not in df_train.columns:
            print(f"[TEST] ✗ Target column '{target_col}' not found!")
            print(f"[TEST] Available columns: {list(df_train.columns)}")
            return False
        
        print(f"[TEST] ✓ Found target column: {target_col}")
        
        # Check for missing values
        null_count = df_train[target_col].isnull().sum()
        print(f"\n[TEST] Missing values in target: {null_count} ({100*null_count/len(df_train):.2f}%)")
        
        # Get unique values
        unique_vals = df_train[target_col].unique()
        print(f"[TEST] Unique values (first 20): {unique_vals[:20]}")
        print(f"[TEST] Total unique values: {len(unique_vals)}")
        
        # Get value counts (excluding NaN)
        print(f"\n[TEST] Class distribution (excluding NaN):")
        value_counts = df_train[target_col].value_counts()
        for val, count in value_counts.items():
            pct = 100 * count / (len(df_train) - null_count)
            print(f"  {val}: {count:,} ({pct:.2f}%)")
        
        # Verify expected IncidentGrade values
        expected = {'TP', 'BP', 'FP'}
        actual = set(df_train[target_col].dropna().unique())
        
        print(f"\n[TEST] Expected grades: {expected}")
        print(f"[TEST] Actual grades (unique): {actual}")
        
        if expected == actual:
            print(f"[TEST] [OK] Target column has expected values")
        else:
            print(f"[TEST] [ERROR] Target column values mismatch!")
            print(f"  Missing: {expected - actual}")
            print(f"  Extra: {actual - expected}")
        
        print("\n" + "="*80)
        print("✓ STEP 2 COMPLETE: Target column validated")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] STEP 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_target()
    sys.exit(0 if success else 1)

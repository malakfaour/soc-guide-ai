"""
Debug script: Test data cleaning.
Validates that missing values are handled correctly.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
from data.loader import load_train_data
from preprocessing.cleaning import clean_data


def debug_cleaning():
    """Test data cleaning."""
    print("\n" + "="*80)
    print("STEP 3: DEBUG DATA CLEANING")
    print("="*80)
    
    try:
        # Load training data
        print("\n[TEST] Loading training data (first 50000 rows)...")
        df_train = load_train_data(nrows=50000)
        
        print(f"[TEST] Input shape: {df_train.shape}")
        print(f"[TEST] Input missing values before cleaning:")
        missing_before = df_train.isnull().sum()
        print(f"  Total NaN cells: {missing_before.sum()}")
        print(f"  Columns with missing: {missing_before[missing_before > 0].index.tolist()}")
        
        # Clean data
        print("\n[TEST] Running data cleaning...")
        df_clean = clean_data(df_train, target_col='IncidentGrade', verbose=True)
        
        print(f"\n[TEST] Output shape: {df_clean.shape}")
        print(f"[TEST] Output missing values after cleaning:")
        missing_after = df_clean.isnull().sum()
        print(f"  Total NaN cells: {missing_after.sum()}")
        
        # Verify no missing values in features
        if 'IncidentGrade' in df_clean.columns:
            features_missing = df_clean.drop(columns=['IncidentGrade']).isnull().sum().sum()
        else:
            features_missing = missing_after.sum()
        
        if features_missing == 0:
            print(f"[TEST] [OK] All feature NaN values handled!")
        else:
            print(f"[TEST] [ERROR] Still have missing values in features: {features_missing}")
        
        # Check data types
        print(f"\n[TEST] Data types:")
        numeric_count = df_clean.select_dtypes(include=['number']).shape[1]
        categorical_count = df_clean.select_dtypes(include=['object']).shape[1]
        print(f"  Numeric columns: {numeric_count}")
        print(f"  Categorical columns: {categorical_count}")
        
        print("\n" + "="*80)
        print("✓ STEP 3 COMPLETE: Data cleaning validated")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] STEP 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_cleaning()
    sys.exit(0 if success else 1)

"""
Debug script: Test dataset loading only.
This script validates that data can be loaded correctly.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from data.loader import load_train_data, load_test_data


def debug_loader():
    """Test loading with minimal dataset sample."""
    print("\n" + "="*80)
    print("STEP 1: DEBUG DATASET LOADING")
    print("="*80)
    
    try:
        # Load training data sample
        print("\n[TEST] Loading training data sample (first 10,000 rows)...")
        df_train = load_train_data(nrows=10000)
        
        print("\n[TEST] Training shape:", df_train.shape)
        print("[TEST] Training columns:", list(df_train.columns))
        
        print("\n[TEST] Training data types:")
        print(df_train.dtypes)
        
        print("\n[TEST] Training missing values:")
        missing = df_train.isnull().sum()
        print(missing[missing > 0])
        
        print("\n[TEST] Training sample:")
        print(df_train.head(3))
        
        # Check for target column
        target_candidates = ['IncidentGrade', 'Grade', 'Target', 'target', 'label', 'Label']
        target_col = None
        for col in target_candidates:
            if col in df_train.columns:
                target_col = col
                break
        
        if target_col:
        print(f"[TEST] [OK] Found target column: {target_col}")
        else:
            print(f"[TEST] [ERROR] Target column not found!")
            print(f"[TEST] Available columns: {list(df_train.columns)}")
        
        print("\n" + "="*80)
        print("✓ STEP 1 COMPLETE: Dataset loading works")
        print("="*80)
        
        return df_train
    
    except Exception as e:
        print(f"\n✗ STEP 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    df = debug_loader()

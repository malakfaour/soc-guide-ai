"""
Final validation script: Test complete preprocessing pipeline.
Validates end-to-end preprocessing and saves processed datasets.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
import numpy as np
from preprocessing.pipeline import run_preprocessing
from utils.versioning import save_dataset_with_version, list_versions, get_current_version


def validate_pipeline():
    """Run complete pipeline and validate outputs."""
    print("\n" + "="*80)
    print("STEP 8: FINAL PIPELINE VALIDATION")
    print("="*80)
    
    try:
        # Run full pipeline
        print("\n[TEST] Running complete preprocessing pipeline (using 100K sample)...")
        X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing(
            apply_scaling=False,  # Using XGBoost/LightGBM mode (no scaling)
            sample_size=100000,   # Use 100K rows for efficiency
            verbose=True
        )
        
        # === Validation 1: Shapes ===
        print("\n[TEST] Validation 1: Data Shapes")
        print(f"  X_train: {X_train.shape} (expected: [N, features])")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_val:   {y_val.shape}")
        print(f"  y_test:  {y_test.shape}")
        
        # Check shapes are consistent
        assert X_train.shape[0] == len(y_train), "X_train and y_train length mismatch"
        assert X_val.shape[0] == len(y_val), "X_val and y_val length mismatch"
        assert X_test.shape[0] == len(y_test), "X_test and y_test length mismatch"
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count mismatch"
        print("  ✓ Shapes validated")
        
        # === Validation 2: Target Distribution ===
        print("\n[TEST] Validation 2: Target Distribution")
        unique_train = np.unique(y_train, return_counts=True)
        print(f"  Training set classes:")
        for val, count in zip(unique_train[0], unique_train[1]):
            pct = 100 * count / len(y_train)
            print(f"    Class {int(val)}: {count:,} ({pct:.2f}%)")
        
        # Check all 3 classes exist
        if len(np.unique(y_train)) == 3:
            print("  ✓ All 3 classes present (0=FP, 1=BP, 2=TP)")
        else:
            print(f"  ✗ Expected 3 classes, found {len(np.unique(y_train))}")
        
        # === Validation 3: Missing Values ===
        print("\n[TEST] Validation 3: Missing Values")
        x_train_missing = X_train.isnull().sum().sum()
        x_val_missing = X_val.isnull().sum().sum()
        x_test_missing = X_test.isnull().sum().sum()
        y_train_missing = y_train.isnull().sum() if hasattr(y_train, 'isnull') else 0
        y_val_missing = y_val.isnull().sum() if hasattr(y_val, 'isnull') else 0
        
        print(f"  X_train NaN: {x_train_missing}")
        print(f"  X_val NaN:   {x_val_missing}")
        print(f"  X_test NaN:  {x_test_missing}")
        print(f"  y_train NaN: {y_train_missing}")
        print(f"  y_val NaN:   {y_val_missing}")
        
        if x_train_missing + x_val_missing + x_test_missing + y_train_missing + y_val_missing == 0:
            print("  ✓ No missing values!")
        else:
            print("  ✗ Missing values detected!")
        
        # === Validation 4: Data Types ===
        print("\n[TEST] Validation 4: Data Types")
        numeric_cols = X_train.select_dtypes(include=['number']).shape[1]
        categorical_cols = X_train.select_dtypes(include=['object']).shape[1]
        print(f"  Numeric columns: {numeric_cols}")
        print(f"  Categorical columns: {categorical_cols}")
        
        if categorical_cols == 0:
            print("  ✓ All features are numeric!")
        else:
            print(f"  ✗ Found {categorical_cols} categorical columns")
        
        # === Validation 5: Infinite Values ===
        print("\n[TEST] Validation 5: Infinite Values")
        inf_train = np.isinf(X_train.select_dtypes(include=['number'])).sum().sum()
        inf_val = np.isinf(X_val.select_dtypes(include=['number'])).sum().sum()
        inf_test = np.isinf(X_test.select_dtypes(include=['number'])).sum().sum()
        
        print(f"  X_train infinities: {inf_train}")
        print(f"  X_val infinities:   {inf_val}")
        print(f"  X_test infinities:  {inf_test}")
        
        if inf_train + inf_val + inf_test == 0:
            print("  ✓ No infinite values!")
        else:
            print("  ✗ Infinite values detected!")
        
        # === Validation 6: Empty Check ===
        print("\n[TEST] Validation 6: Non-Empty Datasets")
        if len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0:
            print(f"  ✓ All datasets non-empty")
        else:
            print(f"  ✗ Empty dataset detected!")
        
        # === Validation 7: Split Ratios ===
        print("\n[TEST] Validation 7: Split Ratios")
        total = len(X_train) + len(X_val) + len(X_test)
        train_ratio = len(X_train) / total
        val_ratio = len(X_val) / total
        test_ratio = len(X_test) / total
        
        print(f"  Train: {train_ratio:.1%} (expected ~70%)")
        print(f"  Val:   {val_ratio:.1%} (expected ~15%)")
        print(f"  Test:  {test_ratio:.1%} (expected ~15%)")
        
        if 0.65 < train_ratio < 0.75 and 0.10 < val_ratio < 0.20 and 0.10 < test_ratio < 0.20:
            print("  ✓ Split ratios are correct")
        else:
            print("  ⚠ Split ratios differ from expected")
        
        # === Save Processed Data with Versioning ===
        print("\n[TEST] Saving processed data with versioning...")
        
        # List existing versions
        existing_versions = list_versions(verbose=True)
        
        # Get current version
        current = get_current_version(verbose=True)
        
        # Save with versioning - will auto-increment version
        try:
            version = save_dataset_with_version(
                X_train, X_val, X_test, 
                y_train, y_val, y_test,
                version=None,  # Auto-determine next version
                force=False,   # Don't overwrite without confirmation
                verbose=True
            )
            
            if version is None:
                print("\n[ERROR] Dataset save cancelled")
                return False
            
        except ValueError as e:
            print(f"\n[ERROR] {str(e)}")
            return False
        
        # === Summary ===
        print("\n" + "="*80)
        print("✓ PIPELINE VALIDATION COMPLETE")
        print("="*80)
        print(f"\nFinal Metrics:")
        print(f"  Total samples: {total:,}")
        print(f"  Number of features: {X_train.shape[1]}")
        print(f"  Number of classes: {len(np.unique(y_train))}")
        print(f"  Dataset version: {version}")
        print(f"  Processed data saved to: data/processed/{version}/")
        print("\n✓ Ready for model training!")
        
        return True
    
    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_pipeline()
    sys.exit(0 if success else 1)

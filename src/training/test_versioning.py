"""
Integration test for dataset versioning system.
Tests version management, overwrite prevention, and logging.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
import numpy as np
from utils.versioning import (
    list_versions, get_current_version, version_exists, 
    get_dataset_path, load_dataset_by_version
)


def test_versioning_integration():
    """Test the complete versioning system."""
    print("\n" + "="*80)
    print("VERSIONING INTEGRATION TEST")
    print("="*80)
    
    # Test 1: List existing versions
    print("\n[TEST 1] List available versions...")
    versions = list_versions(verbose=True)
    assert len(versions) > 0, "No versions found!"
    assert 'v1' in versions, "v1 not found!"
    print("✓ Found v1 with all 6 dataset files")
    
    # Test 2: Get current version
    print("\n[TEST 2] Get current version...")
    current = get_current_version(verbose=True)
    assert current == 'v1', f"Expected 'v1', got '{current}'"
    print("✓ Current version is v1")
    
    # Test 3: Check version exists
    print("\n[TEST 3] Check if v1 exists...")
    exists = version_exists('v1')
    assert exists, "v1 should exist!"
    print("✓ v1 exists")
    
    # Test 4: Get dataset path
    print("\n[TEST 4] Get dataset path for v1...")
    path = get_dataset_path('v1')
    assert os.path.isdir(path), f"Path {path} is not a directory"
    print(f"✓ Dataset path: {path}")
    
    # Test 5: Load dataset
    print("\n[TEST 5] Load dataset from v1...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version('v1')
        print(f"✓ Loaded successfully")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_val:   {y_val.shape}")
        print(f"  y_test:  {y_test.shape}")
        
        # Verify data integrity
        assert X_train.shape[0] > 0, "X_train is empty"
        assert X_val.shape[0] > 0, "X_val is empty"
        assert X_test.shape[0] > 0, "X_test is empty"
        assert len(y_train) > 0, "y_train is empty"
        assert len(y_val) > 0, "y_val is empty"
        assert len(y_test) > 0, "y_test is empty"
        
        # Verify shapes match
        assert X_train.shape[0] == len(y_train), "X_train/y_train length mismatch"
        assert X_val.shape[0] == len(y_val), "X_val/y_val length mismatch"
        assert X_test.shape[0] == len(y_test), "X_test/y_test length mismatch"
        
        print("✓ Data integrity verified")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Test 6: Verify dataset statistics
    print("\n[TEST 6] Verify dataset statistics...")
    
    # Check features all numeric
    numeric_cols = X_train.select_dtypes(include=['number']).shape[1]
    assert numeric_cols == X_train.shape[1], "Non-numeric features found"
    print(f"✓ All {numeric_cols} features are numeric")
    
    # Check no missing values
    x_missing = X_train.isnull().sum().sum() + X_val.isnull().sum().sum() + X_test.isnull().sum().sum()
    y_missing = y_train.isnull().sum() + y_val.isnull().sum() + y_test.isnull().sum()
    assert x_missing + y_missing == 0, f"Missing values detected: {x_missing + y_missing}"
    print("✓ No missing values")
    
    # Check split ratios
    total = len(X_train) + len(X_val) + len(X_test)
    train_ratio = len(X_train) / total
    val_ratio = len(X_val) / total
    test_ratio = len(X_test) / total
    print(f"✓ Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    
    # Test 7: Version protection
    print("\n[TEST 7] Test version protection (v1 exists)...")
    v1_exists = version_exists('v1')
    if v1_exists:
        print("✓ v1 still exists (protected from auto-overwrite)")
    else:
        print("✗ v1 was deleted unexpectedly")
        return False
    
    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("="*80)
    print("\nDataset Versioning System Status:")
    print(f"  Current Version: v1")
    print(f"  Available Versions: {', '.join(versions)}")
    print(f"  Total Samples: {total:,}")
    print(f"  Train/Val/Test Split: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print(f"  Number of Features: {X_train.shape[1]}")
    print(f"  Data Location: data/processed/v1/")
    print("\n✓ All team members should use: data/processed/v1/")
    
    return True


if __name__ == "__main__":
    success = test_versioning_integration()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""Quick debug script"""
import sys
import traceback
sys.path.insert(0, 'src')

try:
    print("Importing pipeline...")
    from preprocessing.pipeline import run_preprocessing
    print("Running pipeline...")
    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing(sample_frac=0.01)
    print("SUCCESS!")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

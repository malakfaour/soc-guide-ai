"""
Complete preprocessing pipeline orchestration.
Combines all steps: load, clean, encode, split, optional scale.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
from data.loader import load_train_data, load_test_data
from preprocessing.cleaning import clean_data
from preprocessing.encoding import encode_data
from data.splitter import split_data
from preprocessing.scaling import scale_data
from utils.artifact_manager import save_artifacts


def run_preprocessing(apply_scaling=False, sample_size=100000, verbose=True, save_artifacts_flag=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        apply_scaling: If True, apply quantile scaling for TabNet. 
                      Default False for XGBoost/LightGBM.
        sample_size: Number of rows to load (for memory efficiency). Default 100000.
                    Set to None to load full dataset.
        verbose: Print debug information at each step
        save_artifacts_flag: If True, save preprocessing artifacts for reproducibility
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    
    Raises:
        Exception: If any step fails
    """
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE: FULL EXECUTION")
    print("="*80)
    
    try:
        # Step 1: Load data
        if verbose:
            print("\n[PIPELINE] Step 1: Loading data...")
        df_train = load_train_data(nrows=sample_size)  # Load with sample size
        
        if verbose:
            print(f"[PIPELINE] Train shape: {df_train.shape}")
        
        # Step 2: Clean data
        if verbose:
            print(f"\n[PIPELINE] Step 2: Cleaning data...")
        df_clean = clean_data(df_train, target_col='IncidentGrade', verbose=verbose)
        
        # Step 3: Encode data
        if verbose:
            print(f"\n[PIPELINE] Step 3: Encoding data...")
        df_encoded, encoders, target_mapping = encode_data(
            df_clean, 
            target_col='IncidentGrade', 
            verbose=verbose,
            return_encoders=True
        )
        
        # Step 4: Split data
        if verbose:
            print(f"\n[PIPELINE] Step 4: Splitting data...")
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df_encoded,
            target_col='IncidentGrade',
            verbose=verbose
        )
        
        # Step 5: Optional scaling
        scaler = None
        if apply_scaling:
            if verbose:
                print(f"\n[PIPELINE] Step 5: Scaling data (QuantileTransformer)...")
            X_train, X_val, X_test, scaler = scale_data(
                X_train, X_val, X_test,
                scale_method='quantile',
                verbose=verbose,
                return_scaler=True
            )
        else:
            if verbose:
                print(f"\n[PIPELINE] Step 5: Skipping scaling (XGBoost/LightGBM mode)")
        
        # Step 6: Save artifacts (optional)
        if save_artifacts_flag:
            if verbose:
                print(f"\n[PIPELINE] Step 6: Saving preprocessing artifacts...")
            save_artifacts(encoders, target_mapping, scaler=scaler, verbose=verbose)
        
        if verbose:
            print("\n" + "="*80)
            print("✓ PIPELINE COMPLETE")
            print("="*80)
            print(f"Final shapes:")
            print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
            print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED at a step: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Test execution
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing(
            apply_scaling=False,
            verbose=True
        )
        print("\n✓ Pipeline execution successful!")
    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {e}")
        sys.exit(1)

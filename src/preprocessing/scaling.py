"""
Data scaling module.
- Optional QuantileTransformer for numeric features
- NOT applied for XGBoost and LightGBM
- Only for TabNet and advanced models
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer


def scale_data(X_train, X_val, X_test, scale_method=None, verbose=True, return_scaler=False):
    """
    Optional scaling of numeric features.
    
    Args:
        X_train: Training features (numeric only)
        X_val: Validation features
        X_test: Test features
        scale_method: 'quantile' or None (default None = no scaling)
        verbose: Print debug information
        return_scaler: If True, also return the scaler object
    
    Returns:
        If return_scaler=False:
            X_train_scaled, X_val_scaled, X_test_scaled
        If return_scaler=True:
            X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    if verbose:
        print(f"\n[SCALING] Scaling configuration:")
        print(f"  Method: {scale_method if scale_method else 'None (no scaling)'}")
    
    if scale_method is None:
        if verbose:
            print(f"[SCALING] Skipping scaling (optimal for XGBoost/LightGBM)")
        if return_scaler:
            return X_train, X_val, X_test, None
        return X_train, X_val, X_test
    
    if scale_method == 'quantile':
        if verbose:
            print(f"[SCALING] Applying QuantileTransformer(output='normal')...")
        
        scaler = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(1000, len(X_train)),
            subsample=100000,
            random_state=42
        )
        
        # Fit on training data only
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform validation and test
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        if verbose:
            print(f"[SCALING] ✓ Scaling completed")
            print(f"[SCALING] Train stats after scaling:")
            print(f"  Mean: {X_train_scaled.mean().mean():.6f}")
            print(f"  Std:  {X_train_scaled.std().mean():.6f}")
        
        if return_scaler:
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    else:
        raise ValueError(f"Unknown scaling method: {scale_method}")


if __name__ == "__main__":
    # Test
    X_train = pd.DataFrame(np.random.randn(1000, 5), columns=[f'feat{i}' for i in range(5)])
    X_val = pd.DataFrame(np.random.randn(200, 5), columns=[f'feat{i}' for i in range(5)])
    X_test = pd.DataFrame(np.random.randn(200, 5), columns=[f'feat{i}' for i in range(5)])
    
    # Test without scaling
    X_train_s, X_val_s, X_test_s = scale_data(X_train, X_val, X_test, scale_method=None)
    print("No scaling test passed!")
    
    # Test with quantile scaling
    X_train_s, X_val_s, X_test_s = scale_data(X_train, X_val, X_test, scale_method='quantile')
    print("Quantile scaling test passed!")

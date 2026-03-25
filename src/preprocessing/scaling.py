"""
<<<<<<< HEAD
Scaling module for GUIDE dataset.

Applies QuantileTransformer for numerical feature scaling.
Required for certain models like TabNet.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from typing import List, Dict, Tuple, Optional


class ScalingPipeline:
    """Handles scaling of numerical features."""
    
    def __init__(self, method: str = 'quantile', output_dist: str = 'normal'):
        """
        Initialize scaler.
        
        Parameters
        ----------
        method : str
            Scaling method ('quantile', 'standard', 'minmax')
        output_dist : str
            Output distribution for QuantileTransformer ('normal' or 'uniform')
        """
        self.method = method
        self.output_dist = output_dist
        self.scaler = None
        self.numerical_cols = None
        self.is_fitted = False
    
    def _create_scaler(self):
        """Create appropriate scaler based on method."""
        if self.method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(
                output_distribution=self.output_dist,
                random_state=42,
                n_quantiles=1000,
                subsample=100000  # For large datasets
            )
        elif self.method == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None
    ) -> 'ScalingPipeline':
        """
        Fit scaler on training data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        numerical_cols : List[str], optional
            List of numerical column names. If None, auto-detects.
        
        Returns
        -------
        ScalingPipeline
            Fitted pipeline (self)
        """
        # Identify numerical columns if not provided
        if numerical_cols is None:
            numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        self.numerical_cols = numerical_cols
        
        if not numerical_cols:
            print("⚠ No numerical columns found to scale")
            return self
        
        # Create and fit scaler
        self._create_scaler()
        self.scaler.fit(X_train[numerical_cols])
        self.is_fitted = True
        
        print(f"✓ Scaler fitted on {len(numerical_cols)} numerical columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
        
        Returns
        -------
        pd.DataFrame
            Scaled DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_scaled = X.copy()
        
        if self.numerical_cols:
            scaled_values = self.scaler.transform(X[self.numerical_cols])
            X_scaled[self.numerical_cols] = scaled_values
        
        return X_scaled
    
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X_train, numerical_cols).transform(X_train)
    
    def get_config(self) -> Dict:
        """Get scaler configuration for reproducibility."""
        return {
            'method': self.method,
            'output_dist': self.output_dist,
            'numerical_cols': self.numerical_cols,
            'scaler_type': type(self.scaler).__name__
        }


def identify_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify numerical columns in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    List[str]
        List of numerical column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = 'quantile',
    output_distribution: str = 'normal'
) -> Tuple[pd.DataFrame, pd.DataFrame, ScalingPipeline]:
    """
    Main scaling pipeline.
    
    Fits scaler on training data, applies to both train and test.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    method : str
        Scaling method ('quantile', 'standard', 'minmax')
    output_distribution : str
        Output distribution for QuantileTransformer
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, ScalingPipeline]
        (X_train_scaled, X_test_scaled, scaler)
    """
    print(f"\n=== SCALING FEATURES ({method}) ===")
    
    # Identify numerical columns
    numerical_cols = identify_numerical_columns(X_train)
    print(f"Found {len(numerical_cols)} numerical columns to scale")
    
    if not numerical_cols:
        print("No numerical columns to scale, returning data as-is")
        return X_train, X_test, None
    
    # Create and fit scaler
    scaler = ScalingPipeline(method=method, output_dist=output_distribution)
    scaler.fit(X_train, numerical_cols)
    
    # Transform both train and test
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Scaling complete")
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def get_scaling_stats(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None
) -> Dict:
    """
    Compute scaling statistics for documentation.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    numerical_cols : List[str], optional
        Columns to compute stats for
    
    Returns
    -------
    Dict
        Scaling statistics
    """
    if numerical_cols is None:
        numerical_cols = identify_numerical_columns(X_train)
    
    stats = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'numerical_cols': numerical_cols,
        'train_stats': {},
        'test_stats': {}
    }
    
    for col in numerical_cols:
        if col in X_train.columns:
            stats['train_stats'][col] = {
                'mean': X_train[col].mean(),
                'std': X_train[col].std(),
                'min': X_train[col].min(),
                'max': X_train[col].max()
            }
            stats['test_stats'][col] = {
                'mean': X_test[col].mean(),
                'std': X_test[col].std(),
                'min': X_test[col].min(),
                'max': X_test[col].max()
            }
    
    return stats
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
>>>>>>> f80310cf3a2ef8383f0d05dcca483e9bcc64aa12

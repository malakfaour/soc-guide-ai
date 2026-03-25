"""
Encoding module for GUIDE dataset.

Implements frequency and target encoding for categorical features.
LabelEncoder applied only to target variable.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional


# High-cardinality categorical features (adjust based on actual data)
HIGH_CARDINALITY_FEATURES = [
    'DetectorId',
    'OrgId',
    'SourceIp',
    'DestinationIp',
    'HashId'
]

TARGET_COLUMN = 'IncidentGrade'


class FrequencyEncoder:
    """Encode categorical features using frequency encoding."""
    
    def __init__(self):
        self.encodings: Dict[str, Dict] = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'FrequencyEncoder':
        """
        Fit frequency encoder on training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame
        columns : List[str]
            Columns to encode
        
        Returns
        -------
        FrequencyEncoder
            Fitted encoder (self)
        """
        for col in columns:
            if col in df.columns:
                # Calculate frequency (value counts normalized)
                freq = df[col].value_counts(normalize=True).to_dict()
                self.encodings[col] = freq
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted frequency encoding.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        df_encoded = df.copy()
        
        for col in self.encodings.keys():
            if col in df_encoded.columns:
                # Map values to their frequencies, use 0 for unseen values
                df_encoded[col] = df_encoded[col].map(
                    self.encodings[col]
                ).fillna(0)
        
        return df_encoded
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, columns).transform(df)


class TargetEncoder:
    """Encode categorical features using target encoding."""
    
    def __init__(self, target_column: str = TARGET_COLUMN):
        self.target_column = target_column
        self.encodings: Dict[str, Dict] = {}
        self.global_mean = None
        self.is_fitted = False
    
    def fit(
        self,
        df: pd.DataFrame,
        columns: List[str],
        smoothing: float = 1.0
    ) -> 'TargetEncoder':
        """
        Fit target encoder on training data.
        
        Uses smoothing to prevent overfitting on rare categories.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame with target column
        columns : List[str]
            Columns to encode
        smoothing : float
            Smoothing factor for rare categories (between 0 and 1)
        
        Returns
        -------
        TargetEncoder
            Fitted encoder (self)
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        # Get global target mean (for smoothing)
        self.global_mean = df[self.target_column].astype('category').cat.codes.mean()
        
        for col in columns:
            if col in df.columns:
                # Calculate target mean per category
                target_means = df.groupby(col)[self.target_column].agg(['mean', 'count'])
                
                # Apply smoothing: blend rare categories toward global mean
                smoothed_means = {}
                for category, (mean, count) in target_means.iterrows():
                    # More smoothing for rare categories (low count)
                    smoothed_value = (
                        (count * mean + smoothing * self.global_mean) /
                        (count + smoothing)
                    )
                    smoothed_means[category] = smoothed_value
                
                self.encodings[col] = smoothed_means
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted target encoding.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        df_encoded = df.copy()
        
        for col in self.encodings.keys():
            if col in df_encoded.columns:
                # Map values to their target-encoded values
                # Use global mean for unseen values
                df_encoded[col] = df_encoded[col].map(
                    self.encodings[col]
                ).fillna(self.global_mean)
        
        return df_encoded
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: List[str],
        smoothing: float = 1.0
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, columns, smoothing=smoothing).transform(df)


def encode_target(y: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Encode target variable using LabelEncoder.
    
    Maps: TP → 2, BP → 1, FP → 0
    
    Parameters
    ----------
    y : pd.Series
        Target variable
    
    Returns
    -------
    Tuple[pd.Series, Dict[str, int]]
        Encoded target and mapping dictionary
    """
    # Define explicit mapping
    mapping = {
        'TP': 2,
        'BP': 1,
        'FP': 0
    }
    
    y_encoded = y.map(mapping)
    
    # Check for unmapped values
    unmapped = y_encoded.isnull().sum()
    if unmapped > 0:
        print(f"⚠ Warning: {unmapped} unmapped target values found")
        print(f"  Unique values: {y.unique()}")
        # Fill any unmapped with mode or 0
        y_encoded.fillna(0, inplace=True)
    
    print(f"✓ Target encoded: {mapping}")
    return y_encoded, mapping


def identify_categorical_columns(
    df: pd.DataFrame,
    exclude_columns: List[str] = None
) -> List[str]:
    """
    Identify categorical columns to encode.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    exclude_columns : List[str], optional
        Columns to exclude from encoding (e.g., target)
    
    Returns
    -------
    List[str]
        List of categorical columns
    """
    if exclude_columns is None:
        exclude_columns = [TARGET_COLUMN]
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in exclude_columns]
    
    return categorical_cols


def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    encoding_method: str = 'frequency',
    smoothing: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main feature encoding pipeline.
    
    Encodes categorical features using specified method.
    Encoder is fit on training data, then applied to test data.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target (needed for target encoding)
    encoding_method : str
        Encoding method: 'frequency' or 'target'
    smoothing : float
        Smoothing parameter for target encoding
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (X_train_encoded, X_test_encoded, encoder_info)
    """
    print(f"\n=== ENCODING FEATURES ({encoding_method}) ===")
    
    # Identify categorical columns
    categorical_cols = identify_categorical_columns(X_train)
    print(f"Found {len(categorical_cols)} categorical columns to encode")
    
    if not categorical_cols:
        print("No categorical columns to encode")
        return X_train, X_test, {'method': encoding_method, 'columns': []}
    
    # Initialize encoder based on method
    if encoding_method.lower() == 'frequency':
        encoder = FrequencyEncoder()
        X_train_encoded = encoder.fit_transform(X_train, categorical_cols)
        X_test_encoded = encoder.transform(X_test)
        print(f"✓ Applied frequency encoding to {len(categorical_cols)} columns")
    
    elif encoding_method.lower() == 'target':
        encoder = TargetEncoder()
        X_train_encoded = encoder.fit_transform(
            pd.concat([X_train, y_train], axis=1),
            categorical_cols,
            smoothing=smoothing
        )
        X_test_encoded = encoder.transform(X_test)
        print(f"✓ Applied target encoding to {len(categorical_cols)} columns")
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")
    
    encoder_info = {
        'method': encoding_method,
        'columns': categorical_cols,
        'encoder': encoder
    }
    
    print(f"✓ Encoding complete")
    return X_train_encoded, X_test_encoded, encoder_info

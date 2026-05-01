"""
Encoding module for GUIDE dataset.

Implements frequency and target encoding for categorical features.
LabelEncoder applied only to target variable.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

TARGET_COLUMN = 'IncidentGrade'


class FrequencyEncoder:
    """Encode categorical features using frequency encoding."""

    def __init__(self):
        self.encodings: Dict[str, Dict] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'FrequencyEncoder':
        for col in columns:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True).to_dict()
                self.encodings[col] = freq
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        df_encoded = df.copy()
        for col, freq_map in self.encodings.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(freq_map).fillna(0)
        return df_encoded

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)


class TargetEncoder:
    """Encode categorical features using target encoding with smoothing."""

    def __init__(self, target_column: str = TARGET_COLUMN):
        self.target_column = target_column
        self.encodings: Dict[str, Dict] = {}
        self.global_mean = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, columns: List[str], smoothing: float = 1.0) -> 'TargetEncoder':
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")

        self.global_mean = df[self.target_column].astype('category').cat.codes.mean()

        for col in columns:
            if col in df.columns:
                target_means = df.groupby(col)[self.target_column].agg(['mean', 'count'])
                smoothed_means = {}
                for category, (mean, count) in target_means.iterrows():
                    smoothed_value = (count * mean + smoothing * self.global_mean) / (count + smoothing)
                    smoothed_means[category] = smoothed_value
                self.encodings[col] = smoothed_means

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        df_encoded = df.copy()
        for col, enc_map in self.encodings.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(enc_map).fillna(self.global_mean)
        return df_encoded

    def fit_transform(self, df: pd.DataFrame, columns: List[str], smoothing: float = 1.0) -> pd.DataFrame:
        return self.fit(df, columns, smoothing=smoothing).transform(df)


def encode_target(y: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Encode target variable. Maps: TP -> 2, BP -> 1, FP -> 0
    """
    mapping = {'TP': 2, 'BP': 1, 'FP': 0}
    y_encoded = y.map(mapping)

    unmapped = y_encoded.isnull().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} unmapped target values. Filling with 0.")
        y_encoded = y_encoded.fillna(0)

    print(f"Target encoded: {mapping}")
    return y_encoded, mapping


def identify_categorical_columns(df: pd.DataFrame, exclude_columns: List[str] = None) -> List[str]:
    if exclude_columns is None:
        exclude_columns = [TARGET_COLUMN]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return [col for col in categorical_cols if col not in exclude_columns]


def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    encoding_method: str = 'frequency',
    smoothing: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main feature encoding pipeline. Fit on train, apply to test.
    """
    print(f"\n=== ENCODING FEATURES ({encoding_method}) ===")

    categorical_cols = identify_categorical_columns(X_train)
    print(f"Found {len(categorical_cols)} categorical columns to encode")

    if not categorical_cols:
        print("No categorical columns to encode")
        return X_train, X_test, {'method': encoding_method, 'columns': []}

    if encoding_method.lower() == 'frequency':
        encoder = FrequencyEncoder()
        X_train_encoded = encoder.fit_transform(X_train, categorical_cols)
        X_test_encoded = encoder.transform(X_test)

    elif encoding_method.lower() == 'target':
        encoder = TargetEncoder()
        X_train_encoded = encoder.fit_transform(
            pd.concat([X_train, y_train], axis=1),
            categorical_cols,
            smoothing=smoothing
        )
        X_test_encoded = encoder.transform(X_test)

    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")

    encoder_info = {
        'method': encoding_method,
        'columns': categorical_cols,
        'encoder': encoder
    }

    print(f"Encoding complete")
    return X_train_encoded, X_test_encoded, encoder_info
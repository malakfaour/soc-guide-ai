"""
Main preprocessing pipeline orchestration.

Coordinates all preprocessing steps in sequence:
1. Load data
2. Clean data
3. Encode features
4. Split dataset
5. Apply scaling (optional)

Returns ready-to-train datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import json
import os

from src.data.loader import load_train_test_data, load_train_data_only
from src.data.splitter import split_data, save_split_indices
from src.preprocessing.cleaning import clean_data, identify_column_types
from src.preprocessing.encoding import encode_features, encode_target
from src.preprocessing.scaling import scale_features


class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    def __init__(
        self,
        train_path: str = "data/raw/GUIDE_Train.csv",
        test_path: str = "data/raw/GUIDE_Test.csv",
        encoding_method: str = "frequency",
        scaling_method: str = "quantile",
        scaling_output_dist: str = "normal",
        test_size: float = 0.2,
        val_size: float = 0.15,
        random_state: int = 42,
        apply_scaling: bool = True,
        numerical_fill_strategy: str = "median",
        categorical_fill_value: str = "unknown"
    ):
        """
        Initialize configuration.
        
        Parameters
        ----------
        train_path : str
            Path to training CSV
        test_path : str
            Path to test CSV
        encoding_method : str
            'frequency' or 'target'
        scaling_method : str
            'quantile', 'standard', or 'minmax'
        scaling_output_dist : str
            'normal' or 'uniform' (for QuantileTransformer)
        test_size : float
            Proportion for test set
        val_size : float
            Proportion of train+val for validation
        random_state : int
            Random seed
        apply_scaling : bool
            Whether to apply scaling
        numerical_fill_strategy : str
            'mean' or 'median'
        categorical_fill_value : str
            Value to fill categorical missing
        """
        self.train_path = train_path
        self.test_path = test_path
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method
        self.scaling_output_dist = scaling_output_dist
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.apply_scaling = apply_scaling
        self.numerical_fill_strategy = numerical_fill_strategy
        self.categorical_fill_value = categorical_fill_value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.__dict__
    
    def save(self, path: str = "configs/preprocessing_config.json"):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Config saved to {path}")
    
    @classmethod
    def load(cls, path: str = "configs/preprocessing_config.json"):
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def run_preprocessing(
    config: PreprocessingConfig = None
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    Dict
]:
    """
    Main preprocessing pipeline.
    
    Executes all preprocessing steps in sequence:
    1. Load data
    2. Clean data
    3. Encode features
    4. Split dataset
    5. Apply scaling (optional)
    
    Parameters
    ----------
    config : PreprocessingConfig, optional
        Configuration object. If None, uses defaults.
    
    Returns
    -------
    Tuple[6 DataFrames/Series + Dict]
        (X_train, X_val, X_test, y_train, y_val, y_test, pipeline_metadata)
        
    Examples
    --------
    >>> config = PreprocessingConfig(encoding_method='target')
    >>> X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config)
    """
    if config is None:
        config = PreprocessingConfig()
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # STEP 1: Load data
    print("\n[1/5] Loading data...")
    df_train, df_test = load_train_test_data(
        train_path=config.train_path,
        test_path=config.test_path
    )
    
    # STEP 2: Clean data
    print("\n[2/5] Cleaning data...")
    df_train = clean_data(
        df_train,
        drop_identifiers=True,
        numerical_strategy=config.numerical_fill_strategy,
        categorical_fill_value=config.categorical_fill_value
    )
    df_test = clean_data(
        df_test,
        drop_identifiers=True,
        numerical_strategy=config.numerical_fill_strategy,
        categorical_fill_value=config.categorical_fill_value
    )
    
    # STEP 3: Separate features and target
    print("\n[3/5] Extracting features and target...")
    target_col = 'IncidentGrade'
    
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    print(f"✓ Features: {X_train.shape[1]} columns")
    print(f"✓ Target: {y_train.name}")
    
    # STEP 4: Encode target
    print("\n[4/5] Encoding target variable...")
    y_train_encoded, target_mapping = encode_target(y_train)
    y_test_encoded, _ = encode_target(y_test)
    
    # STEP 5: Encode features
    print("\n[5/5] Encoding features...")
    X_train_encoded, X_test_encoded, encoder_info = encode_features(
        X_train,
        X_test,
        y_train_encoded,
        encoding_method=config.encoding_method,
        smoothing=1.0
    )
    
    # STEP 6: Split data
    print("\n[6/5] Splitting data (train/val/test)...")
    (X_train_split, X_val_split, X_test_split,
     y_train_split, y_val_split, y_test_split,
     split_info) = split_data(
        X_train_encoded,
        y_train_encoded,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
        stratify=True
    )
    
    # Save split indices
    save_split_indices(split_info, output_dir="data/splits")
    
    # STEP 7: Apply scaling (optional)
    if config.apply_scaling:
        print("\n[7/5] Applying scaling...")
        X_train_split, X_val_split, X_test_split, scaler = apply_scaling(
            X_train_split,
            X_val_split,
            X_test_split,
            method=config.scaling_method,
            output_distribution=config.scaling_output_dist
        )
    else:
        print("\n✓ Skipping scaling (apply_scaling=False)")
        scaler = None
    
    # Prepare metadata
    pipeline_metadata = {
        'config': config.to_dict(),
        'target_mapping': target_mapping,
        'encoder_info': {k: v for k, v in encoder_info.items() if k != 'encoder'},
        'split_info': split_info,
        'scaler_config': scaler.get_config() if scaler else None,
        'shapes': {
            'X_train': X_train_split.shape,
            'X_val': X_val_split.shape,
            'X_test': X_test_split.shape
        }
    }
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nFinal datasets:")
    print(f"  X_train: {X_train_split.shape}")
    print(f"  X_val:   {X_val_split.shape}")
    print(f"  X_test:  {X_test_split.shape}")
    print(f"  y_train: {y_train_split.shape}")
    print(f"  y_val:   {y_val_split.shape}")
    print(f"  y_test:  {y_test_split.shape}")
    
    return (
        X_train_split, X_val_split, X_test_split,
        y_train_split, y_val_split, y_test_split,
        pipeline_metadata
    )


def apply_scaling(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "quantile",
    output_distribution: str = "normal"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Apply scaling to datasets.
    
    Scaler is fit on training data, then applied to val and test.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    X_test : pd.DataFrame
        Test features
    method : str
        Scaling method
    output_distribution : str
        Output distribution (for QuantileTransformer)
    
    Returns
    -------
    Tuple[3 DataFrames + scaler]
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    from src.preprocessing.scaling import ScalingPipeline
    
    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Found {len(numerical_cols)} numerical columns to scale")
    
    if not numerical_cols:
        print("No numerical columns to scale")
        return X_train, X_val, X_test, None
    
    # Create and fit scaler on training data
    scaler = ScalingPipeline(method=method, output_dist=output_distribution)
    scaler.fit(X_train, numerical_cols)
    
    # Apply to all splits
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Scaling applied")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Default preprocessing
    print("Example 1: Default preprocessing")
    config = PreprocessingConfig()
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config)
    
    # Example 2: Custom configuration
    print("\n\nExample 2: Custom configuration (target encoding)")
    config_custom = PreprocessingConfig(
        encoding_method="target",
        apply_scaling=True
    )
    config_custom.save("configs/preprocessing_config_custom.json")
    
    # Example 3: Load configuration from file
    # config_loaded = PreprocessingConfig.load("configs/preprocessing_config_custom.json")
    # X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config_loaded)

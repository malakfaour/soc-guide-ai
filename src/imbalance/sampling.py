"""
Imbalance handling and sampling strategies.

Provides techniques for handling class imbalance including:
- Oversampling
- Undersampling
- SMOTE
- Class weighting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from collections import Counter


class UndersamplingSampler:
    """
    Undersample majority class while preserving minority classes.
    
    Useful for imbalanced datasets where majority class dominates.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_ratio: float = 0.5,
        minority_threshold: int = 10
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and resample data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        target_ratio : float
            Target ratio of samples to keep (0-1)
        minority_threshold : int
            Classes with fewer samples are considered minority
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Resampled (X, y)
        """
        np.random.seed(self.random_state)
        
        # Count class frequencies
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()
        
        # Calculate target majority count
        target_majority_count = int(majority_count * target_ratio)
        
        # Get indices for majority and minority
        majority_indices = y[y == majority_class].index.tolist()
        minority_indices = y[y != majority_class].index.tolist()
        
        # Undersample majority
        sampled_majority_indices = np.random.choice(
            majority_indices,
            size=min(target_majority_count, len(majority_indices)),
            replace=False
        ).tolist()
        
        # Combine
        resampled_indices = sampled_majority_indices + minority_indices
        
        X_resampled = X.loc[resampled_indices]
        y_resampled = y.loc[resampled_indices]
        
        return X_resampled, y_resampled


class WeightedSampler:
    """
    Compute class weights for weighted training.
    
    Useful for models that support sample_weight parameter.
    """
    
    @staticmethod
    def compute_class_weights(y: pd.Series) -> Dict[int, float]:
        """
        Compute class weights using balanced approach.
        
        weight = n_samples / (n_classes * count_per_class)
        
        Parameters
        ----------
        y : pd.Series
            Target variable
        
        Returns
        -------
        Dict[int, float]
            {class_label: weight}
        """
        n_samples = len(y)
        n_classes = len(y.unique())
        class_counts = y.value_counts()
        
        weights = {}
        for class_label, count in class_counts.items():
            weights[class_label] = n_samples / (n_classes * count)
        
        return weights
    
    @staticmethod
    def compute_sample_weights(
        y: pd.Series,
        weights_dict: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Compute sample weights from class weights.
        
        Parameters
        ----------
        y : pd.Series
            Target variable
        weights_dict : Dict, optional
            Pre-computed class weights. If None, computed internally.
        
        Returns
        -------
        np.ndarray
            Sample weights
        """
        if weights_dict is None:
            weights_dict = WeightedSampler.compute_class_weights(y)
        
        sample_weights = np.array([weights_dict[label] for label in y])
        return sample_weights


def analyze_class_imbalance(
    y: pd.Series,
    name: str = "Target"
) -> Dict:
    """
    Analyze class imbalance in target variable.
    
    Parameters
    ----------
    y : pd.Series
        Target variable
    name : str
        Name for reporting
    
    Returns
    -------
    Dict
        Imbalance statistics
    """
    counts = y.value_counts()
    total = len(y)
    
    imbalance_stats = {
        'total_samples': total,
        'n_classes': len(counts),
        'class_counts': counts.to_dict(),
        'class_ratios': (counts / total).to_dict(),
        'imbalance_ratio': counts.max() / counts.min(),
        'minority_ratio': counts.min() / counts.max()
    }
    
    print(f"\n{name} Class Distribution:")
    for class_label, count in counts.items():
        ratio = count / total * 100
        print(f"  {class_label}: {count} samples ({ratio:.1f}%)")
    print(f"Imbalance ratio (max:min): {imbalance_stats['imbalance_ratio']:.2f}:1")
    
    return imbalance_stats


def handle_imbalance(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str = "undersample",
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using specified strategy.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    strategy : str
        'undersample', 'weight', or 'none'
    **kwargs
        Strategy-specific parameters
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Balanced (X, y)
    """
    print(f"\n=== HANDLING IMBALANCE ({strategy}) ===")
    
    # Analyze original imbalance
    analyze_class_imbalance(y, "Original")
    
    if strategy == "undersample":
        sampler = UndersamplingSampler()
        X_balanced, y_balanced = sampler.fit_resample(
            X, y,
            target_ratio=kwargs.get('target_ratio', 0.5),
            minority_threshold=kwargs.get('minority_threshold', 10)
        )
        print(f"✓ Undersampling applied")
    
    elif strategy == "weight":
        # Return original data (weights will be used at training time)
        X_balanced, y_balanced = X, y
        print(f"✓ Weights will be used at training time")
    
    elif strategy == "none":
        X_balanced, y_balanced = X, y
        print(f"✓ No imbalance handling")
    
    else:
        raise ValueError(f"Unknown imbalance strategy: {strategy}")
    
    # Analyze after balancing
    analyze_class_imbalance(y_balanced, "After balancing")
    
    return X_balanced, y_balanced

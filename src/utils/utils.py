"""
Utility functions for preprocessing and modeling.

Common functions used across the pipeline.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path


def ensure_directory(directory: str) -> str:
    """
    Ensure directory exists, create if needed.
    
    Parameters
    ----------
    directory : str
        Directory path
    
    Returns
    -------
    str
        Absolute path to directory
    """
    os.makedirs(directory, exist_ok=True)
    return os.path.abspath(directory)


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns
    -------
    Path
        Project root path
    """
    return Path(__file__).parent.parent.parent


def load_json(filepath: str) -> Dict:
    """
    Load JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to JSON file
    
    Returns
    -------
    Dict
        Loaded JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str) -> None:
    """
    Save data to JSON file.
    
    Parameters
    ----------
    data : Dict
        Data to save
    filepath : str
        Output path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive DataFrame information.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect
    name : str
        Name for reporting
    """
    print(f"\n{name} Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Duplicates: {df.duplicated().sum()}")
    print(f"  Missing: {df.isnull().sum().sum()}")
    print(f"  Dtypes:\n{df.dtypes.value_counts()}")


def get_memory_usage(df: pd.DataFrame) -> float:
    """
    Get DataFrame memory usage in MB.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    
    Returns
    -------
    float
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by optimizing dtypes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool
        Print memory reduction info
    
    Returns
    -------
    pd.DataFrame
        Optimized DataFrame
    """
    initial_memory = get_memory_usage(df)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    final_memory = get_memory_usage(df)
    
    if verbose:
        reduction = (initial_memory - final_memory) / initial_memory * 100
        print(f"Memory reduced from {initial_memory:.2f}MB to {final_memory:.2f}MB ({reduction:.1f}% reduction)")
    
    return df


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for all columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        Summary with dtype, missing, unique counts
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null': df.isnull().sum(),
        'null_pct': df.isnull().sum() / len(df) * 100,
        'unique': df.nunique(),
        'memory_mb': df.memory_usage(deep=True) / 1024**2
    })
    
    return summary.sort_values('memory_mb', ascending=False)


def compare_distributions(
    series1: pd.Series,
    series2: pd.Series,
    name1: str = "Series1",
    name2: str = "Series2"
) -> Dict[str, Any]:
    """
    Compare distributions of two series.
    
    Parameters
    ----------
    series1, series2 : pd.Series
        Series to compare
    name1, name2 : str
        Names for reporting
    
    Returns
    -------
    Dict
        Comparison statistics
    """
    comparison = {
        name1: {
            'mean': series1.mean(),
            'std': series1.std(),
            'min': series1.min(),
            'max': series1.max(),
            'median': series1.median(),
            'q25': series1.quantile(0.25),
            'q75': series1.quantile(0.75)
        },
        name2: {
            'mean': series2.mean(),
            'std': series2.std(),
            'min': series2.min(),
            'max': series2.max(),
            'median': series2.median(),
            'q25': series2.quantile(0.25),
            'q75': series2.quantile(0.75)
        }
    }
    
    return comparison

"""
Data loader for GUIDE dataset.
Handles loading train and test CSV files with proper error handling.
"""
import pandas as pd
import os


def load_data(filepath, sample=False, nrows=None, verbose=True):
    """
    Load CSV file with error handling.
    
    Args:
        filepath: Path to CSV file
        sample: If True, load only first nrows
        nrows: Number of rows to load (None = all)
        verbose: Print debug information
    
    Returns:
        pandas DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    if verbose:
        print(f"\n[LOADER] Loading: {filepath}")
    
    try:
        # Use low_memory=False to avoid mixed type warnings
        df = pd.read_csv(filepath, low_memory=False, nrows=nrows)
        
        if verbose:
            print(f"[LOADER] [OK] Loaded successfully")
            print(f"[LOADER] Shape: {df.shape}")
            print(f"[LOADER] Columns: {list(df.columns)}")
            print(f"[LOADER] Data types:\n{df.dtypes}")
            print(f"[LOADER] Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    except Exception as e:
        print(f"[LOADER] [ERROR] Error loading {filepath}: {str(e)}")
        raise


def load_train_data(sample=False, nrows=10000):
    """Load training dataset."""
    train_path = "data/raw/GUIDE_Train.csv"
    return load_data(train_path, sample=sample, nrows=nrows)


def load_test_data(sample=False, nrows=10000):
    """Load test dataset."""
    test_path = "data/raw/GUIDE_Test.csv"
    return load_data(test_path, sample=sample, nrows=nrows)


if __name__ == "__main__":
    # Test loading
    df_train = load_train_data(nrows=10000)
    print("\n[LOADER] Sample of training data:")
    print(df_train.head())

"""
Dataset versioning management.
Ensures all team members use the exact same dataset versions.
Prevents accidental overwrites and provides version management.
"""
import os
import sys
from pathlib import Path


PROCESSED_DATA_ROOT = "data/processed"


def get_next_version(base_path=PROCESSED_DATA_ROOT, verbose=True):
    """
    Determine the next available version number.
    
    Args:
        base_path: Base path for processed data
        verbose: Print information
    
    Returns:
        Next version string (e.g., 'v1', 'v2', 'v3')
    """
    if not os.path.exists(base_path):
        if verbose:
            print(f"[VERSION] Creating base path: {base_path}")
        os.makedirs(base_path, exist_ok=True)
    
    # Find all existing version directories
    version_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith('v') and item[1:].isdigit():
            version_num = int(item[1:])
            version_dirs.append(version_num)
    
    if not version_dirs:
        next_version = 1
    else:
        next_version = max(version_dirs) + 1
    
    version_str = f"v{next_version}"
    
    if verbose:
        print(f"[VERSION] Current versions: {sorted(version_dirs)}")
        print(f"[VERSION] Next version: {version_str}")
    
    return version_str


def version_exists(version, base_path=PROCESSED_DATA_ROOT):
    """
    Check if a version already exists.
    
    Args:
        version: Version string (e.g., 'v1')
        base_path: Base path for processed data
    
    Returns:
        True if version exists, False otherwise
    """
    version_path = os.path.join(base_path, version)
    return os.path.isdir(version_path)


def get_current_version(base_path=PROCESSED_DATA_ROOT, verbose=True):
    """
    Get the current (latest) version.
    
    Args:
        base_path: Base path for processed data
        verbose: Print information
    
    Returns:
        Current version string (e.g., 'v1') or None if no versions exist
    """
    if not os.path.exists(base_path):
        return None
    
    version_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith('v') and item[1:].isdigit():
            version_num = int(item[1:])
            version_dirs.append((version_num, item))
    
    if not version_dirs:
        return None
    
    current = max(version_dirs)[1]
    
    if verbose:
        print(f"[VERSION] Current dataset version: {current}")
    
    return current


def get_dataset_path(version=None, base_path=PROCESSED_DATA_ROOT):
    """
    Get the full path for a dataset version.
    
    Args:
        version: Version string (e.g., 'v1'). If None, uses current version.
        base_path: Base path for processed data
    
    Returns:
        Full path to version directory
    """
    if version is None:
        version = get_current_version(base_path, verbose=False)
        if version is None:
            raise ValueError("No dataset versions found. Run preprocessing first.")
    
    return os.path.join(base_path, version)


def create_version(version=None, base_path=PROCESSED_DATA_ROOT, verbose=True):
    """
    Create a new dataset version directory.
    
    Args:
        version: Version string (e.g., 'v1'). If None, auto-generates next version.
        base_path: Base path for processed data
        verbose: Print information
    
    Returns:
        Created version string (e.g., 'v1')
    
    Raises:
        ValueError: If version already exists
    """
    if version is None:
        version = get_next_version(base_path, verbose=verbose)
    
    version_path = os.path.join(base_path, version)
    
    if os.path.exists(version_path):
        raise ValueError(
            f"Version {version} already exists at {version_path}. "
            f"Choose a different version or delete existing one manually."
        )
    
    os.makedirs(version_path, exist_ok=True)
    
    if verbose:
        print(f"[VERSION] Created version directory: {version_path}")
    
    return version


def save_dataset_with_version(X_train, X_val, X_test, y_train, y_val, y_test, 
                              version=None, force=False, base_path=PROCESSED_DATA_ROOT, verbose=True):
    """
    Save dataset with automatic versioning.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
        y_train, y_val, y_test: Target Series
        version: Version string (e.g., 'v1'). If None, uses next available.
        force: If True, overwrite existing version (requires confirmation).
        base_path: Base path for processed data
        verbose: Print information
    
    Returns:
        Version string (e.g., 'v1')
    
    Raises:
        ValueError: If version exists and force=False
    """
    if version is None:
        version = get_next_version(base_path, verbose=False)
    
    version_path = os.path.join(base_path, version)
    
    # Check if version exists
    if os.path.exists(version_path):
        if not force:
            raise ValueError(
                f"[VERSION] ERROR: Version '{version}' already exists at {version_path}\n"
                f"[VERSION] To overwrite, set force=True and confirm manually."
            )
        
        # Force=True: require explicit confirmation
        if verbose:
            print(f"\n[VERSION] WARNING: Version '{version}' already exists!")
            print(f"[VERSION] Path: {version_path}")
            print(f"[VERSION] This will OVERWRITE existing data.")
            
            confirm = input("[VERSION] Type 'yes' to confirm overwrite: ").strip().lower()
            if confirm != 'yes':
                print("[VERSION] Overwrite cancelled.")
                return None
    else:
        # Create new version
        os.makedirs(version_path, exist_ok=True)
    
    if verbose:
        print(f"\n[VERSION] Saving dataset to version: {version}")
        print(f"[VERSION] Path: {version_path}")
    
    # Save all files
    files_to_save = [
        ('X_train', X_train),
        ('X_val', X_val),
        ('X_test', X_test),
        ('y_train', y_train),
        ('y_val', y_val),
        ('y_test', y_test)
    ]
    
    for filename, data in files_to_save:
        filepath = os.path.join(version_path, f"{filename}.csv")
        data.to_csv(filepath, index=False)
        
        if verbose:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  [OK] {filename}.csv ({size_mb:.1f} MB)")
    
    if verbose:
        print(f"\n[VERSION] Dataset version {version} created successfully")
    
    return version


def list_versions(base_path=PROCESSED_DATA_ROOT, verbose=True):
    """
    List all available dataset versions.
    
    Args:
        base_path: Base path for processed data
        verbose: Print information
    
    Returns:
        List of version strings sorted by version number
    """
    if not os.path.exists(base_path):
        return []
    
    versions = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith('v') and item[1:].isdigit():
            versions.append(item)
    
    versions = sorted(versions, key=lambda x: int(x[1:]))
    
    if verbose:
        print(f"\n[VERSION] Available dataset versions:")
        for version in versions:
            version_path = os.path.join(base_path, version)
            files = os.listdir(version_path)
            print(f"  {version}: {len(files)} files")
    
    return versions


def load_dataset_by_version(version=None, base_path=PROCESSED_DATA_ROOT, verbose=True):
    """
    Load a specific dataset version.
    
    Args:
        version: Version string (e.g., 'v1'). If None, loads current version.
        base_path: Base path for processed data
        verbose: Print information
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Raises:
        FileNotFoundError: If version not found
    """
    import pandas as pd
    
    if version is None:
        version = get_current_version(base_path, verbose=False)
        if version is None:
            raise FileNotFoundError("No dataset versions found.")
    
    version_path = get_dataset_path(version, base_path)
    
    if not os.path.isdir(version_path):
        raise FileNotFoundError(f"Version {version} not found at {version_path}")
    
    if verbose:
        print(f"\n[VERSION] Loading dataset version: {version}")
        print(f"[VERSION] Path: {version_path}")
    
    # Load all files
    X_train = pd.read_csv(os.path.join(version_path, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(version_path, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(version_path, 'X_test.csv'))
    
    y_train = pd.read_csv(os.path.join(version_path, 'y_train.csv')).squeeze()
    y_val = pd.read_csv(os.path.join(version_path, 'y_val.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(version_path, 'y_test.csv')).squeeze()
    
    if verbose:
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_val:   {y_val.shape}")
        print(f"  y_test:  {y_test.shape}")
        print(f"\n[VERSION] Dataset version {version} loaded successfully")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    print("Dataset Versioning Utility")
    print("=" * 60)
    
    # Test version management
    print("\n1. Get next version...")
    next_v = get_next_version(verbose=True)
    
    print("\n2. List existing versions...")
    versions = list_versions(verbose=True)
    
    print("\n3. Get current version...")
    current = get_current_version(verbose=True)
    
    print("\n✓ Version management utilities ready!")

# Dataset Versioning System

## Overview

All team members must use the **exact same dataset version** to ensure reproducible results. The versioning system prevents accidental data modifications and provides clear version tracking.

## Current Status

**✓ Active Dataset Version: v1**

**Location**: `data/processed/v1/`

**Files**:
- `X_train.csv` (69,631 × 44 features)
- `X_val.csv` (14,922 × 44 features)
- `X_test.csv` (14,922 × 44 features)
- `y_train.csv` (69,631 labels)
- `y_val.csv` (14,922 labels)
- `y_test.csv` (14,922 labels)

**Dataset Statistics**:
- Total Samples: 99,475
- Train/Val/Test Split: 70% / 15% / 15%
- Number of Features: 44 (all numeric)
- Target Classes: 3 (0=FalsePositive, 1=BenignPositive, 2=TruePositive)

## For Team Members

### Loading Dataset

**Option 1: Using Versioning Utility (Recommended)**
```python
from src.utils.versioning import load_dataset_by_version

# Load current version
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version('v1', verbose=True)
```

**Option 2: Direct Loading**
```python
import pandas as pd

X_train = pd.read_csv('data/processed/v1/X_train.csv')
X_val = pd.read_csv('data/processed/v1/X_val.csv')
X_test = pd.read_csv('data/processed/v1/X_test.csv')

y_train = pd.read_csv('data/processed/v1/y_train.csv').squeeze()
y_val = pd.read_csv('data/processed/v1/y_val.csv').squeeze()
y_test = pd.read_csv('data/processed/v1/y_test.csv').squeeze()
```

### Checking Available Versions

```python
from src.utils.versioning import list_versions, get_current_version

# List all versions
versions = list_versions(verbose=True)

# Get current (latest) version
current = get_current_version(verbose=True)
print(f"Using dataset version: {current}")
```

## For Data Scientists (Creating New Versions)

### When to Create a New Version

Create a new version when:
1. Re-running the full preprocessing pipeline with **different parameters**
2. Using a **larger sample** of the full dataset
3. **Fixing bugs** in the preprocessing logic
4. **Adding new features** or columns to the dataset

**DO NOT create a new version:**
- For model training (just use existing v1)
- For hyperparameter tuning (use v1)
- For experiments (use v1)

### Creating a New Version

**Step 1: Run the preprocessing pipeline**
```python
from src.preprocessing.pipeline import run_preprocessing

X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing(
    apply_scaling=False,
    sample_size=None,  # Use full dataset
    verbose=True,
    save_artifacts_flag=True
)
```

**Step 2: Pipeline saves data automatically**

The `test_pipeline.py` script automatically:
- Detects next version (v1 → v2 → v3...)
- Creates versioned directory
- Saves all 6 CSV files
- Prints confirmation message

```bash
python src/training/test_pipeline.py
```

**Output**:
```
[TEST] Saving processed data with versioning...

[VERSION] Available dataset versions:
  v1: 6 files

[VERSION] Current dataset version: v1

[VERSION] Saving dataset to version: v2
[VERSION] Path: data/processed/v2
  [OK] X_train.csv (34.1 MB)
  [OK] X_val.csv (7.3 MB)
  ...
[VERSION] Dataset version v2 created successfully
```

### Overwrite Protection

**If v1 already exists:**

When running the pipeline:
1. System detects v1 exists
2. **Does NOT automatically overwrite**
3. Requires manual confirmation:
   ```
   [VERSION] WARNING: Version 'v1' already exists!
   [VERSION] This will OVERWRITE existing data.
   [VERSION] Type 'yes' to confirm overwrite:
   ```

**To force overwrite** (only if you know what you're doing):
```python
from src.utils.versioning import save_dataset_with_version

save_dataset_with_version(
    X_train, X_val, X_test, y_train, y_val, y_test,
    version='v1',
    force=True,  # Requires confirmation
    verbose=True
)
```

## Version Directory Structure

```
data/
├── raw/                    # Raw CSVs (DO NOT EDIT)
│   ├── GUIDE_Train.csv    # 2.4 GB
│   └── GUIDE_Test.csv     # 1.1 GB
│
├── processed/             # Versioned processed data
│   └── v1/               # Version 1 (CURRENT)
│       ├── X_train.csv
│       ├── X_val.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_val.csv
│       └── y_test.csv
│   
│   └── v2/               # Version 2 (FUTURE - if created)
│       ├── X_train.csv
│       ├── X_val.csv
│       ...
```

## Versioning API Reference

### Loading Data

```python
from src.utils.versioning import load_dataset_by_version

# Load specific version
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version(
    version='v1',           # Load v1
    verbose=True            # Print details
)

# Load current version (auto-detect)
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version(
    version=None,           # Uses get_current_version()
    verbose=True
)
```

### Querying Versions

```python
from src.utils.versioning import (
    list_versions,
    get_current_version,
    version_exists,
    get_dataset_path
)

# List all versions
versions = list_versions(verbose=True)
# Output: ['v1']

# Get current version
current = get_current_version(verbose=True)
# Output: 'v1'

# Check if version exists
exists = version_exists('v1')
# Output: True

# Get path to version
path = get_dataset_path('v1')
# Output: 'data/processed/v1'
```

### Saving Data with Versioning

```python
from src.utils.versioning import save_dataset_with_version

# Auto-increment version
version = save_dataset_with_version(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    version=None,           # Auto-detect (v1 → v2)
    force=False,            # Prevent accidental overwrite
    verbose=True
)
# Result: version = 'v2'
```

## Best Practices

### For All Team Members
- ✅ **ALWAYS reference the current version** when loading data
- ✅ **Use versioning utilities** instead of manual paths
- ✅ **Check the current version** before experiments start
- ✅ **Document which version** was used in your experiments

### For Data Engineering
- ✅ **Version changes from full preprocessing** (re-run pipeline)
- ✅ **Increment version numbers** incrementally (v1 → v2 → v3)
- ✅ **Protect existing versions** from accidental overwrites
- ✅ **Log version creation** with preprocessing parameters

### DO NOT
- ❌ Manually edit CSV files in `data/processed/`
- ❌ Delete version directories without notification
- ❌ Create multiple versions simultaneously
- ❌ Use non-versioned paths like `data/processed/X_train.csv`

## Troubleshooting

### Dataset not found
```
FileNotFoundError: Version v1 not found
```
**Solution**: Verify file path and version exists:
```python
from src.utils.versioning import list_versions
versions = list_versions()
```

### Wrong data shape
```
AssertionError: X_train and y_train length mismatch
```
**Solution**: Ensure you loaded matching versions (same v1 for all)

### Version already exists
```
ValueError: Version v1 already exists
```
**Solution**: Either use a different version or set `force=True` with explicit confirmation

## Integration Examples

### Example 1: Training XGBoost with v1
```python
from src.utils.versioning import load_dataset_by_version
from xgboost import XGBClassifier

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version('v1')

# Train model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate
val_score = model.score(X_val, y_val)
print(f"Validation Accuracy (v1): {val_score:.4f}")
```

### Example 2: Experiment Tracking
```python
from src.utils.versioning import get_current_version
import logging

# Log which dataset version was used
version = get_current_version()
logging.info(f"Experiment using dataset version: {version}")
logging.info(f"Dataset location: data/processed/{version}/")
```

### Example 3: Multi-Model Comparison
```python
from src.utils.versioning import load_dataset_by_version
from src.utils.preprocessing_inference import get_target_mapping

# All models use same data
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version('v1')

# Train multiple models
models = {
    'xgboost': XGBClassifier(),
    'lightgbm': LGBMClassifier(),
    'tabnet': TabNetClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"{name} (v1): {score:.4f}")
```

## Version History

| Version | Created | Samples | Features | Split  | Notes |
|---------|---------|---------|----------|--------|-------|
| v1      | 2024    | 99,475  | 44       | 70/15/15 | Current - Frequency encoded, stratified split |

---

**Status**: ✓ Production Ready

**Last Updated**: 2024

**Dataset Version in Use**: v1

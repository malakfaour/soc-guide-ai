# Preprocessing Pipeline

Complete preprocessing pipeline for the GUIDE dataset, preparing data for XGBoost, LightGBM, and TabNet models.

## Pipeline Overview

```
Raw Data (GUIDE_Train.csv, GUIDE_Test.csv)
    ↓
[1] Load Data
    ↓
[2] Clean Data
    ├─ Drop identifier columns
    ├─ Fill missing values (numerical: median, categorical: 'unknown')
    └─ Validate data quality
    ↓
[3] Encode Features
    ├─ Frequency encoding OR Target encoding (high-cardinality categorical)
    └─ LabelEncoder for target (TP→2, BP→1, FP→0)
    ↓
[4] Split Data
    ├─ Stratified train/val/test split (70%/15%/15%)
    └─ Save split indices for reproducibility
    ↓
[5] Scale Features (Optional)
    └─ QuantileTransformer with normal distribution
    ↓
Ready-to-Train Datasets
```

## Module Documentation

### `loader.py`
Efficient data loading for large datasets.

**Functions:**
- `load_train_test_data()` - Load training and test data
- `load_train_data_only()` - Load training data for CV scenarios

```python
from src.data.loader import load_train_test_data

df_train, df_test = load_train_test_data(
    train_path="data/raw/GUIDE_Train.csv",
    test_path="data/raw/GUIDE_Test.csv"
)
```

### `cleaning.py`
Data cleaning and validation.

**Functions:**
- `clean_data()` - Main cleaning pipeline
- `drop_irrelevant_columns()` - Remove identifiers
- `handle_missing_values()` - Fill missing data
- `identify_column_types()` - Detect numerical/categorical

```python
from src.preprocessing.cleaning import clean_data

df_train = clean_data(
    df_train,
    drop_identifiers=True,
    numerical_strategy='median',
    categorical_fill_value='unknown'
)
```

### `encoding.py`
Feature encoding (frequency and target encoding).

**Classes:**
- `FrequencyEncoder` - Encode categories by frequency
- `TargetEncoder` - Encode categories by target mean (with smoothing)

**Functions:**
- `encode_features()` - Main encoding pipeline
- `encode_target()` - LabelEncoder for target
- `identify_categorical_columns()` - Find categorical features

```python
from src.preprocessing.encoding import encode_features

X_train_enc, X_test_enc, encoder_info = encode_features(
    X_train,
    X_test,
    y_train,
    encoding_method='frequency',  # or 'target'
    smoothing=1.0
)
```

### `scaling.py`
Feature scaling (QuantileTransformer, StandardScaler, MinMaxScaler).

**Classes:**
- `ScalingPipeline` - Handles scaling with fit/transform

**Functions:**
- `scale_features()` - Main scaling pipeline
- `get_scaling_stats()` - Compute statistics

```python
from src.preprocessing.scaling import scale_features

X_train_scaled, X_test_scaled, scaler = scale_features(
    X_train,
    X_test,
    method='quantile',
    output_distribution='normal'
)
```

### `splitter.py`
Stratified train/validation/test splitting.

**Functions:**
- `split_data()` - Main splitting pipeline (returns train/val/test)
- `save_split_indices()` - Save indices for reproducibility
- `load_split_indices()` - Load saved indices
- `stratified_train_val_test_split()` - Core split logic

```python
from src.data.splitter import split_data

X_train, X_val, X_test, y_train, y_val, y_test, split_info = split_data(
    X,
    y,
    test_size=0.2,
    val_size=0.15,
    random_state=42,
    stratify=True
)
```

### `pipeline.py`
Main orchestration script.

**Classes:**
- `PreprocessingConfig` - Configuration management

**Functions:**
- `run_preprocessing()` - Execute full pipeline
- `apply_scaling()` - Apply scaling to train/val/test

```python
from src.preprocessing.pipeline import PreprocessingConfig, run_preprocessing

config = PreprocessingConfig(
    encoding_method='frequency',
    apply_scaling=True
)

X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config)
```

### `imbalance/sampling.py`
Handle class imbalance.

**Classes:**
- `UndersamplingSampler` - Undersample majority class
- `WeightedSampler` - Compute class weights

**Functions:**
- `handle_imbalance()` - Main imbalance handling
- `analyze_class_imbalance()` - Analyze imbalance statistics

```python
from src.imbalance.sampling import handle_imbalance, analyze_class_imbalance

analyze_class_imbalance(y_train)

X_balanced, y_balanced = handle_imbalance(
    X_train,
    y_train,
    strategy='undersample',
    target_ratio=0.5
)
```

### `data/pipeline.py`
Data I/O and validation utilities.

**Functions:**
- `export_preprocessed_data()` - Save processed datasets
- `load_preprocessed_data()` - Load saved datasets
- `validate_data()` - Validate data quality

```python
from src.data.pipeline import export_preprocessed_data, load_preprocessed_data

# Export
export_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)

# Load
X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
```

### `utils/utils.py`
Common utility functions.

**Functions:**
- `ensure_directory()` - Create directories
- `reduce_memory_usage()` - Optimize dtypes
- `get_column_summary()` - Column statistics
- `compare_distributions()` - Compare series

## Usage Examples

### Example 1: Default Preprocessing
```python
from src.preprocessing.pipeline import PreprocessingConfig, run_preprocessing

# Default configuration
config = PreprocessingConfig()
X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

### Example 2: Custom Configuration (Target Encoding)
```python
from src.preprocessing.pipeline import PreprocessingConfig, run_preprocessing

config = PreprocessingConfig(
    encoding_method='target',        # Use target encoding
    apply_scaling=True,
    scaling_method='quantile',
    random_state=42
)
config.save('configs/custom_preprocessing.json')

X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config)
```

### Example 3: Load Configuration from File
```python
from src.preprocessing.pipeline import PreprocessingConfig, run_preprocessing

config = PreprocessingConfig.load('configs/custom_preprocessing.json')
X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config)
```

### Example 4: Manual Step-by-Step
```python
from src.data.loader import load_train_test_data
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import encode_features, encode_target
from src.data.splitter import split_data
from src.preprocessing.scaling import scale_features

# Step 1: Load
df_train, df_test = load_train_test_data()

# Step 2: Clean
df_train = clean_data(df_train)
df_test = clean_data(df_test)

# Step 3: Separate target
X_train = df_train.drop('IncidentGrade', axis=1)
y_train = df_train['IncidentGrade']
X_test = df_test.drop('IncidentGrade', axis=1)
y_test = df_test['IncidentGrade']

# Step 4: Encode target
y_train_enc, mapping = encode_target(y_train)
y_test_enc, _ = encode_target(y_test)

# Step 5: Encode features
X_train_enc, X_test_enc, info = encode_features(X_train, X_test, y_train_enc)

# Step 6: Split
X_train, X_val, X_test, y_train, y_val, y_test, info = split_data(X_train_enc, y_train_enc)

# Step 7: Scale (optional)
X_train, X_val, X_test, scaler = apply_scaling(X_train, X_val, X_test)
```

### Example 5: Handle Class Imbalance
```python
from src.imbalance.sampling import analyze_class_imbalance, handle_imbalance

# Analyze
analyze_class_imbalance(y_train)

# Handle
X_balanced, y_balanced = handle_imbalance(
    X_train,
    y_train,
    strategy='undersample',
    target_ratio=0.5
)
```

## Configuration Options

### PreprocessingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_path` | str | `data/raw/GUIDE_Train.csv` | Path to training data |
| `test_path` | str | `data/raw/GUIDE_Test.csv` | Path to test data |
| `encoding_method` | str | `frequency` | `frequency` or `target` |
| `scaling_method` | str | `quantile` | `quantile`, `standard`, or `minmax` |
| `scaling_output_dist` | str | `normal` | `normal` or `uniform` |
| `test_size` | float | `0.2` | Proportion for test (overall ~15-20%) |
| `val_size` | float | `0.15` | Proportion of train+val for validation |
| `random_state` | int | `42` | Reproducibility seed |
| `apply_scaling` | bool | `True` | Whether to apply scaling |
| `numerical_fill_strategy` | str | `median` | `mean` or `median` |
| `categorical_fill_value` | str | `unknown` | Missing value replacement |

## Output Structure

```
data/
├── raw/
│   ├── GUIDE_Train.csv
│   └── GUIDE_Test.csv
├── interim/
│   └── (cleaned, but not encoded)
├── processed/
│   ├── X_train.parquet
│   ├── X_val.parquet
│   ├── X_test.parquet
│   ├── y_train.parquet
│   ├── y_val.parquet
│   └── y_test.parquet
└── splits/
    ├── train_indices.csv
    ├── val_indices.csv
    ├── test_indices.csv
    └── split_config.json

configs/
└── preprocessing_config.json
```

## Key Design Decisions

1. **No One-Hot Encoding**: Uses frequency/target encoding instead for better handling of high-cardinality features
2. **Stratified Splitting**: Maintains class distribution across train/val/test splits
3. **Fit on Train Only**: All transforms (encoder, scaler) fit on training data to prevent data leakage
4. **Flexible Scaling**: Optional scaling for model compatibility (required for TabNet)
5. **Reproducibility**: Save/load split indices and configurations

## Model Compatibility

- **XGBoost**: Works with or without scaling (prefers raw features)
- **LightGBM**: Works with or without scaling (prefers raw features)
- **TabNet**: Requires scaling (included in pipeline)

## Performance Tips

- Use `frequency` encoding for very high-cardinality features (>100 categories)
- Use `target` encoding for moderate-cardinality features (<100) with smoothing
- For large datasets, use `apply_scaling=False` by default, enable only for TabNet
- Set `nrows` parameter in loader for testing with subset

## Troubleshooting

**Missing Data Not Filled**
- Check `numerical_fill_strategy` (median/mean)
- Verify categorical_fill_value is set correctly

**Target Encoding Not Working**
- Ensure target column 'IncidentGrade' exists
- Verify values are TP, BP, or FP

**Class Imbalance After Split**
- Use `stratify=True` in split_data()
- Check `analyze_class_imbalance()` output

**Memory Issues with Large Dataset**
- Use `reduce_memory_usage()` from utils
- Set `nrows` parameter to test subset
- Reduce `n_quantiles` in QuantileTransformer

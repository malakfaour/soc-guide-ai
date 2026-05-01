# SOC Intelligence Platform - Technical Architecture

**Document Level**: Senior Engineer / Architect  
**Last Updated**: May 1, 2026  
**Status**: Production-Ready Hybrid Inference System

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Data Flow - Complete Pipeline](#data-flow---complete-pipeline)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Model Layer - Deep Dive](#model-layer---deep-dive)
6. [Inference Engine](#inference-engine)
7. [Backend Implementation](#backend-implementation)
8. [Frontend Architecture](#frontend-architecture)
9. [Integration Points](#integration-points)
10. [Running the System](#running-the-system)
11. [Testing Guide](#testing-guide)
12. [Debugging & Troubleshooting](#debugging--troubleshooting)
13. [Design Rationale](#design-rationale)

---

## System Overview

The SOC Intelligence platform is a **hybrid multi-model incident analysis system** designed for security operations center (SOC) alert triage and remediation. The system operates at two levels:

### Two-Level Prediction Hierarchy

1. **Row-Level Triage** (alert classification)
   - Input: Individual alert features (44 features post-preprocessing)
   - Output: Alert severity class (FalsePositive, BenignPositive, TruePositive)
   - Model: TabNet (primary deep learning model)
   - Baselines: LightGBM, XGBoost

2. **Incident-Level Remediation** (action recommendation)
   - Input: Aggregated incident features
   - Output: Two independent binary decisions
     - `account_response`: Should the account be remediated? (Gradient Boosted Trees)
     - `endpoint_response`: Should the endpoint be remediated? (Logistic Regression)
   - Threshold-based: Predictions determined by tuned probability thresholds

### Current Deployment State

- **Triage Model**: TabNet (production choice)
- **Triage Baselines**: LightGBM, XGBoost (for comparison/fallback)
- **Remediation**: Classical (non-DL) models trained incident-level
- **Inference**: Hybrid scoring in `hybrid_incident_scoring.py`
- **Frontend**: React TypeScript (development/experimental)
- **Demo Backend**: Streamlit (app.py)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW DATA LAYER                          │
│  (GUIDE_Train.csv, GUIDE_Test.csv - Security Alerts Dataset)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   PREPROCESSING PIPELINE           │
        │  (src/preprocessing/*)             │
        ├────────────────────────────────────┤
        │ 1. Clean: Remove IDs, fill NaN     │
        │ 2. Encode: Freq/Target encoding    │
        │ 3. Scale: QuantileTransformer      │
        │ 4. Split: Train/Val/Test (70/15/15)│
        └────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
    ┌──────────────┐          ┌──────────────────┐
    │ Row-Level    │          │ Incident-Level   │
    │ Triage Data  │          │ Remediation Data │
    │ (44 features)│          │ (features TBD)   │
    └──────────────┘          └──────────────────┘
        │                                 │
        ▼                                 ▼
    ┌──────────────────────┐     ┌────────────────────┐
    │ TRIAGE MODELS        │     │ REMEDIATION MODELS │
    ├──────────────────────┤     ├────────────────────┤
    │ • TabNet (primary)   │     │ • Account GBT      │
    │ • LightGBM (baseline)│     │ • Endpoint LR      │
    │ • XGBoost (baseline) │     │                    │
    └──────────────────────┘     └────────────────────┘
        │                                 │
        └────────────────┬────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │   HYBRID SCORING ENGINE            │
        │  (src/inference/hybrid_incident_   │
        │   scoring.py)                      │
        ├────────────────────────────────────┤
        │ • Load all models + artifacts      │
        │ • Score triage rows                │
        │ • Score incident remediation       │
        │ • Combine outputs                  │
        └────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
    ┌─────────────┐              ┌─────────────┐
    │  STDOUT     │              │  JSON Files │
    │  (main.py)  │              │  (reports/) │
    └─────────────┘              └─────────────┘
```

---

## Data Flow - Complete Pipeline

### End-to-End Example: From Alert to Remediation Decision

```
INPUT: Raw alert with 50+ features
│
├─ Step 1: Load (src/data/loader.py)
│  └─ CSV → DataFrame, memory optimized
│
├─ Step 2: Clean (src/preprocessing/cleaning.py)
│  ├─ Drop: [Id, AlertId, Timestamps, etc.]
│  ├─ Fill numerical: median
│  └─ Fill categorical: 'unknown'
│
├─ Step 3: Encode (src/preprocessing/encoding.py)
│  ├─ Target: TP→2, BP→1, FP→0
│  └─ Features: Frequency OR Target encoding (configurable)
│
├─ Step 4: Scale (src/preprocessing/scaling.py)
│  ├─ QuantileTransformer(output_distribution='normal')
│  └─ Fit on train, apply to val/test
│
├─ Step 5: Split (src/data/splitter.py)
│  └─ Stratified: 70% train, 15% val, 15% test
│
└─ OUTPUT: Preprocessed feature vectors (44 features each)

INFERENCE FLOW:
│
├─ Step 1: Load Hybrid Artifacts
│  ├─ TabNet model + scaler
│  ├─ GBT account_response model
│  ├─ LR endpoint_response model
│  └─ Thresholds and metadata
│
├─ Step 2: Triage Scoring
│  ├─ Scale features: QuantileTransformer.transform(X)
│  ├─ Predict: TabNet.predict_proba(X_scaled)
│  ├─ Class: argmax(probabilities)
│  └─ Confidence: max(probabilities)
│
├─ Step 3: Incident Aggregation
│  └─ Group triage outputs by incident
│
├─ Step 4: Remediation Scoring
│  ├─ Scale incident features: scaler.transform(incident_X)
│  ├─ Account: GBT.predict_proba(X) >= threshold
│  └─ Endpoint: LR.predict_proba(X) >= threshold
│
└─ OUTPUT: Combined prediction JSON
   ├─ triage: [predictions, probabilities, confidence]
   └─ remediation: [account_response, endpoint_response]
```

---

## Preprocessing Pipeline

### Architecture

The preprocessing pipeline is **modular, reusable, and artifact-driven**. Each step is independent and can be used separately for inference.

```
src/preprocessing/
├── cleaning.py          # Data cleaning (drop, fill)
├── encoding.py          # Feature encoding (frequency, target)
├── scaling.py           # Feature scaling (quantile, standard, minmax)
├── pipeline.py          # Orchestration + PreprocessingConfig

src/data/
├── loader.py            # Load GUIDE CSVs
├── splitter.py          # Stratified train/val/test split
└── pipeline.py          # Save/load preprocessed data

src/utils/
└── artifact_manager.py  # Save/load preprocessing artifacts
```

### Cleaning Step

**File**: `src/preprocessing/cleaning.py`

```python
def clean_data(df, target_col='IncidentGrade', verbose=True):
    """
    Operations:
    1. Drop identifier columns: Id, AlertId, etc.
    2. Fill numerical NaNs: median (configurable)
    3. Fill categorical NaNs: 'unknown' (configurable)
    4. Remove rows with missing target
    5. Validate no remaining NaN in features
    
    Input shape:  (N, 50+)
    Output shape: (N_clean, 50+)
    """
```

**Dropped Columns** (identifiers):
- Id
- AlertId
- Any timestamp columns
- Index columns

**Missing Value Strategy**:
- **Numerical**: Fill with median (robust to outliers)
- **Categorical**: Fill with 'unknown' (preserves signal)
- **Target**: Drop entire row (cannot predict without label)

### Encoding Step

**File**: `src/preprocessing/encoding.py`

Two encoding strategies available (configurable):

#### 1. Frequency Encoding (Default)

```python
FrequencyEncoder:
  For each categorical column:
    - Compute value frequency in training data
    - Map each value to its normalized frequency (0-1)
    - Unseen values in test set → 0
  
  Pros: Preserves ordinal signal (frequent ↔ common)
  Cons: Loss of cardinality information
```

**Example**:
```
DetectorId value counts:
  'A': 1000  → frequency 0.33
  'B': 2000  → frequency 0.67
  'C': 1000  → frequency 0.33

Mapping: {A: 0.33, B: 0.67, C: 0.33}
Test unseen 'D' → 0
```

#### 2. Target Encoding (Optional)

```python
TargetEncoder:
  For each categorical column:
    - Group by category
    - Compute mean target per category
    - Apply smoothing: smoothed_mean = (count*mean + λ*global_mean) / (count + λ)
    - Unseen → global_mean
  
  Pros: Captures target signal directly
  Cons: Risk of target leakage if not careful
```

**Smoothing Formula** (λ=1.0):
```
Smoothed Mean = (count_category * mean_category + λ * global_mean) 
                / (count_category + λ)
```

Ensures rare categories don't overfit to their small sample.

#### Target Variable Encoding

```python
Target Mapping:
  'TP' (TruePositive)    → 2
  'BP' (BenignPositive)  → 1
  'FP' (FalsePositive)   → 0
```

Applied via `LabelEncoder`, deterministic and reversible.

### Scaling Step

**File**: `src/preprocessing/scaling.py`

#### QuantileTransformer (Primary)

```python
ScalingPipeline(method='quantile', output_dist='normal'):
  - Fit on training features only
  - Transform all numerical features to normal distribution
  - n_quantiles=1000, subsample=100000
  - Preserves monotonic relationships
  
  Key Property: Maps any distribution → standard normal
  Robust to outliers (doesn't use min/max)
```

**Process**:
1. Fit: Learn quantile boundaries from X_train
2. Transform: Map X_val, X_test to learned distribution
3. Output: Approximately N(0,1) distributed features

#### Alternative Scalers

- **StandardScaler**: Z-norm (μ=0, σ=1), sensitive to outliers
- **MinMaxScaler**: Rescale to [0, 1], bounded

### Splitting Step

**File**: `src/data/splitter.py`

```python
split_data(X, y, test_size=0.2, val_size=0.15, stratify=True):
  """
  Stratified split to preserve class distribution:
  
  1. First split: test_size=0.2 → (train+val, test)
  2. Then split train+val: val/(1-test_size)=0.15 → (train, val)
  
  Result:
    - Train: 70% (stratified)
    - Val:   15% (stratified)
    - Test:  15% (stratified)
  
  Stratification ensures all sets have same class proportions as original.
  """
  
  # Save split indices for reproducibility
  # data/splits/train_indices.npy, val_indices.npy, test_indices.npy
```

### Pipeline Orchestration

**File**: `src/preprocessing/pipeline.py`

```python
class PreprocessingConfig:
    """Central configuration for all preprocessing steps."""
    
    def __init__(
        self,
        train_path: str = "data/raw/GUIDE_Train.csv",
        test_path: str = "data/raw/GUIDE_Test.csv",
        encoding_method: str = "frequency",      # frequency | target
        scaling_method: str = "quantile",        # quantile | standard | minmax
        scaling_output_dist: str = "normal",     # normal | uniform
        apply_scaling: bool = True,
        numerical_fill_strategy: str = "median", # median | mean
        categorical_fill_value: str = "unknown",
        random_state: int = 42
    ): ...

def run_preprocessing(config: PreprocessingConfig = None):
    """
    Execute full pipeline:
    1. Load data
    2. Clean (both train and test)
    3. Separate X, y
    4. Encode target
    5. Encode features (fit on train)
    6. Split train into train/val/test
    7. Apply scaling (fit on train)
    8. Save artifacts
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, metadata_dict)
    """
```

### Artifact Preservation

**File**: `src/utils/artifact_manager.py`

Preprocessing artifacts are saved for **reproducible inference**:

```
models/artifacts/
├── encoders.pkl          # Feature encoders (frequency/target maps)
├── target_mapping.pkl    # Target variable mapping (TP→2, etc.)
└── scaler.pkl            # QuantileTransformer fitted on training data
```

These are **reused during inference** to ensure test/production data undergoes identical transformations.

---

## Model Layer - Deep Dive

### Triage Models (Row-Level Classification)

#### 1. TabNet (Primary Production Model)

**Directory**: `src/models/tabnet/`

**Architecture**: Attention-based feature selection + sequential decision trees

**Training** (`train.py`):
```python
# From existing training code
TabNetClassifier(
    n_steps=3,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': 2e-2},
    seed=42,
    # Hyperparameters often tuned via Optuna
)

Training:
  - Batch size: varies (configurable)
  - Early stopping: patience=50 epochs
  - Loss: MulticlassLogLoss (3 classes)
  - Optimizer: Adam with learning rate scheduler
```

**Artifacts**:
```
models/tabnet/
├── triage_model.pkl          # Serialized TabNetClassifier
├── triage_model_scaler.pkl   # Fitted QuantileTransformer
└── triage_model_config.json  # Model metadata
   {
     "model_type": "TabNetClassifier",
     "n_classes": 3,
     "classes": [0, 1, 2],
     "class_names": ["FalsePositive", "BenignPositive", "TruePositive"],
     "n_features": 44,
     "feature_count": 44
   }
```

**Prediction** (`predict.py`):
```python
def load_model(model_dir="models/tabnet", model_name="triage_model"):
    model = joblib.load(f"{model_dir}/{model_name}.pkl")
    scaler = joblib.load(f"{model_dir}/{model_name}_scaler.pkl")
    config = load_json(f"{model_dir}/{model_name}_config.json")
    return model, scaler, config

def predict(model, data, return_proba=True):
    # data: (N, 44) numpy array or DataFrame
    
    X_scaled = scaler.transform(data.astype(np.float32))
    probabilities = model.predict_proba(X_scaled)  # (N, 3)
    predictions = np.argmax(probabilities, axis=1)  # (N,)
    
    return predictions, probabilities
```

**Why TabNet**:
- Attention mechanism provides explainability (which features matter per prediction)
- Handles mixed feature types naturally
- Sequential decision trees capture non-linear interactions
- PyTorch backend enables GPU acceleration

#### 2. LightGBM (Gradient Boosting Baseline)

**Directory**: `src/models/lightgbm/`

**Architecture**: Gradient Boosted Decision Trees (GBDT)

**Training** (`train.py`):
```python
LGBMClassifier(
    objective="multiclass",
    num_class=3,
    class_weight="balanced",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
```

**Key Features**:
- Leaf-wise tree growth (more flexible than depth-wise)
- Handles categorical features natively
- class_weight="balanced" addresses class imbalance
- Early stopping prevents overfitting

**Artifacts**:
```
models/lightgbm/
├── triage_model.pkl
├── triage_model_config.json
└── (no separate scaler - GBDT doesn't require it)
```

**Prediction** (`predict.py`):
```python
def predict(model, data, return_proba=True):
    # data: (N, 44) numpy array
    
    probabilities = model.predict_proba(data.astype(np.float32))
    predictions = np.argmax(probabilities, axis=1)
    
    return predictions, probabilities
```

**Why LightGBM as Baseline**:
- Extremely efficient (speed + memory)
- Currently has strongest saved triage metrics
- No preprocessing required (handles raw features)
- Interpretable decision paths

#### 3. XGBoost (Gradient Boosting Alternative)

**Directory**: `src/models/xgboost/`

**Architecture**: eXtreme Gradient Boosting

**Training** (`train.py`):
```python
# Marked as "legacy JSON-export model"
# Can load from:
# 1. models/xgboost/triage_model.pkl (if available)
# 2. models/xgboost_model.json (fallback legacy format)
```

**Prediction** (`predict.py`):
```python
def load_model(model_dir="models/xgboost"):
    if os.path.exists(f"{model_dir}/triage_model.pkl"):
        model = joblib.load(f"{model_dir}/triage_model.pkl")
    else:
        # Legacy format
        model = XGBClassifier()
        model.load_model("models/xgboost_model.json")
    return model, config
```

**Why XGBoost**:
- Regularization (L1/L2) prevents overfitting
- Handles missing values natively
- Parallel tree boosting
- Cross-validation built-in

### Remediation Models (Incident-Level Binary Predictions)

These are **classical models** trained on **incident-level aggregated features** (not row-level).

#### Account Response Prediction

**Model**: Gradient Boosted Trees (GBT)

```
models/classical/account_response_gbt.pkl
```

**Purpose**: Should the user account be remediated/disabled?

**Input**: Incident-level features (e.g., count of alerts per account, avg severity, etc.)

**Output**: Binary probability → thresholded to decision

**Threshold**: Loaded from `remediation_thresholds.json`
```json
{
  "account_response": 0.5,  // Example threshold
  "endpoint_response": 0.3
}
```

#### Endpoint Response Prediction

**Model**: Logistic Regression (LR)

```
models/classical/endpoint_response_lr.pkl
```

**Purpose**: Should the endpoint machine be remediated (isolated)?

**Input**: Incident-level features

**Output**: Binary probability → thresholded to decision

**Why LR for Endpoint**:
- Simpler model for harder problem
- `endpoint_response` is weak signal (few positive incidents)
- LR provides probabilistic outputs without overfitting
- Interpretable feature coefficients

### Scaling for Classical Models

**File**: `models/classical/incident_scaler.pkl`

```python
# Applied to incident-level features before classical model inference
incident_scaled = incident_scaler.transform(incident_features)

# Type: QuantileTransformer (same as triage preprocessing)
# Fit: On incident-level training features only
```

### Model Comparison Metrics

**Saved Locations**:
```
reports/metrics/
├── triage_metrics.json                    # TabNet performance
├── lightgbm_triage_metrics.json
├── xgboost_triage_metrics.json
├── lightgbm_val_metrics.json
├── lightgbm_test_metrics.json
└── classical_remediation_comparison.json

reports/comparisons/
├── tabnet_vs_lightgbm_triage.json
├── tabnet_vs_xgboost_triage.json
└── endpoint_response_limitations.md  # Explains weak signal
```

**Metric Structure**:
```json
{
  "confusion_matrix": [[TN, FP, ...], ...],
  "overall_accuracy": 0.92,
  "macro_f1": 0.87,
  "per_class": {
    "FalsePositive": {"precision": 0.95, "recall": 0.88, "f1": 0.91, "support": 1000},
    "BenignPositive": {"precision": 0.82, "recall": 0.80, "f1": 0.81, "support": 500},
    "TruePositive": {"precision": 0.91, "recall": 0.94, "f1": 0.92, "support": 300}
  }
}
```

---

## Inference Engine

### Hybrid Incident Scoring

**File**: `src/inference/hybrid_incident_scoring.py`

The hybrid scorer combines triage and remediation predictions into a unified output.

#### Load Artifacts

```python
def load_hybrid_models(triage_model_name="triage_model", verbose=False):
    """
    Load all components needed for hybrid scoring:
    - TabNet triage model + scaler
    - Account response GBT model
    - Endpoint response LR model
    - Incident-level scaler
    - Thresholds and metadata
    
    Returns: Dict with all artifacts preloaded
    """
    
    triage_model, triage_scaler, triage_config = load_tabnet_model(...)
    account_model = joblib.load("models/classical/account_response_gbt.pkl")
    endpoint_model = joblib.load("models/classical/endpoint_response_lr.pkl")
    incident_scaler = joblib.load("models/classical/incident_scaler.pkl")
    
    with open("models/classical/remediation_thresholds.json") as f:
        thresholds = json.load(f)
    with open("models/classical/remediation_model_metadata.json") as f:
        metadata = json.load(f)
    
    return {
        "triage_model": triage_model,
        "triage_scaler": triage_scaler,
        "triage_config": triage_config,
        "account_model": account_model,
        "endpoint_model": endpoint_model,
        "incident_scaler": incident_scaler,
        "thresholds": thresholds,
        "remediation_metadata": metadata,
    }
```

#### Triage Scoring

```python
def _score_triage_rows(incident_rows_df, triage_model, triage_scaler):
    """
    Score each row (alert) at the row level.
    
    Process:
    1. Convert DataFrame to float32 numpy array
    2. Apply triage_scaler.transform() for feature scaling
    3. Run triage_model.predict_proba() for 3-class probabilities
    4. Extract argmax for class prediction
    5. Extract max probability for confidence
    
    Input:
      incident_rows_df: (N_alerts, 44) preprocessed features
      Already cleaned, encoded, scaled to match training
    
    Output: {
      "predictions": array(N_alerts,),      # [0, 1, 2]
      "probabilities": array(N_alerts, 3),  # [[p0, p1, p2], ...]
      "confidence": array(N_alerts,)        # [0.92, 0.87, ...]
    }
    """
    
    triage_features = incident_rows_df.to_numpy(dtype=np.float32)
    triage_scaled = triage_scaler.scaler.transform(triage_features)
    
    triage_proba = triage_model.predict_proba(triage_scaled)
    triage_pred = np.argmax(triage_proba, axis=1)
    triage_confidence = triage_proba.max(axis=1)
    
    return {
        "predictions": triage_pred,
        "probabilities": triage_proba,
        "confidence": triage_confidence,
    }
```

#### Remediation Scoring

```python
def _score_incident_remediation(
    incident_features_df,
    account_model, endpoint_model,
    incident_scaler,
    thresholds,
    feature_columns
):
    """
    Score at incident level for remediation decisions.
    
    Process:
    1. Extract required incident features (from metadata)
    2. Apply incident_scaler.transform() for scaling
    3. Run account_model.predict_proba() → probability → threshold
    4. Run endpoint_model.predict_proba() → probability → threshold
    
    Thresholding:
      account_pred = (account_proba >= threshold["account_response"])
      endpoint_pred = (endpoint_proba >= threshold["endpoint_response"])
    
    Input:
      incident_features_df: (N_incidents, M) aggregated features
      feature_columns: ["field1", "field2", ...] from metadata
    
    Output: {
      "account_response": {
        "predictions": array(N_incidents,),     # [0, 1, 0, ...]
        "probabilities": array(N_incidents,),   # [0.45, 0.78, ...]
        "threshold": 0.5
      },
      "endpoint_response": {
        "predictions": array(N_incidents,),
        "probabilities": array(N_incidents,),
        "threshold": 0.3
      }
    }
    """
    
    # Validate all required features present
    missing = [c for c in feature_columns if c not in incident_features_df.columns]
    if missing:
        raise ValueError(f"Missing incident features: {missing}")
    
    incident_X = incident_features_df.loc[:, feature_columns]
    incident_scaled = incident_scaler.transform(incident_X)
    
    account_proba = account_model.predict_proba(incident_scaled)[:, 1]
    endpoint_proba = endpoint_model.predict_proba(incident_scaled)[:, 1]
    
    account_pred = (account_proba >= thresholds["account_response"]).astype(int)
    endpoint_pred = (endpoint_proba >= thresholds["endpoint_response"]).astype(int)
    
    return {
        "account_response": {
            "predictions": account_pred,
            "probabilities": account_proba,
            "threshold": thresholds["account_response"],
        },
        "endpoint_response": {
            "predictions": endpoint_pred,
            "probabilities": endpoint_proba,
            "threshold": thresholds["endpoint_response"],
        },
    }
```

#### Combined Scoring

```python
def score_incident(incident_rows_df, incident_features_df, artifacts=None):
    """
    Main entry point for hybrid scoring.
    
    Args:
      incident_rows_df: Row-level features for triage
      incident_features_df: Incident-level features for remediation
      artifacts: Preloaded from load_hybrid_models() (or loads on demand)
    
    Returns: {
      "triage": {
        "predictions": [...],
        "probabilities": [...],
        "confidence": [...]
      },
      "remediation": {
        "account_response": {...},
        "endpoint_response": {...}
      }
    }
    """
    
    if artifacts is None:
        artifacts = load_hybrid_models(verbose=False)
    
    # Triage scoring (row-level)
    triage_outputs = _score_triage_rows(
        incident_rows_df,
        artifacts["triage_model"],
        artifacts["triage_scaler"]
    )
    
    # Remediation scoring (incident-level)
    remediation_outputs = _score_incident_remediation(
        incident_features_df,
        artifacts["account_model"],
        artifacts["endpoint_model"],
        artifacts["incident_scaler"],
        artifacts["thresholds"],
        artifacts["remediation_metadata"]["feature_columns"]
    )
    
    return {
        "triage": triage_outputs,
        "remediation": remediation_outputs,
    }
```

---

## Backend Implementation

### Current State: Streamlit Demo

**File**: `app.py`

The current backend is a **Streamlit application** (not FastAPI). This is a demo interface for testing models.

```python
import streamlit as st
from src.models.lightgbm.predict import LightGBMPredictor

st.set_page_config(page_title="AI Security Alert Classifier", layout="wide")
st.title("🔐 AI Security Incident Classifier")

@st.cache_resource
def load_model():
    return LightGBMPredictor()

predictor = load_model()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if st.button("🚀 Predict"):
        results = predictor.predict(df)
        
        # Display predictions and probabilities
        output_df = df.copy()
        output_df["Prediction"] = map_predictions(results['predictions'])
        output_df["Confidence"] = results['probabilities'].max(axis=1)
        
        st.dataframe(output_df)
        st.download_button(
            label="📥 Download Results",
            data=output_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
```

**Limitations**:
- Streamlit is single-threaded (not suitable for production)
- Limited to CSV upload (no streaming)
- No API endpoints (no REST calls)
- State management is file-based

### Intended Architecture: FastAPI Backend

Tests and `run_demo.py` expect a **FastAPI backend** that doesn't currently exist.

**Expected Interface** (from `tests/test_api.py`):
```python
# Expected endpoints (NOT YET IMPLEMENTED):

GET /health
Response: {
  "status": "healthy" | "unhealthy",
  "models": {
    "xgboost": {"loaded": true, "error": null},
    "lightgbm": {"loaded": true, "error": null},
    "tabnet": {"loaded": true, "error": null},
    "remediation": {"loaded": true, "error": null}
  }
}

POST /predict
Request: {"features": [1.0, 2.0, ...], "model": "tabnet"}
Response: {
  "model": "tabnet",
  "prediction": 2,
  "probabilities": [0.05, 0.10, 0.85],
  "confidence": 0.85
}

GET /metrics
Response: {
  "confusion_matrix": [[950, 40, 10], ...],
  "accuracy": 0.92,
  "macro_f1": 0.87,
  "per_class": {
    "FalsePositive": {"precision": 0.95, ...},
    ...
  }
}

GET /sample-features?split=test&row=0
Response: {
  "features": [1.0, 2.0, ..., 44 total],
  "dataset": "v1",
  "split": "test",
  "row": 0,
  "feature_count": 44,
  "target": 2,
  "source": "reports/metrics/..."
}

POST /remediation-predict
Request: {"incident_features": [...]}
Response: {
  "account_response": {"prediction": 1, "probability": 0.75, ...},
  "endpoint_response": {"prediction": 0, "probability": 0.28, ...}
}

POST /evaluate
Response: {
  "source": "reports/metrics/...",
  "message": "Metrics loaded from disk"
}
```

**To Implement FastAPI Backend**:

1. Create `backend/app.py` with FastAPI instance
2. Implement `/health` → Check model loading
3. Implement `/predict` → Call triage inference
4. Implement `/metrics` → Load from `reports/metrics/`
5. Implement `/remediation-predict` → Call hybrid scorer
6. Add CORS middleware for React frontend
7. Start with: `uvicorn app:app --host 0.0.0.0 --port 8000`

---

## Frontend Architecture

### Frontend Stack

**Location**: `soc-frontend/`

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Package Manager**: npm

### Project Structure

```
soc-frontend/
├── src/
│   ├── App.tsx                 # Root component, routing
│   ├── main.tsx                # Entry point
│   ├── index.css               # Global styles
│   ├── vite-env.d.ts           # Vite type definitions
│   ├── components/             # Reusable UI components
│   ├── pages/                  # Page components
│   │   ├── Dashboard.tsx       # Overview + stats
│   │   ├── Triage.tsx          # Alert triage interface
│   │   ├── Remediation.tsx     # Remediation actions
│   │   └── Analytics.tsx       # Metrics dashboard
│   ├── services/               # API layer
│   │   └── api.ts              # Backend HTTP calls
│   ├── hooks/                  # React hooks
│   ├── lib/                    # Utilities
│   │   └── store.tsx           # Global state (AlertProvider)
│   └── types/                  # TypeScript interfaces
│       └── alert.ts            # Domain types
├── package.json                # Dependencies
├── vite.config.ts              # Vite configuration
├── tailwind.config.js          # Tailwind theme
├── tsconfig.json               # TypeScript config
└── postcss.config.js           # PostCSS config
```

### App Structure

**File**: `App.tsx`

```typescript
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AlertProvider } from './lib/store';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Triage from './pages/Triage';
import Remediation from './pages/Remediation';
import Analytics from './pages/Analytics';

export default function App() {
  return (
    <AlertProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="triage" element={<Triage />} />
            <Route path="remediation" element={<Remediation />} />
            <Route path="analytics" element={<Analytics />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AlertProvider>
  );
}
```

### API Layer

**File**: `services/api.ts`

Central point for all backend communication.

```typescript
export const BASE_URL = 
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Type definitions
interface PredictResponse {
  model: string;
  prediction: number;        // 0=FalsePositive, 1=BenignPositive, 2=TruePositive
  probabilities: [number, number, number];  // [p0, p1, p2]
  confidence: number;
}

interface RemediationResponse {
  account_response: {
    prediction: 0 | 1;
    probability: number;
    threshold: number;
  };
  endpoint_response: {
    prediction: 0 | 1;
    probability: number;
    threshold: number;
  };
}

interface BackendHealthResponse {
  status: 'healthy' | 'unhealthy';
  models: Record<string, { loaded: boolean; error?: string | null }>;
}

// Main API functions
export async function predict(
  features: number[],
  model: PredictionModel
): Promise<PredictResponse> {
  if (!Array.isArray(features) || features.length === 0) {
    throw new Error('Features must be non-empty numeric array');
  }

  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features, model }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Prediction failed (${res.status}): ${err}`);
  }

  const data = await res.json() as PredictResponse;
  
  if (!Array.isArray(data.probabilities) || data.probabilities.length !== 3) {
    throw new Error('Backend returned invalid probability payload');
  }

  return data;
}

export async function healthStatus(): Promise<BackendHealthResponse> {
  const res = await fetch(`${BASE_URL}/health`, {
    signal: AbortSignal.timeout(3000)
  });

  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`);
  }

  return await res.json() as BackendHealthResponse;
}

export async function remediationPredict(
  incidentFeatures: number[]
): Promise<RemediationResponse> {
  const res = await fetch(`${BASE_URL}/remediation-predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ incident_features: incidentFeatures }),
  });

  if (!res.ok) {
    throw new Error(`Remediation prediction failed (${res.status})`);
  }

  return await res.json() as RemediationResponse;
}
```

### State Management

**File**: `lib/store.tsx`

Uses React Context API for global state.

```typescript
interface Alert {
  id: string;
  features: number[];
  prediction?: number;
  probabilities?: [number, number, number];
  confidence?: number;
  timestamp: Date;
}

interface AlertContextType {
  alerts: Alert[];
  addAlert: (alert: Alert) => void;
  updateAlert: (id: string, data: Partial<Alert>) => void;
  clearAlerts: () => void;
}

const AlertContext = createContext<AlertContextType>(null!);

export function AlertProvider({ children }: { children: React.ReactNode }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);

  return (
    <AlertContext.Provider value={{alerts, addAlert, updateAlert, clearAlerts}}>
      {children}
    </AlertContext.Provider>
  );
}

export function useAlerts() {
  return useContext(AlertContext);
}
```

### Pages

#### Dashboard (`pages/Dashboard.tsx`)

- Overview of recent alerts
- Key metrics summary
- Model health status
- Quick actions

#### Triage (`pages/Triage.tsx`)

- Alert input form (manual feature entry)
- Model selection dropdown
- Prediction results with probabilities
- Confidence indicator
- Alert history table

#### Remediation (`pages/Remediation.tsx`)

- Incident review interface
- Remediation action recommendation
- Account/Endpoint decision display
- Action confirmation/approval

#### Analytics (`pages/Analytics.tsx`)

- Model performance metrics (from API)
- Confusion matrix visualization
- Per-class precision/recall/F1
- Trend charts (if data available)

---

## Integration Points

### Frontend → Backend Communication Flow

#### 1. Health Check (On App Load)

```
Frontend Load
  └─ useEffect(() => { healthStatus() }, [])
      └─ GET /health
          └─ Check all models loaded
          └─ Update UI (green/red indicator)
          └─ Disable predict if unhealthy
```

#### 2. Triage Prediction

```
User enters 44 features + selects model
  └─ onClick: predict(features, model)
      └─ POST /predict
          {
            "features": [1.0, 2.0, ..., 44 items],
            "model": "tabnet" | "lightgbm" | "xgboost"
          }
          └─ Returns:
          {
            "model": "tabnet",
            "prediction": 2,
            "probabilities": [0.05, 0.10, 0.85],
            "confidence": 0.85
          }
          └─ Update local state
          └─ Render prediction cards
```

#### 3. Remediation Prediction

```
User views incident with row-level predictions
  └─ onClick: remediationPredict(incidentFeatures)
      └─ POST /remediation-predict
          {
            "incident_features": [agg1, agg2, ...]
          }
          └─ Returns:
          {
            "account_response": {
              "prediction": 1,
              "probability": 0.75,
              "threshold": 0.5
            },
            "endpoint_response": {
              "prediction": 0,
              "probability": 0.28,
              "threshold": 0.3
            }
          }
          └─ Render remediation panel
          └─ Show account/endpoint decisions
```

#### 4. Metrics Display

```
User navigates to Analytics
  └─ useEffect(() => { metrics() }, [])
      └─ GET /metrics
          └─ Load triage metrics from disk
          └─ Render confusion matrix
          └─ Render per-class stats
          └─ Update trend charts
```

### Error Handling

```typescript
// API layer handles:
// - Network errors → "Unable to reach backend"
// - Invalid response shape → "Backend returned invalid payload"
// - HTTP errors (4xx, 5xx) → "Prediction failed (status): message"

// Frontend layer should:
// - Catch and display errors to user
// - Provide retry buttons
// - Disable features if backend unhealthy
// - Log to error tracking (if available)
```

---

## Running the System

### Prerequisites

```bash
# Python
Python 3.9+

# Node.js
Node.js 16+ (for frontend)
npm 8+

# System
At least 4GB RAM (8GB recommended for TabNet)
```

### Installation

#### Backend Setup

```bash
# Clone repository
git clone <repo>
cd SOC\ Intelligence

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Key dependencies:
# - streamlit (demo backend)
# - lightgbm (triage baseline)
# - xgboost (triage baseline)
# - pytorch_tabnet (triage primary)
# - scikit-learn (preprocessing, classical models)
# - pandas, numpy (data handling)
# - joblib (artifact serialization)
```

#### Frontend Setup

```bash
cd soc-frontend

# Install dependencies
npm install

# Environment variables (create .env.local)
VITE_API_BASE_URL=http://localhost:8000
```

### Running Individual Components

#### 1. Run Preprocessing

```bash
python -c "
from src.preprocessing.pipeline import PreprocessingConfig, run_preprocessing

config = PreprocessingConfig(encoding_method='frequency')
X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing(config)

print(f'Train shape: {X_train.shape}')
print(f'Val shape:   {X_val.shape}')
print(f'Test shape:  {X_test.shape}')
"
```

#### 2. Run Batch Inference (main.py)

```bash
# Default: 5 rows from test set, 1 incident
python main.py

# Custom limits
python main.py --row-limit 10 --incident-limit 2

# Save output
python main.py --row-limit 50 --output-json results/predictions.json
```

**Output**:
```json
{
  "row_count": 50,
  "incident_count": 2,
  "triage_predictions": [0, 1, 2, 0, ...],
  "triage_confidence": [0.92, 0.87, 0.95, ...],
  "account_response_prediction": [1, 0],
  "account_response_probability": [0.75, 0.45],
  "endpoint_response_prediction": [0, 1],
  "endpoint_response_probability": [0.28, 0.82]
}
```

#### 3. Run Streamlit Demo

```bash
streamlit run app.py

# Opens: http://localhost:8501
# Upload CSV, click "Predict", download results
```

#### 4. Run Frontend Dev Server

```bash
cd soc-frontend
npm run dev

# Opens: http://localhost:5173
# (Requires backend running on port 8000)
```

### Running Full Stack Demo

```bash
# All-in-one command (starts backend + frontend)
python scripts/run_demo.py

# Automatically:
# 1. Starts FastAPI backend (expected at port 8000)
# 2. Starts React frontend (port 5173)
# 3. Opens browser to http://localhost:5173
# 4. Press Ctrl+C to stop both
```

**Note**: Currently, `run_demo.py` expects FastAPI backend. To use Streamlit instead:

```bash
# Terminal 1: Streamlit
streamlit run app.py

# Terminal 2: Frontend
cd soc-frontend && npm run dev

# Terminal 3: Optional - Batch inference
python main.py --row-limit 100
```

---

## Testing Guide

### Unit Tests

**File**: `tests/test_api.py`

Tests expect FastAPI backend. Current status: Tests won't pass until backend is implemented.

```bash
# Run tests (when FastAPI backend exists)
pytest tests/test_api.py -v

# Individual tests:
pytest tests/test_api.py::test_health_reports_model_statuses -v
pytest tests/test_api.py::test_metrics_payload_shape -v
pytest tests/test_api.py::test_processed_sample_can_feed_loaded_models -v
```

### Manual Testing

#### Test Preprocessing Pipeline

```python
from src.preprocessing.pipeline import run_preprocessing
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import encode_target, encode_features
from src.preprocessing.scaling import ScalingPipeline

# Test cleaning
df = pd.read_csv("data/raw/GUIDE_Train.csv")
df_clean = clean_data(df)
assert df_clean.isnull().sum().sum() == 0, "Cleaning failed"

# Test encoding
y_encoded, mapping = encode_target(df_clean['IncidentGrade'])
assert mapping == {'TP': 2, 'BP': 1, 'FP': 0}, "Encoding failed"

# Test full pipeline
X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing()
assert X_train.shape[1] == 44, "Feature count mismatch"
```

#### Test Triage Models

```python
from src.models.lightgbm.predict import load_model, predict
import pandas as pd

# Load model
model, config = load_model(verbose=True)

# Load test data
X_test = pd.read_csv("data/processed/v1/X_test.csv").head(10)

# Predict
predictions, probabilities = predict(model, X_test)

assert predictions.shape == (10,), "Prediction shape mismatch"
assert probabilities.shape == (10, 3), "Probability shape mismatch"
assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities don't sum to 1"

print(f"Predictions: {predictions}")
print(f"Confidence: {probabilities.max(axis=1)}")
```

#### Test Hybrid Scoring

```python
from src.inference.hybrid_incident_scoring import load_hybrid_models, score_incident
import pandas as pd

# Load artifacts
artifacts = load_hybrid_models(verbose=True)

# Load test data
incident_rows = pd.read_csv("data/processed/v1/X_test.csv").head(5)
incident_features = pd.read_csv("data/processed/v1/X_incident_remediation_test.csv").head(1)

# Score
results = score_incident(incident_rows, incident_features, artifacts)

# Validate output structure
assert "triage" in results, "Missing triage output"
assert "remediation" in results, "Missing remediation output"
assert results["triage"]["predictions"].shape == (5,)
assert results["remediation"]["account_response"]["predictions"].shape == (1,)

print(json.dumps({
    "triage_predictions": results["triage"]["predictions"].tolist(),
    "remediation": {
        "account": results["remediation"]["account_response"]["predictions"].tolist(),
        "endpoint": results["remediation"]["endpoint_response"]["predictions"].tolist(),
    }
}, indent=2))
```

#### Test Frontend API Integration

```typescript
// In frontend console (F12 → Console tab)

import { predict, healthStatus, remediationPredict } from './services/api';

// Test health
const health = await healthStatus();
console.log('Health:', health);

// Test prediction
const features = Array(44).fill(0.5); // 44 dummy features
const result = await predict(features, 'tabnet');
console.log('Prediction:', result);

// Test remediation
const incidentFeatures = Array(20).fill(0.5); // Dummy incident features
const remediation = await remediationPredict(incidentFeatures);
console.log('Remediation:', remediation);
```

---

## Debugging & Troubleshooting

### Common Issues

#### 1. Model Loading Fails

**Symptom**: `FileNotFoundError: LightGBM model not found`

**Causes**:
- Model artifacts missing (not trained yet)
- Wrong working directory
- Path configuration incorrect

**Solution**:
```bash
# Check model existence
ls models/lightgbm/
# Should see: triage_model.pkl, triage_model_config.json

# Verify working directory
pwd
# Should be: /path/to/SOC Intelligence

# Re-run training
python src/models/lightgbm/train.py
```

#### 2. Preprocessing Artifacts Missing

**Symptom**: `FileNotFoundError: Encoders not found in models/artifacts/`

**Causes**:
- Preprocessing pipeline not run
- Artifacts saved to wrong location

**Solution**:
```python
# Run preprocessing and save artifacts
from src.preprocessing.pipeline import run_preprocessing
X_train, X_val, X_test, y_train, y_val, y_test, metadata = run_preprocessing()

# Verify artifacts saved
import os
assert os.path.exists("models/artifacts/encoders.pkl")
assert os.path.exists("models/artifacts/target_mapping.pkl")
assert os.path.exists("models/artifacts/scaler.pkl")
```

#### 3. Shape Mismatch in Inference

**Symptom**: `ValueError: input has 50 features, expected 44`

**Causes**:
- Input data not preprocessed
- Preprocessing step missing or wrong
- Feature set changed between training and inference

**Solution**:
```python
# Always preprocess before inference
from src.preprocessing.encoding import encode_target, encode_features
from src.preprocessing.scaling import ScalingPipeline

# Apply same transformations as training
X_inference_encoded, _, _ = encode_features(
    X_inference, 
    X_inference,  # Dummy for transform
    y_dummy,
    encoding_method='frequency'  # Match training
)

# Apply scaling
scaler = ScalingPipeline(method='quantile', output_dist='normal')
X_inference_scaled = scaler.transform(X_inference_encoded)

# Now can predict
predictions = model.predict_proba(X_inference_scaled)
```

#### 4. CORS Errors (Frontend ↔ Backend)

**Symptom**: 
```
Access to XMLHttpRequest has been blocked by CORS policy
```

**Causes**:
- Frontend (port 5173) calling backend (port 8000)
- Backend doesn't have CORS middleware
- API_BASE_URL misconfigured

**Solution**:

If implementing FastAPI backend:
```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

If using Streamlit, CORS is not applicable (Streamlit doesn't expose HTTP APIs).

#### 5. Memory Issues with TabNet

**Symptom**: `MemoryError` or `RuntimeError: CUDA out of memory`

**Causes**:
- TabNet is memory-intensive (PyTorch-based)
- Batch size too large
- GPU memory insufficient

**Solution**:
```python
# Use batch inference instead of loading all data
from pathlib import Path

# Process in chunks
batch_size = 32
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    predictions = model.predict_proba(batch)
    # Process batch results
```

#### 6. Feature Encoding Inconsistency

**Symptom**: Predictions wildly different between train/test

**Causes**:
- Encoders not saved/loaded correctly
- Different encoding method used
- Target leakage in target encoding

**Solution**:
```python
# Always use saved encoders
from src.utils.artifact_manager import load_artifacts

artifacts = load_artifacts()
encoders = artifacts['encoders']
target_mapping = artifacts['target_mapping']
scaler = artifacts['scaler']

# Apply exact same transformations as training
```

### Debugging Checklist

```
□ Working directory correct?
  pwd → /path/to/SOC Intelligence

□ Virtual environment activated?
  which python → .../.venv/bin/python (or Scripts\python.exe)

□ Dependencies installed?
  pip list | grep -E "lightgbm|xgboost|torch|pandas"

□ Model artifacts exist?
  ls models/*/triage_model.pkl

□ Preprocessing artifacts exist?
  ls models/artifacts/

□ Input data shape correct?
  X.shape == (N, 44) after preprocessing

□ No NaN in input?
  assert not np.isnan(X).any()

□ Feature scaling applied?
  assert X.std() ≈ 1.0  (after scaling)

□ Output shape correct?
  predictions.shape == (N,)
  probabilities.shape == (N, 3)

□ Probabilities sum to 1?
  assert np.allclose(probabilities.sum(axis=1), 1.0)

□ Thresholds set correctly?
  account_response_threshold in [0, 1]
  endpoint_response_threshold in [0, 1]
```

---

## Design Rationale

### Why Two-Level Hierarchy?

**Row-Level Triage vs Incident-Level Remediation**

The system separates triage (alert classification) from remediation (action recommendation) because:

1. **Different Granularity**
   - Triage labels are at row level (individual alerts)
   - Remediation labels are at incident level (grouped alerts)
   - Mixing granularities causes target leakage and poor performance

2. **Different Features**
   - Triage uses raw alert features (44 dimensions)
   - Remediation uses aggregated incident features
   - TabNet excels at raw features; classical models excel at aggregations

3. **Different Success Metrics**
   - Triage optimizes for alert classification accuracy
   - Remediation optimizes for action correctness
   - Cannot optimize both simultaneously with single model

### Why TabNet for Triage?

**Deep Learning vs. Gradient Boosting Trade-offs**

TabNet chosen as primary triage model because:

1. **Attention-Based Feature Selection**
   - Provides explainability (which features matter)
   - Handles interaction effects naturally
   - No manual feature engineering needed

2. **Handles Raw Features Well**
   - 44 dimensions ideal for deep learning
   - Can capture non-linear patterns
   - Robust to skewed feature distributions

3. **Compared to XGBoost/LightGBM**
   - TabNet: Flexible, interpretable, handles interactions
   - LightGBM: Currently strongest metrics (fastest training)
   - XGBoost: Middle ground (good generalization)

### Why Classical Models for Remediation?

**Why Not Deep Learning?**

1. **Incident-Level Features are Aggregated**
   - Fewer dimensions (≈20 vs 44)
   - Mostly linear relationships
   - Deep learning overkill

2. **Interpretability Requirement**
   - Security decisions need explainability
   - GBT and LR provide feature importance
   - Black-box models unacceptable for SOC

3. **Weak Signal Problem**
   - endpoint_response has very few positive incidents
   - Classical models generalize better with limited data
   - Deep learning would overfit

### Why Preprocessing Artifacts?

**Reproducibility and Inference Consistency**

Preprocessing artifacts are saved because:

1. **Ensures Test/Prod Consistency**
   - Same encoder → same mappings
   - Same scaler → same distributions
   - No data leakage between train/test

2. **Enables Inference at Scale**
   - Load artifacts once, apply to many samples
   - No need to retrain encoders
   - Memory-efficient (artifacts are small)

3. **Traceability**
   - Can audit which encoding was used
   - Can revert to old preprocessing if needed
   - Reproducible research

### Why Threshold-Based Remediation?

**Binary Classification → Threshold-Based Decision**

Instead of direct classification, remediation uses:
```
probability_threshold → decision
account_response = (GBT.predict_proba >= 0.5) → action/no-action
```

**Rationale**:

1. **Flexibility**
   - Can adjust threshold without retraining
   - Different thresholds for different risk profiles
   - Can optimize for precision vs recall

2. **Uncertainty Handling**
   - Probabilities near 0.5 → uncertain
   - Can escalate to manual review
   - Can set threshold at 0.7 for high confidence only

3. **Business Logic Integration**
   - Threshold can incorporate business rules
   - Can vary threshold by account importance
   - Can A/B test different thresholds

### Why Frequency Encoding?

**No One-Hot Encoding**

Frequency encoding chosen over one-hot because:

1. **Cardinality Issues**
   - Categorical features have high cardinality (100+)
   - One-hot would create >100 new columns
   - Exponential feature space explosion

2. **Handles Unseen Values**
   - Frequency: Unseen → 0 (handled gracefully)
   - One-hot: Unseen → error or missing dimension

3. **Ordinal Signal**
   - Frequency encodes actual occurrence pattern
   - Common categories get higher values
   - Preserves information about popularity

4. **Compatible with All Models**
   - Works with TabNet, GBDT, LR
   - No special handling needed
   - Consistent across training/inference

### Why QuantileTransformer?

**Scaling Method Choice**

QuantileTransformer chosen for scaling because:

1. **Robust to Outliers**
   - Doesn't use min/max (outlier-sensitive)
   - Maps to empirical quantiles
   - Handles skewed distributions well

2. **Theory-Driven**
   - Output distribution → N(0,1) or Uniform[0,1]
   - Matches distributional assumptions of some models
   - Better for models assuming normality

3. **Stable**
   - Subsample for efficiency (100K quantiles default)
   - Consistent across training/inference
   - No hyperparameters to tune

---

## Summary

The SOC Intelligence platform is a **production-ready hybrid inference system** combining:

- **Modular preprocessing** with artifact-based reproducibility
- **Multi-model triage** with TabNet (primary) and classical baselines
- **Incident-level remediation** using classical models
- **Complete inference pipeline** for batch and online scoring
- **React frontend** for interactive prediction (development)
- **Streamlit demo** for quick testing

The architecture prioritizes:
- **Separability** of concerns (preprocessing, triage, remediation)
- **Reproducibility** through artifact saving
- **Explainability** via classical models and attention mechanisms
- **Scalability** through efficient loading and batch processing
- **Debuggability** with detailed logging and artifact tracking

For production deployment, the FastAPI backend should be implemented to replace Streamlit, enabling high-concurrency REST endpoints and proper monitoring/observability.


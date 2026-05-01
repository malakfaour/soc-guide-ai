import pandas as pd
import numpy as np
import pickle
import os

print("=" * 80)
print("DATA VERIFICATION FOR LIGHTGBM")
print("=" * 80)

# Load data
print("\n1. LOADING DATA...")
X_train = pd.read_csv('data/processed/v1/X_train.csv')
X_val = pd.read_csv('data/processed/v1/X_val.csv')
X_test = pd.read_csv('data/processed/v1/X_test.csv')
y_train = pd.read_csv('data/processed/v1/y_train.csv')
y_val = pd.read_csv('data/processed/v1/y_val.csv')
y_test = pd.read_csv('data/processed/v1/y_test.csv')

print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_val shape: {X_val.shape}")
print(f"✅ X_test shape: {X_test.shape}")
print(f"✅ y_train shape: {y_train.shape}")
print(f"✅ y_val shape: {y_val.shape}")
print(f"✅ y_test shape: {y_test.shape}")

# Check columns
print(f"\n2. FEATURE INFORMATION")
print(f"Number of features: {X_train.shape[1]}")
print(f"Feature columns: {list(X_train.columns[:10])}... (showing first 10)")

# Check target columns
print(f"\n3. TARGET INFORMATION")
print(f"Target columns: {list(y_train.columns)}")

# Check for missing values
print(f"\n4. MISSING VALUES CHECK")
print(f"Missing in X_train: {X_train.isnull().sum().sum()}")
print(f"Missing in y_train: {y_train.isnull().sum().sum()}")

# Check data types
print(f"\n5. DATA TYPES")
print(f"X_train dtypes: {X_train.dtypes.value_counts().to_dict()}")

# Check triage class distribution
print(f"\n6. TRIAGE CLASS DISTRIBUTION (IMBALANCE CHECK)")
if 'Triage' in y_train.columns:
    triage_dist = y_train['Triage'].value_counts().sort_index()
    print(triage_dist)
    print(f"\nClass imbalance ratio:")
    for cls in triage_dist.index:
        ratio = triage_dist[cls] / len(y_train) * 100
        print(f"  Class {cls}: {ratio:.2f}%")

# Check remediation columns
print(f"\n7. REMEDIATION LABELS")
remediation_cols = [col for col in y_train.columns if col != 'Triage']
print(f"Remediation columns: {remediation_cols}")
if remediation_cols:
    print(f"Number of remediation labels: {len(remediation_cols)}")
    # Check how many samples have remediation labels
    has_remediation = y_train[remediation_cols].notna().any(axis=1).sum()
    print(f"Samples with remediation labels: {has_remediation} ({has_remediation/len(y_train)*100:.2f}%)")

# Load target mapping
print(f"\n8. TARGET MAPPING")
if os.path.exists('models/artifacts/target_mapping.pkl'):
    with open('models/artifacts/target_mapping.pkl', 'rb') as f:
        target_mapping = pickle.load(f)
    print(f"Target mapping: {target_mapping}")

print("\n" + "=" * 80)
print("✅ DATA VERIFICATION COMPLETE")
print("=" * 80)
# Create: verify_data_v2.py
import pandas as pd
import pickle

print("=" * 80)
print("DETAILED DATA VERIFICATION FOR LIGHTGBM")
print("=" * 80)

# Load data
X_train = pd.read_csv('data/processed/v1/X_train.csv')
y_train = pd.read_csv('data/processed/v1/y_train.csv')

# Load target mapping
with open('models/artifacts/target_mapping.pkl', 'rb') as f:
    target_mapping = pickle.load(f)

print(f"\n📊 TARGET MAPPING:")
print(f"   {target_mapping}")

# Reverse mapping for display
reverse_mapping = {v: k for k, v in target_mapping.items()}

print(f"\n📊 CLASS DISTRIBUTION:")
class_counts = y_train['IncidentGrade'].value_counts().sort_index()
print(class_counts)

print(f"\n📊 CLASS DISTRIBUTION WITH LABELS:")
for encoded_val, count in class_counts.items():
    label = reverse_mapping.get(encoded_val, f"Unknown_{encoded_val}")
    percentage = (count / len(y_train)) * 100
    print(f"   {encoded_val} ({label}): {count:,} samples ({percentage:.2f}%)")

print(f"\n📊 IMBALANCE RATIO:")
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count
print(f"   Max/Min ratio: {imbalance_ratio:.2f}:1")
print(f"   ⚠️  This indicates {'SEVERE' if imbalance_ratio > 10 else 'MODERATE' if imbalance_ratio > 3 else 'MILD'} class imbalance")

print(f"\n📊 FEATURE SUMMARY:")
print(f"   Total features: {X_train.shape[1]}")
print(f"   All features are numeric: {(X_train.dtypes == 'float64').all()}")

print("\n" + "=" * 80)
print("✅ READY TO BUILD LIGHTGBM MODEL")
print("=" * 80)
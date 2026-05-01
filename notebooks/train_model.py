from src.preprocessing.preprocess import preprocess_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np

DATA_PATH = "data/raw/GUIDE_Train.csv"

# 🔹 Step 1: Preprocess
X_train, X_test, y_train, y_test, encoders = preprocess_pipeline(DATA_PATH)

# 🔹 Step 2: Handle imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train.value_counts())


# 🔹 Step 3: Initial XGBoost model (for feature importance)
base_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42
)

base_model.fit(X_train, y_train)


# 🔹 Step 4: Feature Selection
importances = base_model.feature_importances_

# keep features with importance > 0.02
mask = importances > 0.01

X_train = X_train.loc[:, mask]
X_test = X_test.loc[:, mask]

print("Selected features:", X_train.shape[1])


# 🔹 Step 5: Hyperparameter tuning (GridSearch)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}

grid = GridSearchCV(
    XGBClassifier(
        objective="multi:softmax",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42
    ),
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

# 🔹 Best model
model = grid.best_estimator_


# 🔹 Step 6: Final evaluation
y_pred = model.predict(X_test)

print("\n✅ Final Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
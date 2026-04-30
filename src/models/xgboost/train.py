# train.py

from xgboost import XGBClassifier
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from .util import load_data
from .plots import plot_confusion_matrix, plot_feature_importance

def compute_sample_weights(y):
    y_array = y.values.ravel()
    classes = np.unique(y_array)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_array
    )

    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[label] for label in y_array])

    return sample_weights


def train_xgboost_model(X_train, y_train, X_val, y_val, sample_weights):
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        early_stopping_rounds=50,
        tree_method="hist",
        device="cuda"
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    return model


if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data()

    print("Computing sample weights...")
    sample_weights = compute_sample_weights(y_train)

    print("Training model...")
    model = train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weights
    )

    print("Saving model...")
    model.save_model("models/xgboost_model.json")

    print("Evaluating model...")
    y_pred = model.predict(X_val)

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    print("DONE")

    # ========================
# 📊 PLOTS
# ========================

    class_names = ["FalsePositive", "BenignPositive", "TruePositive"]

    plot_confusion_matrix(y_val, y_pred, class_names)
    plot_feature_importance(model)
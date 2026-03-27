from xgboost import XGBClassifier
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


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

    return sample_weights, class_weight_dict


def train_xgboost_model(X_train, y_train, X_val, y_val, sample_weights):
    # Convert y to 1D
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

        # ✅ GPU settings (fixed)
         # ✅ correct GPU usage
       
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


def predict_with_threshold(model, X, high_threshold=0.35):
    if hasattr(X, "values"):
        X = X.values

    proba = model.predict_proba(X)

    # Apply threshold for class "High"
    y_pred = np.where(
        proba[:, 2] > high_threshold,
        2,
        np.argmax(proba, axis=1)
    )

    return y_pred, proba
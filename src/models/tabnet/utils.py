import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight


class TabNetScaler:
    def __init__(self, n_quantiles=1000):
        self.scaler = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution='normal',
            random_state=42
        )
        self.is_fitted = False

    def fit_transform_train(self, X):
        X = X.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        self.is_fitted = True
        return X_scaled

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("TabNetScaler must be fitted before calling transform().")
        return self.scaler.transform(X.astype(np.float32)).astype(np.float32)


def scale_tabnet_features(X_train, X_val, X_test):
    scaler = TabNetScaler(n_quantiles=min(1000, len(X_train)))
    X_train = scaler.fit_transform_train(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def compute_tabnet_class_weights(y):
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"compute_tabnet_class_weights requires at least 2 classes, found: {classes}"
        )
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def class_weights_to_sample_weights(class_weights: dict, y: np.ndarray) -> np.ndarray:
    """
    Convert a {class_index: weight} dict into a per-sample weight array.

    TabNet's `weights` parameter must be a 1D array of length == n_train_samples,
    where each value is the weight for that sample's class.  Passing a dict, a
    per-class array, or anything of the wrong length raises:
        "Custom weights should match number of train samples."
    """
    return np.array([class_weights[int(label)] for label in y], dtype=np.float32)


def save_tabnet_model(model, scaler, class_weights, hyperparams=None, model_dir="models/tabnet"):
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model")
    model.save_model(model_path)

    joblib.dump(scaler.scaler, os.path.join(model_dir, "scaler.pkl"))

    config = {"class_weights": {str(k): v for k, v in class_weights.items()}}
    if hyperparams:
        config["hyperparams"] = hyperparams

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {model_dir}")
    return model_path


def load_tabnet_model(model_dir="models/tabnet"):
    from pytorch_tabnet.tab_model import TabNetClassifier

    model_zip    = os.path.join(model_dir, "model.zip")
    scaler_path  = os.path.join(model_dir, "scaler.pkl")
    config_path  = os.path.join(model_dir, "config.json")

    for path in [model_zip, scaler_path, config_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")

    model = TabNetClassifier()
    model.load_model(model_zip)
    scaler = joblib.load(scaler_path)

    with open(config_path) as f:
        config = json.load(f)

    if "class_weights" in config:
        class_weights = {int(k): v for k, v in config["class_weights"].items()}
        hyperparams   = config.get("hyperparams", {})
    else:
        # backwards-compatible with old flat format
        class_weights = {int(k): v for k, v in config.items()}
        hyperparams   = {}

    return model, scaler, class_weights, hyperparams
# predict.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from utils import load_test_data
from remediation import suggest_action


def load_model():
    model = XGBClassifier()
    model.load_model("models/xgboost_model.json")
    return model


def predict(model, data):
    if hasattr(data, "values"):
        data = data.values

    proba = model.predict_proba(data)
    pred = np.argmax(proba, axis=1)

    return pred, proba


if __name__ == "__main__":
    print("Loading model...")
    model = load_model()

    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Running predictions...")
    preds, probs = predict(model, X_test)

    print("\nSample predictions with actions:")
    for i in range(10):
        action = suggest_action(preds[i])
        print(f"Prediction: {preds[i]} | Action: {action}")

    accuracy = (preds == y_test.values.ravel()).mean()
    print(f"\nAccuracy on test set: {accuracy:.4f}")
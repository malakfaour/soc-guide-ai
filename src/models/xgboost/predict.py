# TODO
import pandas as pd
from xgboost import XGBClassifier
import numpy as np

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
    model = load_model()

    # Example input (replace with real data)
    sample = pd.read_csv("data/processed/v1/X_test.csv").iloc[:5]

    preds, probs = predict(model, sample)

    print("Predictions:", preds)
    print("Probabilities:\n", probs)
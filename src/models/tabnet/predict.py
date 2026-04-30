import numpy as np
import pandas as pd

from src.models.tabnet.utils import load_tabnet_model, TabNetScaler


def predict(csv_path):
    print("Loading model...")
    model, scaler, class_weights, hyperparams = load_tabnet_model()

    print("Loading new data...")
    df = pd.read_csv(csv_path)

    # If your dataset has target column, remove it (safe check)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    X = df.values.astype(np.float32)

    print("Applying scaling...")
    X_scaled = scaler.transform(X)

    print("Predicting...")
    y_pred = model.predict(X_scaled)

    print("\n── Predictions ──")
    print(y_pred)

    # Optional: save predictions
    output = pd.DataFrame({"prediction": y_pred})
    output.to_csv("predictions.csv", index=False)
    print("\nSaved to predictions.csv")


if __name__ == "__main__":
    predict("data/processed/v1/X_test.csv")
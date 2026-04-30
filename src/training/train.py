import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.xgboost.train import (
    compute_sample_weights,
    train_xgboost_model,
    predict_with_threshold
)

DATA_PATH = "data/processed/v1/"
MODEL_OUTPUT_PATH = "models/"


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
    X_val   = pd.read_csv(os.path.join(DATA_PATH, "X_val.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))

    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
    y_val   = pd.read_csv(os.path.join(DATA_PATH, "y_val.csv"))
    y_test  = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv"))

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X_test, y_test):
    y_test = y_test.values.ravel()

    y_pred, proba = predict_with_threshold(model, X_test, high_threshold=0.35)

    print("\n" + "="*50)
    print("MODEL EVALUATION ON TEST SET")
    print("="*50)

    print(f"\nAccuracy:      {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 (macro):    {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:\n", cm)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:", per_class_acc)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    plt.savefig(os.path.join(MODEL_OUTPUT_PATH, "confusion_matrix.png"))
    plt.show()

    print(f"\nConfusion matrix saved to {MODEL_OUTPUT_PATH}confusion_matrix.png")


def save_model(model):
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    save_path = os.path.join(MODEL_OUTPUT_PATH, "xgboost_model.json")

    model.save_model(save_path)

    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)

    print("Missing values in X_train:", X_train.isnull().sum().sum())

    # Compute weights
    sample_weights, class_weight_dict = compute_sample_weights(y_train)

    print("\nClass weights:", class_weight_dict)
    print("Sample weights shape:", sample_weights.shape)
    print("Min weight:", sample_weights.min())
    print("Max weight:", sample_weights.max())

    # Train
    print("\nTraining XGBoost model...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val, sample_weights)

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best validation mlogloss: {model.best_score:.5f}")

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model)
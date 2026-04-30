# plots.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=1,
        linecolor="white"
    )

    plt.title("XGBoost Confusion Matrix (Validation Set)", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if class_names:
        plt.xticks(ticks=[0.5,1.5,2.5], labels=class_names)
        plt.yticks(ticks=[0.5,1.5,2.5], labels=class_names)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, top_n=15):
    plt.figure(figsize=(10,6))

    plot_importance(
        model,
        max_num_features=top_n,
        importance_type="gain"
    )

    plt.title("XGBoost Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.show()
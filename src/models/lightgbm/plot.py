# plots.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import lightgbm as lgb


def plot_confusion_matrix(y_true, y_pred):
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

    plt.title("LightGBM Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, top_n=15):
    plt.figure(figsize=(10,6))

    lgb.plot_importance(
        model,
        max_num_features=top_n,
        importance_type="gain"
    )

    plt.title("LightGBM Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.show()
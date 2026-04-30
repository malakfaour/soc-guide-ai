import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.models.tabnet.train import train  # only if you return model
from src.models.tabnet.utils import load_tabnet_model
from src.training.train_tabnet import load_tabnet_data


def run_evaluation():
    print("Loading model...")
    model, scaler, _, _ = load_tabnet_model()

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()

    # =========================
    # 1. CONFUSION MATRIX
    # =========================
    print("Plotting Confusion Matrix...")
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("TabNet Confusion Matrix")
    plt.show()

    # =========================
    # 2. FEATURE IMPORTANCE
    # =========================
    print("Plotting Feature Importance...")
    importances = model.feature_importances_

    plt.figure()
    plt.bar(range(len(importances)), importances)
    plt.title("TabNet Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()

    # =========================
    # 3. TRAINING CURVE (optional)
    # =========================
    if hasattr(model, "history"):
        print("Plotting Training Curve...")
        history = model.history

        plt.figure()
        plt.plot(history["val_0_balanced_accuracy"], label="Validation Accuracy")
        plt.plot(history["loss"], label="Training Loss")
        plt.title("TabNet Training Curve")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    run_evaluation()
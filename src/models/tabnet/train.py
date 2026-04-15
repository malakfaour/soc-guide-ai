import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from src.training.train_tabnet import load_tabnet_data
from src.models.tabnet.utils import (
    scale_tabnet_features,
    compute_tabnet_class_weights,
    class_weights_to_sample_weights,   # per-sample weights for TabNet
    save_tabnet_model,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────
HYPERPARAMS = dict(
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.3,
    lambda_sparse=1e-3,
    max_epochs=200,
    patience=20,
    batch_size=128,
    virtual_batch_size=64,
)


def train():
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()

    print("Scaling...")
    X_train, X_val, X_test, scaler = scale_tabnet_features(X_train, X_val, X_test)

    print("Computing class weights...")
    class_weights = compute_tabnet_class_weights(y_train)

    y_train = y_train.values.ravel() if hasattr(y_train, "values") else y_train.ravel()

    sample_weights = class_weights_to_sample_weights(class_weights, y_train)

    print("DEBUG:")
    print("X_train:", len(X_train))
    print("y_train:", len(y_train))
    print("weights:", len(sample_weights))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TabNetClassifier(
    n_d=HYPERPARAMS["n_d"],
    n_a=HYPERPARAMS["n_a"],
    n_steps=HYPERPARAMS["n_steps"],
    gamma=HYPERPARAMS["gamma"],
    lambda_sparse=HYPERPARAMS["lambda_sparse"],
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-3),   # 🔥 important
    mask_type="entmax",               # 🔥 more stable than sparsemax
    device_name=device,
    verbose=1,
    )

    print("Training...")
    try:
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["balanced_accuracy"],
            max_epochs=HYPERPARAMS["max_epochs"],
            patience=HYPERPARAMS["patience"],
            batch_size=HYPERPARAMS["batch_size"],
            virtual_batch_size=HYPERPARAMS["virtual_batch_size"],
            weights=sample_weights,   # per-sample array, length == len(y_train)
        )
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    # ── Evaluation ───────────────────────────────────────────────────────────
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    print("\n── Test set results ──")
    print(classification_report(y_test, y_pred))

    # ── Save ─────────────────────────────────────────────────────────────────
    print("Saving...")
    save_tabnet_model(model, scaler, class_weights, hyperparams=HYPERPARAMS)
    print("Done!")


if __name__ == "__main__":
    train()

print("Best epoch:", model.best_epoch)
print("Best validation score:", model.best_cost)
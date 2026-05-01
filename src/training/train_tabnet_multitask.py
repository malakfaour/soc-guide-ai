"""
End-to-end multi-task TabNet training pipeline.

Builds remediation targets from the processed v1 action columns,
removes those columns from the model inputs to avoid target leakage,
and trains a shared-encoder TabNet model for:
- Triage: multi-class classification
- Remediation: multi-label recommendation
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from metrics import (
    TriageEvaluator,
    RemediationEvaluator,
    save_triage_metrics,
    save_remediation_metrics,
)
from remediation_targets import (
    ACTION_COLUMNS,
    build_remediation_label_spec,
    encode_remediation_targets,
    export_remediation_targets,
    export_incident_level_remediation_dataset,
)

_MULTITASK_SPEC = importlib.util.spec_from_file_location(
    "tabnet_multitask_module",
    PROJECT_ROOT / "src" / "models" / "tabnet" / "multitask.py",
)
if _MULTITASK_SPEC is None or _MULTITASK_SPEC.loader is None:
    raise ImportError("Could not load TabNet multitask module")
_multitask_module = importlib.util.module_from_spec(_MULTITASK_SPEC)
_MULTITASK_SPEC.loader.exec_module(_multitask_module)

_TABNET_UTILS_SPEC = importlib.util.spec_from_file_location(
    "tabnet_utils",
    PROJECT_ROOT / "src" / "models" / "tabnet" / "utils.py",
)
if _TABNET_UTILS_SPEC is None or _TABNET_UTILS_SPEC.loader is None:
    raise ImportError("Could not load TabNet utility module")
_tabnet_utils = importlib.util.module_from_spec(_TABNET_UTILS_SPEC)
_TABNET_UTILS_SPEC.loader.exec_module(_tabnet_utils)

create_multitask_model = _multitask_module.create_multitask_model
MultiTaskLoss = _multitask_module.MultiTaskLoss
scale_tabnet_features = _tabnet_utils.scale_tabnet_features
compute_tabnet_class_weights = _tabnet_utils.compute_tabnet_class_weights


def load_multitask_data(
    data_dir: str = "data/processed/v1",
    action_columns: List[str] = None,
    target_scheme: str = "incident_family",
    feature_granularity: str = "row",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load processed triage data and derive remediation labels from action columns.
    """
    if action_columns is None:
        action_columns = ACTION_COLUMNS

    data_path = PROJECT_ROOT / data_dir
    if feature_granularity == "incident":
        incident_prefix = "incident_remediation"
        incident_paths = {
            "X_train": data_path / f"X_{incident_prefix}_train.csv",
            "X_val": data_path / f"X_{incident_prefix}_val.csv",
            "X_test": data_path / f"X_{incident_prefix}_test.csv",
            "y_triage_train": data_path / f"y_triage_{incident_prefix}_train.csv",
            "y_triage_val": data_path / f"y_triage_{incident_prefix}_val.csv",
            "y_triage_test": data_path / f"y_triage_{incident_prefix}_test.csv",
            "y_rem_train": data_path / f"y_rem_{incident_prefix}_train.csv",
            "y_rem_val": data_path / f"y_rem_{incident_prefix}_val.csv",
            "y_rem_test": data_path / f"y_rem_{incident_prefix}_test.csv",
        }
        incident_metadata_path = data_path / f"{incident_prefix}_metadata.json"
        if not (all(path.exists() for path in incident_paths.values()) and incident_metadata_path.exists()):
            processed_root = PROJECT_ROOT / "data" / "processed"
            try:
                version_name = data_path.relative_to(processed_root).parts[0]
                export_incident_level_remediation_dataset(
                    version=version_name,
                    verbose=verbose,
                )
            except Exception:
                pass

        if not (all(path.exists() for path in incident_paths.values()) and incident_metadata_path.exists()):
            raise FileNotFoundError(
                "Incident-level remediation dataset is missing and could not be exported"
            )

        with open(incident_metadata_path, "r") as handle:
            incident_metadata = json.load(handle)

        return {
            "X_train_df": pd.read_csv(incident_paths["X_train"]),
            "X_val_df": pd.read_csv(incident_paths["X_val"]),
            "X_test_df": pd.read_csv(incident_paths["X_test"]),
            "y_train": pd.read_csv(incident_paths["y_triage_train"]).iloc[:, 0].to_numpy(dtype=np.int64),
            "y_val": pd.read_csv(incident_paths["y_triage_val"]).iloc[:, 0].to_numpy(dtype=np.int64),
            "y_test": pd.read_csv(incident_paths["y_triage_test"]).iloc[:, 0].to_numpy(dtype=np.int64),
            "y_rem_train": pd.read_csv(incident_paths["y_rem_train"]).to_numpy(dtype=np.float32),
            "y_rem_val": pd.read_csv(incident_paths["y_rem_val"]).to_numpy(dtype=np.float32),
            "y_rem_test": pd.read_csv(incident_paths["y_rem_test"]).to_numpy(dtype=np.float32),
            "label_names": incident_metadata["target_columns"],
            "label_definitions": [
                (name, float(idx)) for idx, name in enumerate(incident_metadata["target_columns"])
            ],
            "baseline_values": {},
            "dropped_feature_columns": [],
            "remediation_metadata": incident_metadata,
            "target_scheme": "incident_family",
            "feature_granularity": "incident",
        }

    split_frames: Dict[str, pd.DataFrame] = {}
    split_targets: Dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        X_path = data_path / f"X_{split}.csv"
        y_path = data_path / f"y_{split}.csv"

        X_df = pd.read_csv(X_path)
        y = pd.read_csv(y_path).iloc[:, 0].to_numpy(dtype=np.int64)

        missing_cols = [col for col in action_columns if col not in X_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing remediation source columns in {X_path.name}: {missing_cols}"
            )

        split_frames[split] = X_df
        split_targets[split] = y

    if target_scheme not in {"incident_family", "family", "granular"}:
        raise ValueError("target_scheme must be 'incident_family', 'family', or 'granular'")

    if target_scheme == "incident_family":
        remediation_prefix = "y_rem_incident_family"
    elif target_scheme == "family":
        remediation_prefix = "y_rem_family"
    else:
        remediation_prefix = "y_rem"
    remediation_paths = {
        "train": data_path / f"{remediation_prefix}_train.csv",
        "val": data_path / f"{remediation_prefix}_val.csv",
        "test": data_path / f"{remediation_prefix}_test.csv",
    }
    metadata_path = data_path / "remediation_targets_metadata.json"

    if not (all(path.exists() for path in remediation_paths.values()) and metadata_path.exists()):
        processed_root = PROJECT_ROOT / "data" / "processed"
        try:
            version_name = data_path.relative_to(processed_root).parts[0]
            export_remediation_targets(version=version_name, verbose=verbose)
        except Exception:
            pass

    if all(path.exists() for path in remediation_paths.values()) and metadata_path.exists():
        remediation_targets = {
            split: pd.read_csv(remediation_paths[split]).to_numpy(dtype=np.float32)
            for split in ["train", "val", "test"]
        }
        with open(metadata_path, "r") as handle:
            remediation_metadata = json.load(handle)
        if target_scheme == "incident_family":
            label_names = remediation_metadata["incident_family_label_names"]
            labels = [(name, float(idx)) for idx, name in enumerate(label_names)]
        elif target_scheme == "family":
            label_names = remediation_metadata["family_label_names"]
            labels = [(name, float(idx)) for idx, name in enumerate(label_names)]
        else:
            label_names = remediation_metadata["label_names"]
            labels = [
                (item["column"], float(item["value"]))
                for item in remediation_metadata["label_definitions"]
            ]
        baseline_values = remediation_metadata["baseline_values"]
    else:
        labels, baseline_values, label_names = build_remediation_label_spec(
            split_frames["train"],
            action_columns,
        )
        remediation_targets = {
            split: encode_remediation_targets(split_frames[split], labels)
            .to_numpy(dtype=np.float32)
            for split in ["train", "val", "test"]
        }
        remediation_metadata = {
            "label_names": label_names,
            "label_definitions": [
                {"column": column, "value": float(value)}
                for column, value in labels
            ],
            "baseline_values": baseline_values,
            "source_columns": action_columns,
        }

    feature_frames = {
        split: split_frames[split].drop(columns=action_columns)
        for split in ["train", "val", "test"]
    }

    return {
        "X_train_df": feature_frames["train"],
        "X_val_df": feature_frames["val"],
        "X_test_df": feature_frames["test"],
        "y_train": split_targets["train"],
        "y_val": split_targets["val"],
        "y_test": split_targets["test"],
        "y_rem_train": remediation_targets["train"],
        "y_rem_val": remediation_targets["val"],
        "y_rem_test": remediation_targets["test"],
        "label_names": label_names,
        "label_definitions": labels,
        "baseline_values": baseline_values,
        "dropped_feature_columns": action_columns,
        "remediation_metadata": remediation_metadata,
        "target_scheme": target_scheme,
        "feature_granularity": "row",
    }


def compute_remediation_pos_weight(y_train: np.ndarray) -> np.ndarray:
    """Compute per-label positive weights for BCEWithLogitsLoss."""
    positives = y_train.sum(axis=0)
    negatives = y_train.shape[0] - positives
    positives = np.maximum(positives, 1.0)
    return (negatives / positives).astype(np.float32)


def _make_loader(
    X: np.ndarray,
    y_triage: np.ndarray,
    y_remediation: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_triage, dtype=torch.long),
        torch.tensor(y_remediation, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    sparse_loss_weight: float,
    mode: str,
) -> Dict[str, Any]:
    """Run one evaluation pass and collect predictions."""
    model.eval()

    total_loss = 0.0
    total_triage_loss = 0.0
    total_remediation_loss = 0.0
    total_samples = 0

    triage_logits_all: List[np.ndarray] = []
    triage_preds_all: List[np.ndarray] = []
    triage_targets_all: List[np.ndarray] = []
    remediation_probs_all: List[np.ndarray] = []
    remediation_preds_all: List[np.ndarray] = []
    remediation_targets_all: List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_triage_batch, y_rem_batch in data_loader:
            X_batch = X_batch.to(device)
            y_triage_batch = y_triage_batch.to(device)
            y_rem_batch = y_rem_batch.to(device)

            triage_logits, remediation_logits, sparse_loss = model(X_batch)
            loss, triage_loss, remediation_loss = criterion(
                triage_logits,
                remediation_logits,
                y_triage_batch,
                y_rem_batch,
            )
            loss = loss + sparse_loss_weight * sparse_loss.mean()

            batch_size = X_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_triage_loss += triage_loss.item() * batch_size
            total_remediation_loss += remediation_loss.item() * batch_size
            total_samples += batch_size

            if mode != "remediation_only":
                triage_proba = torch.softmax(triage_logits, dim=1).cpu().numpy()
                triage_logits_all.append(triage_proba)
                triage_preds_all.append(np.argmax(triage_proba, axis=1))
                triage_targets_all.append(y_triage_batch.cpu().numpy())
            if mode != "triage_only":
                remediation_proba = torch.sigmoid(remediation_logits).cpu().numpy()
                remediation_probs_all.append(remediation_proba)
                remediation_preds_all.append((remediation_proba >= 0.5).astype(np.int64))
                remediation_targets_all.append(y_rem_batch.cpu().numpy().astype(np.int64))

    outputs = {
        "loss": total_loss / max(total_samples, 1),
        "triage_loss": total_triage_loss / max(total_samples, 1),
        "remediation_loss": total_remediation_loss / max(total_samples, 1),
    }
    if mode != "remediation_only":
        outputs.update({
            "triage_proba": np.concatenate(triage_logits_all, axis=0),
            "triage_pred": np.concatenate(triage_preds_all, axis=0),
            "triage_true": np.concatenate(triage_targets_all, axis=0),
        })
    if mode != "triage_only":
        outputs.update({
            "remediation_proba": np.concatenate(remediation_probs_all, axis=0),
            "remediation_pred": np.concatenate(remediation_preds_all, axis=0),
            "remediation_true": np.concatenate(remediation_targets_all, axis=0),
        })
    return outputs


def train_multitask_tabnet(
    max_epochs: int = 40,
    patience: int = 8,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    triage_weight: float = 1.0,
    remediation_weight: float = 1.0,
    sparse_loss_weight: float = 1e-3,
    target_scheme: str = "incident_family",
    remediation_loss_type: str = "focal",
    remediation_focal_gamma: float = 2.0,
    mode: str = "multitask",
    feature_granularity: str = "row",
    model_dir: str = "models/tabnet",
    reports_dir: str = "reports/metrics",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate the multi-task TabNet model."""
    if feature_granularity == "incident" and mode != "remediation_only":
        raise ValueError(
            "Incident-level feature aggregation is currently supported only for remediation_only mode"
        )

    data = load_multitask_data(
        target_scheme=target_scheme,
        feature_granularity=feature_granularity,
        verbose=verbose,
    )

    X_train = data["X_train_df"].to_numpy(dtype=np.float32)
    X_val = data["X_val_df"].to_numpy(dtype=np.float32)
    X_test = data["X_test_df"].to_numpy(dtype=np.float32)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_tabnet_features(
        X_train,
        X_val,
        X_test,
        verbose=False,
    )

    class_weights = compute_tabnet_class_weights(data["y_train"], verbose=False)
    class_weights_array = np.array(
        [class_weights[idx] for idx in sorted(class_weights.keys())],
        dtype=np.float32,
    )
    remediation_pos_weight = compute_remediation_pos_weight(data["y_rem_train"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_multitask_model(
        n_features=X_train_scaled.shape[1],
        n_triage_classes=len(class_weights),
        n_remediations=data["y_rem_train"].shape[1],
        device=str(device),
        mode=mode,
        verbose=verbose,
    )

    criterion = MultiTaskLoss(
        triage_weight=triage_weight,
        remediation_weight=remediation_weight,
        class_weights=class_weights_array,
        remediation_pos_weight=remediation_pos_weight,
        remediation_loss_type=remediation_loss_type,
        remediation_focal_gamma=remediation_focal_gamma,
        mode=mode,
    )
    criterion = criterion.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_loader = _make_loader(
        X_train_scaled,
        data["y_train"],
        data["y_rem_train"],
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _make_loader(
        X_val_scaled,
        data["y_val"],
        data["y_rem_val"],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = _make_loader(
        X_test_scaled,
        data["y_test"],
        data["y_rem_test"],
        batch_size=batch_size,
        shuffle=False,
    )

    best_state: Dict[str, Any] = {}
    best_score = float("-inf")
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, float]] = []

    if verbose:
        print("=" * 60)
        print("Multi-Task TabNet Training")
        print("=" * 60)
        print(f"Train features: {X_train_scaled.shape}")
        print(f"Validation features: {X_val_scaled.shape}")
        print(f"Test features: {X_test_scaled.shape}")
        print(f"Mode: {mode}")
        print(f"Feature granularity: {data['feature_granularity']}")
        print(f"Remediation labels: {len(data['label_names'])}")
        print(f"Target scheme: {data['target_scheme']}")
        print(f"Remediation loss: {remediation_loss_type}")
        print(f"Dropped leakage columns: {data['dropped_feature_columns']}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for X_batch, y_triage_batch, y_rem_batch in train_loader:
            X_batch = X_batch.to(device)
            y_triage_batch = y_triage_batch.to(device)
            y_rem_batch = y_rem_batch.to(device)

            optimizer.zero_grad()
            triage_logits, remediation_logits, sparse_loss = model(X_batch)
            loss, _, _ = criterion(
                triage_logits,
                remediation_logits,
                y_triage_batch,
                y_rem_batch,
            )
            loss = loss + sparse_loss_weight * sparse_loss.mean()
            loss.backward()
            optimizer.step()

            batch_size_actual = X_batch.shape[0]
            running_loss += loss.item() * batch_size_actual
            running_samples += batch_size_actual

        val_outputs = _evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            sparse_loss_weight,
            mode,
        )
        val_triage_f1 = None
        val_rem_f1 = None
        if mode != "remediation_only":
            val_triage_f1 = f1_score(
                val_outputs["triage_true"],
                val_outputs["triage_pred"],
                average="macro",
            )
        if mode != "triage_only":
            val_rem_f1 = f1_score(
                val_outputs["remediation_true"],
                val_outputs["remediation_pred"],
                average="macro",
                zero_division=0,
            )
        if mode == "triage_only":
            val_score = float(val_triage_f1)
        elif mode == "remediation_only":
            val_score = float(val_rem_f1)
        else:
            val_score = (float(val_triage_f1) + float(val_rem_f1)) / 2.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(running_samples, 1),
                "val_loss": val_outputs["loss"],
                "val_triage_macro_f1": None if val_triage_f1 is None else float(val_triage_f1),
                "val_remediation_macro_f1": None if val_rem_f1 is None else float(val_rem_f1),
                "val_score": float(val_score),
            }
        )

        if verbose:
            triage_msg = "n/a" if val_triage_f1 is None else f"{val_triage_f1:.4f}"
            rem_msg = "n/a" if val_rem_f1 is None else f"{val_rem_f1:.4f}"
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={history[-1]['train_loss']:.4f} | "
                f"val_loss={val_outputs['loss']:.4f} | "
                f"triage_f1={triage_msg} | "
                f"rem_f1={rem_msg}"
            )

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best epoch: {best_epoch})"
                    )
                break

    if not best_state:
        raise RuntimeError("Training finished without capturing a best checkpoint")

    model.load_state_dict(best_state["model_state_dict"])

    test_outputs = _evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        sparse_loss_weight,
        mode,
    )

    triage_metrics = None
    remediation_metrics = None
    if mode != "remediation_only":
        triage_evaluator = TriageEvaluator(n_classes=len(class_weights))
        triage_metrics = triage_evaluator.compute_metrics(
            test_outputs["triage_true"],
            test_outputs["triage_pred"],
            test_outputs["triage_proba"],
        )
    if mode != "triage_only":
        remediation_evaluator = RemediationEvaluator(
            n_remediations=data["y_rem_train"].shape[1]
        )
        remediation_evaluator.label_names = data["label_names"]
        remediation_metrics = remediation_evaluator.compute_metrics(
            test_outputs["remediation_true"],
            test_outputs["remediation_pred"],
        )
        remediation_metrics["label_names"] = data["label_names"]

    model_output_dir = PROJECT_ROOT / model_dir
    reports_output_dir = PROJECT_ROOT / reports_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)

    artifact_prefix = {
        "multitask": "multitask",
        "triage_only": "triage_only",
        "remediation_only": "remediation_only",
    }[mode]
    if feature_granularity == "incident":
        artifact_prefix = f"{artifact_prefix}_incident"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": best_state["optimizer_state_dict"],
            "label_names": data["label_names"],
            "label_definitions": [
                {"column": column, "value": float(value)}
                for column, value in data["label_definitions"]
            ],
            "baseline_values": data["baseline_values"],
            "dropped_feature_columns": data["dropped_feature_columns"],
            "class_weights": {str(k): float(v) for k, v in class_weights.items()},
            "remediation_pos_weight": remediation_pos_weight.tolist(),
            "remediation_loss_type": remediation_loss_type,
            "remediation_focal_gamma": remediation_focal_gamma,
            "best_epoch": best_epoch,
            "best_validation_score": float(best_score),
            "target_scheme": target_scheme,
            "mode": mode,
            "feature_granularity": feature_granularity,
        },
        model_output_dir / f"{artifact_prefix}_tabnet.pt",
    )

    with open(model_output_dir / f"{artifact_prefix}_tabnet_config.json", "w") as handle:
        json.dump(
            {
                "n_features": int(X_train_scaled.shape[1]),
                "n_triage_classes": int(len(class_weights)),
                "n_remediation_labels": int(len(data["label_names"])),
                "label_names": data["label_names"],
                "label_definitions": [
                    {"column": column, "value": float(value)}
                    for column, value in data["label_definitions"]
                ],
                "baseline_values": data["baseline_values"],
                "dropped_feature_columns": data["dropped_feature_columns"],
                "remediation_loss_type": remediation_loss_type,
                "remediation_focal_gamma": remediation_focal_gamma,
                "best_epoch": int(best_epoch),
                "best_validation_score": float(best_score),
                "target_scheme": target_scheme,
                "mode": mode,
                "feature_granularity": feature_granularity,
                "train_shape": list(X_train_scaled.shape),
                "val_shape": list(X_val_scaled.shape),
                "test_shape": list(X_test_scaled.shape),
            },
            handle,
            indent=2,
        )

    if triage_metrics is not None:
        save_triage_metrics(
            triage_metrics,
            output_dir=str(reports_output_dir),
            filename=f"{artifact_prefix}_triage_metrics.json",
        )
    if remediation_metrics is not None:
        save_remediation_metrics(
            remediation_metrics,
            output_dir=str(reports_output_dir),
            filename=f"{artifact_prefix}_remediation_metrics.json",
        )

    with open(reports_output_dir / f"{artifact_prefix}_training_history.json", "w") as handle:
        json.dump(history, handle, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_validation_score": float(best_score),
        "triage_metrics": triage_metrics,
        "remediation_metrics": remediation_metrics,
        "label_names": data["label_names"],
        "history": history,
        "mode": mode,
        "feature_granularity": feature_granularity,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-task TabNet")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-dir", type=str, default="models/tabnet")
    parser.add_argument("--reports-dir", type=str, default="reports/metrics")
    parser.add_argument(
        "--remediation-loss-type",
        choices=["bce", "focal"],
        default="focal",
    )
    parser.add_argument("--remediation-focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--target-scheme",
        choices=["incident_family", "family", "granular"],
        default="incident_family",
    )
    parser.add_argument(
        "--mode",
        choices=["multitask", "triage_only", "remediation_only"],
        default="multitask",
    )
    parser.add_argument(
        "--feature-granularity",
        choices=["row", "incident"],
        default="row",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    results = train_multitask_tabnet(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_scheme=args.target_scheme,
        remediation_loss_type=args.remediation_loss_type,
        remediation_focal_gamma=args.remediation_focal_gamma,
        mode=args.mode,
        feature_granularity=args.feature_granularity,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        verbose=not args.quiet,
    )
    summary = {
        "mode": args.mode,
        "feature_granularity": args.feature_granularity,
        "best_epoch": results["best_epoch"],
        "best_validation_score": results["best_validation_score"],
        "triage_macro_f1": None if results["triage_metrics"] is None else results["triage_metrics"]["macro_f1"],
        "remediation_macro_f1": None if results["remediation_metrics"] is None else results["remediation_metrics"]["macro_f1"],
        "account_response_f1": None,
        "endpoint_response_f1": None,
    }
    if results["remediation_metrics"] is not None:
        summary["remediation_subset_accuracy"] = results["remediation_metrics"]["subset_accuracy"]
        summary["account_response_f1"] = results["remediation_metrics"]["per_label_f1"].get("account_response")
        summary["endpoint_response_f1"] = results["remediation_metrics"]["per_label_f1"].get("endpoint_response")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

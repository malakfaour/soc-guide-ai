"""
Validation helper for the multi-task TabNet pipeline.

Runs three checks:
1. Confirms processed v1 remediation targets exist and load correctly
2. Prints remediation label counts for granular and family schemes
3. Optionally runs a 1-epoch smoke training pass
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from utils.versioning import load_dataset_by_version
from train_tabnet_multitask import train_multitask_tabnet
from remediation_targets import (
    export_incident_level_remediation_dataset,
    export_remediation_targets,
)


def validate_processed_targets(version: str = "v1") -> dict:
    """Validate that processed remediation targets are available."""
    dataset = load_dataset_by_version(
        version=version,
        include_remediation=True,
        verbose=True,
    )
    metadata = dataset[9]
    version_path = PROJECT_ROOT / "data" / "processed" / version

    family_files = {
        "y_rem_family_train": version_path / "y_rem_family_train.csv",
        "y_rem_family_val": version_path / "y_rem_family_val.csv",
        "y_rem_family_test": version_path / "y_rem_family_test.csv",
    }
    incident_family_files = {
        "y_rem_incident_family_train": version_path / "y_rem_incident_family_train.csv",
        "y_rem_incident_family_val": version_path / "y_rem_incident_family_val.csv",
        "y_rem_incident_family_test": version_path / "y_rem_incident_family_test.csv",
    }
    incident_dataset_files = {
        "X_incident_remediation_train": version_path / "X_incident_remediation_train.csv",
        "X_incident_remediation_val": version_path / "X_incident_remediation_val.csv",
        "X_incident_remediation_test": version_path / "X_incident_remediation_test.csv",
        "y_triage_incident_remediation_train": version_path / "y_triage_incident_remediation_train.csv",
        "y_triage_incident_remediation_val": version_path / "y_triage_incident_remediation_val.csv",
        "y_triage_incident_remediation_test": version_path / "y_triage_incident_remediation_test.csv",
        "y_rem_incident_remediation_train": version_path / "y_rem_incident_remediation_train.csv",
        "y_rem_incident_remediation_val": version_path / "y_rem_incident_remediation_val.csv",
        "y_rem_incident_remediation_test": version_path / "y_rem_incident_remediation_test.csv",
    }
    incident_dataset_metadata = version_path / "incident_remediation_metadata.json"
    if any(
        not path.exists()
        for path in {**family_files, **incident_family_files}.values()
    ):
        export_remediation_targets(version=version, verbose=True)
    if any(not path.exists() for path in incident_dataset_files.values()) or not incident_dataset_metadata.exists():
        export_incident_level_remediation_dataset(version=version, verbose=True)
    missing_family = [
        name
        for name, path in {**family_files, **incident_family_files, **incident_dataset_files}.items()
        if not path.exists()
    ]
    if missing_family:
        raise FileNotFoundError(f"Missing remediation family files: {missing_family}")

    family_counts = {}
    for name, path in family_files.items():
        family_counts[name] = pd.read_csv(path).sum().to_dict()
    incident_family_counts = {}
    for name, path in incident_family_files.items():
        incident_family_counts[name] = pd.read_csv(path).sum().to_dict()
    with open(incident_dataset_metadata, "r") as handle:
        incident_dataset_summary = json.load(handle)

    summary = {
        "granular_label_names": metadata["label_names"],
        "granular_train_counts": metadata["train_positive_counts"],
        "family_label_names": metadata["family_label_names"],
        "family_train_counts": metadata["family_train_positive_counts"],
        "family_val_counts": metadata["family_val_positive_counts"],
        "family_test_counts": metadata["family_test_positive_counts"],
        "family_file_checks": {name: str(path) for name, path in family_files.items()},
        "incident_family_label_names": metadata["incident_family_label_names"],
        "incident_family_train_counts": metadata["incident_family_train_positive_counts"],
        "incident_family_val_counts": metadata["incident_family_val_positive_counts"],
        "incident_family_test_counts": metadata["incident_family_test_positive_counts"],
        "incident_family_file_checks": {
            name: str(path) for name, path in incident_family_files.items()
        },
        "incident_dataset_shapes": {
            "train": incident_dataset_summary["train_shape"],
            "val": incident_dataset_summary["val_shape"],
            "test": incident_dataset_summary["test_shape"],
        },
        "incident_dataset_target_counts": {
            "train": incident_dataset_summary["train_target_counts"],
            "val": incident_dataset_summary["val_target_counts"],
            "test": incident_dataset_summary["test_target_counts"],
        },
        "incident_dataset_file_checks": {
            name: str(path) for name, path in incident_dataset_files.items()
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-task TabNet pipeline")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--target-scheme",
        choices=["incident_family", "family", "granular"],
        default="incident_family",
    )
    parser.add_argument(
        "--feature-granularity",
        choices=["row", "incident"],
        default="row",
    )
    parser.add_argument(
        "--remediation-loss-type",
        choices=["bce", "focal"],
        default="focal",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VALIDATE MULTI-TASK TABNET")
    print("=" * 60)
    print("\n[1/2] Validating processed remediation targets...")
    validate_processed_targets(args.version)

    if args.skip_train:
        print("\n[2/2] Smoke training skipped by request")
        return

    print("\n[2/2] Running 1-epoch smoke training...")
    results = train_multitask_tabnet(
        max_epochs=1,
        patience=1,
        batch_size=4096,
        target_scheme=args.target_scheme,
        remediation_loss_type=args.remediation_loss_type,
        mode="remediation_only" if args.feature_granularity == "incident" else "multitask",
        feature_granularity=args.feature_granularity,
        verbose=True,
    )
    print("\n[SUMMARY]")
    print(json.dumps(
        {
            "target_scheme": args.target_scheme,
            "feature_granularity": args.feature_granularity,
            "remediation_loss_type": args.remediation_loss_type,
            "best_epoch": results["best_epoch"],
            "triage_macro_f1": results["triage_metrics"]["macro_f1"],
            "remediation_macro_f1": results["remediation_metrics"]["macro_f1"],
            "remediation_subset_accuracy": results["remediation_metrics"]["subset_accuracy"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

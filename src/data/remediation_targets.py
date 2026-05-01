"""
Utilities for exporting remediation targets as first-class processed artifacts.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_COLUMNS = ["ActionGrouped", "ActionGranular"]
ENCODER_ARTIFACT_PATH = PROJECT_ROOT / "models" / "artifacts" / "encoders.pkl"
RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "GUIDE_Train.csv"
REMEDIATION_FAMILY_RULES = {
    "account_response": {
        "ActionGrouped": {"ContainAccount"},
        "ActionGranular": {
            "update stsrefreshtokenvalidfrom timestamp.",
            "account password changed",
            "change user password.",
            "reset user password.",
            "forcepasswordresetremediation",
            "set force change user password.",
            "account disabled",
            "disable account.",
            "disableuser",
            "msecidentitiessuspenduser",
            "account deleted",
            "delete user.",
            "msecidentitiesconfirmusercompromised",
        },
    },
    "endpoint_response": {
        "ActionGrouped": {"IsolateDevice", "Stop Virtual Machines"},
        "ActionGranular": {
            "isolateresponse",
            "quarantinefile",
            "delete virtualmachines",
        },
    },
}
INCIDENT_FEATURE_EXPORT_PREFIX = "incident_remediation"
INCIDENT_DATASET_RANDOM_STATE = 42
RAW_INCIDENT_FEATURE_COLUMNS = [
    "IncidentId",
    "IncidentGrade",
    "Category",
    "EntityType",
    "ResourceType",
    "DeviceId",
    "DeviceName",
    "ResourceIdName",
    "SuspicionLevel",
    "LastVerdict",
]
SUSPICION_SCORE_MAP = {
    "unknown": 0,
    "Suspicious": 1,
    "Incriminated": 2,
}
VERDICT_SCORE_MAP = {
    "unknown": 0,
    "NoThreatsFound": 0,
    "Suspicious": 1,
    "Malicious": 2,
}
INCIDENT_GRADE_TO_INDEX = {
    "FalsePositive": 0,
    "BenignPositive": 1,
    "TruePositive": 2,
}


def _load_action_name_lookup(
    artifact_path: Path = ENCODER_ARTIFACT_PATH,
) -> Dict[str, Dict[float, List[str]]]:
    """Load reverse mappings from encoded action values to raw names."""
    if not artifact_path.exists():
        return {}

    encoders = joblib.load(artifact_path)
    lookups: Dict[str, Dict[float, List[str]]] = {}

    for column in ACTION_COLUMNS:
        mapping = encoders.get(column, {})
        reverse: Dict[float, List[str]] = {}
        for raw_name, encoded_value in mapping.items():
            reverse.setdefault(float(encoded_value), []).append(str(raw_name))
        lookups[column] = reverse

    return lookups


def _label_display_name(
    column: str,
    value: float,
    action_name_lookup: Dict[str, Dict[float, List[str]]],
) -> str:
    """Render a readable remediation label."""
    aliases: List[str] = []
    for encoded_value, names in action_name_lookup.get(column, {}).items():
        if np.isclose(float(value), float(encoded_value), rtol=0.0, atol=1e-12):
            aliases = names
            break

    aliases = [alias for alias in aliases if alias != "unknown"]
    if aliases:
        return f"{column}={'|'.join(sorted(aliases))}"

    return f"{column}={format(float(value), '.12g')}"


def build_remediation_label_spec(
    X_train: pd.DataFrame,
    action_columns: List[str] = None,
) -> Tuple[List[Tuple[str, float]], Dict[str, float], List[str]]:
    """Build the remediation label space from processed action columns."""
    if action_columns is None:
        action_columns = ACTION_COLUMNS

    labels: List[Tuple[str, float]] = []
    baseline_values: Dict[str, float] = {}

    for column in action_columns:
        counts = X_train[column].value_counts(dropna=False)
        if counts.empty:
            raise ValueError(f"No values found for action column: {column}")

        baseline_values[column] = float(counts.index[0])
        for value in counts.index[1:]:
            labels.append((column, float(value)))

    action_name_lookup = _load_action_name_lookup()
    label_names = [
        _label_display_name(column, value, action_name_lookup)
        for column, value in labels
    ]
    return labels, baseline_values, label_names


def encode_remediation_targets(
    X_split: pd.DataFrame,
    labels: List[Tuple[str, float]],
) -> pd.DataFrame:
    """Encode remediation targets as a multi-hot dataframe."""
    output = {}

    for column, value in labels:
        output_key = f"{column}::{format(float(value), '.12g')}"
        output[output_key] = np.isclose(
            X_split[column].to_numpy(dtype=np.float64),
            value,
            rtol=0.0,
            atol=1e-12,
        ).astype(np.int64)

    return pd.DataFrame(output, index=X_split.index)


def export_remediation_targets(
    version: str = "v1",
    base_path: str = "data/processed",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Export remediation target CSVs for an existing processed dataset version.
    """
    version_path = PROJECT_ROOT / base_path / version
    if not version_path.exists():
        raise FileNotFoundError(f"Processed dataset version not found: {version_path}")

    X_train = pd.read_csv(version_path / "X_train.csv")
    X_val = pd.read_csv(version_path / "X_val.csv")
    X_test = pd.read_csv(version_path / "X_test.csv")

    missing_cols = [col for col in ACTION_COLUMNS if col not in X_train.columns]
    if missing_cols:
        raise ValueError(f"Missing action columns in processed data: {missing_cols}")

    labels, baseline_values, label_names = build_remediation_label_spec(X_train)
    y_rem_train = encode_remediation_targets(X_train, labels)
    y_rem_val = encode_remediation_targets(X_val, labels)
    y_rem_test = encode_remediation_targets(X_test, labels)

    y_rem_family_train = encode_remediation_family_targets(X_train)
    y_rem_family_val = encode_remediation_family_targets(X_val)
    y_rem_family_test = encode_remediation_family_targets(X_test)
    incident_family_lookup = build_incident_family_lookup()
    y_rem_incident_family_train = (
        X_train[["IncidentId"]]
        .merge(incident_family_lookup, left_on="IncidentId", right_index=True, how="left")
        .drop(columns=["IncidentId"])
        .fillna(0)
        .astype(int)
    )
    y_rem_incident_family_val = (
        X_val[["IncidentId"]]
        .merge(incident_family_lookup, left_on="IncidentId", right_index=True, how="left")
        .drop(columns=["IncidentId"])
        .fillna(0)
        .astype(int)
    )
    y_rem_incident_family_test = (
        X_test[["IncidentId"]]
        .merge(incident_family_lookup, left_on="IncidentId", right_index=True, how="left")
        .drop(columns=["IncidentId"])
        .fillna(0)
        .astype(int)
    )

    incident_export = export_incident_level_remediation_dataset(
        version=version,
        base_path=base_path,
        verbose=verbose,
    )

    y_rem_train.to_csv(version_path / "y_rem_train.csv", index=False)
    y_rem_val.to_csv(version_path / "y_rem_val.csv", index=False)
    y_rem_test.to_csv(version_path / "y_rem_test.csv", index=False)
    y_rem_family_train.to_csv(version_path / "y_rem_family_train.csv", index=False)
    y_rem_family_val.to_csv(version_path / "y_rem_family_val.csv", index=False)
    y_rem_family_test.to_csv(version_path / "y_rem_family_test.csv", index=False)
    y_rem_incident_family_train.to_csv(
        version_path / "y_rem_incident_family_train.csv", index=False
    )
    y_rem_incident_family_val.to_csv(
        version_path / "y_rem_incident_family_val.csv", index=False
    )
    y_rem_incident_family_test.to_csv(
        version_path / "y_rem_incident_family_test.csv", index=False
    )

    metadata = {
        "label_names": label_names,
        "label_definitions": [
            {"column": column, "value": float(value)}
            for column, value in labels
        ],
        "baseline_values": baseline_values,
        "source_columns": ACTION_COLUMNS,
        "train_positive_counts": {
            label_name: int(y_rem_train.iloc[:, idx].sum())
            for idx, label_name in enumerate(label_names)
        },
        "val_positive_counts": {
            label_name: int(y_rem_val.iloc[:, idx].sum())
            for idx, label_name in enumerate(label_names)
        },
        "test_positive_counts": {
            label_name: int(y_rem_test.iloc[:, idx].sum())
            for idx, label_name in enumerate(label_names)
        },
        "family_label_names": list(y_rem_family_train.columns),
        "family_train_positive_counts": {
            column: int(y_rem_family_train[column].sum())
            for column in y_rem_family_train.columns
        },
        "family_val_positive_counts": {
            column: int(y_rem_family_val[column].sum())
            for column in y_rem_family_val.columns
        },
        "family_test_positive_counts": {
            column: int(y_rem_family_test[column].sum())
            for column in y_rem_family_test.columns
        },
        "incident_family_label_names": list(y_rem_incident_family_train.columns),
        "incident_family_train_positive_counts": {
            column: int(y_rem_incident_family_train[column].sum())
            for column in y_rem_incident_family_train.columns
        },
        "incident_family_val_positive_counts": {
            column: int(y_rem_incident_family_val[column].sum())
            for column in y_rem_incident_family_val.columns
        },
        "incident_family_test_positive_counts": {
            column: int(y_rem_incident_family_test[column].sum())
            for column in y_rem_incident_family_test.columns
        },
    }

    with open(version_path / "remediation_targets_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    if verbose:
        print(f"[REMEDIATION] Exported remediation targets to {version_path}")
        print(f"[REMEDIATION] Labels: {len(label_names)}")
        for label_name in label_names:
            print(
                f"  {label_name}: "
                f"train={metadata['train_positive_counts'][label_name]}, "
                f"val={metadata['val_positive_counts'][label_name]}, "
                f"test={metadata['test_positive_counts'][label_name]}"
            )
        print(f"[REMEDIATION] Family labels: {len(y_rem_family_train.columns)}")
        for family_name in y_rem_family_train.columns:
            print(
                f"  {family_name}: "
                f"train={metadata['family_train_positive_counts'][family_name]}, "
                f"val={metadata['family_val_positive_counts'][family_name]}, "
                f"test={metadata['family_test_positive_counts'][family_name]}"
            )
        print(f"[REMEDIATION] Incident family labels: {len(y_rem_incident_family_train.columns)}")
        for family_name in y_rem_incident_family_train.columns:
            print(
                f"  {family_name}: "
                f"train={metadata['incident_family_train_positive_counts'][family_name]}, "
                f"val={metadata['incident_family_val_positive_counts'][family_name]}, "
                f"test={metadata['incident_family_test_positive_counts'][family_name]}"
            )
        print("[REMEDIATION] Incident-level remediation dataset:")
        print(f"  train={incident_export['train_shape']}")
        print(f"  val={incident_export['val_shape']}")
        print(f"  test={incident_export['test_shape']}")

    return {
        "y_rem_train_path": str(version_path / "y_rem_train.csv"),
        "y_rem_val_path": str(version_path / "y_rem_val.csv"),
        "y_rem_test_path": str(version_path / "y_rem_test.csv"),
        "y_rem_family_train_path": str(version_path / "y_rem_family_train.csv"),
        "y_rem_family_val_path": str(version_path / "y_rem_family_val.csv"),
        "y_rem_family_test_path": str(version_path / "y_rem_family_test.csv"),
        "y_rem_incident_family_train_path": str(version_path / "y_rem_incident_family_train.csv"),
        "y_rem_incident_family_val_path": str(version_path / "y_rem_incident_family_val.csv"),
        "y_rem_incident_family_test_path": str(version_path / "y_rem_incident_family_test.csv"),
        "X_incident_remediation_train_path": incident_export["X_train_path"],
        "X_incident_remediation_val_path": incident_export["X_val_path"],
        "X_incident_remediation_test_path": incident_export["X_test_path"],
        "y_triage_incident_train_path": incident_export["y_triage_train_path"],
        "y_triage_incident_val_path": incident_export["y_triage_val_path"],
        "y_triage_incident_test_path": incident_export["y_triage_test_path"],
        "y_rem_incident_level_train_path": incident_export["y_rem_train_path"],
        "y_rem_incident_level_val_path": incident_export["y_rem_val_path"],
        "y_rem_incident_level_test_path": incident_export["y_rem_test_path"],
        "incident_metadata_path": incident_export["metadata_path"],
        "metadata_path": str(version_path / "remediation_targets_metadata.json"),
        "label_names": label_names,
    }


def encode_remediation_family_targets(
    X_split: pd.DataFrame,
) -> pd.DataFrame:
    """Encode broader remediation families from processed action columns."""
    action_name_lookup = _load_action_name_lookup()
    columns = {}

    for family_name, family_rules in REMEDIATION_FAMILY_RULES.items():
        family_mask = np.zeros(len(X_split), dtype=bool)
        for action_column, allowed_names in family_rules.items():
            for encoded_value, aliases in action_name_lookup.get(action_column, {}).items():
                alias_set = {alias for alias in aliases if alias != "unknown"}
                if alias_set & allowed_names:
                    family_mask |= np.isclose(
                        X_split[action_column].to_numpy(dtype=np.float64),
                        float(encoded_value),
                        rtol=0.0,
                        atol=1e-12,
                    )
        columns[family_name] = family_mask.astype(np.int64)

    return pd.DataFrame(columns, index=X_split.index)


def build_incident_family_lookup(
    raw_train_path: Path = RAW_TRAIN_PATH,
) -> pd.DataFrame:
    """Aggregate remediation families at the incident level from raw GUIDE data."""
    raw = pd.read_csv(
        raw_train_path,
        usecols=["IncidentId", "ActionGrouped", "ActionGranular"],
    )
    raw["ActionGrouped"] = raw["ActionGrouped"].fillna("unknown")
    raw["ActionGranular"] = raw["ActionGranular"].fillna("unknown")

    incident_actions = raw.groupby("IncidentId").agg(
        {
            "ActionGrouped": lambda s: set(s),
            "ActionGranular": lambda s: set(s),
        }
    )

    columns = {}
    for family_name, family_rules in REMEDIATION_FAMILY_RULES.items():
        columns[family_name] = incident_actions.apply(
            lambda row: int(
                any(
                    (set(row[action_column]) & allowed_names)
                    for action_column, allowed_names in family_rules.items()
                )
            ),
            axis=1,
        )

    return pd.DataFrame(columns, index=incident_actions.index)


def _safe_string(value: Any) -> str:
    """Convert raw values to normalized strings."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _iter_processed_incident_ids(version_path: Path) -> Iterable[int]:
    """Yield unique processed IncidentIds across existing row-level splits."""
    seen: set[int] = set()
    for split in ["train", "val", "test"]:
        incident_series = pd.read_csv(
            version_path / f"X_{split}.csv",
            usecols=["IncidentId"],
        )["IncidentId"].astype(int)
        for incident_id in incident_series.tolist():
            if incident_id not in seen:
                seen.add(incident_id)
                yield incident_id


def _choose_stratify_labels(target_df: pd.DataFrame) -> pd.Series:
    """Choose a stable stratification key for incident-level splits."""
    label_keys = target_df.astype(int).astype(str).agg("|".join, axis=1)
    if label_keys.value_counts().min() >= 2:
        return label_keys

    endpoint = target_df["endpoint_response"].astype(int)
    if endpoint.value_counts().min() >= 2:
        return endpoint

    account = target_df["account_response"].astype(int)
    if account.value_counts().min() >= 2:
        return account

    return pd.Series(np.zeros(len(target_df), dtype=np.int64), index=target_df.index)


def _split_incident_dataset(
    incident_df: pd.DataFrame,
    random_state: int = INCIDENT_DATASET_RANDOM_STATE,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Create leakage-free incident train/val/test splits."""
    stratify_labels = _choose_stratify_labels(
        incident_df[["account_response", "endpoint_response"]]
    )
    all_ids = incident_df.index.to_numpy()

    train_ids, temp_ids = train_test_split(
        all_ids,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify_labels.loc[all_ids],
    )
    temp_stratify = stratify_labels.loc[temp_ids]
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_stratify,
    )
    return pd.Index(train_ids), pd.Index(val_ids), pd.Index(test_ids)


def _encode_dominant_category(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, int]]:
    """Encode dominant category using train-only mappings."""
    category_order = (
        train_series.fillna("unknown").astype(str).value_counts().index.tolist()
    )
    mapping = {category: idx + 1 for idx, category in enumerate(category_order)}
    mapping["unknown"] = 0

    def _map(series: pd.Series) -> pd.Series:
        normalized = series.fillna("unknown").astype(str)
        return normalized.map(lambda value: mapping.get(value, 0)).astype(np.int64)

    return _map(train_series), _map(val_series), _map(test_series), mapping


def _aggregate_incident_features(
    incident_ids: set[int],
    raw_train_path: Path = RAW_TRAIN_PATH,
    chunksize: int = 250000,
) -> pd.DataFrame:
    """Aggregate raw GUIDE rows into one feature vector per incident."""
    incident_stats: Dict[int, Dict[str, Any]] = {}
    category_counts: Dict[int, Counter] = defaultdict(Counter)
    entity_types: Dict[int, set[str]] = defaultdict(set)
    incident_grade_counts: Dict[int, Counter] = defaultdict(Counter)

    for chunk in pd.read_csv(
        raw_train_path,
        usecols=RAW_INCIDENT_FEATURE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["IncidentId"] = chunk["IncidentId"].astype(np.int64)
        chunk = chunk[chunk["IncidentId"].isin(incident_ids)].copy()
        if chunk.empty:
            continue

        for column in [
            "IncidentGrade",
            "Category",
            "EntityType",
            "ResourceType",
            "DeviceId",
            "DeviceName",
            "ResourceIdName",
            "SuspicionLevel",
            "LastVerdict",
        ]:
            chunk[column] = chunk[column].map(_safe_string)

        chunk["machine_context"] = (
            chunk["EntityType"].isin(["Machine", "Process", "File", "AzureResource"])
            | chunk["ResourceType"].ne("unknown")
            | chunk["DeviceId"].ne("unknown")
            | chunk["DeviceName"].ne("unknown")
            | chunk["ResourceIdName"].ne("unknown")
        )
        chunk["device_context"] = (
            chunk["DeviceId"].ne("unknown")
            | chunk["DeviceName"].ne("unknown")
            | chunk["ResourceIdName"].ne("unknown")
        )
        chunk["vm_resource"] = chunk["ResourceType"].str.contains(
            "Virtual Machine",
            case=False,
            na=False,
        )
        chunk["process_entity"] = chunk["EntityType"].eq("Process")
        chunk["file_entity"] = chunk["EntityType"].eq("File")
        chunk["machine_entity"] = chunk["EntityType"].isin(["Machine", "AzureResource"])
        chunk["suspicion_score"] = chunk["SuspicionLevel"].map(
            lambda value: SUSPICION_SCORE_MAP.get(value, 0)
        )
        chunk["verdict_score"] = chunk["LastVerdict"].map(
            lambda value: VERDICT_SCORE_MAP.get(value, 0)
        )

        local_stats = chunk.groupby("IncidentId").agg(
            alert_count=("IncidentId", "size"),
            machine_entity_count=("machine_context", "sum"),
            device_context_count=("device_context", "sum"),
            vm_resource_count=("vm_resource", "sum"),
            has_process_entity=("process_entity", "max"),
            has_file_entity=("file_entity", "max"),
            has_machine_entity=("machine_entity", "max"),
            max_suspicion_score=("suspicion_score", "max"),
            max_verdict_score=("verdict_score", "max"),
        )

        for row in local_stats.itertuples():
            stats = incident_stats.setdefault(
                int(row.Index),
                {
                    "alert_count": 0,
                    "machine_entity_count": 0,
                    "device_context_count": 0,
                    "vm_resource_count": 0,
                    "has_process_entity": 0,
                    "has_file_entity": 0,
                    "has_machine_entity": 0,
                    "max_suspicion_score": 0,
                    "max_verdict_score": 0,
                },
            )
            stats["alert_count"] += int(row.alert_count)
            stats["machine_entity_count"] += int(row.machine_entity_count)
            stats["device_context_count"] += int(row.device_context_count)
            stats["vm_resource_count"] += int(row.vm_resource_count)
            stats["has_process_entity"] = max(
                stats["has_process_entity"],
                int(row.has_process_entity),
            )
            stats["has_file_entity"] = max(
                stats["has_file_entity"],
                int(row.has_file_entity),
            )
            stats["has_machine_entity"] = max(
                stats["has_machine_entity"],
                int(row.has_machine_entity),
            )
            stats["max_suspicion_score"] = max(
                stats["max_suspicion_score"],
                int(row.max_suspicion_score),
            )
            stats["max_verdict_score"] = max(
                stats["max_verdict_score"],
                int(row.max_verdict_score),
            )

        for (incident_id, category), count in (
            chunk.groupby(["IncidentId", "Category"]).size().items()
        ):
            if category != "unknown":
                category_counts[int(incident_id)][str(category)] += int(count)

        for incident_id, values in chunk.groupby("IncidentId")["EntityType"]:
            entity_types[int(incident_id)].update(
                value for value in values.unique().tolist() if value != "unknown"
            )

        for (incident_id, grade), count in (
            chunk.groupby(["IncidentId", "IncidentGrade"]).size().items()
        ):
            if grade != "unknown":
                incident_grade_counts[int(incident_id)][str(grade)] += int(count)

    records: List[Dict[str, Any]] = []
    for incident_id in sorted(incident_ids):
        stats = incident_stats.get(
            incident_id,
            {
                "alert_count": 0,
                "machine_entity_count": 0,
                "device_context_count": 0,
                "vm_resource_count": 0,
                "has_process_entity": 0,
                "has_file_entity": 0,
                "has_machine_entity": 0,
                "max_suspicion_score": 0,
                "max_verdict_score": 0,
            },
        )
        alert_count = max(int(stats["alert_count"]), 1)
        dominant_category = "unknown"
        if category_counts[incident_id]:
            dominant_category = category_counts[incident_id].most_common(1)[0][0]
        dominant_grade = "BenignPositive"
        if incident_grade_counts[incident_id]:
            dominant_grade = incident_grade_counts[incident_id].most_common(1)[0][0]

        records.append(
            {
                "IncidentId": incident_id,
                "alert_count": int(stats["alert_count"]),
                "machine_entity_count": int(stats["machine_entity_count"]),
                "machine_entity_ratio": float(stats["machine_entity_count"]) / float(alert_count),
                "device_context_count": int(stats["device_context_count"]),
                "device_context_ratio": float(stats["device_context_count"]) / float(alert_count),
                "vm_resource_count": int(stats["vm_resource_count"]),
                "dominant_category": dominant_category,
                "max_suspicion_score": int(stats["max_suspicion_score"]),
                "max_verdict_score": int(stats["max_verdict_score"]),
                "max_severity_score": int(
                    max(stats["max_suspicion_score"], stats["max_verdict_score"])
                ),
                "unique_entity_types": int(len(entity_types[incident_id])),
                "has_process_entity": int(stats["has_process_entity"]),
                "has_file_entity": int(stats["has_file_entity"]),
                "has_machine_entity": int(stats["has_machine_entity"]),
                "incident_triage_label": INCIDENT_GRADE_TO_INDEX.get(dominant_grade, 1),
            }
        )

    incident_df = pd.DataFrame.from_records(records).set_index("IncidentId")
    return incident_df


def export_incident_level_remediation_dataset(
    version: str = "v1",
    base_path: str = "data/processed",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build and export a leakage-free incident-level remediation dataset."""
    version_path = PROJECT_ROOT / base_path / version
    if not version_path.exists():
        raise FileNotFoundError(f"Processed dataset version not found: {version_path}")

    incident_ids = set(_iter_processed_incident_ids(version_path))
    incident_features = _aggregate_incident_features(incident_ids)
    incident_targets = build_incident_family_lookup().reindex(incident_features.index).fillna(0).astype(int)
    incident_df = incident_features.join(incident_targets, how="left")
    train_ids, val_ids, test_ids = _split_incident_dataset(incident_df)

    train_df = incident_df.loc[train_ids].copy()
    val_df = incident_df.loc[val_ids].copy()
    test_df = incident_df.loc[test_ids].copy()

    (
        train_df["dominant_category_code"],
        val_df["dominant_category_code"],
        test_df["dominant_category_code"],
        category_mapping,
    ) = _encode_dominant_category(
        train_df["dominant_category"],
        val_df["dominant_category"],
        test_df["dominant_category"],
    )

    feature_columns = [
        "alert_count",
        "machine_entity_count",
        "machine_entity_ratio",
        "device_context_count",
        "device_context_ratio",
        "vm_resource_count",
        "dominant_category_code",
        "max_suspicion_score",
        "max_verdict_score",
        "max_severity_score",
        "unique_entity_types",
        "has_process_entity",
        "has_file_entity",
        "has_machine_entity",
    ]
    target_columns = ["account_response", "endpoint_response"]

    export_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    output_paths = {}
    for split, frame in export_frames.items():
        X_path = version_path / f"X_{INCIDENT_FEATURE_EXPORT_PREFIX}_{split}.csv"
        y_triage_path = version_path / f"y_triage_{INCIDENT_FEATURE_EXPORT_PREFIX}_{split}.csv"
        y_rem_path = version_path / f"y_rem_{INCIDENT_FEATURE_EXPORT_PREFIX}_{split}.csv"

        frame.loc[:, feature_columns].to_csv(X_path, index=False)
        frame.loc[:, ["incident_triage_label"]].to_csv(y_triage_path, index=False)
        frame.loc[:, target_columns].to_csv(y_rem_path, index=False)
        output_paths[split] = {
            "X_path": str(X_path),
            "y_triage_path": str(y_triage_path),
            "y_rem_path": str(y_rem_path),
        }

    metadata = {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "dominant_category_mapping": category_mapping,
        "train_shape": [int(train_df.shape[0]), len(feature_columns)],
        "val_shape": [int(val_df.shape[0]), len(feature_columns)],
        "test_shape": [int(test_df.shape[0]), len(feature_columns)],
        "train_target_counts": {
            column: int(train_df[column].sum()) for column in target_columns
        },
        "val_target_counts": {
            column: int(val_df[column].sum()) for column in target_columns
        },
        "test_target_counts": {
            column: int(test_df[column].sum()) for column in target_columns
        },
        "source_incident_count": int(len(incident_ids)),
        "random_state": INCIDENT_DATASET_RANDOM_STATE,
        "split_note": "Incident-level 70/15/15 split with stratification on remediation labels",
    }
    metadata_path = version_path / f"{INCIDENT_FEATURE_EXPORT_PREFIX}_metadata.json"
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    if verbose:
        print("[REMEDIATION] Exported incident-level remediation dataset")
        print(f"  Incidents: {len(incident_ids)}")
        print(f"  Train: {metadata['train_shape']}")
        print(f"  Val:   {metadata['val_shape']}")
        print(f"  Test:  {metadata['test_shape']}")
        for column in target_columns:
            print(
                f"  {column}: "
                f"train={metadata['train_target_counts'][column]}, "
                f"val={metadata['val_target_counts'][column]}, "
                f"test={metadata['test_target_counts'][column]}"
            )

    return {
        "X_train_path": output_paths["train"]["X_path"],
        "X_val_path": output_paths["val"]["X_path"],
        "X_test_path": output_paths["test"]["X_path"],
        "y_triage_train_path": output_paths["train"]["y_triage_path"],
        "y_triage_val_path": output_paths["val"]["y_triage_path"],
        "y_triage_test_path": output_paths["test"]["y_triage_path"],
        "y_rem_train_path": output_paths["train"]["y_rem_path"],
        "y_rem_val_path": output_paths["val"]["y_rem_path"],
        "y_rem_test_path": output_paths["test"]["y_rem_path"],
        "metadata_path": str(metadata_path),
        "train_shape": metadata["train_shape"],
        "val_shape": metadata["val_shape"],
        "test_shape": metadata["test_shape"],
    }


if __name__ == "__main__":
    export_remediation_targets()

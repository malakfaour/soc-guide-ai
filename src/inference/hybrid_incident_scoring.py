"""
Hybrid inference for SOC incidents.

Combines:
- Row-level TabNet triage predictions
- Incident-level classical remediation predictions
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "tabnet"))

from utils import load_tabnet_model


CLASSICAL_MODEL_DIR = PROJECT_ROOT / "models" / "classical"
TRIAGE_MODEL_DIR = PROJECT_ROOT / "models" / "tabnet"


def load_hybrid_models(
    triage_model_name: str = "triage_model",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Load the row-level triage model and incident-level remediation models."""
    triage_model, triage_scaler, triage_config = load_tabnet_model(
        model_dir=str(TRIAGE_MODEL_DIR),
        model_name=triage_model_name,
        verbose=verbose,
    )

    account_model = joblib.load(CLASSICAL_MODEL_DIR / "account_response_gbt.pkl")
    endpoint_model = joblib.load(CLASSICAL_MODEL_DIR / "endpoint_response_lr.pkl")
    incident_scaler = joblib.load(CLASSICAL_MODEL_DIR / "incident_scaler.pkl")

    with open(CLASSICAL_MODEL_DIR / "remediation_thresholds.json", "r") as handle:
        thresholds = json.load(handle)
    with open(CLASSICAL_MODEL_DIR / "remediation_model_metadata.json", "r") as handle:
        remediation_metadata = json.load(handle)

    return {
        "triage_model": triage_model,
        "triage_scaler": triage_scaler,
        "triage_config": triage_config,
        "account_model": account_model,
        "endpoint_model": endpoint_model,
        "incident_scaler": incident_scaler,
        "thresholds": thresholds,
        "remediation_metadata": remediation_metadata,
    }


def _score_triage_rows(
    incident_rows_df: pd.DataFrame,
    triage_model: Any,
    triage_scaler: Any,
) -> Dict[str, Any]:
    """Run row-level triage inference."""
    triage_features = incident_rows_df.to_numpy(dtype=np.float32)
    scaler_impl = getattr(triage_scaler, "scaler", triage_scaler)
    triage_scaled = scaler_impl.transform(triage_features)
    triage_proba = triage_model.predict_proba(triage_scaled)
    triage_pred = np.argmax(triage_proba, axis=1)
    triage_confidence = triage_proba.max(axis=1)

    return {
        "predictions": triage_pred,
        "probabilities": triage_proba,
        "confidence": triage_confidence,
    }


def _score_incident_remediation(
    incident_features_df: pd.DataFrame,
    account_model: Any,
    endpoint_model: Any,
    incident_scaler: Any,
    thresholds: Dict[str, float],
    feature_columns: list[str],
) -> Dict[str, Any]:
    """Run incident-level remediation inference."""
    missing_features = [
        column for column in feature_columns if column not in incident_features_df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Incident feature dataframe is missing required columns: {missing_features}"
        )

    incident_features = incident_features_df.loc[:, feature_columns]
    incident_scaled = incident_scaler.transform(incident_features)

    account_proba = account_model.predict_proba(incident_scaled)[:, 1]
    endpoint_proba = endpoint_model.predict_proba(incident_scaled)[:, 1]

    account_pred = account_proba >= float(thresholds["account_response"])
    endpoint_pred = endpoint_proba >= float(thresholds["endpoint_response"])

    return {
        "account_response": {
            "predictions": account_pred.astype(int),
            "probabilities": account_proba,
            "threshold": float(thresholds["account_response"]),
        },
        "endpoint_response": {
            "predictions": endpoint_pred.astype(int),
            "probabilities": endpoint_proba,
            "threshold": float(thresholds["endpoint_response"]),
        },
    }


def score_incident(
    incident_rows_df: pd.DataFrame,
    incident_features_df: pd.DataFrame,
    artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Score one incident using the hybrid pipeline.

    Parameters
    ----------
    incident_rows_df
        Preprocessed row-level features for the triage TabNet model.
    incident_features_df
        Incident-level aggregated features for the classical remediation models.
    artifacts
        Optional preloaded artifacts from `load_hybrid_models()`.
    """
    if artifacts is None:
        artifacts = load_hybrid_models(verbose=False)

    triage_outputs = _score_triage_rows(
        incident_rows_df=incident_rows_df,
        triage_model=artifacts["triage_model"],
        triage_scaler=artifacts["triage_scaler"],
    )
    remediation_outputs = _score_incident_remediation(
        incident_features_df=incident_features_df,
        account_model=artifacts["account_model"],
        endpoint_model=artifacts["endpoint_model"],
        incident_scaler=artifacts["incident_scaler"],
        thresholds=artifacts["thresholds"],
        feature_columns=artifacts["remediation_metadata"]["feature_columns"],
    )

    return {
        "triage": triage_outputs,
        "remediation": remediation_outputs,
    }

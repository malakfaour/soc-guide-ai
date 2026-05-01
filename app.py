<<<<<<< HEAD
"""
FastAPI application for SOC incident triage and remediation predictions.

Endpoints:
  POST /predict      — expects pre-encoded numeric features (original)
  POST /predict-raw  — accepts raw alert fields, runs full preprocessing pipeline
  GET  /health
  GET  /features/triage
  GET  /features/remediation
  GET  /classes
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
=======
<<<<<<< HEAD
import streamlit as st
import pandas as pd
from src.models.lightgbm.predict import LightGBMPredictor

# Page config
st.set_page_config(
    page_title="AI Security Alert Classifier",
    layout="wide"
)

# Title
st.title("🔐 AI Security Incident Classifier")
st.write("Predict whether an alert is TruePositive, FalsePositive, or Benign")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Upload your dataset and classify alerts instantly")

# Load model (cached)
@st.cache_resource
def load_model():
    return LightGBMPredictor()

predictor = load_model()

# Upload section
st.header("📂 Upload Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    if st.button("🚀 Predict"):
        results = predictor.predict(df)

        predictions = results['predictions']
        probs = results['probabilities']

        st.success("✅ Predictions completed!")

        # Mapping classes
        class_map = {
            0: "FalsePositive",
            1: "BenignPositive",
            2: "TruePositive"
        }

        # Create output dataframe
        output_df = df.copy()

        output_df["Prediction"] = [class_map[p] for p in predictions]
        output_df["Confidence"] = probs.max(axis=1)

        # Add probability columns
        prob_df = pd.DataFrame(
            probs,
            columns=["FalsePositive", "BenignPositive", "TruePositive"]
        )

        output_df = pd.concat([output_df, prob_df], axis=1)

        # Show results
        st.subheader("📊 Results Preview")
        st.dataframe(output_df.head())

        # Show chart
        st.subheader("📊 Prediction Probabilities (First 10 rows)")
        st.bar_chart(prob_df.head(10))

        # Show single sample nicely
        st.subheader("🔍 Example Prediction")
        sample = output_df.iloc[0]

        st.write(f"**Prediction:** {sample['Prediction']}")
        st.write(f"**Confidence:** {sample['Confidence']:.2f}")

        # Download button
        st.download_button(
            label="📥 Download Results",
            data=output_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

else:
    st.info("📥 Upload a CSV file to start prediction")
=======
from fastapi import FastAPI
from pydantic import BaseModel
>>>>>>> feature/lightgbm
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Constants ─────────────────────────────────────────────────────────────────
TRIAGE_CLASS_LABELS = {0: "FP", 1: "BP", 2: "TP"}
TRIAGE_CLASS_FULL   = {0: "False Positive", 1: "Benign Positive", 2: "True Positive"}

TRIAGE_FEATURES = [
    "Id", "OrgId", "IncidentId", "AlertId", "Timestamp", "DetectorId",
    "AlertTitle", "Category", "MitreTechniques", "ActionGrouped",
    "ActionGranular", "EntityType", "EvidenceRole", "DeviceId", "Sha256",
    "IpAddress", "Url", "AccountSid", "AccountUpn", "AccountObjectId",
    "AccountName", "DeviceName", "NetworkMessageId", "EmailClusterId",
    "RegistryKey", "RegistryValueName", "RegistryValueData", "ApplicationId",
    "ApplicationName", "OAuthApplicationId", "ThreatFamily", "FileName",
    "FolderPath", "ResourceIdName", "ResourceType", "Roles", "OSFamily",
    "OSVersion", "AntispamDirection", "SuspicionLevel", "LastVerdict",
    "CountryCode", "State", "City",
]

REMEDIATION_FEATURES = [
    "alert_count", "machine_entity_count", "machine_entity_ratio",
    "device_context_count", "device_context_ratio", "vm_resource_count",
    "dominant_category_code", "max_suspicion_score", "max_verdict_score",
    "max_severity_score", "unique_entity_types", "has_process_entity",
    "has_file_entity", "has_machine_entity",
]

# ── Pydantic Models ───────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    """Original endpoint: pre-encoded numeric features."""
    features: Dict[str, float] = Field(..., description="Dict of feature names → numeric values")
    include_remediation: bool = Field(default=True)

    class Config:
        json_schema_extra = {
            "example": {
                "features": {"OrgId": 77.0, "IncidentId": 105914.0},
                "include_remediation": True,
            }
        }


class RawAlertRequest(BaseModel):
    """
    New endpoint: raw alert fields exactly as they appear in the GUIDE dataset.
    All fields are optional — missing ones are filled with sensible defaults
    before preprocessing so the model can always run.
    """
    # Core identifiers (will be frequency-encoded)
    OrgId:            Optional[Any] = None
    IncidentId:       Optional[Any] = None
    AlertId:          Optional[Any] = None
    DetectorId:       Optional[Any] = None
    Category:         Optional[str] = None
    MitreTechniques:  Optional[str] = None
    ActionGrouped:    Optional[str] = None
    ActionGranular:   Optional[str] = None
    EntityType:       Optional[str] = None
    EvidenceRole:     Optional[str] = None
    DeviceId:         Optional[Any] = None

    # Forensic fields
    Sha256:           Optional[str] = None
    IpAddress:        Optional[str] = None
    Url:              Optional[str] = None
    FileName:         Optional[str] = None
    FolderPath:       Optional[str] = None
    ThreatFamily:     Optional[str] = None
    RegistryKey:      Optional[str] = None
    RegistryValueName:Optional[str] = None
    RegistryValueData:Optional[str] = None

    # Account / identity
    AccountSid:       Optional[str] = None
    AccountUpn:       Optional[str] = None
    AccountObjectId:  Optional[str] = None
    AccountName:      Optional[str] = None

    # Network / email
    NetworkMessageId: Optional[Any] = None
    EmailClusterId:   Optional[Any] = None

    # App
    ApplicationId:    Optional[Any] = None
    ApplicationName:  Optional[str] = None
    OAuthApplicationId:Optional[Any]= None

    # Device / OS
    DeviceName:       Optional[str] = None
    ResourceIdName:   Optional[str] = None
    ResourceType:     Optional[str] = None
    Roles:            Optional[str] = None
    OSFamily:         Optional[str] = None
    OSVersion:        Optional[str] = None

    # Numeric / scored fields
    SuspicionLevel:   Optional[float] = None
    LastVerdict:      Optional[str] = None
    AntispamDirection:Optional[str] = None

    # Geo
    CountryCode:      Optional[str] = None
    State:            Optional[str] = None
    City:             Optional[str] = None

    # Timestamp & misc
    Timestamp:        Optional[Any] = None
    Id:               Optional[Any] = None

    # Optional: include_remediation
    include_remediation: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "Category": "Malware",
                "MitreTechniques": "T1059.001",
                "IpAddress": "185.234.219.3",
                "FileName": "powershell.exe",
                "ThreatFamily": "Cobalt Strike",
                "OSFamily": "Windows",
                "SuspicionLevel": 3.0,
                "LastVerdict": "Malicious",
            }
        }


class RemediationPrediction(BaseModel):
    prediction: int
    probability: float
    threshold: float


<<<<<<< HEAD
class PredictionResponse(BaseModel):
    triage_class:    int
    triage_label:    str   # FP / BP / TP
    triage_full:     str   # False Positive / Benign Positive / True Positive
    confidence:      float
    class_probabilities: Dict[str, float]
    remediation_predictions: Optional[Dict[str, RemediationPrediction]] = None
    warnings: List[str] = []


# ── Model Manager ─────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.triage_model    = None
        self.account_model   = None
        self.endpoint_model  = None
        self.incident_scaler = None
        self.thresholds      = None
        self.is_loaded       = False

    def load_models(self, verbose: bool = True) -> None:
        models_dir   = PROJECT_ROOT / "models"
        xgboost_dir  = models_dir / "xgboost"
        classical_dir= models_dir / "classical"

        self.triage_model = joblib.load(xgboost_dir / "triage_model.pkl")
        if verbose: print("✓ Triage XGBoost model loaded")

        self.account_model   = joblib.load(classical_dir / "account_response_gbt.pkl")
        self.endpoint_model  = joblib.load(classical_dir / "endpoint_response_lr.pkl")
        self.incident_scaler = joblib.load(classical_dir / "incident_scaler.pkl")
        if verbose: print("✓ Remediation models loaded")

        with open(classical_dir / "remediation_thresholds.json") as f:
            self.thresholds = json.load(f)
        if verbose: print("✓ Thresholds loaded")

        self.is_loaded = True

    def predict_triage(self, X: np.ndarray):
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        proba = self.triage_model.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        return preds, proba

    def predict_remediation(self, incident_features: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        missing = set(REMEDIATION_FEATURES) - set(incident_features.columns)
        if missing:
            raise ValueError(f"Missing remediation features: {missing}")

        scaled = self.incident_scaler.transform(incident_features)
        acc_proba  = self.account_model.predict_proba(scaled)[:, 1]
        end_proba  = self.endpoint_model.predict_proba(scaled)[:, 1]
        acc_thresh = float(self.thresholds["account_response"])
        end_thresh = float(self.thresholds["endpoint_response"])

        return {
            "account_response": {
                "prediction":  int((acc_proba >= acc_thresh).astype(int)[0]),
                "probability": float(acc_proba[0]),
                "threshold":   acc_thresh,
            },
            "endpoint_response": {
                "prediction":  int((end_proba >= end_thresh).astype(int)[0]),
                "probability": float(end_proba[0]),
                "threshold":   end_thresh,
            },
        }


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SOC Incident Triage & Remediation API",
    description="Predicts incident triage severity and remediation actions",
    version="2.0.0",
)

# Allow the frontend (any origin during development) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_models(verbose=True)
        print("\n✓ API ready")
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}")
        raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_response(triage_pred, triage_proba, include_remediation, features_df, warnings) -> PredictionResponse:
    """Shared logic to build a PredictionResponse from model output."""
    triage_class = int(triage_pred[0])
    confidence   = float(triage_proba[0].max())

    response = PredictionResponse(
        triage_class=triage_class,
        triage_label=TRIAGE_CLASS_LABELS[triage_class],
        triage_full =TRIAGE_CLASS_FULL[triage_class],
        confidence  =confidence,
        class_probabilities={
            "FP": float(triage_proba[0, 0]),
            "BP": float(triage_proba[0, 1]),
            "TP": float(triage_proba[0, 2]),
        },
        warnings=warnings,
    )

    if include_remediation:
        try:
            rem_pred = model_manager.predict_remediation(features_df)
            response.remediation_predictions = {
                "account_response":  RemediationPrediction(**rem_pred["account_response"]),
                "endpoint_response": RemediationPrediction(**rem_pred["endpoint_response"]),
            }
        except ValueError as e:
            response.warnings.append(f"Remediation skipped: {e}")

    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy" if model_manager.is_loaded else "initializing",
            "models_loaded": model_manager.is_loaded}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TriageRequest) -> PredictionResponse:
    """
    Original endpoint — caller supplies pre-encoded numeric features.
    Use /predict-raw if you have raw alert fields instead.
    """
    if not model_manager.is_loaded:
        raise HTTPException(503, "Models not loaded")

    try:
        features_df = pd.DataFrame([request.features])
        missing = [f for f in TRIAGE_FEATURES if f not in features_df.columns]
        if missing:
            raise HTTPException(400, f"Missing required features: {missing}")

        X = features_df[TRIAGE_FEATURES].values.astype(np.float32)
        triage_pred, triage_proba = model_manager.predict_triage(X)

        return _build_response(triage_pred, triage_proba,
                               request.include_remediation, features_df, [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")


@app.post("/predict-raw", response_model=PredictionResponse)
async def predict_raw(request: RawAlertRequest) -> PredictionResponse:
    """
    NEW — accepts raw GUIDE-style alert fields (strings, nulls, etc.)
    and runs the full preprocessing pipeline before prediction.

    The frontend sends human-readable values like:
      { "Category": "Malware", "IpAddress": "185.234.219.3", ... }

    This endpoint:
      1. Converts the request to a single-row DataFrame
      2. Runs clean_data() to fill missing values
      3. Applies saved frequency encoders from models/artifacts/
      4. Aligns columns to match training feature order
      5. Runs triage (and optionally remediation) prediction
      6. Returns the same PredictionResponse shape as /predict
    """
    if not model_manager.is_loaded:
        raise HTTPException(503, "Models not loaded")

    try:
        # ── Import preprocessing pipeline ──────────────────────────────────
        from utils.preprocessing_inference import apply_preprocessing_artifacts

        # ── Build a single-row DataFrame from the request ──────────────────
        raw_dict = request.dict(exclude={"include_remediation"})
        row_df = pd.DataFrame([raw_dict])

        warnings: List[str] = []

        # ── Run full preprocessing pipeline ────────────────────────────────
        # apply_preprocessing_artifacts cleans + frequency-encodes every column
        # using the saved artifacts from models/artifacts/
        X_processed = apply_preprocessing_artifacts(
            row_df,
            target_col=None,     # no target at inference time
            apply_scaling=False, # XGBoost doesn't need scaling
            verbose=False,
        )

        # ── Align to training feature order ────────────────────────────────
        # Add any columns the model expects that weren't in the request (fill 0)
        for col in TRIAGE_FEATURES:
            if col not in X_processed.columns:
                X_processed[col] = 0.0
                warnings.append(f"Feature '{col}' not provided — defaulted to 0")

        # Drop any extra columns not seen during training
        extra_cols = [c for c in X_processed.columns if c not in TRIAGE_FEATURES]
        if extra_cols:
            X_processed = X_processed.drop(columns=extra_cols)

        # Enforce exact column order
        X_processed = X_processed[TRIAGE_FEATURES]
        X = X_processed.values.astype(np.float32)

        # ── Predict ────────────────────────────────────────────────────────
        triage_pred, triage_proba = model_manager.predict_triage(X)

        return _build_response(
            triage_pred, triage_proba,
            request.include_remediation,
            X_processed,
            warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Raw prediction failed: {e}")


@app.get("/features/triage")
async def get_triage_features():
    return {"features": TRIAGE_FEATURES, "count": len(TRIAGE_FEATURES)}


@app.get("/features/remediation")
async def get_remediation_features():
    return {"features": REMEDIATION_FEATURES, "count": len(REMEDIATION_FEATURES)}


@app.get("/classes")
async def get_classes():
    return {"classes": TRIAGE_CLASS_LABELS, "full_labels": TRIAGE_CLASS_FULL, "n_classes": 3}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
=======
    return {
        "prediction": pred,
        "probabilities": proba.tolist()
    }
>>>>>>> origin/main
>>>>>>> feature/lightgbm

"""
FastAPI backend for SOC triage model inference.

Contract:
  POST /predict
    Request:  {"features": number[], "model": "xgboost" | "lightgbm" | "tabnet"}
    Response: {"prediction": number, "probabilities": number[], "model": string}

  GET /health
"""

from __future__ import annotations

import json
import csv
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, cast

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "v1"
MODEL_NAMES = ("xgboost", "lightgbm", "tabnet")


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Preprocessed numeric feature vector.")
    model: Literal["xgboost", "lightgbm", "tabnet"]


class PredictResponse(BaseModel):
    prediction: int
    probabilities: List[float]
    model: str


class RemediationRecommendation(BaseModel):
    prediction: int
    probability: float
    threshold: float


class RemediationRequest(BaseModel):
    incident_features: List[float] = Field(..., description="Incident-level numeric feature vector.")


class RemediationResponse(BaseModel):
    account_response: RemediationRecommendation
    endpoint_response: RemediationRecommendation


class FeatureSampleResponse(BaseModel):
    features: List[float]
    dataset: str
    split: Literal["train", "val", "test"]
    row: int
    feature_count: int
    target: Optional[int] = None
    source: str


class MetricsClassStats(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class MetricsResponse(BaseModel):
    confusion_matrix: List[List[int]]
    accuracy: float
    macro_f1: float
    class_distribution: Dict[str, int]
    confidence_distribution: List[Dict[str, Any]]
    alerts_over_time: List[Dict[str, Any]]
    per_class: Dict[str, MetricsClassStats]


class EvaluationResponse(MetricsResponse):
    source: str
    message: str


class ModelStatus(BaseModel):
    loaded: bool
    error: Optional[str] = None


class ModelRegistry:
    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.tabnet_scaler: Any | None = None
        self.account_model: Any | None = None
        self.endpoint_model: Any | None = None
        self.incident_scaler: Any | None = None
        self.remediation_metadata: Dict[str, Any] = {}
        self.remediation_thresholds: Dict[str, float] = {}
        self.remediation_status = ModelStatus(loaded=False)
        self.metrics_payloads: Dict[str, Dict[str, Any]] = {}
        self.metrics_sources: Dict[str, str] = {}
        self.status: Dict[str, ModelStatus] = {
            model_name: ModelStatus(loaded=False) for model_name in MODEL_NAMES
        }

    def load_all(self) -> None:
        self._load_xgboost()
        self._load_lightgbm()
        self._load_tabnet()
        self._load_remediation()

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _set_loaded(self, model_name: str, model: Any, config: Dict[str, Any]) -> None:
        self.models[model_name] = model
        self.configs[model_name] = config
        self.status[model_name] = ModelStatus(loaded=True)

    def _set_failed(self, model_name: str, exc: Exception) -> None:
        self.status[model_name] = ModelStatus(loaded=False, error=f"{type(exc).__name__}: {exc}")

    def _load_xgboost(self) -> None:
        try:
            model_path = MODEL_ROOT / "xgboost" / "triage_model.pkl"
            config_path = MODEL_ROOT / "xgboost" / "triage_model_config.json"
            model = joblib.load(model_path)
            config = self._load_json(config_path)
            self._set_loaded("xgboost", model, config)
        except Exception as exc:
            self._set_failed("xgboost", exc)

    def _load_lightgbm(self) -> None:
        try:
            model_path = MODEL_ROOT / "lightgbm" / "triage_model.pkl"
            config_path = MODEL_ROOT / "lightgbm" / "triage_model_config.json"
            model = joblib.load(model_path)
            config = self._load_json(config_path)
            self._set_loaded("lightgbm", model, config)
        except Exception as exc:
            self._set_failed("lightgbm", exc)

    def _load_tabnet(self) -> None:
        try:
            from src.models.tabnet.utils import load_tabnet_model

            model, scaler, config = load_tabnet_model(
                model_dir=str(MODEL_ROOT / "tabnet"),
                model_name="triage_model",
                verbose=False,
            )
            self.tabnet_scaler = scaler
            self._set_loaded("tabnet", model, config)
        except Exception as exc:
            self._set_failed("tabnet", exc)

    def _load_remediation(self) -> None:
        try:
            classical_dir = MODEL_ROOT / "classical"
            self.account_model = joblib.load(classical_dir / "account_response_gbt.pkl")
            self.endpoint_model = joblib.load(classical_dir / "endpoint_response_lr.pkl")
            self.incident_scaler = joblib.load(classical_dir / "incident_scaler.pkl")
            self.remediation_metadata = self._load_json(classical_dir / "remediation_model_metadata.json")
            self.remediation_thresholds = self._load_json(classical_dir / "remediation_thresholds.json")
            self.remediation_status = ModelStatus(loaded=True)
        except Exception as exc:
            self.remediation_status = ModelStatus(loaded=False, error=f"{type(exc).__name__}: {exc}")

    def _metrics_path_for_model(self, model_name: str) -> Path:
        metrics_dir = PROJECT_ROOT / "reports" / "metrics"
        if model_name == "tabnet":
            return metrics_dir / "triage_metrics.json"
        return metrics_dir / f"{model_name}_triage_metrics.json"

    def _load_test_features(self) -> np.ndarray:
        feature_path = PROCESSED_DATA_ROOT / "X_test.csv"
        with open(feature_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            rows = [[float(value) for value in row] for row in reader]
        return np.asarray(rows, dtype=np.float32)

    def _load_test_timestamps(self) -> List[str]:
        feature_path = PROCESSED_DATA_ROOT / "X_test.csv"
        raw_timestamps: List[float] = []
        with open(feature_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                processed_timestamp = row.get("Timestamp")
                if processed_timestamp is None:
                    raise HTTPException(status_code=503, detail="Processed test features are missing the Timestamp column.")
                raw_timestamps.append(float(processed_timestamp))

        timestamp_array = np.asarray(raw_timestamps, dtype=np.float32)
        if timestamp_array.size == 0:
            return []

        bucket_count = min(12, int(timestamp_array.size))
        if bucket_count <= 1:
            return ["Window 01"] * int(timestamp_array.size)

        edges = np.linspace(float(timestamp_array.min()), float(timestamp_array.max()), bucket_count + 1)
        bucket_indexes = np.digitize(timestamp_array, edges[1:-1], right=False)
        return [f"Window {index + 1:02d}" for index in bucket_indexes.tolist()]

    def _predict_probabilities_for_metrics(self, model_name: str, features: np.ndarray) -> np.ndarray:
        model_status = self.status.get(model_name)
        if model_status is None or not model_status.loaded:
            message = "Model is unavailable." if model_status is None else (model_status.error or "Model is unavailable.")
            raise HTTPException(status_code=503, detail=f"Metrics unavailable because model '{model_name}' is not loaded: {message}")

        model = self.models[model_name]
        if model_name == "tabnet":
            scaled = self.tabnet_scaler.transform(features, split_name="Metrics")
            probabilities = model.predict_proba(scaled)
        else:
            probabilities = model.predict_proba(features)
        return np.asarray(probabilities, dtype=np.float32)

    def _build_confidence_distribution(self, probabilities: np.ndarray) -> List[Dict[str, Any]]:
        confidences = np.max(probabilities, axis=1)
        bins = np.linspace(0.0, 1.0, 11)
        counts, edges = np.histogram(confidences, bins=bins)
        distribution: List[Dict[str, Any]] = []
        for index, count in enumerate(counts):
            start = int(round(edges[index] * 100))
            end = int(round(edges[index + 1] * 100))
            distribution.append({
                "bucket": f"{start}-{end}%",
                "count": int(count),
            })
        return distribution

    def _build_alerts_over_time(self, probabilities: np.ndarray, timestamps: List[str]) -> List[Dict[str, Any]]:
        predictions = np.argmax(probabilities, axis=1)
        if len(predictions) != len(timestamps):
            raise HTTPException(
                status_code=503,
                detail="Metrics artifact mismatch: processed test rows do not align with processed timestamp buckets.",
            )

        labels = ("FalsePositive", "BenignPositive", "TruePositive")
        grouped: Dict[str, Dict[str, int]] = {}
        for date_key, prediction in zip(timestamps, predictions):
            bucket = grouped.setdefault(date_key, {label: 0 for label in labels})
            bucket[labels[int(prediction)]] += 1

        return [
            {"date": date_key, **grouped[date_key]}
            for date_key in sorted(grouped.keys())
        ]

    def _build_metrics_payload(self, model_name: str) -> Dict[str, Any]:
        metrics_path = self._metrics_path_for_model(model_name)
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail=f"Metrics file not found for model '{model_name}'.")

        raw_metrics = self._load_json(metrics_path)
        per_class = {
            "FalsePositive": {
                "precision": raw_metrics["per_class_metrics"]["Class_0"]["precision"],
                "recall": raw_metrics["per_class_metrics"]["Class_0"]["recall"],
                "f1": raw_metrics["per_class_metrics"]["Class_0"]["f1"],
                "support": raw_metrics["per_class_metrics"]["Class_0"]["support"],
            },
            "BenignPositive": {
                "precision": raw_metrics["per_class_metrics"]["Class_1"]["precision"],
                "recall": raw_metrics["per_class_metrics"]["Class_1"]["recall"],
                "f1": raw_metrics["per_class_metrics"]["Class_1"]["f1"],
                "support": raw_metrics["per_class_metrics"]["Class_1"]["support"],
            },
            "TruePositive": {
                "precision": raw_metrics["per_class_metrics"]["Class_2"]["precision"],
                "recall": raw_metrics["per_class_metrics"]["Class_2"]["recall"],
                "f1": raw_metrics["per_class_metrics"]["Class_2"]["f1"],
                "support": raw_metrics["per_class_metrics"]["Class_2"]["support"],
            },
        }

        test_features = self._load_test_features()
        timestamps = self._load_test_timestamps()
        probabilities = self._predict_probabilities_for_metrics(model_name, test_features)

        payload = {
            "confusion_matrix": raw_metrics["confusion_matrix"],
            "accuracy": raw_metrics.get("overall_accuracy", raw_metrics.get("accuracy")),
            "macro_f1": raw_metrics["macro_f1"],
            "class_distribution": {
                label: int(stats["support"])
                for label, stats in per_class.items()
            },
            "confidence_distribution": self._build_confidence_distribution(probabilities),
            "alerts_over_time": self._build_alerts_over_time(probabilities, timestamps),
            "per_class": per_class,
        }
        self.metrics_sources[model_name] = str(metrics_path.relative_to(PROJECT_ROOT))
        return payload

    def health_payload(self) -> Dict[str, Any]:
        online = any(status.loaded for status in self.status.values())
        status_to_dict = lambda status: status.model_dump() if hasattr(status, "model_dump") else status.dict()
        return {
            "status": "healthy" if online else "unhealthy",
            "models": {
                name: status_to_dict(status)
                for name, status in self.status.items()
            } | {"remediation": status_to_dict(self.remediation_status)},
        }

    def expected_feature_count(self, model_name: str) -> int:
        config = self.configs.get(model_name, {})
        feature_names = config.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            return len(feature_names)

        train_shape = config.get("train_shape")
        if isinstance(train_shape, list) and len(train_shape) >= 2:
            return int(train_shape[1])

        # TabNet config in this repo does not currently store feature names.
        return 44

    def validate_features(self, model_name: str, features: List[float]) -> np.ndarray:
        if not features:
            raise HTTPException(status_code=400, detail="Features array cannot be empty.")

        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 1:
            raise HTTPException(status_code=400, detail="Features must be a one-dimensional numeric array.")
        if np.isnan(array).any() or np.isinf(array).any():
            raise HTTPException(status_code=400, detail="Features array contains invalid numeric values.")

        expected = self.expected_feature_count(model_name)
        if array.shape[0] != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' expects {expected} features, but received {array.shape[0]}.",
            )

        return array.reshape(1, -1)

    def predict(self, model_name: str, features: List[float]) -> PredictResponse:
        model_status = self.status.get(model_name)
        if model_status is None:
            raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'.")
        if not model_status.loaded:
            message = model_status.error or "Model is unavailable."
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' is not loaded: {message}")

        data = self.validate_features(model_name, features)
        model = self.models[model_name]

        try:
            if model_name == "tabnet":
                scaled = self.tabnet_scaler.transform(data, split_name="Inference")
                probabilities = model.predict_proba(scaled)
            else:
                probabilities = model.predict_proba(data)

            prediction = int(np.argmax(probabilities, axis=1)[0])
            probability_list = np.asarray(probabilities[0], dtype=np.float32).tolist()
            return PredictResponse(
                prediction=prediction,
                probabilities=probability_list,
                model=model_name,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed for '{model_name}': {exc}") from exc

    def remediation_feature_count(self) -> int:
        feature_columns = self.remediation_metadata.get("feature_columns", [])
        return len(feature_columns)

    def predict_remediation(self, incident_features: List[float]) -> RemediationResponse:
        if not self.remediation_status.loaded:
            message = self.remediation_status.error or "Remediation models are unavailable."
            raise HTTPException(status_code=503, detail=f"Remediation models are not loaded: {message}")

        array = np.asarray(incident_features, dtype=np.float32)
        expected = self.remediation_feature_count()

        if array.ndim != 1:
            raise HTTPException(status_code=400, detail="Incident features must be a one-dimensional numeric array.")
        if array.shape[0] != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Remediation expects {expected} incident features, but received {array.shape[0]}.",
            )
        if np.isnan(array).any() or np.isinf(array).any():
            raise HTTPException(status_code=400, detail="Incident features contain invalid numeric values.")

        try:
            scaled = self.incident_scaler.transform(array.reshape(1, -1))
            account_probability = float(self.account_model.predict_proba(scaled)[:, 1][0])
            endpoint_probability = float(self.endpoint_model.predict_proba(scaled)[:, 1][0])
            account_threshold = float(self.remediation_thresholds["account_response"])
            endpoint_threshold = float(self.remediation_thresholds["endpoint_response"])

            return RemediationResponse(
                account_response=RemediationRecommendation(
                    prediction=int(account_probability >= account_threshold),
                    probability=account_probability,
                    threshold=account_threshold,
                ),
                endpoint_response=RemediationRecommendation(
                    prediction=int(endpoint_probability >= endpoint_threshold),
                    probability=endpoint_probability,
                    threshold=endpoint_threshold,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Remediation prediction failed: {exc}") from exc

    def metrics(self, model_name: Literal["xgboost", "lightgbm", "tabnet"]) -> MetricsResponse:
        if model_name not in self.metrics_payloads:
            self.metrics_payloads[model_name] = self._build_metrics_payload(model_name)
        return MetricsResponse(**self.metrics_payloads[model_name])

    def evaluate(self, model_name: Literal["xgboost", "lightgbm", "tabnet"]) -> EvaluationResponse:
        metrics = self.metrics(model_name)
        payload = metrics.model_dump() if hasattr(metrics, "model_dump") else metrics.dict()
        return EvaluationResponse(
            **payload,
            source=self.metrics_sources.get(model_name, "reports/metrics"),
            message=f"Loaded saved evaluation metrics and derived backend chart series for '{model_name}'.",
        )

    def sample_features(self, split: Literal["train", "val", "test"], row: int) -> FeatureSampleResponse:
        if row < 0:
            raise HTTPException(status_code=400, detail="Row index must be zero or greater.")

        feature_path = PROCESSED_DATA_ROOT / f"X_{split}.csv"
        target_path = PROCESSED_DATA_ROOT / f"y_{split}.csv"
        if not feature_path.exists():
            raise HTTPException(status_code=404, detail=f"Processed feature file not found: {feature_path.name}")

        try:
            with open(feature_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header:
                    raise HTTPException(status_code=503, detail=f"Processed feature file is empty: {feature_path.name}")

                selected_row = None
                for index, values in enumerate(reader):
                    if index == row:
                        selected_row = values
                        break

            if selected_row is None:
                raise HTTPException(status_code=404, detail=f"Row {row} not found in {feature_path.name}.")

            features = [float(value) for value in selected_row]
            target = None
            if target_path.exists():
                with open(target_path, "r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    next(reader, None)
                    for index, values in enumerate(reader):
                        if index == row:
                            if values:
                                target = int(float(values[0]))
                            break

            return FeatureSampleResponse(
                features=features,
                dataset="triage",
                split=split,
                row=row,
                feature_count=len(features),
                target=target,
                source=str(feature_path.relative_to(PROJECT_ROOT)),
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=f"Processed row contains non-numeric values: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load processed sample: {exc}") from exc


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    registry.load_all()
    for model_name, status in registry.status.items():
        if status.loaded:
            print(f"[startup] loaded {model_name}")
        else:
            print(f"[startup] failed {model_name}: {status.error}")
    yield


app = FastAPI(
    title="SOC Intelligence Inference API",
    version="1.0.0",
    description="Inference API for XGBoost, LightGBM, and TabNet SOC triage models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return registry.health_payload()


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    return registry.predict(request.model, request.features)


@app.post("/remediation-predict", response_model=RemediationResponse)
async def remediation_predict(request: RemediationRequest) -> RemediationResponse:
    return registry.predict_remediation(request.incident_features)


@app.get("/metrics", response_model=MetricsResponse)
async def metrics(
    model: Literal["xgboost", "lightgbm", "tabnet"] = "xgboost",
) -> MetricsResponse:
    return registry.metrics(cast(Literal["xgboost", "lightgbm", "tabnet"], model))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    model: Literal["xgboost", "lightgbm", "tabnet"] = "xgboost",
) -> EvaluationResponse:
    return registry.evaluate(cast(Literal["xgboost", "lightgbm", "tabnet"], model))


@app.get("/sample-features", response_model=FeatureSampleResponse)
async def sample_features(
    split: Literal["train", "val", "test"] = "test",
    row: int = 0,
) -> FeatureSampleResponse:
    return registry.sample_features(split, row)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

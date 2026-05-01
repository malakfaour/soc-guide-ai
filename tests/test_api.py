from fastapi.testclient import TestClient

from app import app


def test_health_reports_model_statuses():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "unhealthy"}
    assert {"xgboost", "lightgbm", "tabnet", "remediation"}.issubset(payload["models"])


def test_metrics_payload_shape():
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in payload["confusion_matrix"])
    assert 0 <= payload["accuracy"] <= 1
    assert 0 <= payload["macro_f1"] <= 1
    assert {"FalsePositive", "BenignPositive", "TruePositive"}.issubset(payload["per_class"])


def test_processed_sample_can_feed_loaded_models():
    with TestClient(app) as client:
        sample_response = client.get("/sample-features?split=test&row=0")
        assert sample_response.status_code == 200
        sample = sample_response.json()
        assert sample["feature_count"] == 44
        assert len(sample["features"]) == 44

        health = client.get("/health").json()
        loaded_models = [
            name
            for name, status in health["models"].items()
            if name in {"xgboost", "lightgbm", "tabnet"} and status["loaded"]
        ]
        assert loaded_models, "At least one triage model should be loaded for demo inference."

        for model in loaded_models:
            prediction_response = client.post(
                "/predict",
                json={"features": sample["features"], "model": model},
            )
            assert prediction_response.status_code == 200
            prediction = prediction_response.json()
            assert prediction["model"] == model
            assert prediction["prediction"] in {0, 1, 2}
            assert len(prediction["probabilities"]) == 3


def test_evaluate_returns_saved_metrics_source():
    with TestClient(app) as client:
        response = client.post("/evaluate")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"].startswith("reports")
    assert payload["message"]

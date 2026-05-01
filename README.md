# SOC Intelligence

<<<<<<< HEAD
> AI-powered security operations intelligence for alert triage, model training, and explainable incident classification.

## 🚀 Overview

SOC Intelligence is a machine learning project for security alert classification and incident triage. It combines a modular preprocessing pipeline, multiple model implementations, evaluation utilities, explainability tooling, and a lightweight FastAPI inference service to support experimentation and deployment workflows for SOC use cases.

The repository is organized for end-to-end work across data preparation, model training, inference, and reporting, with local artifacts stored under the project workspace.

## ✨ Features

- FastAPI inference API for serving model predictions
- Modular preprocessing pipeline for loading, cleaning, encoding, splitting, and scaling data
- Support for multiple model families including XGBoost, LightGBM, and TabNet
- Evaluation utilities for triage metrics and performance reporting
- TabNet explainability with feature importance and attention-mask visualizations
- Artifact management for encoders, mappings, scalers, and model assets
- Structured project layout for training, tuning, validation, and reporting workflows

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Frontend / Swagger UI] --> B[FastAPI API]
    B --> C[Worker Layer<br/>Preprocessing, Training, Inference, Evaluation]
    C --> D[Data & Artifact Store<br/>CSV files, model files, reports, artifacts]
```

### Flow Summary

- `Frontend`: Currently represented by FastAPI Swagger UI and any future dashboard/client.
- `API`: Receives prediction requests and exposes interactive docs.
- `Worker`: Runs preprocessing, model inference, training, tuning, and evaluation scripts.
- `Database`: In the current project state, storage is file-based under `data/`, `models/`, and `reports/` rather than an external database service.

## 🛠️ Tech Stack

- `Python`
- `FastAPI` + `Pydantic`
- `XGBoost`
- `LightGBM`
- `TabNet / PyTorch`
- `scikit-learn`
- `pandas` and `numpy`
- `matplotlib` and `seaborn`
- `SHAP`

## 📁 Project Structure

```text
.
+-- app.py                  # FastAPI inference entry point
+-- requirements.txt        # Python dependencies
+-- src/
|   +-- data/               # Data loading and splitting
|   +-- preprocessing/      # Cleaning, encoding, scaling, pipeline orchestration
|   +-- models/             # Model-specific training and prediction logic
|   +-- training/           # Training workflows
|   +-- tuning/             # Hyperparameter tuning
|   +-- evaluation/         # Metrics and validation helpers
|   +-- explainability/     # TabNet explainability utilities
|   '-- utils/              # Shared helpers and artifact management
+-- data/                   # Raw and processed datasets
+-- models/                 # Saved model files and preprocessing artifacts
+-- reports/                # Metrics and generated outputs
'-- docs/                   # Supporting technical documentation
```

## ⚙️ Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd "SOC Intelligence"
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install backend dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn xgboost lightgbm joblib pytorch-tabnet
```

### 3. Prepare the database / storage layer

This project currently uses local file-based storage instead of an external database.

- Place raw datasets in `data/raw/`
- Saved models live in `models/`
- Preprocessing artifacts are stored in `models/artifacts/`
- Metrics and reports are written to `reports/`

Expected raw dataset paths:

```text
data/raw/GUIDE_Train.csv
data/raw/GUIDE_Test.csv
```

### 4. Frontend setup

There is no separate frontend app in the repository at the moment. For local development, use the built-in FastAPI Swagger UI as the primary interface for testing and exploring the API.

## ▶️ Run Locally

### Start the API

```bash
uvicorn app:app --reload
```

The API will be available at:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

### Run supporting workflows

Depending on your use case, you can also execute standalone scripts for evaluation, explainability, or validation from the project root, for example:

```bash
python evaluation_integration_example.py
python final_validation_evaluation.py
python validate_explainability.py
```

## 📚 API Documentation

Interactive API documentation is automatically available through Swagger UI:

- `GET /docs` for Swagger UI

Once the API is running, open:

```text
http://127.0.0.1:8000/docs
```

## 🔮 Future Improvements

- Add a dedicated frontend dashboard for analysts and SOC operators
- Introduce an external database or feature store for production-scale persistence
- Containerize the API and worker workflows with Docker
- Add background job orchestration for training and batch inference
- Improve dependency management and lock the full runtime stack
- Expand automated tests and CI/CD coverage
- Add authentication, rate limiting, and production deployment configs

## 📝 Notes

- The repository already includes saved model artifacts and evaluation outputs for experimentation.
- The current architecture is well-suited for local development, research, and iterative model improvement.
- For production deployment, the next step would typically be adding a dedicated worker queue, persistent database, and deployment automation.
=======
SOC incident analysis project with a finalized hybrid modeling pipeline.

## Final Architecture

The project now uses a task-specific final design:

- `TabNet` for row-level triage
- `LightGBM` as a triage baseline
- `XGBoost` as a triage baseline
- `Gradient Boosted Trees` for incident-level `account_response`
- `Logistic Regression` for incident-level `endpoint_response`

The runnable hybrid path reuses:
- [main.py](/c:/Users/malty/Projects/SOC%20Intelligence/main.py)
- [hybrid_incident_scoring.py](/c:/Users/malty/Projects/SOC%20Intelligence/src/inference/hybrid_incident_scoring.py)

## Current Best-Use Summary

Triage:
- `LightGBM` currently has the strongest saved triage metrics
- `XGBoost` is the next strongest triage baseline
- `TabNet` remains the deep-learning triage model used by the hybrid pipeline

Remediation:
- `Gradient Boosted Trees` is the selected model for `account_response`
- `Logistic Regression` is the selected model for `endpoint_response`
- remediation stays incident-level because that matches the label granularity

## Saved Artifacts

TabNet triage:
- [triage_model.zip](/c:/Users/malty/Projects/SOC%20Intelligence/models/tabnet/triage_model.zip)
- [triage_model_config.json](/c:/Users/malty/Projects/SOC%20Intelligence/models/tabnet/triage_model_config.json)
- [triage_model_scaler.pkl](/c:/Users/malty/Projects/SOC%20Intelligence/models/tabnet/triage_model_scaler.pkl)

LightGBM triage:
- [triage_model.pkl](/c:/Users/malty/Projects/SOC%20Intelligence/models/lightgbm/triage_model.pkl)
- [triage_model_config.json](/c:/Users/malty/Projects/SOC%20Intelligence/models/lightgbm/triage_model_config.json)

XGBoost triage:
- [triage_model.pkl](/c:/Users/malty/Projects/SOC%20Intelligence/models/xgboost/triage_model.pkl)
- [triage_model_config.json](/c:/Users/malty/Projects/SOC%20Intelligence/models/xgboost/triage_model_config.json)

Classical remediation:
- [account_response_gbt.pkl](/c:/Users/malty/Projects/SOC%20Intelligence/models/classical/account_response_gbt.pkl)
- [endpoint_response_lr.pkl](/c:/Users/malty/Projects/SOC%20Intelligence/models/classical/endpoint_response_lr.pkl)
- [incident_scaler.pkl](/c:/Users/malty/Projects/SOC%20Intelligence/models/classical/incident_scaler.pkl)
- [remediation_thresholds.json](/c:/Users/malty/Projects/SOC%20Intelligence/models/classical/remediation_thresholds.json)
- [remediation_model_metadata.json](/c:/Users/malty/Projects/SOC%20Intelligence/models/classical/remediation_model_metadata.json)

## Key Reports

Metrics:
- [triage_metrics.json](/c:/Users/malty/Projects/SOC%20Intelligence/reports/metrics/triage_metrics.json)
- [lightgbm_triage_metrics.json](/c:/Users/malty/Projects/SOC%20Intelligence/reports/metrics/lightgbm_triage_metrics.json)
- [xgboost_triage_metrics.json](/c:/Users/malty/Projects/SOC%20Intelligence/reports/metrics/xgboost_triage_metrics.json)
- [classical_remediation_comparison.json](/c:/Users/malty/Projects/SOC%20Intelligence/reports/metrics/classical_remediation_comparison.json)

Comparisons:
- [tabnet_vs_lightgbm_triage.json](/c:/Users/malty/Projects/SOC%20Intelligence/reports/comparisons/tabnet_vs_lightgbm_triage.json)
- [tabnet_vs_xgboost_triage.json](/c:/Users/malty/Projects/SOC%20Intelligence/reports/comparisons/tabnet_vs_xgboost_triage.json)
- [endpoint_response_limitations.md](/c:/Users/malty/Projects/SOC%20Intelligence/reports/comparisons/endpoint_response_limitations.md)

Figures:
- [feature_importance.png](/c:/Users/malty/Projects/SOC%20Intelligence/reports/figures/feature_importance.png)
- [step_importance.png](/c:/Users/malty/Projects/SOC%20Intelligence/reports/figures/step_importance.png)
- [tabnet_explainability_report.json](/c:/Users/malty/Projects/SOC%20Intelligence/reports/figures/tabnet_explainability_report.json)

## Final Validation Commands

Run the saved-model inference paths:

```powershell
python src\models\lightgbm\predict.py
python src\models\xgboost\predict.py
python main.py --row-limit 3 --incident-limit 1
```

Re-run training baselines only if needed:

```powershell
python src\models\lightgbm\train.py
python src\models\xgboost\train.py
python compare_remediation_baselines.py
```

## Final Notes

- Triage and remediation are intentionally separated now.
- `endpoint_response` remains the weakest label because it is low-signal and has very few positive incidents.
- The project is best understood as a hybrid system, not a single-model system.
>>>>>>> origin/main

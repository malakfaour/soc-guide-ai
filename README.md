# SOC Intelligence Platform

Production-style SOC triage demo that combines a FastAPI ML backend with a React analyst console. The system scores preprocessed security alert feature vectors with XGBoost, LightGBM, or TabNet, surfaces real evaluation metrics, and supports a triage-to-remediation workflow.

## What It Demonstrates

- **Multi-model inference**: runtime model selection for XGBoost, LightGBM, and TabNet.
- **Real backend integration**: React calls FastAPI endpoints for prediction, health, metrics, remediation, and processed demo samples.
- **Preprocessing contract**: frontend sends only numeric preprocessed feature arrays; preprocessing artifacts stay backend-side.
- **Analytics dashboard**: model accuracy, macro F1, per-class metrics, confusion matrix, and class distribution from saved evaluation outputs.
- **SOC workflow**: run a prediction, add it to the triage queue, inspect probabilities, and send selected alerts to remediation.
- **Demo-ready samples**: backend can serve rows directly from `data/processed/v1/X_test.csv` for repeatable testing.

## Architecture

```text
React + Vite frontend
  -> /predict
  -> /metrics
  -> /sample-features
  -> /remediation-predict
FastAPI backend
  -> saved ML models
  -> processed datasets
  -> evaluation reports
```

Key paths:

- `app.py`: FastAPI inference API.
- `soc-frontend/`: React SOC console.
- `models/`: saved triage and remediation models.
- `models/artifacts/`: preprocessing artifacts.
- `data/processed/v1/`: preprocessed train/validation/test feature files.
- `reports/metrics/`: saved model evaluation metrics.
- `tests/`: lightweight backend API tests.

## Tech Stack

- Backend: FastAPI, Pydantic, NumPy, pandas, scikit-learn, joblib.
- ML: XGBoost, LightGBM, PyTorch TabNet, classical scikit-learn remediation models.
- Frontend: React, TypeScript, Vite, Tailwind CSS, Recharts.
- Testing: pytest, FastAPI TestClient.

## Setup

```powershell
cd "C:\Users\malty\Projects\SOC Intelligence"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

cd soc-frontend
npm install
```

Optional frontend API override:

```powershell
Copy-Item .env.example .env
```

## One-Command Demo

From the project root:

```powershell
.\run_demo.bat
```

This starts:

- FastAPI backend at `http://localhost:8000`
- React frontend at `http://localhost:5173`
- Browser opens automatically

Stop both services with `Ctrl+C`.

## Manual Run

Backend:

```powershell
cd "C:\Users\malty\Projects\SOC Intelligence"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```powershell
cd "C:\Users\malty\Projects\SOC Intelligence\soc-frontend"
npm run dev -- --host 0.0.0.0
```

## Demo Flow

1. Open the frontend and go to **Triage**.
2. Choose a model: XGBoost, LightGBM, or TabNet.
3. Click **Load processed test row** to fetch a real 44-feature row from `X_test.csv`.
4. Click **Run Prediction**.
5. Add the result to the triage queue.
6. Open the queued prediction and send it to remediation.
7. Visit **Analytics** to show real `/metrics` evaluation graphs.

## API

- `GET /health`: backend and model loading status.
- `POST /predict`: score a preprocessed numeric feature vector.
- `GET /sample-features`: return a real processed feature row for demo/testing.
- `POST /remediation-predict`: score incident-level remediation needs.
- `GET /metrics`: return saved evaluation metrics.
- `POST /evaluate`: return saved evaluation metrics with source metadata.

Prediction request:

```json
{
  "features": [0.1, 0.2, 0.3],
  "model": "lightgbm"
}
```

Prediction response:

```json
{
  "prediction": 1,
  "probabilities": [0.05, 0.91, 0.04],
  "model": "lightgbm"
}
```

## Testing

Backend:

```powershell
cd "C:\Users\malty\Projects\SOC Intelligence"
.\.venv\Scripts\python.exe -m pytest
```

Frontend:

```powershell
cd "C:\Users\malty\Projects\SOC Intelligence\soc-frontend"
npm run typecheck
npm run build
```

## Screenshots

Add final screenshots before publishing:

- `docs/screenshots/dashboard.png`
- `docs/screenshots/triage.png`
- `docs/screenshots/analytics.png`
- `docs/screenshots/remediation.png`

## Notes

- The frontend intentionally does not preprocess raw CSVs.
- Raw GUIDE data belongs in the preprocessing pipeline, not `/predict`.
- TabNet requires `pytorch-tabnet`; install `requirements.txt` before demoing all three models.
- Large raw datasets are ignored by git. Keep saved models and processed demo data available for local demos.

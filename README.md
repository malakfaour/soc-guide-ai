# SOC Intelligence

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

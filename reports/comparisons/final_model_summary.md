# Final Model Summary

## Final Project Structure

The project is finalized around a hybrid task split:

- `TabNet` -> row-level triage
- `LightGBM` -> triage baseline
- `XGBoost` -> triage baseline
- `Gradient Boosted Trees` -> incident-level `account_response`
- `Logistic Regression` -> incident-level `endpoint_response`

## Triage Results

Based on the saved metric files:

| Model | Macro-F1 | Accuracy |
|---|---:|---:|
| LightGBM | 0.9132 | 0.9186 |
| XGBoost | 0.8848 | 0.8919 |
| TabNet | 0.7834 | 0.7904 |

Current triage ranking:

1. `LightGBM`
2. `XGBoost`
3. `TabNet`

## Remediation Results

Incident-level remediation comparison showed that classical models outperform the TabNet remediation baseline.

Selected final remediation models:

- `account_response` -> `Gradient Boosted Trees`
- `endpoint_response` -> `Logistic Regression`

Key F1 scores:

| Model | account_response F1 | endpoint_response F1 | remediation macro-F1 |
|---|---:|---:|---:|
| TabNet baseline | 0.0729 | 0.0085 | 0.0407 |
| Logistic Regression | 0.1232 | 0.0584 | 0.0908 |
| Gradient Boosted Trees | 0.1585 | 0.0400 | 0.0993 |

## Final Recommendation

Use the system as a hybrid pipeline:

- `TabNet` for deep-learning triage inference
- classical incident-level models for remediation inference

Keep `LightGBM` and `XGBoost` as triage comparison baselines and reporting benchmarks.

## Remaining Limitation

`endpoint_response` is still a low-signal label.

Current incident-level positives:

- train: `89`
- validation: `19`
- test: `19`

This is a data limitation, not a sign that the final architecture is broken.

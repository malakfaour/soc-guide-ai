`endpoint_response` remains a low-signal remediation label in the current incident-level dataset.

Current incident-level positive counts:
- train: `89`
- validation: `19`
- test: `19`

Interpretation:
- this is a data limitation, not primarily a modeling limitation
- even after correcting the row-vs-incident granularity mismatch, endpoint remediation remains sparse
- classical models improved over TabNet, but performance is still constrained by low positive volume

Recommended stance for the project:
- document `endpoint_response` as a low-signal task under current data volume
- keep the current classical remediation model and tuned threshold
- prioritize more endpoint-positive incidents or richer endpoint supervision in future work

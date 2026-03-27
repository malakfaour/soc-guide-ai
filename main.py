"""
Project entrypoint for hybrid SOC incident scoring.

Expects already-processed feature CSVs and reuses the existing hybrid pipeline:
- TabNet for row-level triage
- Classical incident-level models for remediation
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "inference"))

from hybrid_incident_scoring import load_hybrid_models, score_incident


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run hybrid SOC scoring on processed features")
    parser.add_argument(
        "--row-features",
        default="data/processed/v1/X_test.csv",
        help="CSV containing processed row-level triage features",
    )
    parser.add_argument(
        "--incident-features",
        default="data/processed/v1/X_incident_remediation_test.csv",
        help="CSV containing processed incident-level remediation features",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=5,
        help="Number of alert rows to score from the row-level CSV",
    )
    parser.add_argument(
        "--incident-limit",
        type=int,
        default=1,
        help="Number of incident rows to score from the incident-level CSV",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the combined output as JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Load processed features and run the existing hybrid scorer."""
    args = parse_args()

    row_path = PROJECT_ROOT / args.row_features
    incident_path = PROJECT_ROOT / args.incident_features

    row_features = pd.read_csv(row_path).head(args.row_limit)
    incident_features = pd.read_csv(incident_path).head(args.incident_limit)

    artifacts = load_hybrid_models(verbose=False)
    outputs = score_incident(
        incident_rows_df=row_features,
        incident_features_df=incident_features,
        artifacts=artifacts,
    )

    result = {
        "row_features_path": str(row_path),
        "incident_features_path": str(incident_path),
        "row_count": int(len(row_features)),
        "incident_count": int(len(incident_features)),
        "triage_predictions": outputs["triage"]["predictions"].tolist(),
        "triage_confidence": outputs["triage"]["confidence"].tolist(),
        "account_response_prediction": outputs["remediation"]["account_response"]["predictions"].tolist(),
        "account_response_probability": outputs["remediation"]["account_response"]["probabilities"].tolist(),
        "endpoint_response_prediction": outputs["remediation"]["endpoint_response"]["predictions"].tolist(),
        "endpoint_response_probability": outputs["remediation"]["endpoint_response"]["probabilities"].tolist(),
    }

    if args.output_json:
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

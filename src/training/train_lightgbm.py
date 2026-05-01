"""
LightGBM Training Entry Point
Main script for training LightGBM triage model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightgbm.train import train_triage_model
from models.lightgbm.predict import LightGBMPredictor

from evaluation.metrics import TriageEvaluator  # ✅ FIXED
import json
import pandas as pd


def train_and_evaluate():
    print("="*80)
    print("LIGHTGBM TRIAGE MODEL - TRAINING PIPELINE")
    print("="*80)
    
    # STEP 1: Train model
    print("\nSTEP 1: TRAINING MODEL")
    model = train_triage_model(
        data_dir='data/processed/v1',
        save_dir='models/lightgbm',
        use_class_weights=True
    )
    
    # STEP 2: Load data
    print("\nSTEP 2: LOADING DATA")
    X_train = pd.read_csv('data/processed/v1/X_train.csv')
    X_val = pd.read_csv('data/processed/v1/X_val.csv')
    X_test = pd.read_csv('data/processed/v1/X_test.csv')

    y_train = pd.read_csv('data/processed/v1/y_train.csv').iloc[:, 0]
    y_val = pd.read_csv('data/processed/v1/y_val.csv').iloc[:, 0]
    y_test = pd.read_csv('data/processed/v1/y_test.csv').iloc[:, 0]
    
    # STEP 3: Validation Evaluation
    print("\nSTEP 3: VALIDATION EVALUATION")
    predictor = LightGBMPredictor()
    evaluator = TriageEvaluator(n_classes=3)

    val_results = predictor.predict(X_val)
    val_metrics = evaluator.compute_metrics(
        y_true=y_val.values,
        y_pred=val_results['predictions'],
        y_proba=val_results['probabilities']
    )

    print(evaluator.format_results(val_metrics))

    os.makedirs('reports/metrics', exist_ok=True)
    with open('reports/metrics/lightgbm_val_metrics.json', 'w') as f:
        json.dump(val_metrics, f, indent=2)

    # STEP 4: Test Evaluation
    print("\nSTEP 4: TEST EVALUATION")

    test_results = predictor.predict(X_test)
    test_metrics = evaluator.compute_metrics(
        y_true=y_test.values,
        y_pred=test_results['predictions'],
        y_proba=test_results['probabilities']
    )

    print(evaluator.format_results(test_metrics))

    with open('reports/metrics/lightgbm_test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # SUMMARY
    print("\nTRAINING SUMMARY")
    print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")

    print("\n✅ DONE")

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate()
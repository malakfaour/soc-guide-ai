#!/usr/bin/env python
"""
Model Comparison Visualization Integration Guide.

Shows how to generate comprehensive visualizations comparing TabNet (DL),
XGBoost, and LightGBM models with matplotlib including:
- Performance metrics comparison
- Per-class breakdown
- Confusion matrices
- TabNet neural network architecture diagram
- ROC curves
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from visualization.model_comparisons import ModelComparison, load_metrics_from_files


def example_1_basic_visualization():
    """Example 1: Basic visualization with mock data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Model Comparison (With Mock Data)")
    print("=" * 80)
    
    print("""
This example shows how to generate visualizations using sample metrics data.
Useful for testing or when metrics haven't been computed yet.
    """)
    
    # Create mock metrics for demonstration
    mock_metrics = {
        'TabNet': {
            'macro_f1': 0.847,
            'overall_accuracy': 0.852,
            'per_class_metrics': {
                'Class_0': {'precision': 0.91, 'recall': 0.88, 'f1': 0.895, 'support': 2500},
                'Class_1': {'precision': 0.82, 'recall': 0.85, 'f1': 0.835, 'support': 1200},
                'Class_2': {'precision': 0.78, 'recall': 0.82, 'f1': 0.800, 'support': 800},
            },
            'confusion_matrix': [
                [2200, 200, 100],
                [180, 1020, 0],
                [50, 100, 650],
            ]
        },
        'XGBoost': {
            'macro_f1': 0.813,
            'overall_accuracy': 0.820,
            'per_class_metrics': {
                'Class_0': {'precision': 0.88, 'recall': 0.86, 'f1': 0.870, 'support': 2500},
                'Class_1': {'precision': 0.80, 'recall': 0.82, 'f1': 0.810, 'support': 1200},
                'Class_2': {'precision': 0.75, 'recall': 0.80, 'f1': 0.775, 'support': 800},
            },
            'confusion_matrix': [
                [2150, 250, 100],
                [200, 984, 16],
                [60, 100, 640],
            ]
        },
        'LightGBM': {
            'macro_f1': 0.825,
            'overall_accuracy': 0.832,
            'per_class_metrics': {
                'Class_0': {'precision': 0.89, 'recall': 0.87, 'f1': 0.880, 'support': 2500},
                'Class_1': {'precision': 0.81, 'recall': 0.84, 'f1': 0.825, 'support': 1200},
                'Class_2': {'precision': 0.76, 'recall': 0.81, 'f1': 0.785, 'support': 800},
            },
            'confusion_matrix': [
                [2175, 225, 100],
                [190, 1008, 2],
                [55, 105, 640],
            ]
        }
    }
    
    # Create comparator and generate visualizations
    comparator = ModelComparison(output_dir="reports/figures")
    results = comparator.generate_all_comparisons(mock_metrics)
    
    print("\n✓ Generated visualizations:")
    for name, path in results.items():
        print(f"  • {name}: {path}")


def example_2_load_from_files():
    """Example 2: Load metrics from saved files and visualize."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Load Metrics from Files (Real Data)")
    print("=" * 80)
    
    print("""
This example shows how to load metrics from JSON files (saved by evaluators)
and create comparison visualizations.
    """)
    
    # Load metrics from saved files
    metrics = load_metrics_from_files("reports/metrics")
    
    if not metrics:
        print("✗ No metrics files found!")
        print("  Expected files:")
        print("    - reports/metrics/triage_metrics.json (TabNet)")
        print("    - reports/metrics/xgboost_triage_metrics.json")
        print("    - reports/metrics/lightgbm_triage_metrics.json")
        return
    
    print(f"\n✓ Loaded metrics for {len(metrics)} model(s):")
    for model_name in metrics.keys():
        macro_f1 = metrics[model_name].get('macro_f1', 'N/A')
        print(f"  • {model_name}: Macro-F1 = {macro_f1}")
    
    # Generate all comparisons
    comparator = ModelComparison(output_dir="reports/figures")
    results = comparator.generate_all_comparisons(metrics)
    
    print("\n✓ Generated comparison visualizations:")
    for name, path in results.items():
        print(f"  • {name.replace('_', ' ').title()}: {path}")


def example_3_individual_visualizations():
    """Example 3: Generate individual visualizations with custom settings."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Individual Visualizations (Advanced)")
    print("=" * 80)
    
    print("""
This example shows how to generate individual visualizations with
custom parameters.
    """)
    
    # Create mock metrics
    mock_metrics = {
        'TabNet': {
            'macro_f1': 0.847,
            'overall_accuracy': 0.852,
            'per_class_metrics': {
                'Critical': {'precision': 0.91, 'recall': 0.88, 'f1': 0.895, 'support': 2500},
                'High': {'precision': 0.82, 'recall': 0.85, 'f1': 0.835, 'support': 1200},
                'Medium': {'precision': 0.78, 'recall': 0.82, 'f1': 0.800, 'support': 800},
            },
            'confusion_matrix': [
                [2200, 200, 100],
                [180, 1020, 0],
                [50, 100, 650],
            ]
        },
        'XGBoost': {
            'macro_f1': 0.813,
            'overall_accuracy': 0.820,
            'per_class_metrics': {
                'Critical': {'precision': 0.88, 'recall': 0.86, 'f1': 0.870, 'support': 2500},
                'High': {'precision': 0.80, 'recall': 0.82, 'f1': 0.810, 'support': 1200},
                'Medium': {'precision': 0.75, 'recall': 0.80, 'f1': 0.775, 'support': 800},
            },
            'confusion_matrix': [
                [2150, 250, 100],
                [200, 984, 16],
                [60, 100, 640],
            ]
        },
        'LightGBM': {
            'macro_f1': 0.825,
            'overall_accuracy': 0.832,
            'per_class_metrics': {
                'Critical': {'precision': 0.89, 'recall': 0.87, 'f1': 0.880, 'support': 2500},
                'High': {'precision': 0.81, 'recall': 0.84, 'f1': 0.825, 'support': 1200},
                'Medium': {'precision': 0.76, 'recall': 0.81, 'f1': 0.785, 'support': 800},
            },
            'confusion_matrix': [
                [2175, 225, 100],
                [190, 1008, 2],
                [55, 105, 640],
            ]
        }
    }
    
    comparator = ModelComparison()
    
    # 1. Metrics comparison with custom size
    print("\n1. Metrics Comparison (Large)...")
    comparator.plot_metrics_comparison(
        mock_metrics,
        title="SOC Intelligence - Model Performance Metrics",
        figsize=(14, 7)
    )
    
    # 2. Per-class metrics
    print("2. Per-Class Metrics...")
    comparator.plot_per_class_metrics(
        mock_metrics,
        figsize=(16, 6)
    )
    
    # 3. Confusion matrices
    print("3. Confusion Matrices...")
    comparator.plot_confusion_matrices(
        mock_metrics,
        figsize=(16, 4)
    )
    
    # 4. TabNet architecture with custom parameters
    print("4. TabNet Neural Architecture Diagram...")
    comparator.plot_tabnet_neural_architecture(
        n_features=14,      # Number of input features
        n_classes=3,        # Number of classes (Critical, High, Medium)
        n_decision_steps=8, # Number of decision steps
        figsize=(14, 10)
    )
    
    # 5. ROC curves
    print("5. ROC Curves...")
    comparator.plot_roc_curves(mock_metrics)
    
    print("\n✓ All individual visualizations generated!")


def example_4_custom_models():
    """Example 4: Using visualizations with custom model data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Integration with Training Pipeline")
    print("=" * 80)
    
    print("""
This example shows how to integrate the visualization module into your
model training pipeline to automatically generate comparison plots.
    """)
    
    code_example = """
# In your training script (e.g., src/training/train.py)

from src.evaluation.metrics import evaluate_tabnet_triage
from src.visualization import ModelComparison

# Train model and get predictions
model, predictions = train_tabnet_model(X_train, X_val, X_test, y_train, y_val, y_test)

# Evaluate
metrics, formatted = evaluate_tabnet_triage(y_test, predictions['y_pred'])

print(formatted)

# Generate visualizations
if all_models_trained:
    # Load metrics from all three models
    all_metrics = {
        'TabNet': metrics,
        'XGBoost': xgb_metrics,
        'LightGBM': lgbm_metrics,
    }
    
    # Generate comparison plots
    comparator = ModelComparison()
    visualizations = comparator.generate_all_comparisons(all_metrics)
    
    # Browser or report generation
    print("View visualizations in: reports/figures/")
    """
    
    print(code_example)
    
    print("\n✓ Integration example shown above!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON VISUALIZATION - INTEGRATION GUIDE")
    print("=" * 80)
    
    print("""
This module provides comprehensive model comparison visualizations for:
  • TabNet (Deep Learning) - with neural architecture diagram
  • XGBoost (Gradient Boosting)
  • LightGBM (Gradient Boosting)

Features:
  ✓ Metrics comparison (F1, Accuracy)
  ✓ Per-class metrics breakdown (Precision, Recall, F1)
  ✓ Confusion matrices for all models
  ✓ TabNet neural network architecture diagram
  ✓ ROC curves
  ✓ Easy integration with evaluation pipeline
  ✓ Automatic matplotlib visualization generation
    """)
    
    # Run examples
    example_1_basic_visualization()
    example_2_load_from_files()
    example_3_individual_visualizations()
    example_4_custom_models()
    
    # Summary
    print("\n" + "=" * 80)
    print("USAGE SUMMARY")
    print("=" * 80)
    
    print("""
Quick Start:

1. Basic usage (mock data):
    from src.visualization import ModelComparison
    
    comparator = ModelComparison()
    comparator.generate_all_comparisons(metrics_dict)

2. Load from files:
    from src.visualization import load_metrics_from_files
    
    metrics = load_metrics_from_files()
    # Now use with ModelComparison

3. Individual plots:
    # TabNet architecture diagram
    comparator.plot_tabnet_neural_architecture(
        n_features=14,
        n_classes=3,
        n_decision_steps=8
    )
    
    # Model comparison
    comparator.plot_metrics_comparison(metrics)
    
    # Confusion matrices
    comparator.plot_confusion_matrices(metrics)

Output Directory:
  reports/figures/ (created automatically)
    • 01_metrics_comparison.png
    • 02_per_class_metrics.png
    • 03_confusion_matrices.png
    • 04_tabnet_architecture.png (neural network diagram)
    • 05_roc_curves.png

Integration:
  Add to your training pipeline after model evaluation:
  
    comparator = ModelComparison()
    results = comparator.generate_all_comparisons(all_metrics)

For more details, see the docstrings in src/visualization/model_comparisons.py
    """)
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

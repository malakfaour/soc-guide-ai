# Model Comparison Visualization Module

## Overview

The visualization module provides comprehensive matplotlib-based visualizations for comparing three models:
- **TabNet** (Deep Learning)
- **XGBoost** (Gradient Boosting)
- **LightGBM** (Gradient Boosting)

## Key Features

### 1. **Metrics Comparison** 📊
Compares macro-F1 scores and accuracy across all three models side-by-side with:
- Color-coded bars for each model
- Value labels on bars
- Grid lines for readability

### 2. **Per-Class Metrics** 📈
Breaks down precision, recall, and F1 scores by class for all models:
- Row-level visualization of each metric type
- Grouped bars by model
- Normalized 0-1 scale

### 3. **Confusion Matrices** 🎯
Displays confusion matrices for all models with:
- Normalized values (0-1 scale)
- Absolute counts
- Color heat maps
- Side-by-side comparison

### 4. **TabNet Neural Architecture Diagram** 🧠
**This is the showpiece!** A detailed neural network architecture visualization specifically for TabNet:
- Input layer (14 features)
- Sequential decision steps (default 8)
- Feature selection and masking
- Aggregation layer
- Output layer (3 classes)
- Color-coded components
- Informative labels and arrows

**Why TabNet?** It's an attention-based model that learns which features are important at each step, making it interpretable. The diagram shows:
- How features flow through decision steps
- Feature reuse across steps
- Ensemble aggregation
- Sequential attention mechanism

### 5. **ROC Curves** 📉
Comparative ROC curves for all models with:
- True Positive vs False Positive rates
- Random classifier baseline
- Model-specific styling

## Installation

Ensure matplotlib is installed:
```bash
pip install matplotlib
```

All other dependencies (numpy, sklearn) are already in requirements.txt.

## Quick Start

### Basic Usage with Mock Data

```python
from src.visualization import ModelComparison

# Create mock metrics
mock_metrics = {
    'TabNet': {'macro_f1': 0.847, 'overall_accuracy': 0.852, ...},
    'XGBoost': {'macro_f1': 0.813, 'overall_accuracy': 0.820, ...},
    'LightGBM': {'macro_f1': 0.825, 'overall_accuracy': 0.832, ...},
}

# Generate all visualizations
comparator = ModelComparison(output_dir="reports/figures")
results = comparator.generate_all_comparisons(mock_metrics)

# All outputs saved to reports/figures/
```

### Load from Saved Metrics Files

```python
from src.visualization import ModelComparison, load_metrics_from_files

# Load metrics from JSON files
metrics = load_metrics_from_files("reports/metrics")

# Generate comparisons
comparator = ModelComparison()
results = comparator.generate_all_comparisons(metrics)

# View results in reports/figures/
```

### Generate Specific Visualizations

```python
comparator = ModelComparison()

# Only metrics comparison
comparator.plot_metrics_comparison(metrics, figsize=(14, 7))

# Only TabNet architecture
comparator.plot_tabnet_neural_architecture(
    n_features=14,
    n_classes=3,
    n_decision_steps=8
)

# Only confusion matrices
comparator.plot_confusion_matrices(metrics)

# Only per-class metrics
comparator.plot_per_class_metrics(metrics)

# Only ROC curves
comparator.plot_roc_curves(metrics)
```

## Integration with Training Pipeline

Add to your training script after all models have been trained:

```python
from src.evaluation.metrics import evaluate_tabnet_triage, evaluate_tabnet_remediation
from src.visualization import ModelComparison
import json

# Train three models and collect metrics
tabnet_metrics, _ = evaluate_tabnet_triage(y_test, tabnet_predictions)
xgb_metrics = evaluate_xgboost(y_test, xgb_predictions)
lgbm_metrics = evaluate_lightgbm(y_test, lgbm_predictions)

# Combine into single dictionary
all_metrics = {
    'TabNet': tabnet_metrics,
    'XGBoost': xgb_metrics,
    'LightGBM': lgbm_metrics,
}

# Generate comparison plots
comparator = ModelComparison(output_dir="reports/figures")
visualizations = comparator.generate_all_comparisons(all_metrics)

print("✓ Comparison visualizations saved to reports/figures/")
```

## Output Files

All visualizations are saved to `reports/figures/`:

```
reports/figures/
├── 01_metrics_comparison.png       # F1 and accuracy comparison
├── 02_per_class_metrics.png        # Precision/Recall/F1 by class
├── 03_confusion_matrices.png       # Confusion matrices for all models
├── 04_tabnet_architecture.png      # Neural network architecture diagram ⭐
└── 05_roc_curves.png               # ROC curves comparison
```

## Metrics Dictionary Format

Expected format for `metrics_dict` parameter:

```python
metrics_dict = {
    'TabNet': {
        'macro_f1': float,                    # 0.0-1.0
        'overall_accuracy': float,            # 0.0-1.0
        'per_class_metrics': {
            'Class_0': {
                'precision': float,
                'recall': float,
                'f1': float,
                'support': int
            },
            'Class_1': {...},
            'Class_2': {...}
        },
        'confusion_matrix': [
            [int, int, int],
            [int, int, int],
            [int, int, int]
        ]
    },
    'XGBoost': {...},
    'LightGBM': {...}
}
```

## Customization

### Adjust Figure Size

```python
comparator.plot_metrics_comparison(metrics, figsize=(16, 8))
```

### Change TabNet Architecture Parameters

```python
# For different numbers of features or classes
comparator.plot_tabnet_neural_architecture(
    n_features=20,        # Your dataset's feature count
    n_classes=5,          # Your number of classes
    n_decision_steps=10   # TabNet decision steps
)
```

### Adjust Colors

Edit `self.colors` dictionary in `ModelComparison.__init__()`:

```python
self.colors = {
    'tabnet': '#FF6B6B',      # Red
    'xgboost': '#4ECDC4',     # Teal
    'lightgbm': '#95E1D3',    # Mint
}
```

## Example: Running Visualization Guide

```bash
# From project root
python visualization_integration_guide.py
```

This will:
1. Generate example visualizations with mock data
2. Show how to load metrics from files
3. Demonstrate individual plot generation
4. Show integration with training pipeline

## Advanced Usage

### Create Custom Comparison Report

```python
from src.visualization import ModelComparison, load_metrics_from_files
import datetime

metrics = load_metrics_from_files()
comparator = ModelComparison(output_dir=f"reports/figures/{datetime.date.today()}")
results = comparator.generate_all_comparisons(metrics)

# Create index HTML (optional)
html_index = f"""
<html>
<body>
<h1>Model Comparison Report</h1>
<h2>Generated: {datetime.datetime.now()}</h2>
<h3>TabNet Neural Architecture</h3>
<img src="04_tabnet_architecture.png" width="100%">
<h3>Performance Metrics</h3>
<img src="01_metrics_comparison.png" width="70%">
<h3>Confusion Matrices</h3>
<img src="03_confusion_matrices.png" width="100%">
</body>
</html>
"""

with open(comparator.output_dir / "index.html", "w") as f:
    f.write(html_index)
```

## Extending the Module

### Add Custom Visualizations

```python
class CustomComparison(ModelComparison):
    def plot_custom_metric(self, metrics_dict):
        """Add your own visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        # Your custom plotting code
        output_path = self.output_dir / "06_custom_metric.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
```

## Dependencies

- **matplotlib**: Visualization and plotting
- **numpy**: Numerical arrays and confusion matrix operations
- **scikit-learn**: Metrics (already included)

All are in `requirements.txt`.

## Files

- `src/visualization/model_comparisons.py` - Main module (320+ lines)
- `src/visualization/__init__.py` - Package initialization
- `visualization_integration_guide.py` - Usage examples and guide

## Notes

1. **Neural Network Architecture**: The TabNet diagram is fully customizable for different numbers of features, classes, and decision steps.

2. **Matplotlib Backend**: Uses default backend. For headless systems, add before import:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   ```

3. **High DPI**: All outputs are 300 DPI for publication-quality images.

4. **Color Accessibility**: Colors chosen for color-blind accessibility where possible.

5. **Memory Efficient**: Uses `plt.close()` after each figure to prevent memory leaks.

## Troubleshooting

### Matplotlib backend issues
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

### Missing metrics files
Make sure to run model training and evaluation first:
```bash
python src/models/tabnet/train.py
python src/models/xgboost/train.py
python src/models/lightgbm/train.py
```

### Output directory permissions
Ensure `reports/figures/` directory is writable.

## Future Enhancements

- [ ] Interactive Plotly visualizations
- [ ] Precision-Recall curves
- [ ] Learning curves
- [ ] Feature importance comparison
- [ ] Training time comparison
- [ ] Model size comparison
- [ ] Inference speed comparison
- [ ] HTML report generation

---

For questions or issues, refer to the docstrings in `src/visualization/model_comparisons.py` or run the integration guide for examples.

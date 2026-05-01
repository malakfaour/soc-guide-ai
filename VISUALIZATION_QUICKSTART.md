# Model Comparison Visualization - Quick Start Guide

## 📊 What You Just Created

You now have a complete visualization suite for comparing three models with matplotlib:

### **Files Created:**

1. **`src/visualization/model_comparisons.py`** (320+ lines)
   - Main visualization module with `ModelComparison` class
   - Methods for all visualization types
   - Fully customizable and extensible

2. **`notebooks/MODEL_COMPARISON_VISUALIZATIONS.ipynb`** ⭐
   - Complete Jupyter notebook showing end-to-end workflow
   - Trains 3 models: Random Forest, XGBoost, Deep Learning
   - Generates 5 matplotlib visualizations
   - NOW READY TO RUN!

3. **`visualization_integration_guide.py`**
   - 4 comprehensive examples
   - Shows how to integrate with your pipeline
   - Instructions for custom visualizations

4. **`docs/VISUALIZATION_GUIDE.md`**
   - Complete documentation
   - API reference
   - Troubleshooting section

## 🚀 Quick Start

### Run the Jupyter Notebook:
```bash
# Navigate to notebooks directory
cd notebooks

# Open and run the notebook
jupyter notebook MODEL_COMPARISON_VISUALIZATIONS.ipynb
```

### Or use the Python module directly:
```python
from src.visualization import ModelComparison, load_metrics_from_files

# Load your model metrics
metrics = load_metrics_from_files("reports/metrics")

# Generate all visualizations
comparator = ModelComparison()
results = comparator.generate_all_comparisons(metrics)

# Outputs saved to: reports/figures/
```

## 📈 Visualizations Generated

| # | Visualization | Purpose | File |
|---|---|---|---|
| 1 | **Metrics Comparison** | Bar charts for Accuracy, Precision, Recall, F1 | `01_metrics_comparison.png` |
| 2 | **Confusion Matrices** | Side-by-side confusion matrices for all models | `02_confusion_matrices.png` |
| 3 | **🧠 Neural Network Architecture** | Detailed DL model architecture diagram | `03_neural_network_architecture.png` |
| 4 | **Training History** | Loss & accuracy curves over epochs | `04_training_history.png` |
| 5 | **Comparison Dashboard** | Comprehensive subplot comparison + radar chart | `05_comprehensive_comparison.png` |

## 🧠 The Neural Network Graph (Special Feature)

The neural network architecture diagram shows:

```
INPUT (14 features)
    ↓
DENSE 1: 128 neurons [ReLU + BatchNorm + Dropout]
    ↓
DENSE 2: 64 neurons [ReLU + BatchNorm + Dropout]
    ↓
DENSE 3: 32 neurons [ReLU + BatchNorm]
    ↓
DENSE 4: 16 neurons [ReLU]
    ↓
OUTPUT: 3 classes [Softmax]
```

**Color-coded layers with:**
- ✓ Node connections
- ✓ Layer statistics
- ✓ Activation functions
- ✓ Parameter counts
- ✓ Regularization methods

## 📝 Example: Use in Training Pipeline

Add to any training script:

```python
from src.visualization import ModelComparison
from src.evaluation.metrics import evaluate_tabnet_triage

# After training all models...

# Collect metrics
all_metrics = {
    'TabNet': evaluate_tabnet_triage(y_test, tabnet_pred),
    'XGBoost': evaluate_xgboost(y_test, xgb_pred),
    'LightGBM': evaluate_lightgbm(y_test, lgbm_pred),
}

# Generate visualizations
comparator = ModelComparison(output_dir="reports/figures")
results = comparator.generate_all_comparisons(all_metrics)

print("✓ Visualizations saved!")
```

## 🎨 Customization

### Change Colors:
```python
comparator.colors = {
    'tabnet': '#YOUR_COLOR_1',
    'xgboost': '#YOUR_COLOR_2',
    'lightgbm': '#YOUR_COLOR_3',
}
```

### Adjust Figure Sizes:
```python
comparator.plot_metrics_comparison(metrics, figsize=(16, 8))
```

### Customize Neural Network Diagram:
```python
comparator.plot_tabnet_neural_architecture(
    n_features=20,        # Your dataset features
    n_classes=5,          # Your number of classes
    n_decision_steps=10   # TabNet decision steps
)
```

## 📊 Data Format Required

Your metrics dictionary should look like:

```python
{
    'ModelName': {
        'macro_f1': 0.85,
        'overall_accuracy': 0.82,
        'per_class_metrics': {
            'Class_0': {
                'precision': 0.90,
                'recall': 0.88,
                'f1': 0.89,
                'support': 2500
            },
            # ... more classes
        },
        'confusion_matrix': [
            [2200, 200, 100],
            [180, 1020, 0],
            [50, 100, 650],
        ]
    }
}
```

## 🔍 Integration Points

### With `src/evaluation/metrics.py`:
```python
from src.evaluation.metrics import evaluate_tabnet_triage
metrics, _ = evaluate_tabnet_triage(y_test, y_pred)
```

### With `src/models/`:
- Automatically works with TabNet predictions
- Compatible with XGBoost output
- Works with any classifier predictions

### Output Location:
All visualizations → `reports/figures/` (auto-created)

## 📚 Files You Can Study

1. **`src/visualization/model_comparisons.py`**
   - Main implementation (fully documented)
   - Customize by editing this file

2. **`notebooks/MODEL_COMPARISON_VISUALIZATIONS.ipynb`**
   - Step-by-step examples
   - Run individual cells to explore

3. **`visualization_integration_guide.py`**
   - 4 different usage patterns
   - Copy-paste ready code

## 🐛 Troubleshooting

### Matplotlib backend issues:
```python
import matplotlib
matplotlib.use('Agg')
```

### Missing reports/figures directory:
Created automatically by `ModelComparison()`.

### Can't import modules:
```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## 🎯 Next Steps

1. ✅ **Run the notebook** - Execute `MODEL_COMPARISON_VISUALIZATIONS.ipynb`
2. ✅ **View outputs** - Check `reports/figures/` for PNG files
3. ✅ **Customize colors** - Edit visualization colors to match your brand
4. ✅ **Integrate** - Add to your training pipeline
5. ✅ **Extend** - Create custom visualization methods

## 💡 Pro Tips

- 📌 **High DPI**: All outputs are 300 DPI for publication quality
- 🎨 **Color Blind**: Colors chosen for accessibility
- 💾 **Memory Efficient**: Uses `plt.close()` to prevent leaks  
- 🔄 **Reproducible**: Set random seeds for consistent results
- 📋 **Extensible**: Inherit from `ModelComparison` to add custom plots

## 📞 Support

- Refer to docstrings in `src/visualization/model_comparisons.py`
- Check `docs/VISUALIZATION_GUIDE.md` for detailed API
- Review examples in `visualization_integration_guide.py`

---

**Ready to visualize?** Run the notebook now:
```bash
python notebooks/MODEL_COMPARISON_VISUALIZATIONS.ipynb
```

Or use in Python:
```python
from src.visualization import ModelComparison
comparator = ModelComparison()
# Generate visualizations...
```

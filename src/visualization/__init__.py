"""
Visualization module for SOC Intelligence project.

Provides comprehensive model comparison visualizations including:
- Performance metrics comparison
- Per-class metrics breakdown
- Confusion matrices
- Neural network architecture diagrams
- ROC curves
"""

from .model_comparisons import ModelComparison, load_metrics_from_files

__all__ = [
    'ModelComparison',
    'load_metrics_from_files',
]

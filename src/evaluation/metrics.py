"""
Evaluation metrics for TabNet triage and remediation models.

Metrics for:
- Triage: macro-F1, per-class precision/recall/F1, confusion matrix
- Remediation: Hamming loss, per-label F1
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    hamming_loss,
)
import json
from pathlib import Path


class TriageEvaluator:
    """
    Evaluation metrics for multi-class triage classification.
    """
    
    def __init__(self, n_classes: int = 3):
        """
        Initialize triage evaluator.
        
        Parameters
        ----------
        n_classes : int
            Number of triage classes (default: 3)
        """
        self.n_classes = n_classes
        self.class_labels = [f"Class_{i}" for i in range(n_classes)]
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive triage evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels (n_samples,)
        y_pred : np.ndarray
            Predicted labels (n_samples,)
        y_proba : np.ndarray, optional
            Predicted probabilities (n_samples, n_classes)
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - macro_f1: macro-averaged F1 score
            - per_class_metrics: precision, recall, F1 for each class
            - confusion_matrix: confusion matrix
            - support: number of samples per class
        """
        # Macro-F1 score
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.n_classes))
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.n_classes)))
        
        # Build per-class metrics dictionary
        per_class_metrics = {}
        for i in range(self.n_classes):
            per_class_metrics[self.class_labels[i]] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        
        # Results dictionary
        results = {
            "macro_f1": float(macro_f1),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "overall_accuracy": float(np.mean(y_pred == y_true)),
        }
        
        return results
    
    def format_results(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics for display.
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary from compute_metrics()
        
        Returns
        -------
        str
            Formatted metrics string
        """
        lines = [
            "\n" + "=" * 70,
            "TRIAGE EVALUATION METRICS",
            "=" * 70,
            f"\nOverall Macro-F1: {metrics['macro_f1']:.4f}",
            f"Overall Accuracy: {metrics['overall_accuracy']:.4f}",
            "\nPer-Class Metrics:",
            "-" * 70,
        ]
        
        # Header
        lines.append(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
        lines.append("-" * 70)
        
        # Per-class data
        for class_name, metrics_dict in metrics["per_class_metrics"].items():
            lines.append(
                f"{class_name:<12} "
                f"{metrics_dict['precision']:<12.4f} "
                f"{metrics_dict['recall']:<12.4f} "
                f"{metrics_dict['f1']:<12.4f} "
                f"{metrics_dict['support']:<12d}"
            )
        
        # Confusion matrix
        lines.append("\nConfusion Matrix:")
        lines.append("-" * 70)
        cm = np.array(metrics["confusion_matrix"])
        for i, row in enumerate(cm):
            lines.append(f"{self.class_labels[i]}: {row.tolist()}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


class RemediationEvaluator:
    """
    Evaluation metrics for multi-label remediation classification.
    """
    
    def __init__(self, n_remediations: int):
        """
        Initialize remediation evaluator.
        
        Parameters
        ----------
        n_remediations : int
            Number of remediation actions
        """
        self.n_remediations = n_remediations
        self.label_names = [f"Remediation_{i}" for i in range(n_remediations)]
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive remediation evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels (n_samples, n_remediations) - binary array
        y_pred : np.ndarray
            Predicted labels (n_samples, n_remediations) - binary array
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - hamming_loss: Hamming loss (fraction of wrong labels)
            - per_label_f1: F1 score for each label
            - micro_f1: Micro-averaged F1
            - macro_f1: Macro-averaged F1
            - label_support: Number of positive samples per label
        """
        # Hamming loss
        h_loss = hamming_loss(y_true, y_pred)
        
        # Per-label F1 scores
        per_label_f1 = {}
        per_label_precision = {}
        per_label_recall = {}
        label_support = {}
        
        for i in range(self.n_remediations):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, i],
                y_pred[:, i],
                average='binary',
                zero_division=0
            )
            per_label_precision[self.label_names[i]] = float(precision)
            per_label_recall[self.label_names[i]] = float(recall)
            per_label_f1[self.label_names[i]] = float(f1)
            label_support[self.label_names[i]] = int(np.sum(y_true[:, i]))
        
        # Micro and macro F1 (across all labels)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        results = {
            "hamming_loss": float(h_loss),
            "subset_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
            "per_label_precision": per_label_precision,
            "per_label_recall": per_label_recall,
            "per_label_f1": per_label_f1,
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "label_support": label_support,
        }
        
        return results
    
    def format_results(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics for display.
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary from compute_metrics()
        
        Returns
        -------
        str
            Formatted metrics string
        """
        lines = [
            "\n" + "=" * 70,
            "REMEDIATION EVALUATION METRICS",
            "=" * 70,
            f"\nHamming Loss: {metrics['hamming_loss']:.4f}",
            f"Subset Accuracy: {metrics['subset_accuracy']:.4f}",
            f"Micro-Averaged F1: {metrics['micro_f1']:.4f}",
            f"Macro-Averaged F1: {metrics['macro_f1']:.4f}",
            "\nPer-Label F1:",
            "-" * 70,
        ]
        
        # Header
        lines.append(f"{'Label':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Positive Samples':<20}")
        lines.append("-" * 70)
        
        # Per-label data
        for label_name in self.label_names:
            precision = metrics["per_label_precision"][label_name]
            recall = metrics["per_label_recall"][label_name]
            f1 = metrics["per_label_f1"][label_name]
            support = metrics["label_support"][label_name]
            lines.append(
                f"{label_name:<25} "
                f"{precision:<12.4f} "
                f"{recall:<12.4f} "
                f"{f1:<12.4f} "
                f"{support:<20d}"
            )
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def save_triage_metrics(
    metrics: Dict[str, Any],
    output_dir: str = "reports/metrics",
    filename: str = "triage_metrics.json",
) -> Path:
    """
    Save triage metrics to JSON file.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics dictionary from TriageEvaluator.compute_metrics()
    output_dir : str
        Output directory path
    filename : str
        Output filename
    
    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return file_path


def save_remediation_metrics(
    metrics: Dict[str, Any],
    output_dir: str = "reports/metrics",
    filename: str = "remediation_metrics.json",
) -> Path:
    """
    Save remediation metrics to JSON file.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics dictionary from RemediationEvaluator.compute_metrics()
    output_dir : str
        Output directory path
    filename : str
        Output filename
    
    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return file_path


def evaluate_tabnet_triage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    output_dir: str = "reports/metrics",
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluate triage predictions and save metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities
    output_dir : str
        Output directory for metrics
    
    Returns
    -------
    Tuple[Dict[str, Any], str]
        (metrics dictionary, formatted string)
    """
    evaluator = TriageEvaluator(n_classes=len(np.unique(y_true)))
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    formatted = evaluator.format_results(metrics)
    
    # Save to file
    save_triage_metrics(metrics, output_dir)
    
    return metrics, formatted


def evaluate_tabnet_remediation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = "reports/metrics",
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluate remediation predictions and save metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (n_samples, n_remediations)
    y_pred : np.ndarray
        Predicted labels (n_samples, n_remediations)
    output_dir : str
        Output directory for metrics
    
    Returns
    -------
    Tuple[Dict[str, Any], str]
        (metrics dictionary, formatted string)
    """
    n_remediations = y_true.shape[1]
    evaluator = RemediationEvaluator(n_remediations)
    metrics = evaluator.compute_metrics(y_true, y_pred)
    formatted = evaluator.format_results(metrics)
    
    # Save to file
    save_remediation_metrics(metrics, output_dir)
    
    return metrics, formatted

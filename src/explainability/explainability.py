"""
TabNet explainability module.

Provides feature importance analysis and visualization for TabNet models
using model.explain(X) output and attention masks across decision steps.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
import json


class TabNetExplainer:
    """
    Extract and analyze feature importance from TabNet models.
    
    TabNet produces attention masks for each decision step, showing which
    features are being used at each step of the decision process.
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize TabNet explainer.
        
        Parameters
        ----------
        model : TabNetClassifier or TabNetRegressor
            Trained TabNet model with explain() method
        feature_names : List[str], optional
            List of feature names. If None, uses feature_X format.
        """
        self.model = model
        self.n_features = model.input_dim
        
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        else:
            self.feature_names = feature_names
    
    def get_feature_masks(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature masks and predictions from model.explain().
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            feature_masks: Attention masks (n_samples, n_steps, n_features)
            predictions: Model predictions
        """
        explain_output, aux_output = self.model.explain(X)

        # Current pytorch-tabnet returns:
        #   explain_output -> aggregated explanation (n_samples, n_features)
        #   aux_output -> dict of decision-step masks
        if isinstance(aux_output, dict):
            ordered_masks = [aux_output[key] for key in sorted(aux_output.keys())]
            feature_masks = np.stack(ordered_masks, axis=1)
            predictions = np.asarray(explain_output)
            return feature_masks, predictions

        # Fallback for older assumptions.
        feature_masks = np.asarray(explain_output)
        predictions = np.asarray(aux_output)
        return feature_masks, predictions
    
    def aggregate_feature_importance(
        self,
        feature_masks: np.ndarray,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Aggregate feature masks across samples and steps.
        
        Parameters
        ----------
        feature_masks : np.ndarray
            Attention masks (n_samples, n_steps, n_features)
        aggregation : str
            Aggregation method: 'mean', 'max', 'sum'
        
        Returns
        -------
        np.ndarray
            Aggregated importance (n_features,)
        """
        if aggregation == 'mean':
            # Average across samples and steps
            importance = np.mean(feature_masks, axis=(0, 1))
        elif aggregation == 'max':
            # Max across samples and steps
            importance = np.max(feature_masks, axis=(0, 1))
        elif aggregation == 'sum':
            # Sum across samples and steps
            importance = np.sum(feature_masks, axis=(0, 1))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return importance
    
    def get_step_importance(
        self,
        feature_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Get feature importance for each decision step.
        
        Parameters
        ----------
        feature_masks : np.ndarray
            Attention masks (n_samples, n_steps, n_features)
        
        Returns
        -------
        np.ndarray
            Step importance (n_steps, n_features)
        """
        # Average across samples for each step
        step_importance = np.mean(feature_masks, axis=0)
        return step_importance
    
    def get_top_features(
        self,
        feature_masks: np.ndarray,
        top_k: int = 10,
        aggregation: str = 'mean'
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most important features.
        
        Parameters
        ----------
        feature_masks : np.ndarray
            Attention masks (n_samples, n_steps, n_features)
        top_k : int
            Number of top features to return
        aggregation : str
            Aggregation method
        
        Returns
        -------
        List[Tuple[str, float]]
            List of (feature_name, importance) sorted by importance
        """
        importance = self.aggregate_feature_importance(feature_masks, aggregation)
        
        # Get top-k indices
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        # Return feature names and importance
        top_features = [
            (self.feature_names[idx], float(importance[idx]))
            for idx in top_indices
        ]
        
        return top_features
    
    def explain_instance(
        self,
        X: np.ndarray,
        instance_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a single instance.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        instance_idx : int
            Index of instance to explain
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - instance_masks: Feature masks for this instance (n_steps, n_features)
            - step_features: Top features used at each step
            - prediction: Model prediction for this instance
        """
        feature_masks, predictions = self.get_feature_masks(X)
        
        # Get masks for this instance
        instance_masks = feature_masks[instance_idx]  # (n_steps, n_features)
        
        # Get top features per step
        step_features = {}
        for step_idx, step_mask in enumerate(instance_masks):
            top_indices = np.argsort(step_mask)[-5:][::-1]
            top_features = [
                (self.feature_names[idx], float(step_mask[idx]))
                for idx in top_indices
            ]
            step_features[f"step_{step_idx}"] = top_features
        
        explanation = {
            "instance_masks": instance_masks.tolist(),
            "step_features": step_features,
            "prediction": float(predictions[instance_idx]),
        }
        
        return explanation


def plot_feature_importance(
    explainer: TabNetExplainer,
    feature_masks: np.ndarray,
    top_k: int = 15,
    output_path: Optional[str] = None,
    title: str = "TabNet Feature Importance"
) -> plt.Figure:
    """
    Plot top-k most important features.
    
    Parameters
    ----------
    explainer : TabNetExplainer
        Explainer instance
    feature_masks : np.ndarray
        Feature masks from explain()
    top_k : int
        Number of top features to plot
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    top_features = explainer.get_top_features(feature_masks, top_k)
    feature_names = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(feature_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    
    ax.barh(y_pos, importances, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel("Average Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, val) in enumerate(zip(y_pos, importances)):
        ax.text(val, idx, f"{val:.4f}", va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [Saved] {output_path}")
    
    return fig


def plot_step_importance(
    explainer: TabNetExplainer,
    feature_masks: np.ndarray,
    top_k: int = 8,
    output_path: Optional[str] = None,
    title: str = "TabNet Feature Importance by Step"
) -> plt.Figure:
    """
    Plot feature importance for each decision step.
    
    Parameters
    ----------
    explainer : TabNetExplainer
        Explainer instance
    feature_masks : np.ndarray
        Feature masks from explain()
    top_k : int
        Number of top features per step
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    step_importance = explainer.get_step_importance(feature_masks)
    n_steps = step_importance.shape[0]
    
    # Create subplots
    fig, axes = plt.subplots(
        (n_steps + 1) // 2, 2,
        figsize=(14, 3 * ((n_steps + 1) // 2))
    )
    
    if n_steps == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    # Plot each step
    for step_idx, step_mask in enumerate(step_importance):
        ax = axes[step_idx]
        
        # Get top-k features for this step
        top_indices = np.argsort(step_mask)[-top_k:][::-1]
        top_feature_names = [explainer.feature_names[idx] for idx in top_indices]
        top_importance = step_mask[top_indices]
        
        # Plot
        colors = plt.cm.plasma(np.linspace(0, 1, len(top_feature_names)))
        ax.barh(range(len(top_feature_names)), top_importance, color=colors)
        ax.set_yticks(range(len(top_feature_names)))
        ax.set_yticklabels(top_feature_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=10)
        ax.set_title(f"Step {step_idx}", fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_steps, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [Saved] {output_path}")
    
    return fig


def plot_mask_heatmap(
    feature_masks: np.ndarray,
    feature_names: Optional[List[str]] = None,
    sample_indices: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    title: str = "TabNet Feature Mask Heatmap"
) -> plt.Figure:
    """
    Plot heatmap of feature masks across steps and samples.
    
    Parameters
    ----------
    feature_masks : np.ndarray
        Feature masks (n_samples, n_steps, n_features)
    feature_names : List[str], optional
        Feature names for y-axis
    sample_indices : List[int], optional
        Indices of samples to plot. If None, uses first 10.
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_samples, n_steps, n_features = feature_masks.shape
    
    # Select samples
    if sample_indices is None:
        sample_indices = list(range(min(10, n_samples)))
    
    # Average masks across steps
    avg_masks = np.mean(feature_masks[sample_indices], axis=1)  # (n_samples_selected, n_features)
    
    # Get feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, len(sample_indices)))
    
    sns.heatmap(
        avg_masks,
        cmap='YlOrRd',
        cbar_kws={'label': 'Average Importance'},
        xticklabels=feature_names,
        yticklabels=[f"Sample_{idx}" for idx in sample_indices],
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [Saved] {output_path}")
    
    return fig


def save_explanation_report(
    explanation: Dict[str, Any],
    output_path: str
) -> Path:
    """
    Save explanation dictionary as JSON.
    
    Parameters
    ----------
    explanation : Dict[str, Any]
        Explanation dictionary from explainer
    output_path : str
        Path to save JSON file
    
    Returns
    -------
    Path
        Path to saved file
    """
    output_file = Path(output_path)
    
    with open(output_file, 'w') as f:
        json.dump(explanation, f, indent=2)
    
    return output_file


def explain_tabnet_model(
    model,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "reports/figures",
    top_k: int = 15,
    include_heatmap: bool = True,
    include_step_plots: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive TabNet model explanations and visualizations.
    
    Parameters
    ----------
    model : TabNetClassifier/Regressor
        Trained TabNet model
    X_test : np.ndarray
        Test features for explanation
    feature_names : List[str], optional
        Feature names
    output_dir : str
        Directory to save plots
    top_k : int
        Number of top features to show
    include_heatmap : bool
        Whether to create heatmap visualization
    include_step_plots : bool
        Whether to create per-step plots
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with plot paths and explanation data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[EXPLAINABILITY] Analyzing TabNet model")
    print(f"  Samples: {X_test.shape[0]}")
    print(f"  Features: {X_test.shape[1]}")
    
    # Initialize explainer
    explainer = TabNetExplainer(model, feature_names)
    
    # Get feature masks
    print("\n[STEP 1] Extracting feature masks...")
    feature_masks, predictions = explainer.get_feature_masks(X_test)
    print(f"  [Extracted] Masks shape: {feature_masks.shape} (samples, steps, features)")
    
    # Generate plots
    print("\n[STEP 2] Generating visualizations...")
    
    plots = {}
    
    # 1. Feature importance
    fig_path = output_path / "feature_importance.png"
    plot_feature_importance(
        explainer, feature_masks, top_k=top_k,
        output_path=str(fig_path),
        title="TabNet Feature Importance (Aggregated)"
    )
    plots['feature_importance'] = str(fig_path)
    plt.close()
    
    # 2. Step importance
    if include_step_plots:
        fig_path = output_path / "step_importance.png"
        plot_step_importance(
            explainer, feature_masks, top_k=8,
            output_path=str(fig_path),
            title="Feature Importance by Decision Step"
        )
        plots['step_importance'] = str(fig_path)
        plt.close()
    
    # 3. Heatmap
    if include_heatmap:
        fig_path = output_path / "feature_mask_heatmap.png"
        plot_mask_heatmap(
            feature_masks,
            feature_names=explainer.feature_names,
            sample_indices=list(range(min(10, X_test.shape[0]))),
            output_path=str(fig_path),
            title="Feature Mask Heatmap (First 10 Samples)"
        )
        plots['mask_heatmap'] = str(fig_path)
        plt.close()
    
    # Get top features
    print("\n[STEP 3] Computing feature importance...")
    top_features = explainer.get_top_features(feature_masks, top_k=top_k)
    print(f"  [Computed] Top features identified")
    for i, (name, importance) in enumerate(top_features[:5], 1):
        print(f"      {i}. {name}: {importance:.4f}")
    
    # Prepare results
    results = {
        'plots': plots,
        'top_features': [(name, float(imp)) for name, imp in top_features],
        'n_samples': X_test.shape[0],
        'n_features': X_test.shape[1],
        'n_steps': feature_masks.shape[1],
    }
    
    print("\n[COMPLETE] Explainability analysis finished")
    
    return results

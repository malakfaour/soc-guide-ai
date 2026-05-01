"""
Generate final report figures from saved metrics and artifacts.

This script reuses existing outputs and does not retrain any models.
It generates:
- confusion matrix heatmaps for TabNet, LightGBM, and XGBoost
- a triage model comparison bar chart
- a hybrid pipeline architecture diagram
- a small note artifact when TabNet training history is unavailable
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["FalsePositive", "BenignPositive", "TruePositive"]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_confusion_matrix_figure(model_name: str, metrics_path: Path, output_path: Path) -> None:
    metrics = load_json(metrics_path)
    confusion_matrix = np.array(metrics["confusion_matrix"])

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        cbar=True,
    )
    plt.title(f"{model_name} Triage Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_model_comparison_figure(output_path: Path) -> None:
    tabnet = load_json(METRICS_DIR / "triage_metrics.json")
    lightgbm = load_json(METRICS_DIR / "lightgbm_triage_metrics.json")
    xgboost = load_json(METRICS_DIR / "xgboost_triage_metrics.json")

    model_names = ["TabNet", "LightGBM", "XGBoost"]
    macro_f1 = [tabnet["macro_f1"], lightgbm["macro_f1"], xgboost["macro_f1"]]
    accuracy = [
        tabnet["overall_accuracy"],
        lightgbm["overall_accuracy"],
        xgboost["overall_accuracy"],
    ]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, macro_f1, width, label="Macro-F1", color="#205072")
    bars2 = ax.bar(x + width / 2, accuracy, width, label="Accuracy", color="#329D9C")

    ax.set_title("Triage Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def draw_box(ax, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    rect = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="#1B1B1B", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11, wrap=True)


def draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2, color="#1B1B1B"),
    )


def save_hybrid_architecture_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.06, 0.64, 0.22, 0.17, "Processed Row-Level Features\nX_test / live alert rows", "#F4D35E")
    draw_box(ax, 0.39, 0.64, 0.22, 0.17, "TabNet Triage Model\n3-class alert triage", "#9AD1D4")
    draw_box(ax, 0.73, 0.64, 0.21, 0.17, "Triage Output\nFP / BP / TP", "#B8E0A5")

    draw_box(ax, 0.06, 0.20, 0.22, 0.17, "Processed Incident Features\nincident remediation dataset", "#F4D35E")
    draw_box(ax, 0.34, 0.30, 0.18, 0.14, "GBT\naccount_response", "#EE964B")
    draw_box(ax, 0.34, 0.12, 0.18, 0.14, "Logistic Regression\nendpoint_response", "#EE964B")
    draw_box(ax, 0.73, 0.20, 0.21, 0.17, "Remediation Output\naccount / endpoint", "#B8E0A5")

    draw_arrow(ax, 0.28, 0.725, 0.39, 0.725)
    draw_arrow(ax, 0.61, 0.725, 0.73, 0.725)
    draw_arrow(ax, 0.28, 0.285, 0.34, 0.37)
    draw_arrow(ax, 0.28, 0.285, 0.34, 0.19)
    draw_arrow(ax, 0.52, 0.37, 0.73, 0.285)
    draw_arrow(ax, 0.52, 0.19, 0.73, 0.285)

    ax.text(0.5, 0.93, "Final Hybrid SOC Pipeline", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.86, "TabNet handles triage, classical incident-level models handle remediation", ha="center", va="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_training_history_note(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    message = (
        "TabNet training-curve plot unavailable\n\n"
        "A real epoch-by-epoch history file was not saved with the current final model artifacts.\n"
        "The project keeps final metrics, confusion matrices, and explainability outputs,\n"
        "but not a reusable training-history JSON for the saved TabNet run."
    )
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="#888888"),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    save_confusion_matrix_figure(
        "TabNet",
        METRICS_DIR / "triage_metrics.json",
        FIGURES_DIR / "tabnet_confusion_matrix.png",
    )
    save_confusion_matrix_figure(
        "LightGBM",
        METRICS_DIR / "lightgbm_triage_metrics.json",
        FIGURES_DIR / "lightgbm_confusion_matrix.png",
    )
    save_confusion_matrix_figure(
        "XGBoost",
        METRICS_DIR / "xgboost_triage_metrics.json",
        FIGURES_DIR / "xgboost_confusion_matrix.png",
    )
    save_model_comparison_figure(FIGURES_DIR / "triage_model_comparison.png")
    save_hybrid_architecture_figure(FIGURES_DIR / "hybrid_pipeline_architecture.png")
    save_training_history_note(FIGURES_DIR / "tabnet_training_curve_note.png")

    summary = {
        "generated_figures": {
            "tabnet_confusion_matrix": str(FIGURES_DIR / "tabnet_confusion_matrix.png"),
            "lightgbm_confusion_matrix": str(FIGURES_DIR / "lightgbm_confusion_matrix.png"),
            "xgboost_confusion_matrix": str(FIGURES_DIR / "xgboost_confusion_matrix.png"),
            "triage_model_comparison": str(FIGURES_DIR / "triage_model_comparison.png"),
            "hybrid_pipeline_architecture": str(FIGURES_DIR / "hybrid_pipeline_architecture.png"),
            "tabnet_training_curve_note": str(FIGURES_DIR / "tabnet_training_curve_note.png"),
        }
    }

    with open(FIGURES_DIR / "final_figures_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

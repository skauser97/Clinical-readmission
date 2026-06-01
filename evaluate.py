"""
evaluate.py
-----------
Model evaluation: AUC-ROC, calibration curve, confusion matrix,
classification report, and Brier score.

All plots are saved to results/ as high-res PNGs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Union

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.calibration import calibration_curve


# ------------------------------------------------------------------
# Plot style
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

RISK_COLOR = "#e63946"   # red — high risk
SAFE_COLOR = "#457b9d"   # blue — low risk


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute full evaluation suite and optionally save plots.

    Parameters
    ----------
    model
        Fitted sklearn-compatible classifier (must have predict_proba).
    X_test : np.ndarray
    y_test : np.ndarray
    threshold : float
        Decision threshold for binary classification.
    save_dir : Path or None
        Directory to save plots. If None, plots are shown interactively.
    verbose : bool
        Print metrics to stdout.

    Returns
    -------
    dict with keys: auc_roc, auc_pr, brier_score, f1, accuracy
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    accuracy = (y_pred == y_test).mean()

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "brier_score": round(float(brier), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "threshold": threshold,
        "n_test": len(y_test),
        "prevalence": round(float(y_test.mean()), 4),
    }

    if verbose:
        print("\n" + "=" * 50)
        print("  EVALUATION RESULTS")
        print("=" * 50)
        print(f"  AUC-ROC        : {auc_roc:.3f}")
        print(f"  AUC-PR         : {auc_pr:.3f}")
        print(f"  Brier Score    : {brier:.3f}  (lower is better)")
        print(f"  F1 (high-risk) : {f1:.3f}")
        print(f"  Accuracy       : {accuracy:.3f}")
        print("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=["Low Risk", "High Risk"]))

    # Save metrics
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Generate and save plots
    _plot_full_evaluation(y_test, y_prob, y_pred, metrics, save_dir)

    return metrics


def _plot_full_evaluation(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    save_dir: Optional[Path],
):
    """Generate 2x2 evaluation figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # ---- 1. ROC Curve ----
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color=RISK_COLOR, lw=2,
             label=f"ClinicalBERT (AUC = {metrics['auc_roc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    # ---- 2. Precision-Recall Curve ----
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    baseline = y_test.mean()
    ax2.plot(recall, precision, color=RISK_COLOR, lw=2,
             label=f"ClinicalBERT (AP = {metrics['auc_pr']:.3f})")
    ax2.axhline(y=baseline, color="k", linestyle="--", lw=1, alpha=0.5,
                label=f"No-skill baseline ({baseline:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    # ---- 3. Calibration Curve ----
    ax3 = fig.add_subplot(gs[1, 0])
    fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax3.plot(mean_pred, fraction_pos, "s-", color=RISK_COLOR, lw=2,
             label="ClinicalBERT")
    ax3.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Fraction of Positives")
    ax3.set_title(f"Calibration Curve  (Brier = {metrics['brier_score']:.3f})")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])

    # ---- 4. Confusion Matrix ----
    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y_test, y_pred)
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    ax4.set_title(f"Confusion Matrix  (threshold={metrics['threshold']})")
    tick_marks = [0, 1]
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(["Low Risk", "High Risk"])
    ax4.set_yticklabels(["Low Risk", "High Risk"])
    ax4.set_ylabel("True Label")
    ax4.set_xlabel("Predicted Label")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight="bold")

    fig.suptitle("Clinical Readmission Risk — Model Evaluation", fontsize=14, y=1.01)

    if save_dir is not None:
        out_path = Path(save_dir) / "evaluation.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[evaluate] Plots saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_risk_distribution(
    y_prob: np.ndarray,
    y_test: np.ndarray,
    save_dir: Optional[Path] = None,
):
    """
    Plot predicted probability distributions for each class.
    Shows how well-separated the model's scores are.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    low_risk_probs = y_prob[y_test == 0]
    high_risk_probs = y_prob[y_test == 1]

    ax.hist(low_risk_probs, bins=30, alpha=0.6, color=SAFE_COLOR,
            label=f"Low Risk (n={len(low_risk_probs)})", density=True)
    ax.hist(high_risk_probs, bins=30, alpha=0.6, color=RISK_COLOR,
            label=f"High Risk (n={len(high_risk_probs)})", density=True)

    ax.axvline(x=0.5, color="black", linestyle="--", lw=1.5,
               label="Default threshold (0.5)")
    ax.set_xlabel("Predicted Readmission Probability")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution by True Label")
    ax.legend()

    if save_dir is not None:
        out_path = Path(save_dir) / "risk_distribution.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[evaluate] Risk distribution saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)


def save_evaluation_plots(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
):
    """Convenience wrapper — saves all evaluation outputs."""
    metrics = evaluate_model(model, X_test, y_test, save_dir=save_dir)
    y_prob = model.predict_proba(X_test)[:, 1]
    plot_risk_distribution(y_prob, y_test, save_dir=save_dir)
    return metrics

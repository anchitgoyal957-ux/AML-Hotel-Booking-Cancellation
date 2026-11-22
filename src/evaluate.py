"""
evaluate.py - Test set evaluation, metrics calculation, PR/ROC curves,
and the cost-based overbooking threshold decision.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

# Cost Model Parameters (in USD / EUR)
COST_WALKED_GUEST_FP = 250.0  # False Positive: predicted cancel, guest arrives -> walked to another hotel
COST_EMPTY_ROOM_FN = 100.0    # False Negative: predicted arrival, guest cancels -> room sits empty (lost ADR)

def evaluate_models(fitted_pipelines, X_test, y_test, results_dir="results"):
    """
    Evaluate fitted models on the held-out test set once.
    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC.
    Saves to results/metrics.csv.
    """
    records = []
    predictions = {}

    for name, pipe in fitted_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4)
        })
        predictions[name] = {"prob": y_prob, "pred": y_pred}

    metrics_df = pd.DataFrame(records).sort_values(by="ROC_AUC", ascending=False)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    return metrics_df, predictions

def plot_curves(predictions, y_test, save_path="results/figures/roc_pr_curves.png"):
    """
    Plot ROC curves and Precision-Recall curves side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    for name, data in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, data["prob"])
        auc = roc_auc_score(y_test, data["prob"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (0.50)")
    axes[0].set_title("ROC Curves (False Positive Rate vs True Positive Rate)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # PR Curves
    baseline_rate = y_test.mean()
    for name, data in predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, data["prob"])
        pr_auc = average_precision_score(y_test, data["prob"])
        axes[1].plot(rec, prec, label=f"{name} (AP = {pr_auc:.3f})")
    axes[1].axhline(baseline_rate, color="k", linestyle="--", alpha=0.6, label=f"Class Prevalence ({baseline_rate:.2f})")
    axes[1].set_title("Precision-Recall Curves (Asymmetric Costs / Imbalance)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Coverage of Cancellations)")
    axes[1].set_ylabel("Precision (Accuracy of Cancellation Calls)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def optimize_overbooking_threshold(
    y_true, y_prob,
    cost_fp: float = COST_WALKED_GUEST_FP,
    cost_fn: float = COST_EMPTY_ROOM_FN,
    thresholds: np.ndarray = None,
    save_path: str = "results/figures/threshold_cost_sweep.png"
):
    """
    Sweep classification threshold and compute total operational expected cost:
    Expected Cost = (FP * cost_fp) + (FN * cost_fn)
    
    Theoretical Bayes-optimal threshold for asymmetric loss:
    p* = cost_fp / (cost_fp + cost_fn)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    costs = []
    fp_counts = []
    fn_counts = []
    tp_counts = []
    tn_counts = []

    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_t, labels=[0, 1]).ravel()
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(total_cost)
        fp_counts.append(fp)
        fn_counts.append(fn)
        tp_counts.append(tp)
        tn_counts.append(tn)

    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    optimal_threshold = thresholds[opt_idx]
    min_cost = costs[opt_idx]

    # Baseline cost at 0.5 threshold
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    cost_05 = costs[idx_05]

    # Theoretical threshold
    bayes_threshold = cost_fp / (cost_fp + cost_fn)

    # Plot Cost vs Threshold
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, costs, color="#1f77b4", lw=2.5, label="Total Operational Loss ($)")
    ax1.axvline(optimal_threshold, color="#d62728", linestyle="--", lw=2,
                label=f"Recommended Threshold ({optimal_threshold:.2f}, Min Cost: ${min_cost:,.0f})")
    ax1.axvline(0.50, color="#7f7f7f", linestyle=":", lw=1.5,
                label=f"Default Threshold (0.50, Cost: ${cost_05:,.0f})")
    ax1.axvline(bayes_threshold, color="#2ca02c", linestyle="-.", lw=1.5,
                label=f"Bayes-optimal Analytic Ratio ({bayes_threshold:.2f})")

    ax1.set_xlabel("Decision Threshold (Predicted Cancellation Probability)", fontsize=11)
    ax1.set_ylabel("Total Operational Cost ($)", fontsize=11, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Overbooking Decision: Cost Curve vs. Cancellation Probability Threshold", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Dual axis for FP (walks) and FN (empty rooms)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, fp_counts, color="#ff7f0e", linestyle="--", alpha=0.7, label="Walked Guests (FP)")
    ax2.plot(thresholds, fn_counts, color="#9467bd", linestyle="--", alpha=0.7, label="Empty Rooms (FN)")
    ax2.set_ylabel("Incident Count (FP / FN)", fontsize=11, color="#333333")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix comparison at 0.5 vs recommended
    cm_05 = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    cm_opt = confusion_matrix(y_true, (y_prob >= optimal_threshold).astype(int))

    return {
        "optimal_threshold": float(optimal_threshold),
        "bayes_threshold": float(bayes_threshold),
        "min_cost": float(min_cost),
        "cost_at_05": float(cost_05),
        "cost_savings": float(cost_05 - min_cost),
        "cm_at_05": cm_05,
        "cm_at_opt": cm_opt,
        "walked_guests_05": int(fp_counts[idx_05]),
        "walked_guests_opt": int(fp_counts[opt_idx]),
        "empty_rooms_05": int(fn_counts[idx_05]),
        "empty_rooms_opt": int(fn_counts[opt_idx])
    }

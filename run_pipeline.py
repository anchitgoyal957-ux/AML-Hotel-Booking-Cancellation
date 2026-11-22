"""
run_pipeline.py - End-to-end execution script for P03 Hotel Booking Cancellations.
Runs data loading, cleaning, leakage audit, stratified split, model training,
test-set evaluation, cost-matrix optimization, and outputs all results and figures.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_raw_data, audit_leakage, clean_data
from src.features import get_train_test_split
from src.models import get_model_pipelines, tune_decision_tree_depth, cross_validate_models
from src.evaluate import evaluate_models, plot_curves, optimize_overbooking_threshold

def main():
    print("=" * 70)
    print(" P03: Hotel-Booking Cancellations & Overbooking Decision Pipeline ")
    print("=" * 70)

    # 1. Load Data
    data_path = Path("data/raw/hotel_bookings.csv")
    print(f"\n[1/6] Ingesting raw data from {data_path}...")
    df_raw = load_raw_data(str(data_path))
    print(f"      Raw data shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    # 2. Leakage Audit & Cleaning
    print("\n[2/6] Auditing leakage features and executing cleaning policy...")
    audit_table = audit_leakage(df_raw)
    print(audit_table.to_string(index=False))

    df_clean = clean_data(df_raw, drop_leakage=True)
    print(f"      Cleaned data shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    print(f"      Overall cancellation rate: {df_clean['is_canceled'].mean():.4%}")

    # Save cleaned data to data/processed
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(proc_dir / "hotel_bookings_cleaned.csv", index=False)
    print(f"      Cleaned dataset saved to {proc_dir / 'hotel_bookings_cleaned.csv'}")

    # 3. Stratified Train/Test Split
    print("\n[3/6] Performing stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = get_train_test_split(df_clean, test_size=0.2, random_state=42)
    print(f"      Training set: {X_train.shape[0]:,} rows (Cancel rate: {y_train.mean():.4%})")
    print(f"      Testing set:  {X_test.shape[0]:,} rows (Cancel rate: {y_test.mean():.4%})")

    # 4. Model Construction & CV
    print("\n[4/6] Building model pipelines & tuning decision tree depth...")
    best_depth, best_dt_score, _ = tune_decision_tree_depth(X_train, y_train, depths=[3, 5, 8, 10, 15], cv_folds=5)
    print(f"      Tuned Decision Tree max_depth: {best_depth} (CV ROC-AUC: {best_dt_score:.4f})")

    pipelines = get_model_pipelines(dt_max_depth=best_depth, random_state=42)

    print("\n      Running 5-fold cross-validation across all 5 candidate models on training set...")
    cv_summary = cross_validate_models(pipelines, X_train, y_train, cv_folds=5, scoring="roc_auc")
    for name, res in cv_summary.items():
        print(f"      -> {name:<22}: CV ROC-AUC = {res['mean_score']:.4f} (+/- {res['std_score']:.4f})")

    # Fit all pipelines on full training set
    print("\n      Fitting all 5 pipelines on full training set...")
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        print(f"      [✓] {name} fitted.")

    # 5. Held-out Test Set Evaluation
    print("\n[5/6] Evaluating on held-out test set (touched once)...")
    metrics_df, predictions = evaluate_models(pipelines, X_test, y_test, results_dir="results")
    print("\n" + metrics_df.to_string(index=False))

    plot_curves(predictions, y_test, save_path="results/figures/roc_pr_curves.png")
    print("      Saved ROC & Precision-Recall curves to results/figures/roc_pr_curves.png")

    # 6. Cost-based Overbooking Threshold Optimization
    print("\n[6/6] Executing overbooking threshold sweep with stated cost model...")
    # Using best ensemble model (HistGradientBoosting or Random Forest)
    best_model_name = metrics_df.iloc[0]["Model"]
    best_probs = predictions[best_model_name]["prob"]

    decision_res = optimize_overbooking_threshold(
        y_test, best_probs,
        cost_fp=250.0,  # Walked guest penalty
        cost_fn=100.0,  # Empty room penalty
        save_path="results/figures/threshold_cost_sweep.png"
    )

    print(f"      Model used: {best_model_name}")
    print(f"      Theoretical Bayes-optimal threshold ratio: {decision_res['bayes_threshold']:.3f}")
    print(f"      Empirically recommended threshold:         {decision_res['optimal_threshold']:.3f}")
    print(f"      Expected cost at default 0.50:             ${decision_res['cost_at_05']:,.2f}")
    print(f"      Expected cost at recommended threshold:    ${decision_res['min_cost']:,.2f}")
    print(f"      Operational savings achieved:              ${decision_res['cost_savings']:,.2f}")
    print(f"      Walked guests (FP): {decision_res['walked_guests_05']} (at 0.5) -> {decision_res['walked_guests_opt']} (at recommended)")
    print(f"      Empty rooms   (FN): {decision_res['empty_rooms_05']} (at 0.5) -> {decision_res['empty_rooms_opt']} (at recommended)")
    print("=" * 70)
    print(" Pipeline completed successfully! All deliverables generated.")
    print("=" * 70)

if __name__ == "__main__":
    main()

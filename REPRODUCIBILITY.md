# Reproducibility Guide: P03 Hotel-Booking Cancellations & Overbooking Decision

This document details the exact environment, dependencies, random seeds, and end-to-end execution sequence required to reproduce all findings, metrics, figures, and overbooking threshold recommendations from a clean checkout.

---

## 1. Environment and Hardware

- **Python Version**: Python 3.11 (tested on Python 3.11.x 64-bit).
- **Virtual Environment Setup**:
  ```bash
  # Create a clean virtual environment
  python -m venv .venv

  # Activate the environment
  # On Windows (PowerShell):
  .venv\Scripts\Activate.ps1
  # On Windows (Command Prompt):
  .venv\Scripts\activate.bat
  # On Linux / macOS:
  source .venv/bin/activate

  # Install pinned requirements
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

---

## 2. Dataset Information and Verification

- **Source**: Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.
- **Kaggle Distribution**: [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Local Path**: `data/raw/hotel_bookings.csv`
- **Expected Dimensions**: 119,390 rows × 32 columns.
- **File Size**: ~16.85 MB (16,855,599 bytes).

---

## 3. Pinned Random Seeds

To guarantee deterministic reproduction of train/test splits, cross-validation folds, and stochastic estimators, the seed is set to `42` everywhere:

| Scope / Component | Function / Class | Seed Parameter | Purpose |
|---|---|---|---|
| Train/Test Split | `sklearn.model_selection.train_test_split` | `random_state=42` | Stratified 80/20 partition preservation |
| Cross-Validation | `sklearn.model_selection.StratifiedKFold` | `random_state=42`, `shuffle=True` | Identical 5 folds across all estimators |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `random_state=42` | Solver convergence determinism |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | `random_state=42` | Split tie-breaking determinism |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `random_state=42` | Bootstrap sampling & feature subsampling |
| HistGradientBoosting | `sklearn.ensemble.HistGradientBoostingClassifier` | `random_state=42` | Binning & tree split selection |

---

## 4. Execution Sequence

All analyses can be reproduced either interactively via Jupyter Notebooks or via the automated standalone Python pipeline.

### Option A: End-to-End Command-Line Pipeline (Fastest)

Run the master script from the repository root:
```bash
python run_pipeline.py
```
This script will sequentially:
1. Ingest raw data from `data/raw/hotel_bookings.csv`.
2. Execute the leakage audit and drop the 5 post-booking leakage columns.
3. Impute missing values and drop zero-guest and extreme ADR outliers.
4. Export the clean dataset to `data/processed/hotel_bookings_cleaned.csv`.
5. Execute the stratified 80/20 train/test split.
6. Tune Decision Tree depth via 5-fold CV.
7. Run 5-fold cross-validation on all 5 candidate model families on `X_train`.
8. Fit final models and evaluate once on `X_test`.
9. Export test metrics to `results/metrics.csv`.
10. Generate ROC/PR diagnostic curves to `results/figures/roc_pr_curves.png`.
11. Perform the cost-matrix threshold sweep and generate `results/figures/threshold_cost_sweep.png`.

### Option B: Interactive Jupyter Notebooks

Execute notebooks in numerical order from the `notebooks/` directory:

1. **`01_eda.ipynb` (or `01-eda.ipynb`)**:
   - Original exploratory data analysis examining class balance overall and per hotel, cancellation rate against lead time, deposit type, market segment, and previous cancellations.
2. **`02_preprocessing_baseline.ipynb`**:
   - Executes Phase B (Checkpoint 1). Implements leakage drop, cleaning decisions, country grouping policy, stratified split, and evaluates the Baseline Logistic Regression pipeline.
3. **`03_models.ipynb`**:
   - Executes Phase C (Checkpoint 2). Implements the 5-model protocol, Decision Tree CV depth tuning, scaling justification for KNN, 5-fold cross-validation, and full-training fits.
4. **`04_evaluation.ipynb`**:
   - Executes Phase D. Evaluates the held-out test set, exports `results/metrics.csv`, plots comparative ROC and Precision-Recall curves, conducts the overbooking cost-matrix threshold optimization, and compares feature importances.

---

## 5. Expected Output Deliverables

After execution, verify the presence of the following artifacts:

- `results/metrics.csv`: Performance metrics for all 5 models (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
- `results/figures/roc_pr_curves.png`: Side-by-side ROC and Precision-Recall comparison plots.
- `results/figures/threshold_cost_sweep.png`: Total expected operational cost curve vs. cancellation probability threshold.
- `results/figures/feature_importance_comparison.png`: Logistic regression coefficients vs. Random Forest Gini importances.
- `report/project_report.md`: Complete technical and managerial report.

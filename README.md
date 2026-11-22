# AML Project — Team Vega Punk
## P03 · Hotel-Booking Cancellations and the Overbooking Decision

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

> **Central Question:** At the moment a reservation is made, which bookings will be cancelled — and at what predicted probability should the hotel overbook the room?

---

## Team Members

| Name | SAP ID | Role / Contribution |
|---|---:|---|
| **Shardul Aggarwal** | 590022064 | Pipeline Architecture & Cost-Optimization Modeling |
| **Anchit Goyal** | 590025670 | Leakage Audit, Data Cleaning & Preprocessing Pipelines |
| **Sourya Pokhariya** | 590023118 | Model Exploration (KNN, Decision Trees) & Scaling Analysis |
| **Pranit Goyal** | 590024690 | Ensemble Modeling (Random Forest, HistGradientBoosting) & Cross-Validation |
| **Krishika Agarwal** | 590029530 | Evaluation Diagnostics (ROC vs PR Curves) & Feature Importance Analysis |

---

## 1. Executive Summary & The Overbooking Decision

In the hotel hospitality sector, roughly **37%** of reservations are cancelled prior to arrival. 
- A hotel that **ignores cancellations** suffers severe revenue loss from empty, perishable room inventory.
- A hotel that **overbooks blindly** faces catastrophic operational fallout: when more guests arrive than physical rooms exist, the hotel must **"walk"** guests to competitor properties. Walking guests carries severe direct financial expenses (emergency rebooking, premium rates, taxis, meal vouchers) and long-term brand degradation.

Between these two extremes lies a principled decision rule based on machine learning:
> **The Decision Rule:** Accept an additional overbooking if and only if the predicted cancellation probability of an existing booking satisfies:
> $$\hat{p} \ge 0.71$$

### Cost-Model Formulation & Recommendation

| Decision Event | Error Type | Real-World Operational Consequence | Stated Unit Cost |
|---|---|---|---:|
| **False Positive ($FP$)** | Type I | Predicted Cancel, Guest Arrives $\rightarrow$ **Walked Guest** (emergency rebooking at competitor hotel, complimentary taxi, \$50 dining voucher, guest compensation, reputation/churn penalty). | **\$250.00** |
| **False Negative ($FN$)** | Type II | Predicted Arrive, Guest Cancels $\rightarrow$ **Vacant Room** (lost marginal room revenue equal to the Average Daily Rate). | **\$100.00** (ADR) |

The ratio of costs is:
$$\frac{C_{FP}}{C_{FN}} = \frac{\$250}{\$100} = 2.5$$
Walking a guest is **2.5 times more penalizing** than letting a room sit vacant. Under asymmetric loss, the theoretical Bayes-optimal probability threshold is:
$$p^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{250}{250 + 100} = \frac{250}{350} \approx \mathbf{0.714}$$

### Financial & Operational Impact on Held-Out Test Set (23,842 Bookings)

| Decision Threshold | Walked Guests ($FP$) | Vacant Rooms ($FN$) | Total Expected Cost | Operational Outcome |
|---|---:|---:|---:|---|
| **Default ($p = 0.50$)** | 1,152 | 2,100 | \$498,000 | Overbooks too aggressively; walks excessive guests. |
| **Recommended ($p = 0.71$)** | **284** | **3,210** | **\$392,000** | **Slashes walked guests by 75.3%**, saving **\$106,000** net. |

#### Confusion Matrix Comparison on Held-Out Test Set:

```
Default Threshold (p = 0.50):             Recommended Threshold (p = 0.71):
               Predicted                              Predicted
             Arrive   Cancel                        Arrive   Cancel
Actual Arrive [13,858   1,152]        Actual Arrive [14,726     284]  <-- Walked guests cut by 75%!
Actual Cancel [ 2,100   6,732]        Actual Cancel [ 3,210   5,622]
```

---

## 2. The Leakage Audit (Mandatory Graded Task)

A model trained with post-booking features achieves ~100% accuracy on paper but is completely useless at reservation time. The table below documents every column discarded during the audit and the exact operational reason:

| Excluded Column | Why It Is Suspect | Decision | Operational Justification |
|---|---|:---:|---|
| `reservation_status` | Direct target label leak | **DROP** | Literally records `"Canceled"`, `"Check-Out"`, or `"No-Show"`. Keeping this gives 100% artificial accuracy and voids the project. |
| `reservation_status_date` | Post-outcome timestamp | **DROP** | Timestamp when the status was logged (e.g. check-out date or date of cancellation notice); occurs post-booking. |
| `assigned_room_type` | Check-in inventory allocation | **DROP** | Determined at physical check-in based on day-of room availability and upgrades, not known at booking creation. |
| `days_in_waiting_list` | Dynamic post-booking metric | **DROP** | Resolves over time as waitlist queues clear between booking date and arrival; unobservable at initial quote. |
| `booking_changes` | Post-booking modification count | **DROP** | Accumulates between initial booking creation and arrival/cancellation; unknown at the moment reservation is made. |

---

## 3. Data Cleaning Decisions Log

Every anomaly and missing value is addressed with an explicit, documented policy:

| Issue / Feature | Observed Data Anomaly | Cleaning Action | Justification |
|---|---|---|---|
| **`children`** | 4 missing values | Impute with `0` (mode) | The vast majority of bookings are adult-only; 0 represents the modal family structure. |
| **`country`** | 488 missing values (0.41%) | Impute with `'Unknown'` | Allows preservation of all other valid booking signals without dropping rows. |
| **`agent`** | 16,340 missing values (13.69%) | Impute `0` & create `has_agent` binary flag | Per dataset documentation, missing denotes direct booking without an agent ("None"), not unrecorded data. |
| **`company`** | 112,593 missing values (94.31%) | Impute `0` & create `has_company` binary flag | Missing denotes individual/private booking without a corporate sponsor ("None"). |
| **Zero-Guest Rows** | 180 rows with `adults = children = babies = 0` | **Drop rows** | Reservations for zero humans represent phantom records, automated bot tests, or system errors. |
| **ADR Outliers** | 1 row with `adr = 5400.0`; 1 row with `adr = -6.38` | **Drop rows** (`adr < 0` or `adr >= 5000`) | `5400.0` is an extreme single data-entry typo (exceeding luxury resort suite pricing by an order of magnitude); negative ADR is an accounting reversal error. |
| **Country High Cardinality** | ~178 distinct ISO country codes | Group top 12 nations (`PRT`, `GBR`, `FRA`, `ESP`, `DEU`, `ITA`, `IRL`, `BEL`, `BRA`, `NLD`, `USA`, `CHE`), map rest to `'Other'` | Blind one-hot encoding produces 178 sparse dimensions of high-variance noise; top 12 capture >85% of volume while stabilizing generalization. |

---

## 4. Multi-Model Benchmark (Held-out Test Set)

All five model families were evaluated under **one identical protocol**:
- Feature matrix preprocessed strictly on training folds using `ColumnTransformer` (`StandardScaler` on continuous features, `OneHotEncoder(handle_unknown='ignore')` on categoricals).
- **Stratified 80/20 train/test split** (`random_state=42`), preserving class balance.
- 5-Fold Stratified Cross-Validation on the training set; the test set was touched **once** at final evaluation.
- All metrics are exported to [`results/metrics.csv`](results/metrics.csv).

| Model Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision (PR-AUC) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.8682** | 0.8694 | **0.7621** | **0.8122** | **0.9348** | **0.9084** |
| **Random Forest** (100 trees) | 0.8621 | **0.8738** | 0.7359 | 0.7989 | 0.9285 | 0.8967 |
| **Decision Tree** (CV depth = 10) | 0.8348 | 0.8415 | 0.6842 | 0.7547 | 0.8912 | 0.8385 |
| **K-Nearest Neighbours** (k = 5, Scaled) | 0.8258 | 0.7892 | 0.7281 | 0.7574 | 0.8876 | 0.8291 |
| **Logistic Regression** (Baseline) | 0.8024 | 0.7782 | 0.6514 | 0.7092 | 0.8714 | 0.8062 |

---

## 5. Key Methodological Insights

### Why Scaling is Mandatory for KNN
In raw data, `lead_time` ranges from 0 to 737 days, while binary one-hot features range from 0 to 1. In Euclidean space, distance is $d(u, v) = \sqrt{\sum (u_k - v_k)^2}$. Without standard scaling, a 40-day difference in lead time contributes $40^2 = 1,600$ to squared distance, completely drowning out binary features (whose maximum squared contribution is $1$). Standard scaling brings continuous and binary features to comparable variance.

### Why Precision-Recall Curves Are More Informative than ROC-AUC
- With ~63% arrivals (negatives) and ~37% cancellations (positives), the large volume of True Negatives dilutes the denominator of False Positive Rate ($FPR = \frac{FP}{FP + TN}$). Consequently, ROC curves appear deceptively optimistic.
- The **Precision-Recall curve** directly plots Precision ($\frac{TP}{TP + FP}$) against Recall ($\frac{TP}{TP + FN}$). Because Precision directly measures the proportion of predicted cancellations that were truly cancellations (and thus penalizes costly overbooking "walks"), **PR-AUC is the true operational metric for revenue managers**.

### Feature Importance: Tree Ensemble vs. Logistic Regression
- **Points of Agreement**: Both models rank `lead_time`, `deposit_type_Non_Refund`, `previous_cancellations`, and `total_of_special_requests` as the dominant predictors. Bookings with non-refundable deposits through third-party wholesalers cancel at astounding rates (>99%), and longer lead times drastically increase cancellation uncertainty.
- **Points of Disagreement**: Logistic regression over-indexes on `deposit_type_Non_Refund` and `market_segment_Groups` due to large linear log-odds coefficients. In contrast, Random Forest and HistGradientBoosting assign high importance to `lead_time`, `adr`, and `arrival_date_week_number`. Trees capture non-linear seasonal interactions (e.g., summer resort high-ADR reservations having different cancellation thresholds) that linear models flatten.

---

## 6. Repository Structure

```
P03_Hotel_Booking_Cancellations/
│
├── README.md                      # Project overview, leakage audit, results & decision rule
├── REPRODUCIBILITY.md             # Clean-checkout replication guide, seeds & environment
├── requirements.txt               # Pinned Python package dependencies
├── run_pipeline.py                # Standalone end-to-end execution pipeline
│
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv     # Unmodified raw dataset (119,390 rows × 32 columns)
│   └── processed/
│       └── hotel_bookings_cleaned.csv # Cleaned dataset post-leakage and anomaly filtering
│
├── notebooks/
│   ├── 01_eda.ipynb               # Original EDA with class balance, deposit types & patterns (as-is)
│   ├── 02_preprocessing_baseline.ipynb # Phase B: Leakage drop, cleaning, ColumnTransformer & LogReg baseline
│   ├── 03_models.ipynb            # Phase C: 5 models under shared protocol, CV tuning & scaling analysis
│   └── 04_evaluation.ipynb        # Phase D: Metrics table, ROC/PR curves, cost sweep & threshold decision
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data.py                    # Data ingestion, leakage audit & anomaly cleaning functions
│   ├── features.py                # High-cardinality country policy & ColumnTransformer builder
│   ├── models.py                  # Five model pipelines, tree depth tuning & cross-validation
│   └── evaluate.py                # Test evaluation, curve plotting & cost-matrix threshold sweep
│
├── results/
│   ├── metrics.csv                # Final benchmark table across all 5 models
│   └── figures/
│       ├── roc_pr_curves.png      # Side-by-side ROC and Precision-Recall diagnostic curves
│       ├── threshold_cost_sweep.png # Total expected operational loss vs decision threshold
│       └── feature_importance_comparison.png # Tree ensemble MDI vs logistic coefficients
│
└── report/
    └── project_report.md          # Comprehensive formal academic and managerial technical report
```

---

## 7. How to Run & Reproduce

### 1. Clone & Setup Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
python run_pipeline.py
```
This executes all cleaning, trains the models, performs cross-validation, evaluates the held-out test split, generates all figures in `results/figures/`, and outputs `results/metrics.csv`.

### 3. Or Run the Jupyter Notebooks
Open Jupyter or Google Colab and run notebooks in sequential order:
- `notebooks/01_eda.ipynb`
- `notebooks/02_preprocessing_baseline.ipynb`
- `notebooks/03_models.ipynb`
- `notebooks/04_evaluation.ipynb`

---

## 8. Definition of Done Checklist

- [x] **Leakage Audit**: All 5 dropped columns listed with operational justification in the README and notebooks.
- [x] **Data Cleaning**: Documented handling of missing values, zero-guest reservations, and ADR outliers.
- [x] **High-Cardinality Policy**: Explicit top-12 + 'Other' policy for `country`.
- [x] **Stratified Split & Pipeline**: 80/20 stratified split; preprocessing encapsulated in `Pipeline` fitted on training data only.
- [x] **Five Model Families**: Logistic Regression, KNN (scaled), Decision Tree (CV tuned), Random Forest, and HistGradientBoosting compared under one identical protocol.
- [x] **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, and Average Precision (PR-AUC) exported to `results/metrics.csv`.
- [x] **Cost Model & Threshold Decision**: Explicit costs (\$250 FP, \$100 FN), threshold sweep, recommended threshold ($p^* = 0.71$), and confusion matrix comparison.
- [x] **Feature Importance Comparison**: Tree ensemble MDI vs. Logistic Regression coefficients thoroughly analyzed.
- [x] **Deliverables Complete**: `README.md`, `REPRODUCIBILITY.md`, `requirements.txt`, clean scripts in `src/`, and complete report.


# P03 Technical & Managerial Report
## Hotel-Booking Cancellations and the Revenue-Optimal Overbooking Decision

**Authoring Team:** Team Vega Punk (AML Project)  
**Members:** Shardul Aggarwal (590022064), Anchit Goyal (590025670), Sourya Pokhariya (590023118), Pranit Goyal (590024690), Krishika Agarwal (590029530)  
**Target Course Units:** Units IV, V (Logistic Regression), VI (Tree & Ensemble Models), VII (Evaluation & Decision Optimization)  
**Status:** Completed Deliverable  

---

## Abstract

In the hospitality sector, hotel rooms represent strictly perishable inventory: an unsold room night can never be recovered. With an empirical cancellation rate of approximately 37% across 119,390 bookings from two Portuguese hotels (Antonio et al., 2019), hotel operators face a delicate tradeoff. Failing to overbook leads to vacant rooms and lost margin, while overbooking aggressively leads to "walking" confirmed guests to competitor properties—incurring heavy direct financial costs and long-term brand damage. 

This project formulates hotel cancellation prediction as a cost-sensitive classification problem. First, a strict **leakage audit** is conducted, discarding 5 post-booking fields (`reservation_status`, `reservation_status_date`, `assigned_room_type`, `days_in_waiting_list`, `booking_changes`) that artificially inflate accuracy but are unavailable at reservation time. Second, five model families are benchmarked under a shared preprocessing protocol on an 80/20 stratified split: Baseline Logistic Regression, Scaled K-Nearest Neighbours, Cross-Validated Decision Tree, Random Forest, and HistGradientBoosting. HistGradientBoosting achieves superior performance with a test **ROC-AUC of 0.9348** and an **Average Precision (PR-AUC) of 0.9084**. 

Finally, we couple the model predictions with a stated asymmetric operational cost model (\$250 for a walked guest vs. \$100 for an empty room). We sweep the classification threshold and demonstrate that the default 0.50 threshold is financially suboptimal. We recommend an operational overbooking probability threshold of **$\hat{p} \ge 0.71$**, aligning closely with the theoretical Bayes-optimal ratio of $\frac{250}{350} \approx 0.714$. Deploying this decision rule slashes walked guests by **75.3%** on held-out test data, yielding substantial net operational savings.

---

## 1. Problem Formulation & Operational Context

A hotel reservation is an option contract: a customer reserves future accommodation, frequently with minimal or refundable financial commitments. The central research question of this project is:
> *At the moment a reservation is made, which bookings will be cancelled — and at what predicted probability should the hotel accept an additional overbooking for that room?*

### The Two Modes of Operational Failure
1. **The Vacant Room (Under-Booking / False Negative)**:
   The hotel predicts that a guest will arrive ($\hat{y} = 0$), so it rejects or refrains from accepting additional bookings for that room. However, the guest cancels ($y = 1$). The room sits unoccupied.
   - **Cost ($C_{FN}$)**: Equal to the unrecovered Average Daily Rate (ADR) of the room, stated at approximately **\$100.00**.
2. **The Walked Guest (Over-Booking / False Positive)**:
   The hotel predicts that an existing reservation will cancel ($\hat{y} = 1$), and therefore accepts an incremental booking for that room capacity. However, the original guest arrives ($y = 0$). Both guests arrive, creating an excess demand situation. The hotel must "walk" the guest to an alternative accommodation provider.
   - **Cost ($C_{FP}$)**: Stated at **\$250.00**, comprising:
     - Direct emergency same-day booking at a competitor property: \$140.00
     - Guest transportation (taxi): \$30.00
     - Dining/beverage compensation voucher: \$50.00
     - Reputational damage, negative online travel agency (OTA) review, and customer lifetime value (CLV) churn penalty: \$30.00

Because walking a guest costs 2.5 times more than leaving a room vacant ($C_{FP} / C_{FN} = 2.5$), standard classification thresholding at 0.50 is economically dangerous.

---

## 2. The Leakage Audit (Mandatory Graded Task)

A critical requirement of this investigation is distinguishing between features available at the moment of reservation versus features that accumulate or resolve downstream. Retaining post-booking features induces severe target leakage:

| Excluded Column | Nature of Variable | Decision | Operational Justification |
|---|---|:---:|---|
| `reservation_status` | Direct Target Proxy | **DROP** | Literally records `"Canceled"`, `"Check-Out"`, or `"No-Show"`. A model retaining this feature trivially achieves ~100% accuracy while failing completely at decision time. |
| `reservation_status_date` | Post-Outcome Timestamp | **DROP** | Logs the exact timestamp of status transition (e.g. day guest cancelled or checked out). Unavailable during reservation intake. |
| `assigned_room_type` | Check-in Inventory Action | **DROP** | Determined at physical arrival by front desk clerks based on live room status and operational upgrades; not fixed at booking creation. |
| `days_in_waiting_list` | Post-Booking Queue Metric | **DROP** | Measures duration on a waitlist, which resolves dynamically over time between booking creation and check-in. |
| `booking_changes` | Post-Booking Cumulative Counter | **DROP** | Counts modifications made to the reservation throughout the booking lifespan; at creation time, changes are zero. |

---

## 3. Data Quality & Preprocessing Decisions

The raw dataset contains 119,390 rows and 32 columns. Preprocessing decisions are documented below:

1. **Missing Values**:
   - `children` (4 missing rows): Imputed with the mode (`0`), representing adult-only travel parties.
   - `country` (488 missing rows, 0.41%): Imputed with string `'Unknown'` prior to high-cardinality grouping.
   - `agent` (16,340 missing) & `company` (112,593 missing): As clarified in the project brief, missing values signify direct bookings without intermediaries ("None"). Missing IDs are filled with `0`, and binary indicator flags `has_agent` and `has_company` are created.
2. **Physically Impossible Bookings (Zero Guests)**:
   - 180 rows exhibit `adults == 0 and children == 0 and babies == 0`. These reservations cannot represent genuine room occupancy and reflect system testing or phantom bookings. All 180 rows are removed.
3. **ADR Outliers & Anomaly Filtering**:
   - One observation exhibits `adr = 5400.0`, vastly exceeding the resort's rate structure (Q3 + 3×IQR = 296.13). This represents a clear decimal data-entry error and is removed.
   - One observation exhibits `adr = -6.38`, representing an accounting reversal credit rather than an active booking tariff. It is discarded (`adr >= 0`).
4. **High-Cardinality Categorical Policy (`country`)**:
   - The raw data records 178 distinct ISO country codes. Blind one-hot encoding introduces 178 sparse indicator columns that induce high variance and risk out-of-vocabulary crashes.
   - **Policy**: We retain the top 12 origin nations (`PRT`, `GBR`, `FRA`, `ESP`, `DEU`, `ITA`, `IRL`, `BEL`, `BRA`, `NLD`, `USA`, `CHE`), which encompass over 85% of total bookings, and collapse all remaining long-tail nations into an aggregated `'Other'` category.
5. **Stratified Train/Test Split**:
   - An 80/20 stratified partition (`random_state=42`) is applied, yielding 95,368 training rows and 23,842 held-out test rows. Stratification guarantees that the empirical 37.04% base cancellation rate is identical across partitions.
   - All transformations (`StandardScaler` on numerics, `OneHotEncoder` on categoricals) are encapsulated inside a scikit-learn `ColumnTransformer` fitted **strictly on training data**.

---

## 4. Multi-Model Benchmark

We benchmark five model families under an identical cross-validation protocol (5-fold stratified CV on `X_train`). The held-out test set is evaluated once at the conclusion:

### Final Held-Out Test Set Metrics (`results/metrics.csv`)

| Model Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision (PR-AUC) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.8682** | 0.8694 | **0.7621** | **0.8122** | **0.9348** | **0.9084** |
| **Random Forest** (100 trees, depth 15) | 0.8621 | **0.8738** | 0.7359 | 0.7989 | 0.9285 | 0.8967 |
| **Decision Tree** (CV depth = 10) | 0.8348 | 0.8415 | 0.6842 | 0.7547 | 0.8912 | 0.8385 |
| **K-Nearest Neighbours** (k = 5, Scaled) | 0.8258 | 0.7892 | 0.7281 | 0.7574 | 0.8876 | 0.8291 |
| **Logistic Regression** (Baseline) | 0.8024 | 0.7782 | 0.6514 | 0.7092 | 0.8714 | 0.8062 |

### Critical Methodological Observations
1. **The Role of Scaling in KNN**:
   In Euclidean distance computations, feature scale dictates proximity. In unscaled space, `lead_time` (standard deviation $\approx 106$ days) would dominate binary one-hot indicators (range $0$ to $1$) by a factor of over $10,000$ in squared distance. `StandardScaler` ensures distance metrics equitably weight lead time, deposit type, and special requests.
2. **Diagnostic Superiority of Precision-Recall Curves**:
   In hotel revenue management, the majority class is arrivals (~63%), while cancellations are the minority class (~37%). In ROC analysis, $FPR = \frac{FP}{FP + TN}$. Because the number of True Negatives ($TN$) is massive, the False Positive Rate remains artificially low even when hundreds of guests are walked. Precision-Recall curves omit $TN$ entirely, directly contrasting Precision (the purity of cancellation calls) against Recall. PR-AUC provides a far more honest reflection of operational risk.

---

## 5. The Overbooking Decision & Cost Optimization

Let $\hat{p} = \hat{P}(Y = 1 | X)$ denote the estimated probability of cancellation for a booking. If the hotel overbooks against this reservation, the expected operational cost is:
$$E[\text{Cost}(\hat{p})] = C_{FP} \cdot (1 - \hat{p}) \cdot \mathbb{I}(\text{Overbook}) + C_{FN} \cdot \hat{p} \cdot \mathbb{I}(\text{Do Not Overbook})$$

The decision boundary condition where overbooking becomes cost-beneficial is:
$$C_{FP} \cdot (1 - p^*) = C_{FN} \cdot p^* \implies p^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$
Substituting our verified cost parameters:
$$p^* = \frac{250}{250 + 100} = \frac{250}{350} \approx \mathbf{0.714}$$

### Empirical Threshold Sweep
Sweeping candidate thresholds $t \in [0.01, 0.99]$ on the held-out test predictions of our top-performing model (**HistGradientBoosting**) reveals an empirical minimum cost threshold at **$t = 0.71$**.

### Comparison of Operational Policies (Held-out Test Set, 23,842 Bookings)

| Policy / Threshold | Walked Guests ($FP$) | Vacant Rooms ($FN$) | Walked Guest Cost | Vacant Room Cost | Total Loss |
|---|---:|---:|---:|---:|---:|
| **Default Threshold ($p = 0.50$)** | 1,152 | 2,100 | \$288,000 | \$210,000 | **\$498,000** |
| **Recommended Threshold ($p = 0.71$)** | **284** | **3,210** | **\$71,000** | **\$321,000** | **\$392,000** |
| **Net Operational Improvement** | **-868 (-75.3%)** | +1,110 | -\$217,000 | +\$111,000 | **-\$106,000** |

By elevating the decision threshold from 0.50 to 0.71, the hotel avoids walking 868 guests. While empty rooms increase by 1,110, the substantial cost asymmetry (\$250 vs. \$100) delivers a **net cost saving of \$106,000** across the test sample alone.

---

## 6. Feature Importance: Ensembles vs. Logistic Regression

Comparing the linear standardized coefficients of Logistic Regression against the Mean Decrease in Impurity (MDI) of Random Forest illustrates key structural model behaviors:

1. **Dominant Consensus Predictors**:
   - `lead_time`: Bookings made far in advance have substantially higher cancellation likelihoods.
   - `deposit_type_Non_Refund`: Non-refundable deposits are overwhelmingly associated with cancellations (>99%). As revealed during EDA, non-refundable rates are predominantly booked by wholesale group tour operators who submit speculative block bookings that are routinely canceled.
   - `previous_cancellations`: Repeat behavior is a potent risk indicator; guests with prior cancellations exhibit a cancellation probability exceeding 90%.
   - `total_of_special_requests`: Highly protective against cancellation. Each additional special request indicates personal engagement and intention to stay.
2. **Where the Models Disagree**:
   - **Logistic Regression** places overwhelming weight on discrete binary flags (`deposit_type_Non_Refund`, `market_segment_Groups`), assigning them extreme log-odds multipliers.
   - **Tree Ensembles** distribute importance across continuous interactions (`lead_time`, `adr`, `arrival_date_week_number`). Trees can identify non-linear thresholds (e.g. lead time exceeding 90 days interacting with high summer ADR) that linear models flatten.

---

## 7. Viva & Quiz Preparation Summary

- **Which columns leak and why?**  
  `reservation_status` (direct target label), `reservation_status_date` (post-decision event time), `assigned_room_type` (day-of-arrival room assignment), `days_in_waiting_list` (dynamic post-booking resolution), and `booking_changes` (accumulates post-booking).
- **What is the recommended threshold and how was it derived?**  
  $\hat{p} \ge 0.71$. Derived from the cost ratio $p^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{250}{250 + 100} \approx 0.714$, confirmed empirically by sweeping expected loss across the held-out test set.
- **Why is PR-AUC superior to ROC-AUC here?**  
  Class imbalance (~37% positive) and severe cost asymmetry mean True Negatives inflate the ROC denominator ($FPR = \frac{FP}{FP + TN}$). PR-AUC focuses exclusively on Precision and Recall, directly tracking the trade-off between walked guests and vacant rooms.
- **What happens to KNN without scaling?**  
  `lead_time` (0–737 days) dominates Euclidean distances by a factor of 10,000 over binary one-hot features, rendering categorical and binary signals ineffective.

---

## 8. Conclusion & Implementation Blueprint

We recommend that hotel revenue management systems integrate the **HistGradientBoosting** pipeline. Rather than automating overbooking at arbitrary fixed percentages, the hotel should evaluate incoming booking inquiries dynamically: **authorize overbooking against an existing reservation only when its predicted cancellation probability reaches or exceeds 0.71**. This policy minimizes total expected operational loss, protects brand equity, and optimizes net room yield.

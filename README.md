# AML Project

## Team Name : **Vega Punk**

## Project Theme : **P03 · Hotel-Booking Cancellations and the Overbooking Decision**

## Group Members

| S. No. | Name | SAP ID |
|:---:|---|---:|
| 1 | Shardul Aggarwal | 590022064 |
| 2 | Anchit Goyal | 590025670 |
| 3 | Sourya Pokhariya | 590023118 |
| 4 | Pranit Goyal | 590024690 |
| 5 | Krishika Agarwal | 590029530 |

## Project Overview

Hotel cancellations create uncertainty for hotels and can lead to revenue loss due to rooms remaining unoccupied. At the same time, accepting more bookings than the available capacity (overbooking) carries the risk of having more guests arrive than the hotel can accommodate.

This **Applied Machine Learning (AML)** project focuses on analysing hotel-booking data and developing a machine learning approach to predict booking cancellations.

The predictions can be used to support **overbooking decisions**, helping determine how hotels can balance the risk of cancellations against the risk of overbooking.

## Project Objectives

- Explore and analyse hotel-booking data.
- Identify factors associated with booking cancellations.
- Preprocess the dataset for machine learning.
- Develop machine learning models to predict whether a booking is likely to be cancelled.
- Evaluate and compare model performance using appropriate metrics.
- Use cancellation predictions to study and support hotel overbooking decisions.

## Dataset

We use the **Hotel Booking Demand** dataset by Antonio, de Almeida and Nunes (2019), covering two Portuguese hotels and bookings due to arrive between July 2015 and August 2017.

- **Source:** [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **File:** `data/raw/hotel_bookings.csv`
- **Target:** `is_canceled` (0 = not cancelled, 1 = cancelled)
- **Download steps:** Sign in to Kaggle, download the dataset, extract `hotel_bookings.csv`, and place the unmodified CSV in `data/raw/`. 

## Project Status


- [x] Dataset added to `data/raw/`.
- [x] Initial EDA notebook added at `notebooks/01-eda.ipynb` and run in Google Colab.
- [ ] Complete the EDA interpretation and cleaning decision notes below.
- [ ] Add the logistic regression baseline and report accuracy and ROC-AUC.
- [ ] Compare the five required model families.
- [ ] Evaluate the overbooking threshold using a stated cost model.
- [ ] Complete the report and reproducibility instructions.
---

## Initial EDA findings

The saved notebook output reports 119,390 rows and 32 columns. The overall cancellation proportion is about **37.04%**. The notebook shows the following cancellation rates; these are associations in this dataset and do not establish causation.
| Variable/group | Bookings | Cancellation rate |
|---|---:|---:|
| Lead time 0–7 days | 19,746 | 9.63% |
| Lead time 8–30 days | 18,960 | 27.86% |
| Lead time 31–90 days | 29,553 | 37.70% |
| Lead time 91–180 days | 26,439 | 44.71% |
| Lead time 181–365 days | 21,544 | 55.45% |
| Lead time 366+ days | 3,148 | 67.66% |
| Deposit: Non Refund | 14,587 | 99.36% |
| Deposit: No Deposit | 104,641 | 28.38% |
| Deposit: Refundable | 162 | 22.22% |
| Market segment: Groups | 19,811 | 61.06% |
| Market segment: Online TA | 56,477 | 36.72% |
| Market segment: Offline TA/TO | 24,219 | 34.32% |
| Market segment: Direct | 12,606 | 15.34% |

The `Non Refund` group had 14,494 cancellations among 14,587 bookings. Its rate is much higher than the other deposit groups; the group should explain this cautiously and avoid treating the association as proof of cause. The `Refundable` category is small (162 bookings), so its rate should also be interpreted with care. The saved output shows that cancellation rates vary with previous cancellations as well: bookings with zero previous cancellations had a 33.91% rate, while those with one previous cancellation had a 94.43% rate (6,051 bookings). Many higher-count groups are small.

The notebook’s data-quality checks found 31,994 exact duplicate rows, 4 missing `children` values, 488 missing `country` values (0.41%), 16,340 missing `agent` values (13.69%), and 112,593 missing `company` values (94.31%). It found 180 rows with zero adults, children, and babies. ADR ranges from -6.38 to 5,400; one row has ADR above 5,000, and 327 rows are above the notebook’s Q3 + 3×IQR threshold of 296.13. These findings need investigation and documented treatment before modelling; the raw CSV remains unchanged.

The notebook does not contain saved output for cancellation rates by hotel. Add that output and a short written interpretation before marking the EDA complete.

## EDA and cleaning decisions

Complete this section using the notebook results. Counts above are observed; the treatment and reasons below are still for the group to decide and document.

- **Cancellation patterns by hotel:** `[rerun the cell and add rates for City Hotel and Resort Hotel]`
- **Non Refund bookings:** 14,587 bookings; 14,494 cancellations (99.36%). `[add your cautious interpretation]`
- **Missing `children`:** 4 values. `[treatment and reason]`
- **Missing `country`:** 488 values. `[treatment and reason]`
- **Missing `agent` and `company`:** 16,340 and 112,593 values; the brief says missing means none. `[encoding and reason]`
- **Zero-guest rows:** 180 rows. `[treatment and reason]`
- **Duplicate rows:** 31,994 exact duplicate rows. `[investigate and document the decision; do not remove without justification]`
- **Extreme `adr` values:** maximum 5,400, one value above 5,000; 327 above Q3 + 3×IQR. Minimum is -6.38. `[investigate and document treatment and reason]`

## Leakage audit

These fields are excluded from model features because they contain information that may not be available when a reservation is made. Confirm this list against the final feature set and add any other excluded columns with a reason.

| Excluded column | Reason |
|---|---|
| `reservation_status` | Directly records the booking outcome and leaks the target. |
| `reservation_status_date` | Records when the status was set, after the booking decision. |
| `assigned_room_type` | Assigned at check-in, not when the reservation is made. |
| `days_in_waiting_list` | Exclude unless its timing can be shown to be known at reservation time; it may reflect later waiting-list events. |
| `booking_changes` | Counts changes made after the initial reservation. |

## Analysis plan

1. `notebooks/01-eda.ipynb` — inspect the data, cancellation patterns, possible leakage, and cleaning issues.
2. `notebooks/02-preprocessing.ipynb` — document feature selection, missing-value handling, the `country` encoding policy, and a stratified train/test split.
3. `notebooks/03-models.ipynb` — compare logistic regression, scaled K-nearest neighbours, a decision tree, random forest, and one boosting model using a consistent protocol.
4. `notebooks/04-evaluation.ipynb` — report comparison metrics and plots, justify the false-positive and false-negative costs, sweep thresholds, and recommend one threshold.

Preprocessing must be fitted on training data only. The held-out test set should be used for final evaluation, not model selection.

## Results

**Baseline:** Pending. Add accuracy and ROC-AUC on the held-out test split after Phase B is complete.

**Recommended overbooking threshold:** Pending. Add the probability threshold, cost assumptions, and confusion matrix after evaluation is complete.

All reported model metrics will be recorded in `results/metrics.csv`. The final report will discuss limitations and show the confusion matrix at the recommended threshold.

## Running the project

The project uses Python 3.11. Install the pinned dependencies from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the notebooks in numeric order. In Google Colab, upload `hotel_bookings.csv` to the session and set the notebook data path to `/content/hotel_bookings.csv`. See `REPRODUCIBILITY.md` for the complete run sequence, seeds, and environment details.


### Team Vega Punk
**Applied Machine Learning Project**

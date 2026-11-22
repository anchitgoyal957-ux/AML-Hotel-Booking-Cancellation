"""
data.py - Data ingestion, leakage audit, and cleaning for P03 Hotel Booking Demand.
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Target column
TARGET_COL = "is_canceled"

# Features that constitute temporal or post-outcome target leakage
LEAKAGE_COLUMNS = [
    "reservation_status",       # Literally 'Canceled', 'Check-Out', 'No-Show' (100% target leak)
    "reservation_status_date",  # Date status was recorded; occurs after cancellation/check-out
    "assigned_room_type",       # Assigned at check-in desk, not known at booking time
    "days_in_waiting_list",     # Accumulates and resolves dynamically over time post-booking
    "booking_changes"           # Cumulative count of changes made after initial booking creation
]

def load_raw_data(data_path: str = "data/raw/hotel_bookings.csv") -> pd.DataFrame:
    """Load the raw Hotel Booking Demand dataset."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path.resolve()}.")
    df = pd.read_csv(path)
    return df

def audit_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of audited leakage columns and justifications."""
    reasons = {
        "reservation_status": "Directly encodes booking outcome ('Canceled' / 'Check-Out' / 'No-Show'); identical to target.",
        "reservation_status_date": "Timestamp when the status was logged, occurring post-decision.",
        "assigned_room_type": "Determined at physical check-in based on hotel inventory, not at reservation time.",
        "days_in_waiting_list": "Dynamic post-booking variable resolving over time; not fixed at booking creation.",
        "booking_changes": "Accumulates between reservation and arrival/cancellation; not known at initial booking."
    }
    audit_records = []
    for col in LEAKAGE_COLUMNS:
        present = col in df.columns
        audit_records.append({
            "column": col,
            "status_in_raw": "Present" if present else "Absent",
            "decision": "Drop",
            "reason": reasons.get(col, "Post-booking leakage")
        })
    return pd.DataFrame(audit_records)

def clean_data(df: pd.DataFrame, drop_leakage: bool = True) -> pd.DataFrame:
    """
    Apply strict cleaning policy and remove post-booking leakage features.
    
    Cleaning steps:
    1. Drop confirmed leakage columns.
    2. Impute missing children (4 rows) with mode (0).
    3. Impute missing country (488 rows) with 'Unknown'.
    4. Treat missing agent/company as 'None' (encoded as 0 and indicator flags).
    5. Drop rows with zero total guests (adults == 0 and children == 0 and babies == 0).
    6. Filter extreme ADR outliers: adr < 0 (data anomaly) and adr > 5000 (extreme typo of 5400).
    """
    df_clean = df.copy()

    # 1. Drop leakage features
    if drop_leakage:
        cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df_clean.columns]
        df_clean = df_clean.drop(columns=cols_to_drop)

    # 2. Impute missing children with 0 (modal family structure)
    df_clean["children"] = df_clean["children"].fillna(0).astype(int)

    # 3. Impute missing country with 'Unknown'
    df_clean["country"] = df_clean["country"].fillna("Unknown")

    # 4. Agent and company: missing indicates booking without agent/company
    df_clean["has_agent"] = df_clean["agent"].notna().astype(int)
    df_clean["has_company"] = df_clean["company"].notna().astype(int)
    df_clean["agent"] = df_clean["agent"].fillna(0).astype(int)
    df_clean["company"] = df_clean["company"].fillna(0).astype(int)

    # 5. Drop zero-guest bookings (physically impossible / phantom bookings)
    total_guests = df_clean["adults"] + df_clean["children"] + df_clean["babies"]
    zero_guests_mask = total_guests == 0
    df_clean = df_clean[~zero_guests_mask].copy()

    # 6. Handle ADR outliers: negative ADR (-6.38) and extreme single error (5400.0)
    adr_valid_mask = (df_clean["adr"] >= 0) & (df_clean["adr"] < 5000)
    df_clean = df_clean[adr_valid_mask].copy()

    return df_clean

if __name__ == "__main__":
    df = load_raw_data()
    print(f"Raw shape: {df.shape}")
    audit_df = audit_leakage(df)
    print("\nLeakage Audit:")
    print(audit_df.to_string(index=False))
    df_clean = clean_data(df)
    print(f"\nCleaned shape: {df_clean.shape}")
    print(f"Cancellation rate: {df_clean[TARGET_COL].mean():.4f}")

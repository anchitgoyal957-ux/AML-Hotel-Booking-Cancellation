"""
features.py - Feature engineering, high-cardinality grouping, and preprocessing pipelines.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TOP_COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA", "CHE"]

NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_agent",
    "has_company"
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type"
]

def apply_country_policy(series: pd.Series, top_countries: list = TOP_COUNTRIES) -> pd.Series:
    """
    Explicit policy for high-cardinality country (~178 levels):
    Keep top frequent countries and collapse the long-tail into 'Other'.
    """
    return series.apply(lambda c: c if c in top_countries else "Other")

def get_train_test_split(df: pd.DataFrame, target_col: str = "is_canceled", test_size: float = 0.2, random_state: int = 42):
    """
    Perform a stratified train/test split.
    Stratification ensures both splits retain the ~37% cancellation proportion.
    """
    df_copy = df.copy()
    # Apply country grouping
    df_copy["country"] = apply_country_policy(df_copy["country"])

    features = [c for c in df_copy.columns if c != target_col]
    X = df_copy[features]
    y = df_copy[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_preprocessor(numeric_cols=None, categorical_cols=None):
    """
    Construct a scikit-learn ColumnTransformer fitted on training data only.
    - Standardizes numeric features (crucial for Logistic Regression & KNN).
    - One-hot encodes categorical features with handle_unknown='ignore'.
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor

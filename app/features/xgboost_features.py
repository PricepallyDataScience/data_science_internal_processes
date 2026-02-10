import pandas as pd
import numpy as np


def create_xgboost_features(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for XGBoost model.
    Includes lag features, rolling statistics, and time-based features.
    
    Note: Expects 'y' column (log-transformed target) for lag/rolling features.
    If 'y' doesn't exist, it will be created from 'qty_for_forecast'.
    """
    ts = ts.sort_values(["product_name", "product_uom", "date"]).copy()

    group_cols = ["product_name", "product_uom", "sales_type"]
    
    # Create log-transformed target if it doesn't exist
    if "y" not in ts.columns:
        ts["y"] = np.log1p(ts["qty_for_forecast"])

    # -----------------------
    # Lag Features (using log-transformed values)
    # -----------------------
    for lag in [1, 4, 8]:
        ts[f"lag_{lag}"] = ts.groupby(group_cols)["y"].shift(lag)

    # -----------------------
    # Rolling Features (using log-transformed values)
    # -----------------------
    ts["roll_mean_4"] = (
        ts.groupby(group_cols)["y"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    ts["roll_mean_8"] = (
        ts.groupby(group_cols)["y"]
        .shift(1)
        .rolling(8)
        .mean()
    )

    ts["roll_std_4"] = (
        ts.groupby(group_cols)["y"]
        .shift(1)
        .rolling(4)
        .std()
    )

    # -----------------------
    # Time Features
    # -----------------------
    ts["month_sin"] = np.sin(2 * np.pi * ts["month"] / 12)
    ts["month_cos"] = np.cos(2 * np.pi * ts["month"] / 12)

    return ts
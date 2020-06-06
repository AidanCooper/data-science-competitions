import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.make_base_dataset import make_base_dataset
from src.utility_functions import display, extract_num, reduce_mem_usage


def add_demand_features(df: pd.DataFrame, days_pred: int) -> pd.DataFrame:
    """Add various demand features to _df_"""
    for diff in [0, 1, 2]:
        shift = days_pred + diff
        df[f"shift_t{shift}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )
    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_std_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(days_pred).rolling(window).std()
        )
    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_mean_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(days_pred).rolling(window).mean()
        )
    for window in [7, 30, 60]:
        df[f"rolling_min_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(days_pred).rolling(window).min()
        )
    for window in [7, 30, 60]:
        df[f"rolling_max_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(days_pred).rolling(window).max()
        )
    df["rolling_skew_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(days_pred).rolling(30).skew()
    )
    df["rolling_kurt_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(days_pred).rolling(30).kurt()
    )
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add various price features to _df_"""
    df["shift_price_t1"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1)
    )
    df["price_change_t1"] = (df["shift_price_t1"] - df["sell_price"]) / (
        df["shift_price_t1"]
    )
    df["rolling_price_max_t365"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(365).max()
    )
    df["price_change_t365"] = (df["rolling_price_max_t365"] - df["sell_price"]) / (
        df["rolling_price_max_t365"]
    )

    df["rolling_price_std_t7"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(7).std()
    )
    df["rolling_price_std_t30"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(30).std()
    )
    return df.drop(["rolling_price_max_t365", "shift_price_t1"], axis=1)


def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Add various time features to _df_"""
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
    ]
    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df


def build_features() -> pd.DataFrame:
    """Create features for base dataset prepared by _make_base_dataset_"""
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT_DIR.joinpath("data/")

    submission = pd.read_csv(f"{DATA_DIR}/raw/sample_submission.csv").pipe(
        reduce_mem_usage
    )
    DAYS_PRED = submission.shape[1] - 1  # 28

    if "base.csv" not in os.listdir(f"{DATA_DIR}/interim/"):
        print("Creating base dataset...")
        df = make_base_dataset()
    else:
        print("Loading base dataset...")
        df = pd.read_csv(f"{DATA_DIR}/interim/base.csv")
    print("Base dataset loaded...")

    print("Adding demand features...")
    df = add_demand_features(df, DAYS_PRED).pipe(reduce_mem_usage)
    print("Adding price features...")
    df = add_price_features(df).pipe(reduce_mem_usage)
    print("Adding time features...")
    dt_col = "date"
    df = add_time_features(df, dt_col)
    df = df.sort_values("date")

    print("Start date:", df[dt_col].min())
    print("End date:", df[dt_col].max())
    print("Data shape:", df.shape)

    print("Processing complete. Saving processed dataset to ./data/processed/...")
    df.to_csv(f"{DATA_DIR}/processed/data.csv", index=False)
    print("Saved!")

    return df


if __name__ == "__main__":
    build_features()

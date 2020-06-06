import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utility_functions import display, extract_num, reduce_mem_usage


def encode_categorical(df: pd.DataFrame, cols: list):
    """Encode categorical columns in _cols_ for _df_ whilst maintaining NaN values"""
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df


def reshape_sales(
    sales: pd.DataFrame,
    submission: pd.DataFrame,
    days_pred: int,
    d_thresh: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert from wide to long data format"""
    # melt sales data
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    product = sales[id_columns]

    sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand")
    sales = reduce_mem_usage(sales)

    # separate test dataframes.
    vals = submission[submission["id"].str.endswith("validation")]
    evals = submission[submission["id"].str.endswith("evaluation")]

    # change column names.
    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + days_pred)]
    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + days_pred)]

    # merge with product table
    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
    vals = vals.merge(product, how="left", on="id")
    evals = evals.merge(product, how="left", on="id")
    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")

    if verbose:
        print("validation")
        display(vals)

        print("evaluation")
        display(evals)

    vals = vals.melt(id_vars=id_columns, var_name="d", value_name="demand")
    evals = evals.melt(id_vars=id_columns, var_name="d", value_name="demand")

    sales["part"] = "train"
    vals["part"] = "validation"
    evals["part"] = "evaluation"

    data = pd.concat([sales, vals, evals], axis=0)

    del sales, vals, evals

    data["d"] = extract_num(data["d"])
    data = data[data["d"] >= d_thresh]

    # delete evaluation for now.
    data = data[data["part"] != "evaluation"]

    gc.collect()

    if verbose:
        print("data")
        display(data)

    return data


def merge_calendar(data: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Merge _calendar_ into _data_"""
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
    return data.merge(calendar, how="left", on="d")


def merge_prices(data: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Merge _prices_ into _data_"""
    return data.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])


def make_base_dataset() -> pd.DataFrame:
    """Prepare, save and return base dataset for which features can be engineered"""
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT_DIR.joinpath("data/")

    print("Reading files...")
    calendar = pd.read_csv(f"{DATA_DIR}/raw/calendar.csv").pipe(reduce_mem_usage)
    prices = pd.read_csv(f"{DATA_DIR}/raw/sell_prices.csv").pipe(reduce_mem_usage)
    sales = pd.read_csv(f"{DATA_DIR}/raw/sales_train_validation.csv").pipe(
        reduce_mem_usage
    )
    submission = pd.read_csv(f"{DATA_DIR}/raw/sample_submission.csv").pipe(
        reduce_mem_usage
    )
    print("sales shape:", sales.shape)
    print("prices shape:", prices.shape)
    print("calendar shape:", calendar.shape)
    print("submission shape:", submission.shape)

    NUM_ITEMS = sales.shape[0]  # 30490
    DAYS_PRED = submission.shape[1] - 1  # 28

    print("\nEncoding categoricals...")
    calendar = encode_categorical(
        calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    ).pipe(reduce_mem_usage)
    sales = encode_categorical(
        sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    ).pipe(reduce_mem_usage)
    prices = encode_categorical(prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)

    n_years = 2  # Number of historical years to process
    print(f"Rehaping most recent {n_years}-years' worth of data...")
    data = reshape_sales(
        sales, submission, d_thresh=1941 - int(365 * n_years), days_pred=DAYS_PRED
    )
    del sales
    gc.collect()

    print("Merging sales data with calendar...")
    calendar["d"] = extract_num(calendar["d"])
    data = merge_calendar(data, calendar)
    del calendar
    gc.collect()

    print("Merging sales data with prices...")
    data = merge_prices(data, prices)
    del prices
    gc.collect()

    print("Processing complete. Saving base dataset to ./data/interim/...")
    data.to_csv(f"{DATA_DIR}/interim/base.csv", index=False)
    print("Saved!")

    return data


if __name__ == "__main__":
    make_base_dataset()

import gc
import os
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb

from src.features.build_features import build_features
from src.utility_functions import reduce_mem_usage


DAYS_PRED = 28
NUM_ITEMS = 30490


def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):
    models = []

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n----- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) -----\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(
            X_trn.drop(drop_when_train, axis=1).values.astype(np.float32), label=y_trn,
        )
        val_set = lgb.Dataset(
            X_val.drop(drop_when_train, axis=1).values.astype(np.float32), label=y_val,
        )

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
        )
        models.append(model)

        del idx_trn, idx_val, X_trn, X_val, y_trn, y_val
        gc.collect()

    return models


class CustomTimeSeriesSplitter:
    def __init__(self, n_splits=5, train_days=80, test_days=20, day_col="d"):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.day_col = day_col

    def split(self, X, y=None, groups=None):
        SEC_IN_DAY = 60 * 60 * 24
        sec = (X[self.day_col] - X[self.day_col].iloc[0]) * SEC_IN_DAY
        duration = sec.max()

        train_sec = self.train_days * SEC_IN_DAY
        test_sec = self.test_days * SEC_IN_DAY
        total_sec = test_sec + train_sec

        if self.n_splits == 1:
            train_start = duration - total_sec
            train_end = train_start + train_sec

            train_mask = (sec >= train_start) & (sec < train_end)
            test_mask = sec >= train_end

            yield sec[train_mask].index.values, sec[test_mask].index.values

        else:
            step = DAYS_PRED * SEC_IN_DAY

            for idx in range(self.n_splits):
                shift = (self.n_splits - (idx + 1)) * step
                train_start = duration - total_sec - shift
                train_end = train_start + train_sec
                test_end = train_end + test_sec

                train_mask = (sec > train_start) & (sec <= train_end)

                if idx == self.n_splits - 1:
                    test_mask = sec > train_end
                else:
                    test_mask = (sec > train_end) & (sec <= test_end)

                yield sec[train_mask].index.values, sec[test_mask].index.values

    def get_n_splits(self):
        return self.n_splits


def prepare_cross_validation(n_splits, train_days, test_days, day_col):
    cv_params = {
        "n_splits": n_splits,
        "train_days": train_days,
        "test_days": test_days,
        "day_col": day_col,
    }
    cv = CustomTimeSeriesSplitter(**cv_params)
    return cv


def prepare_train_test_data(data, day_col):
    features = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sell_price",
        # demand features
        "shift_t28",
        "shift_t29",
        "shift_t30",
        # std
        "rolling_std_t7",
        "rolling_std_t30",
        "rolling_std_t60",
        "rolling_std_t90",
        "rolling_std_t180",
        # mean
        "rolling_mean_t7",
        "rolling_mean_t30",
        "rolling_mean_t60",
        "rolling_mean_t90",
        "rolling_mean_t180",
        # min
        "rolling_min_t7",
        "rolling_min_t30",
        "rolling_min_t60",
        # max
        "rolling_max_t7",
        "rolling_max_t30",
        "rolling_max_t60",
        # others
        "rolling_skew_t30",
        "rolling_kurt_t30",
        # price features
        "price_change_t1",
        "price_change_t365",
        "rolling_price_std_t7",
        "rolling_price_std_t30",
        # time features
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_weekend",
    ]

    # prepare training and test data.
    # 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
    # 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
    # 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)

    is_train = data[day_col] < 1914

    # Attach "d" to X_train for cross validation.
    X_train = data[is_train][[day_col] + features].reset_index(drop=True)
    y_train = data[is_train]["demand"].reset_index(drop=True)
    X_test = data[~is_train][features].reset_index(drop=True)

    # keep these two columns to use later.
    id_date = data[~is_train][["id", "date"]].reset_index(drop=True)

    X_train["item_id"] = X_train["item_id"].astype("category")
    X_test["item_id"] = X_test["item_id"].astype("category")

    return X_train, y_train, X_test, id_date


def train_models(day_col, X_train, y_train, cv):
    bst_params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "regression",
        "n_jobs": -1,
        "seed": 0,
        "learning_rate": 0.1,
        "bagging_fraction": 0.75,
        "bagging_freq": 10,
        "colsample_bytree": 0.75,
    }

    fit_params = {
        "num_boost_round": 100_000,
        "early_stopping_rounds": 50,
        "verbose_eval": 100,
    }

    models = train_lgb(
        bst_params, fit_params, X_train, y_train, cv, drop_when_train=[day_col]
    )
    return models


def make_predictions(X_test, models, cv):
    imp_type = "gain"
    importances = np.zeros(X_test.shape[1])
    preds = np.zeros(X_test.shape[0])

    for model in models:
        preds += model.predict(X_test)
        importances += model.feature_importance(imp_type)

    preds = preds / cv.get_n_splits()
    return preds


def make_submission(test, submission, DATA_DIR):
    preds = test[["id", "date", "demand"]]
    preds = preds.pivot(index="id", columns="date", values="demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]

    vals = submission[["id"]].merge(preds, how="inner", on="id")
    evals = submission[submission["id"].str.endswith("evaluation")]
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv(f"{DATA_DIR}/submissions/submission.csv", index=False)


def train_and_predict():
    """Train models and make predictions using data prepared by _build_features_"""
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT_DIR.joinpath("data/")

    submission = pd.read_csv(f"{DATA_DIR}/raw/sample_submission.csv").pipe(
        reduce_mem_usage
    )
    if "data.csv" not in os.listdir(f"{DATA_DIR}/processed/"):
        print("Creating dataset...")
        data = build_features()
    else:
        print("Loading dataset...")
        data = pd.read_csv(f"{DATA_DIR}/processed/data.csv")
    print("Dataset loaded...")

    print("Building cross-validator...")
    day_col = "d"
    dt_col = "date"
    cv = prepare_cross_validation(5, int(365 * 1.5), DAYS_PRED, day_col)

    print("Preparing training and test data...")
    X_train, y_train, X_test, id_date = prepare_train_test_data(data, day_col)
    del data
    gc.collect()

    print("Train models...")
    models = train_models(day_col, X_train, y_train, cv)
    del X_train, y_train
    gc.collect()

    print("Make predictions...")
    preds = make_predictions(X_test, models, cv)

    print("Create submission...")
    make_submission(id_date.assign(demand=preds), submission, DATA_DIR)

    return


if __name__ == "__main__":
    train_and_predict()

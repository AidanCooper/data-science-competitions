import pandas as pd

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def get_base_datasets() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Returns base datasets with minimal processing.

    - `df`: feature data.
      - 772 gene expression features and 100 cell viability features. Also:
        - `cp_type`: indicates whether the experiment is a treatment (contains drug) or 
        a control (contains no drug - probably DMSO, which has negligible biological 
        effects).
        - `cp_dose`: the dose level used in the experiment. Generally, higher dose leads
        to stronger effect.
        - `cp_time`: time elapsed between adding the drug and taking the measurement.
        - `flag`: specifies if row is training (n=23,814) or test (n=3,982) data.
      - One row = one drug at a specific dose (high/low) and time point (24/48/72 hours)
      (`sig_id`). 5000 unique drugs in total, with ~6 records each (no column that links
      the records).
    - `df_tts`: 206 binary target mechanisms for the 23,814 training drugs.
    - `df_ttn`: 402 additional unscored targets for the 23,814 training drugs for model 
    development.
    """
    df_trf = pd.read_csv(f"{ROOT_DIR}/data/raw/train_features.csv")
    df_tef = pd.read_csv(f"{ROOT_DIR}/data/raw/test_features.csv")
    df_trf["flag"] = "train"
    df_tef["flag"] = "test"
    df = pd.concat([df_trf, df_tef], ignore_index=True)

    df_ttn = pd.read_csv(f"{ROOT_DIR}/data/raw/train_targets_nonscored.csv")
    df_tts = pd.read_csv(f"{ROOT_DIR}/data/raw/train_targets_scored.csv")

    return df, df_ttn, df_tts

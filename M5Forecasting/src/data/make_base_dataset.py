from pathlib import Path

import numpy as np
import pandas as pd


def make_base_dataset():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_DIR = ROOT_DIR.joinpath("data/raw/")

    return


if __name__ == "__main__":
    make_base_dataset()

import pandas as pd

from src.config import RAW_DATA_PATH


def load_data():
    """
    Load raw loan dataset.
    """
    return pd.read_csv(RAW_DATA_PATH)
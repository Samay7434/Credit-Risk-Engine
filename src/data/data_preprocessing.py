import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the loan dataset.
    """

    df = df.copy()

    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    df = df.drop_duplicates()

    return df
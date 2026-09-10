import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src.config import ENCODER_PATH


def encode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical columns and save encoders.
    """

    df = df.copy()

    categorical_columns = df.select_dtypes(include=["object"]).columns

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    joblib.dump(encoders, ENCODER_PATH)

    return df
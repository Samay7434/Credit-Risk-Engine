from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def prepare_features(df):
    """
    Separate features and target
    and create train/test split.
    """

    X = df.drop("Default", axis=1)

    y = df["Default"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
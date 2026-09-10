from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE


def train_model(X_train, y_train, params=None):
    """
    Train a LightGBM model.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    params : dict | None

    Returns
    -------
    LGBMClassifier
    """

    if params is None:
        params = {}

    model = LGBMClassifier(
    random_state=RANDOM_STATE,
    verbose=-1,

    class_weight="balanced",

    **params,
    )

    model.fit(X_train, y_train)

    return model
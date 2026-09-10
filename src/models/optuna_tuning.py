from src.data.loader import load_data
from src.data.data_preprocessing import clean_dataset

from src.utils.encoder import encode_dataset

from src.features.feature_engineering import prepare_features

from src.models.optimizer import optimize_model


def main():

    df = load_data()

    df = clean_dataset(df)

    df = encode_dataset(df)

    X_train, X_test, y_train, y_test = prepare_features(df)

    best_params = optimize_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("=" * 60)
    print("BEST PARAMETERS")
    print("=" * 60)

    print(best_params)


if __name__ == "__main__":
    main()
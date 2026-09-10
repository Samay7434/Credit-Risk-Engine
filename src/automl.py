import joblib
import mlflow
import mlflow.lightgbm

from src.config import MODEL_PATH

from src.data.loader import load_data
from src.data.data_preprocessing import clean_dataset

from src.utils.encoder import encode_dataset
from src.utils.metrics import evaluate_model

from src.features.feature_engineering import prepare_features

from src.models.optimizer import optimize_model
from src.models.trainer import train_model


def main():

    print("=" * 60)
    print("AUTOML PIPELINE")
    print("=" * 60)

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

    model = train_model(
        X_train,
        y_train,
        best_params,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    mlflow.set_experiment(
        "Credit Risk Engine"
    )

    with mlflow.start_run():

        mlflow.log_params(best_params)

        for k, v in metrics.items():

            if k != "confusion_matrix":
                mlflow.log_metric(k, v)

        mlflow.lightgbm.log_model(
            model,
            name="best_model",
        )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nTraining Completed.")

    print("\nMetrics")

    print(metrics)


if __name__ == "__main__":
    main()
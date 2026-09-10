import joblib
import mlflow
import mlflow.lightgbm

from src.config import MODEL_PATH

from src.data.loader import load_data
from src.data.data_preprocessing import clean_dataset

from src.utils.encoder import encode_dataset
from src.utils.metrics import evaluate_model

from src.features.feature_engineering import prepare_features

from src.models.trainer import train_model


def main():

    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)

    # Load
    df = load_data()

    # Clean
    df = clean_dataset(df)

    # Encode
    df = encode_dataset(df)

    # Split
    X_train, X_test, y_train, y_test = prepare_features(df)

    # Train
    model = train_model(
        X_train,
        y_train,
    )

    # Evaluate
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # MLflow
    mlflow.set_experiment("Credit Risk Engine")

    with mlflow.start_run():

        mlflow.log_param(
            "model",
            "LightGBM",
        )

        for metric_name, metric_value in metrics.items():

            if metric_name != "confusion_matrix":
                mlflow.log_metric(
                    metric_name,
                    metric_value,
                )

        mlflow.lightgbm.log_model(
            model,
            name="lightgbm_model",
        )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nMODEL PERFORMANCE\n")

    for key, value in metrics.items():

        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
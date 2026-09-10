import optuna

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

from src.config import RANDOM_STATE


def optimize_model(
    X_train,
    X_test,
    y_train,
    y_test,
    n_trials=30,
):

    def objective(trial):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves",
                20,
                150,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                15,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),

            "random_state": RANDOM_STATE,

            "verbose": -1,
        }

        model = LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
        )

        probability = model.predict_proba(
            X_test,
        )[:, 1]

        return roc_auc_score(
            y_test,
            probability,
        )

    study = optuna.create_study(
        direction="maximize",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    return study.best_params
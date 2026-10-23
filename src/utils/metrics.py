from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate probabilities at an explicit threshold (0.5 by default)."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "threshold": float(threshold),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ),
    }

    return metrics

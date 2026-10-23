"""Batch inference helpers and command-line entry point."""

import pandas as pd

from src.config import INPUT_DATA_PATH, OUTPUT_DATA_PATH
from src.models.predictor import CreditRiskPredictor
from src.utils.decision_engine import loan_decision
from src.utils.risk_category import risk_category
from src.utils.risk_score import calculate_risk_score


def score_applications(frame: pd.DataFrame, predictor=None) -> pd.DataFrame:
    """Score applications; optional identifiers and observed labels are never model inputs."""
    if frame.empty:
        raise ValueError("The uploaded CSV contains no applications.")
    predictors = predictor or CreditRiskPredictor()
    expected = predictors.feature_names
    missing = [name for name in expected if name not in frame.columns]
    metadata_columns = {"LoanID", "Default"}
    extra = [name for name in frame.columns if name not in expected and name not in metadata_columns]
    if missing or extra:
        bits = []
        if missing:
            bits.append("missing columns: " + ", ".join(missing))
        if extra:
            bits.append("unexpected columns: " + ", ".join(extra))
        raise ValueError("Invalid CSV schema (" + "; ".join(bits) + ").")

    if "Default" in frame:
        observed = pd.to_numeric(frame["Default"], errors="coerce")
        if observed.isna().any() or not observed.isin([0, 1]).all():
            raise ValueError("The optional Default column must contain only 0 or 1 values.")

    results = frame.copy()
    probabilities = []
    for index, row in results[expected].iterrows():
        try:
            _, probability = predictors.predict(row.to_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid application at row {index + 1}: {exc}") from exc
        probabilities.append(probability)

    results["DefaultProbability"] = probabilities
    results["RiskScore"] = results["DefaultProbability"].map(calculate_risk_score)
    results["RiskCategory"] = results["RiskScore"].map(risk_category)
    results["LoanDecision"] = results["RiskScore"].map(loan_decision)
    return results


def main():
    applications = pd.read_csv(INPUT_DATA_PATH)
    result = score_applications(applications)
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Scored {len(result):,} applications to {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    main()

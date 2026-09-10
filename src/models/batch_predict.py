import pandas as pd

from src.models.predictor import CreditRiskPredictor

from src.utils.risk_score import calculate_risk_score
from src.utils.decision_engine import loan_decision

from src.config import (
    INPUT_DATA_PATH,
    OUTPUT_DATA_PATH,
)

predictor = CreditRiskPredictor()


def main():

    df = pd.read_csv(INPUT_DATA_PATH)

    probabilities = []
    scores = []
    decisions = []

    for _, row in df.iterrows():

        customer = row.to_dict()

        prediction, probability = predictor.predict(customer)

        risk_score = calculate_risk_score(probability)

        decision = loan_decision(risk_score)

        probabilities.append(probability)

        scores.append(risk_score)

        decisions.append(decision)

    df["DefaultProbability"] = probabilities

    from src.utils.risk_category import risk_category

    df["RiskScore"] = scores

    df["RiskCategory"] = [

        risk_category(score)

        for score in scores

    ]

    df["LoanDecision"] = decisions

    df.to_csv(
        OUTPUT_DATA_PATH,
        index=False,
    )

    print("Underwriting report generated successfully.")

if __name__ == "__main__":
    main()
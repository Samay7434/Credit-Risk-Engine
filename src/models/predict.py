from src.models.predictor import CreditRiskPredictor
from src.utils.risk_score import calculate_risk_score
from src.utils.decision_engine import loan_decision


def main():

    predictor = CreditRiskPredictor()

    customer = {

        "Age":35,
        "Income":65000,
        "LoanAmount":18000,
        "CreditScore":720,
        "MonthsEmployed":60,
        "NumCreditLines":4,
        "InterestRate":9.5,
        "LoanTerm":36,
        "DTIRatio":0.32,
        "Education":"Bachelor's",
        "EmploymentType":"Full-time",
        "MaritalStatus":"Married",
        "HasMortgage":"Yes",
        "HasDependents":"No",
        "LoanPurpose":"Home",
        "HasCoSigner":"Yes",
    }

    prediction, probability = predictor.predict(
        customer
    )

    risk_score = calculate_risk_score(probability)

    decision = loan_decision(risk_score)

    print("=" * 60)
    print("CREDIT UNDERWRITING REPORT")
    print("=" * 60)

    print(f"Probability of Default : {probability:.2%}")
    print(f"Risk Score            : {risk_score}/1000")
    print(f"Loan Decision         : {decision}")

if __name__ == "__main__":
    main()
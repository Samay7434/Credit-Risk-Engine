import pandas as pd

from src.config import OUTPUT_DATA_PATH


def portfolio_summary():

    df = pd.read_csv(OUTPUT_DATA_PATH)

    total = len(df)

    approved = (df["LoanDecision"] == "Approved").sum()

    review = (df["LoanDecision"] == "Manual Review").sum()

    rejected = (df["LoanDecision"] == "Rejected").sum()

    avg_probability = df["DefaultProbability"].mean()

    avg_score = df["RiskScore"].mean()

    print("=" * 70)
    print("LOAN PORTFOLIO ANALYTICS")
    print("=" * 70)

    print(f"Total Applications      : {total}")
    print(f"Approved                : {approved}")
    print(f"Manual Review           : {review}")
    print(f"Rejected                : {rejected}")

    print()

    print(f"Average Default Prob.   : {avg_probability:.2%}")
    print(f"Average Risk Score      : {avg_score:.2f}")

    print()

    print("=" * 70)
    print("DECISION DISTRIBUTION")
    print("=" * 70)

    print(df["LoanDecision"].value_counts())


if __name__ == "__main__":
    portfolio_summary()
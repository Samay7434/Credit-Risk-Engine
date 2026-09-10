from src.utils.risk_category import risk_category


def generate_report(customer,
                    probability,
                    risk_score,
                    decision):

    report = {}

    report["ApplicationID"] = "CR-2026-000001"

    report["Probability"] = round(
        probability * 100,
        2,
    )

    report["RiskScore"] = risk_score

    report["RiskCategory"] = risk_category(
        risk_score
    )

    report["Decision"] = decision

    approval = []

    risks = []

    policy = []

    # -------------------------
    # Credit Score
    # -------------------------

    if customer["CreditScore"] >= 700:

        approval.append(
            f"Excellent Credit Score ({customer['CreditScore']})"
        )

        policy.append("✔ Credit Score Passed")

    elif customer["CreditScore"] >= 600:

        approval.append(
            f"Good Credit Score ({customer['CreditScore']})"
        )

        policy.append("✔ Credit Score Passed")

    else:

        risks.append(
            f"Low Credit Score ({customer['CreditScore']})"
        )

        policy.append("❌ Credit Score Failed")

    # -------------------------
    # DTI
    # -------------------------

    if customer["DTIRatio"] <= 0.35:

        approval.append(
            f"Healthy Debt-to-Income Ratio ({customer['DTIRatio']:.0%})"
        )

        policy.append("✔ DTI Passed")

    else:

        risks.append(
            f"High Debt-to-Income Ratio ({customer['DTIRatio']:.0%})"
        )

        policy.append("❌ DTI Failed")

    # -------------------------
    # Employment
    # -------------------------

    if customer["MonthsEmployed"] >= 24:

        approval.append(
            f"Stable Employment ({customer['MonthsEmployed']} months)"
        )

    else:

        risks.append(
            "Limited Employment History"
        )

    # -------------------------
    # Loan Amount
    # -------------------------

    if customer["LoanAmount"] > 30000:

        risks.append(
            f"Large Loan Amount (${customer['LoanAmount']:,})"
        )

    else:

        policy.append("✔ Loan Amount Passed")

    # -------------------------
    # Interest Rate
    # -------------------------

    if customer["InterestRate"] > 10:

        risks.append(
            f"High Interest Rate ({customer['InterestRate']}%)"
        )

    report["ApprovalFactors"] = approval

    report["RiskFactors"] = risks

    report["PolicyChecks"] = policy

    return report
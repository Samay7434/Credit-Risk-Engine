def loan_decision(risk_score: int) -> str:

    if risk_score < 300:
        return "Approved"

    elif risk_score < 600:
        return "Manual Review"

    return "Rejected"
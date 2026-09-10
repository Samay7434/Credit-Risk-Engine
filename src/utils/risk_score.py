def calculate_risk_score(default_probability: float) -> int:
    """
    Convert default probability into a 0-1000 risk score.
    """

    return round(default_probability * 1000)
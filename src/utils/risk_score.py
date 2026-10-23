def calculate_risk_score(default_probability: float) -> int:
    """
    Convert default probability into a 0-1000 risk score.
    """

    probability = float(default_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("default_probability must be between 0 and 1")
    return round(probability * 1000)

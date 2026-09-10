def risk_category(score):

    if score < 300:
        return "Low"

    elif score < 700:
        return "Medium"

    return "High"
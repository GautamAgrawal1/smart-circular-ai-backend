def decide_action(condition_score, remaining_life_months, price):
    """
    Decide whether to Reuse, Repair, or Recycle
    """

    # High-quality product → Reuse
    if condition_score >= 0.7 and remaining_life_months >= 12:
        return {
            "action": "Reuse",
            "reason": "Good condition and long remaining life"
        }

    # Medium-quality product → Repair
    if 0.4 <= condition_score < 0.7 and remaining_life_months >= 6:
        return {
            "action": "Repair",
            "reason": "Moderate condition with usable remaining life"
        }

    # Low-quality or near end-of-life → Recycle
    return {
        "action": "Recycle",
        "reason": "Low condition or insufficient remaining life"
    }

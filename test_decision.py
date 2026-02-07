from decision_engine import decide_action

# Example inputs (from previous phases)
condition_score = 0.33
remaining_life_months = 8.4
recommended_price = 6200

decision = decide_action(
    condition_score,
    remaining_life_months,
    recommended_price
)

print("Decision:", decision["action"])
print("Reason:", decision["reason"])

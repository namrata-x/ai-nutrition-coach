def verify_results(agent_calories: int, classifier_range: str) -> dict:

    if agent_calories < 400:
        agent_range = "low"
    elif agent_calories <= 700:
        agent_range = "medium"
    else:
        agent_range = "high"

    if agent_range == classifier_range:
        return {
            "status": "verified",
            "confidence": "high",
            "agent_range": agent_range,
            "classifier_range": classifier_range
        }

    adjacent_pairs = [{"low", "medium"}, {"medium", "high"}]
    if {agent_range, classifier_range} in adjacent_pairs:
        return {
            "status": "minor_disagreement",
            "confidence": "medium",
            "agent_range": agent_range,
            "classifier_range": classifier_range,
            "note": "Estimate near a range boundary"
        }

    return {
        "status": "disagreement",
        "confidence": "low",
        "agent_range": agent_range,
        "classifier_range": classifier_range,
        "note": "Results inconsistent — please review meal description"
    }

from src.classification.labels import Classification
from src.classification.rule_detector import detect_by_rules
from src.classification.pii_detector import detect_pii
from src.classification.llm_scorer import score_chunk


def classify_chunk(text: str) -> dict:
    """
    Combine rule-based, PII, and LLM classification results.

    Final classification is the most restrictive detector result.
    Final score is the highest severity score from any detector.
    """

    rule_result = detect_by_rules(text)
    pii_result = detect_pii(text)
    llm_result = score_chunk(text)

    final_classification = max(
        rule_result["classification"],
        pii_result["classification"],
        llm_result["classification"],
    )

    final_score = max(
        rule_result["score"],
        pii_result["score"],
        llm_result["score"],
    )

    reasons = []

    if rule_result["findings"]:
        reasons.extend(
            finding["reason"]
            for finding in rule_result["findings"]
        )

    if pii_result["findings"]:
        reasons.extend(
            finding["reason"]
            for finding in pii_result["findings"]
        )

    if llm_result["reason"]:
        reasons.append(
            f"LLM assessment: {llm_result['reason']}"
        )

    return {
        "score": final_score,
        "classification": final_classification,
        "reason": " ".join(reasons),
        "rule_result": rule_result,
        "pii_result": pii_result,
        "llm_result": llm_result,
    }


def classify_chunk_metadata(text: str) -> dict:
    """
    Return classification fields ready to attach to chunk metadata.
    """

    result = classify_chunk(text)

    return {
        "classification": result["classification"].name,
        "classification_score": result["score"],
        "classification_reason": result["reason"],
    }


if __name__ == "__main__":
    samples = [
        "The employee salary report contains monthly compensation.",
        "Employees must submit leave requests through the internal portal.",
        "Our office is open Monday through Friday.",
        "Contact the employee at vishal@example.com.",
        "The production API key must remain secret.",
    ]

    for sample in samples:
        result = classify_chunk(sample)

        print("\nDocument:", sample)
        print("Final score:", result["score"])
        print("Final classification:",
              result["classification"].name)
        print("Reason:", result["reason"])
        print("Rule matches:",
              result["rule_result"]["matches"])
        print("PII matches:",
              result["pii_result"]["matches"])
        print("Metadata:",
              classify_chunk_metadata(sample))

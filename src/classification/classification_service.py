
import time

from src.classification.labels import Classification
from src.classification.rule_detector import detect_by_rules
from src.classification.pii_detector import detect_pii
from src.classification.llm_scorer import score_chunk


def classify_chunk(text: str) -> dict:
    """
    Combine rule-based, PII, and Grok LLM classification results.

    Final classification is the most restrictive detector result.
    Final score is the highest severity score from any detector.
    """

    start_time = time.perf_counter()

    # Step 1: Rule-based classification
    rule_result = detect_by_rules(text)

    # Step 2: PII detection using spaCy and configured patterns
    pii_result = detect_pii(text)

    # Step 3: Contextual classification using Grok
    llm_result = score_chunk(text)

    # Step 4: Select the most restrictive classification
    final_classification = max(
        rule_result["classification"],
        pii_result["classification"],
        llm_result["classification"],
    )

    # Step 5: Select the highest severity score
    final_score = max(
        rule_result["score"],
        pii_result["score"],
        llm_result["score"],
    )

    # Step 6: Combine reasons from all detectors
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

    # Step 7: Calculate total classification time
    classification_time = time.perf_counter() - start_time

    return {
        "score": final_score,
        "classification": final_classification,
        "reason": " ".join(reasons),
        "rule_result": rule_result,
        "pii_result": pii_result,
        "llm_result": llm_result,
        "classification_time_seconds": round(
            classification_time, 3
        ),
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

        # Run the complete classification pipeline only once
        result = classify_chunk(sample)

        # Build metadata from the existing result.
        # This avoids making another Grok API request.
        metadata = {
            "classification": result["classification"].name,
            "classification_score": result["score"],
            "classification_reason": result["reason"],
        }

        print("\n" + "=" * 70)
        print("Document:", sample)

        print("\nFinal classification:")
        print("Score:", result["score"])
        print("Label:", result["classification"].name)
        print("Reason:", result["reason"])

        print("\nIndividual detector results:")

        print("Rule detector:")
        print("Score:", result["rule_result"]["score"])
        print(
            "Classification:",
            result["rule_result"]["classification"].name
        )
        print("Matches:", result["rule_result"]["matches"])

        print("\nPII detector:")
        print("Score:", result["pii_result"]["score"])
        print(
            "Classification:",
            result["pii_result"]["classification"].name
        )
        print("Matches:", result["pii_result"]["matches"])

        print("\nGrok LLM detector:")
        print("Score:", result["llm_result"]["score"])
        print(
            "Classification:",
            result["llm_result"]["classification"].name
        )
        print("Reason:", result["llm_result"]["reason"])
        print("Confidence:", result["llm_result"]["confidence"])

        print("\nClassification time:")
        print(
            result["classification_time_seconds"],
            "seconds"
        )

        print("\nFinal metadata:")
        print(metadata)

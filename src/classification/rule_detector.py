
import re

from src.classification.labels import Classification


RULES = {
    "salary": {
        "pattern": r"\b(salary|payroll|compensation|annual\s+pay)\b",
        "classification": Classification.CONFIDENTIAL,
        "score": 85,
        "reason": "Salary, payroll, or compensation information detected.",
    },
    "financial": {
        "pattern": (
            r"\b(bank\s+account|credit\s+card|financial\s+statement)\b"
        ),
        "classification": Classification.CONFIDENTIAL,
        "score": 90,
        "reason": "Sensitive financial information detected.",
    },
    "credentials": {
        "pattern": (
            r"\b(password|secret\s+key|api\s+key|access\s+token)\b"
        ),
        "classification": Classification.CONFIDENTIAL,
        "score": 100,
        "reason": "Potential credential or authentication secret detected.",
    },
    "confidential_label": {
        "pattern": (
            r"\b(strictly\s+confidential|highly\s+confidential)\b"
        ),
        "classification": Classification.CONFIDENTIAL,
        "score": 80,
        "reason": "The document contains a strong confidentiality label.",
    },
    "internal_label": {
        "pattern": (
            r"\b(internal\s+use\s+only|employee\s+only|proprietary)\b"
        ),
        "classification": Classification.INTERNAL,
        "score": 40,
        "reason": "The document is marked for internal use.",
    },
    "hr": {
        "pattern": (
            r"\b(performance\s+review|employee\s+evaluation|"
            r"disciplinary\s+action)\b"
        ),
        "classification": Classification.CONFIDENTIAL,
        "score": 80,
        "reason": "Sensitive employee or HR information detected.",
    },
    "business": {
        "pattern": (
            r"\b(internal\s+report|business\s+strategy|"
            r"unreleased\s+product)\b"
        ),
        "classification": Classification.INTERNAL,
        "score": 45,
        "reason": "Non-public business information detected.",
    },
}


CREDENTIAL_VALUE_PATTERNS = [
    r"\b(password|passwd|api[_ -]?key|secret[_ -]?key|"
    r"access[_ -]?token|auth[_ -]?token)\b"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})",

    r"\b(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16})\b",
]


# Require contextual evidence of actual compensation data.
SALARY_VALUE_PATTERNS = [
    r"\b(?:salary|compensation|annual\s+pay)\b"
    r"\s*(?:is|was|:|=)\s*\$?\d[\d,]*(?:\.\d{2})?\b",
    r"\b(?:paid|earns?|earned)\s+\$?\d[\d,]*(?:\.\d{2})?\b",
    r"\b(?:monthly|annual|yearly)\s+(?:salary|compensation|pay)"
    r"\s*(?:of|is|:|=)\s*\$?\d[\d,]*(?:\.\d{2})?\b",
]


def contains_actual_credential(text: str) -> bool:
    """Return True when text appears to contain a credential value."""
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in CREDENTIAL_VALUE_PATTERNS
    )


def contains_salary_value(text: str) -> bool:
    """Return True when text appears to disclose an actual salary amount."""
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in SALARY_VALUE_PATTERNS
    )


def detect_by_rules(text: str) -> dict:
    """
    Detect sensitivity indicators using predefined rules.

    Actual credential values receive score 100.
    Actual salary amounts receive score 85.
    General credential mentions receive score 40.
    General salary mentions are recorded but not automatically classified
    as confidential.
    """

    matches = []
    findings = []

    actual_credential = contains_actual_credential(text)
    actual_salary = contains_salary_value(text)

    for category, rule in RULES.items():
        if not re.search(rule["pattern"], text, re.IGNORECASE):
            continue

        matches.append(category)

        if category == "credentials":
            if actual_credential:
                classification = Classification.CONFIDENTIAL
                score = 100
                reason = (
                    "A potential exposed credential value was detected."
                )
            else:
                classification = Classification.INTERNAL
                score = 40
                reason = (
                    "The document discusses credentials but no actual "
                    "credential value was detected."
                )

        elif category == "salary":
            if actual_salary:
                classification = Classification.CONFIDENTIAL
                score = 85
                reason = (
                    "An actual salary or compensation amount was detected."
                )
            else:
                # Mention alone is not sufficient to classify as confidential.
                continue

        else:
            classification = rule["classification"]
            score = rule["score"]
            reason = rule["reason"]

        findings.append({
            "category": category,
            "classification": classification,
            "score": score,
            "reason": reason,
        })

    if not findings:
        return {
            "classification": Classification.PUBLIC,
            "score": 0,
            "matches": matches,
            "findings": [],
        }

    classification = max(
        finding["classification"]
        for finding in findings
    )

    score = max(
        finding["score"]
        for finding in findings
    )

    return {
        "classification": classification,
        "score": score,
        "matches": matches,
        "findings": findings,
    }


if __name__ == "__main__":
    samples = [
        "The production API key must remain secret.",
        "Never share your API key with anyone.",
        'api_key = "sk-123456789012345678901234"',
        "This internal use only document contains employee salary.",
        "The employee salary is $85,000 annually.",
        "This article explains why companies should protect employee salary information.",
        "Our office is open Monday through Friday.",
    ]

    for sample in samples:
        result = detect_by_rules(sample)

        print("\nDocument:", sample)
        print("Classification:", result["classification"].name)
        print("Score:", result["score"])
        print("Matched rules:", result["matches"])
        print("Findings:", result["findings"])

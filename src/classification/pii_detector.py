
import re

import spacy

from src.classification.labels import Classification


# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


# Structured PII patterns
PII_PATTERNS = {
    "email": (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": (
        r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?[6-9]\d{9}(?!\w)"
    ),
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
}


# Known organization, product, and document terms
NON_PERSON_ENTITIES = {
    "fluid intelligence",
    "knowledge base",
}


# Domains reserved for examples and documentation
PLACEHOLDER_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "test.com",
    "invalid",
}


PUBLIC_CONTEXT = (
    "public contact",
    "contact us",
    "customer support",
    "support email",
    "sales email",
    "official website",
    "publicly available",
    "published contact",
    "official contact",
)

CONFIDENTIAL_CONTEXT = (
    "private email",
    "personal email",
    "private phone",
    "personal phone",
    "confidential",
    "personal details",
    "private details",
)

INTERNAL_CONTEXT = (
    "employee email",
    "employee phone",
    "internal contact",
    "staff contact",
)


def get_context(text: str, start: int, end: int) -> str:
    """Return surrounding text to help interpret a detected entity."""

    left = max(0, start - 100)
    right = min(len(text), end + 100)

    return text[left:right].lower()


def is_placeholder_email(email: str) -> bool:
    """Check whether an email uses a reserved example domain."""

    domain = email.rsplit("@", 1)[-1].lower()

    return (
        domain in PLACEHOLDER_EMAIL_DOMAINS
        or domain.endswith(".example")
    )


def classify_contact(context: str) -> tuple:
    """Classify contact information using nearby context."""

    if any(
        keyword in context
        for keyword in CONFIDENTIAL_CONTEXT
    ):
        return Classification.CONFIDENTIAL, 85

    if any(
        keyword in context
        for keyword in INTERNAL_CONTEXT
    ):
        return Classification.INTERNAL, 40

    if any(
        keyword in context
        for keyword in PUBLIC_CONTEXT
    ):
        return Classification.PUBLIC, 5

    return Classification.INTERNAL, 40


def detect_pii(text: str) -> dict:
    """Detect common PII patterns and named persons."""

    matches = []
    findings = []

    # Step 1: Detect structured PII using regular expressions
    for category, pattern in PII_PATTERNS.items():

        for match in re.finditer(pattern, text, re.IGNORECASE):

            value = match.group()

            context = get_context(
                text,
                match.start(),
                match.end(),
            )

            # PAN and Aadhaar are always confidential
            if category in ("pan", "aadhaar"):

                classification = Classification.CONFIDENTIAL
                score = 95

                reason = (
                    f"Detected {category.upper()}, a sensitive "
                    "personal identifier."
                )

            # Placeholder emails are not treated as real PII
            elif (
                category == "email"
                and is_placeholder_email(value)
            ):

                classification = Classification.PUBLIC
                score = 0

                reason = (
                    "Detected an example or placeholder email "
                    "address, not treated as real personal information."
                )

            else:

                classification, score = classify_contact(context)

                reason = (
                    f"Detected {category}. "
                    "Classification based on surrounding context."
                )

            matches.append(category)

            findings.append({
                "category": category,
                "value": value,
                "classification": classification,
                "score": score,
                "reason": reason,
            })

    # Step 2: Detect person names using spaCy NER
    doc = nlp(text)

    persons = []

    for entity in doc.ents:

        if entity.label_ != "PERSON":
            continue

        person = entity.text.strip()
        normalized_person = person.casefold()

        # Ignore known organization/product/document terms
        if normalized_person in NON_PERSON_ENTITIES:
            continue

        if not person:
            continue

        if person not in persons:

            persons.append(person)
            matches.append("person_name")

            findings.append({
                "category": "person_name",
                "value": person,
                "classification": Classification.INTERNAL,
                "score": 40,
                "reason": (
                    "A person's name was detected. "
                    "Names alone are treated as INTERNAL."
                ),
            })

    # Step 3: Determine the most restrictive result
    if findings:

        classification = max(
            finding["classification"]
            for finding in findings
        )

        score = max(
            finding["score"]
            for finding in findings
        )

    else:

        classification = Classification.PUBLIC
        score = 0

    return {
        "classification": classification,
        "score": score,
        "matches": list(dict.fromkeys(matches)),
        "persons": persons,
        "findings": findings,
    }


if __name__ == "__main__":

    samples = [
        "Fluid Intelligence — Knowledge Base",

        "Section: Fluid Intelligence — Knowledge Base",

        "Contact us at support@company.com for assistance.",

        "Employee email: vishal@company.com",

        "Contact the employee at vishal@example.com.",

        "The employee's private email is vishal@gmail.com.",

        "PAN number: ABCDE1234F",

        "Our office is open Monday through Friday.",

    ]

    for sample in samples:

        result = detect_pii(sample)

        print("\n" + "=" * 70)
        print("Document:", sample)

        print(
            "Classification:",
            result["classification"].name
        )

        print("Score:", result["score"])
        print("Matches:", result["matches"])
        print("Persons:", result["persons"])
        print("Findings:", result["findings"])

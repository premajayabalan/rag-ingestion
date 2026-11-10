"""DeepEval metric configuration."""
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)


def build_metrics(threshold: float = 0.7):
    """Create standard RAG metrics. DeepEval's configured judge model is used."""
    return [
        AnswerRelevancyMetric(threshold=threshold, include_reason=True),
        FaithfulnessMetric(threshold=threshold, include_reason=True),
        ContextualPrecisionMetric(threshold=threshold, include_reason=True),
        ContextualRecallMetric(threshold=threshold, include_reason=True),
        HallucinationMetric(threshold=threshold, include_reason=True),
    ]

def retrieval_precision_at_k(retrieved_titles: list[str], relevant_titles: list[str], k: int | None = None) -> float:
    """Fraction of the top-k retrieved chunks that come from a relevant document."""
    top = retrieved_titles[:k] if k else retrieved_titles
    if not top:
        return 0.0
    rel = {t.lower() for t in relevant_titles}
    return sum(1 for t in top if t.lower() in rel) / len(top)


def detection_rate(detected_flags: list[bool]) -> float:
    return sum(detected_flags) / len(detected_flags) if detected_flags else 0.0


def false_positive_rate(control_detected_flags: list[bool]) -> float:
    """Non-entailment rate on unchanged control sentences."""
    return detection_rate(control_detected_flags)

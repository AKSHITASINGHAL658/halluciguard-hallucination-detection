from app import config
from app.core.sentence_localizer import is_abstention, is_filler, localize_hallucinations
from app.core.span_checker import find_wrong_spans


def verify_sentences(sentences: list[str], chunks: list[dict], use_claims: bool | None = None,
                     use_spans: bool = True, threshold: float | None = None) -> list[dict]:
    """The full sentence check used by /query: NLI (optionally per atomic claim) + wrong numbers/names + filler.

    Adds to each localize_hallucinations() dict: is_filler, is_abstention, nli_flagged (NLI rejection before
    overrides), wrong_spans. Filler is never flagged; an abstention ("not in the provided context") is neutral:
    never flagged, never supported. Any wrong span flags the sentence and switches its source to the passage the
    sentence is about.
    """
    out = localize_hallucinations(sentences, chunks, use_claims=use_claims, threshold=threshold)
    for s in out:
        s["is_filler"] = is_filler(s["text"])
        s["is_abstention"] = not s["is_filler"] and is_abstention(s["text"])
        if s["is_filler"]:  # no checkable claim: never a hallucination
            s["is_hallucination"] = False
        if s["is_abstention"]:  # says the answer isn't in the documents: neutral, not backed by them
            s["is_hallucination"], s["label"] = False, "neutral"
        s["nli_flagged"] = s["is_hallucination"]
        spans, ref = ([], None) if (s["is_filler"] or s["is_abstention"] or not use_spans) else find_wrong_spans(
            s["text"], chunks, s["best_chunk_id"], s["max_entailment"])
        s["wrong_spans"] = spans
        if spans and ref:
            s["is_hallucination"] = True
            s["best_chunk_id"], s["best_chunk_text"] = ref["chunk_id"], ref["text"]
    return out

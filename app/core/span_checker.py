import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.core.trust_engine import _nlp, _normalize_entity

# standalone numbers only: not part of a word ("COP28"), not a [1] citation, not followed by letters ("2030s")
NUM_RE = re.compile(r"(?<![\w.\[])\d+(?:,\d{3})*(?:\.\d+)?(?!\w)")
_WORD_RE = re.compile(r"[a-z]+")
NAME_TYPES = {"PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"}


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in NUM_RE.finditer(text)}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 2}


def pick_reference_chunk(sentence: str, chunks: list[dict], nli_best_id: int | None = None) -> dict | None:
    """Chunk the sentence is *about*: most shared non-numeric content words (ties -> the NLI best chunk).

    Chosen lexically rather than by NLI entailment, because NLI can prefer a loosely related chunk.
    """
    if not chunks:
        return None
    words = _content_words(sentence)
    return max(chunks, key=lambda c: (len(words & _content_words(c["text"])), c["chunk_id"] == nli_best_id))


def find_wrong_spans(sentence: str, chunks: list[dict], nli_best_id: int | None = None) -> tuple[list[dict], dict | None]:
    """Spans of the sentence that its source does not back up.

    - numbers (incl. years) missing from the reference chunk
    - named entities (people, orgs, places...) missing from every surviving chunk
    Returns ([{text, start, end, kind, reason}], reference_chunk).
    """
    ref = pick_reference_chunk(sentence, chunks, nli_best_id)
    if ref is None:
        return [], None
    ref_nums = _numbers(ref["text"])
    all_text = " ".join(c["text"] for c in chunks).lower()
    spans = []
    for m in NUM_RE.finditer(sentence):
        if m.group(0).replace(",", "") not in ref_nums:
            spans.append({"text": m.group(0), "start": m.start(), "end": m.end(), "kind": "number",
                          "reason": "number not found in the source passage"})
    for e in _nlp()(sentence).ents:
        if e.label_ not in NAME_TYPES:
            continue
        name = _normalize_entity(e.text) or e.text
        if name.lower() not in all_text:
            spans.append({"text": e.text, "start": e.start_char, "end": e.end_char, "kind": "entity",
                          "reason": "name not found in any source passage"})
    return sorted(spans, key=lambda s: s["start"]), ref

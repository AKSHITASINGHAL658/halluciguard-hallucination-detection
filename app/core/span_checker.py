import math
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app import config
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
    """Chunk the sentence is *about*, chosen by meaning rather than NLI entailment (NLI can prefer a loosely
    related chunk). Score = embedding cosine of the number-free sentence + a small boost for how often the
    sentence's content words occur in the chunk; remaining ties -> the NLI best chunk.
    """
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    import numpy as np

    from app.core.embeddings import embed_texts

    words = _content_words(sentence)
    stripped = re.sub(r"\d[\d,.]*", " ", sentence)
    vecs = embed_texts([stripped] + [c["text"] for c in chunks])
    sent_vec = np.asarray(vecs[0])

    def score(i: int, c: dict) -> tuple:
        cos = float(sent_vec @ np.asarray(vecs[i + 1]))
        tokens = _WORD_RE.findall(c["text"].lower())
        freq = sum(math.log1p(tokens.count(w)) for w in words) / max(1, len(words))
        return (cos + 0.03 * freq, c["chunk_id"] == nli_best_id)

    return max(enumerate(chunks), key=lambda ic: score(*ic))[1]


def find_wrong_spans(sentence: str, chunks: list[dict], nli_best_id: int | None = None,
                     nli_entailment: float = 0.0) -> tuple[list[dict], dict | None]:
    """Spans of the sentence that its source does not back up.

    - numbers (incl. years) missing from the reference chunk (unless the NLI-preferred chunk has them and NLI
      entailment >= SPAN_STRONG_ENTAILMENT)
    - named entities (people, orgs, places...) missing from every surviving chunk
    Returns ([{text, start, end, kind, reason}], reference_chunk).
    """
    ref = pick_reference_chunk(sentence, chunks, nli_best_id)
    if ref is None:
        return [], None
    ref_nums = _numbers(ref["text"])
    if nli_best_id is not None and nli_entailment >= config.SPAN_STRONG_ENTAILMENT:
        for c in chunks:
            if c["chunk_id"] == nli_best_id:
                ref_nums |= _numbers(c["text"])
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

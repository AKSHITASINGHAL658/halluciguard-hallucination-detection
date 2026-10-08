import math
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load(config.SPACY_MODEL)


def entity_score(answer: str, chunks: list[dict]) -> tuple[float, list[dict]]:
    """Fraction of spaCy entities in the answer found (case-insensitive) in the surviving chunk text."""
    context = " ".join(c["text"] for c in chunks).lower()
    seen, entities = set(), []
    for e in _nlp()(answer).ents:
        key = (e.text.lower(), e.label_)
        if key in seen:
            continue
        seen.add(key)
        ok = e.text.lower() in context
        entities.append({"text": e.text, "entity_type": e.label_, "supported": ok, "score": 1.0 if ok else 0.0})
    if not entities:
        return 1.0, []
    return sum(e["supported"] for e in entities) / len(entities), entities


def sentence_score(sentence_results: list[dict]) -> float:
    """Mean max-entailment probability x (1 - hallucination ratio)."""
    if not sentence_results:
        return 0.0
    mean_ent = sum(s["max_entailment"] for s in sentence_results) / len(sentence_results)
    ratio = sum(s["is_hallucination"] for s in sentence_results) / len(sentence_results)
    return mean_ent * (1 - ratio)


def doc_score(sentence_results: list[dict], chunk_authority: dict[int, float]) -> float:
    """Mean authority of the best chunks of supported sentences; 0 if none supported."""
    vals = [chunk_authority.get(s["best_chunk_id"], config.DEFAULT_AUTHORITY)
            for s in sentence_results if not s["is_hallucination"]]
    return sum(vals) / len(vals) if vals else 0.0


def token_prob_component(token_logprobs: list[float] | None, avg_logprob: float | None) -> float | None:
    """Geometric-mean token probability, 0-1; None if no logprobs."""
    if token_logprobs:
        return math.exp(sum(token_logprobs) / len(token_logprobs))
    if avg_logprob is not None:
        return math.exp(avg_logprob)
    return None


def aggregate_trust(sentence: float, entity: float, document: float, token_prob: float | None) -> float:
    w = dict(config.TRUST_WEIGHTS)
    if token_prob is None:
        w.pop("token_prob")
    total = sum(w.values())
    parts = {"sentence": sentence, "entity": entity, "document": document, "token_prob": token_prob}
    return sum(w[k] / total * parts[k] for k in w)

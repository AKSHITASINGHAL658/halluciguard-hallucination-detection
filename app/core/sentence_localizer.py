import nltk

from app import config
from app.core.nli_verifier import batch_verify


def split_sentences(text: str) -> list[str]:
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if s.strip()]


def localize_hallucinations(sentences: list[str], chunks: list[dict]) -> list[dict]:
    """Run every sentence against every chunk; keep the chunk with the highest entailment prob.

    is_hallucination = max entailment < 0.5. Label = contradiction if contradiction > 0.5 against any chunk,
    else the best chunk's label.
    """
    if not sentences:
        return []
    pairs = [(s, c["text"]) for s in sentences for c in chunks]
    results = batch_verify(pairs) if pairs else []
    n_c = len(chunks)
    out = []
    for i, sent in enumerate(sentences):
        row = results[i * n_c:(i + 1) * n_c]
        if not row:
            out.append({"index": i, "text": sent, "label": "neutral", "confidence": 0.0, "probs": {},
                        "max_entailment": 0.0, "is_hallucination": True, "best_chunk_id": None,
                        "best_chunk_text": None})
            continue
        best = max(range(n_c), key=lambda j: row[j]["probs"]["entailment"])
        max_ent = row[best]["probs"]["entailment"]
        max_con = max(r["probs"]["contradiction"] for r in row)
        if max_con > config.CONTRADICTION_THRESHOLD:
            label = "contradiction"
            con_idx = max(range(n_c), key=lambda j: row[j]["probs"]["contradiction"])
            best = best if max_ent >= config.HALLUCINATION_ENTAIL_THRESHOLD else con_idx
            confidence = max_con
        else:
            label = row[best]["label"]
            confidence = row[best]["probs"][label]
        out.append({"index": i, "text": sent, "label": label, "confidence": confidence,
                    "probs": row[best]["probs"], "max_entailment": max_ent,
                    "is_hallucination": max_ent < config.HALLUCINATION_ENTAIL_THRESHOLD,
                    "best_chunk_id": chunks[best]["chunk_id"], "best_chunk_text": chunks[best]["text"]})
    return out

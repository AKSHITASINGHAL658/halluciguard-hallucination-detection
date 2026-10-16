import re

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


def _evaluate_unit(text: str, row: list[dict], chunks: list[dict]) -> dict:
    """Verdict for one claim/sentence from its NLI results against every chunk."""
    n_c = len(chunks)
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
    return {"claim": text, "label": label, "confidence": confidence, "probs": row[best]["probs"],
            "max_entailment": max_ent, "is_hallucination": max_ent < config.HALLUCINATION_ENTAIL_THRESHOLD,
            "best_chunk_id": chunks[best]["chunk_id"], "best_chunk_text": chunks[best]["text"]}


def localize_hallucinations(sentences: list[str], chunks: list[dict], use_claims: bool | None = None,
                            threshold: float | None = None) -> list[dict]:
    """Run every sentence (or each atomic claim of it) against every chunk; keep the chunk with the highest
    entailment prob.

    is_hallucination = max entailment < threshold (a sentence is flagged if any of its claims is). Label =
    contradiction if contradiction > 0.5 against any chunk, else the best chunk's label. With claim splitting
    the sentence takes the verdict of its weakest claim (lowest max entailment).
    """
    if not sentences:
        return []
    use_claims = config.USE_CLAIMS if use_claims is None else use_claims
    thr = config.HALLUCINATION_ENTAIL_THRESHOLD if threshold is None else threshold
    from app.core.claim_extractor import extract_claims

    units = []  # (sentence index, claim text)
    for i, sent in enumerate(sentences):
        for claim in (extract_claims(sent) if use_claims and chunks else [sent]):
            units.append((i, claim))
    n_c = len(chunks)
    results = batch_verify([(c, ch["text"]) for _, c in units for ch in chunks]) if n_c else []

    per_sentence: dict[int, list[dict]] = {i: [] for i in range(len(sentences))}
    for u, (i, claim) in enumerate(units):
        row = results[u * n_c:(u + 1) * n_c]
        if row:
            v = _evaluate_unit(claim, row, chunks)
            v["is_hallucination"] = v["max_entailment"] < thr
            per_sentence[i].append(v)
    out = []
    for i, sent in enumerate(sentences):
        vs = per_sentence[i]
        if not vs:
            out.append({"index": i, "text": sent, "label": "neutral", "confidence": 0.0, "probs": {},
                        "max_entailment": 0.0, "is_hallucination": True, "best_chunk_id": None,
                        "best_chunk_text": None, "claims": [sent], "weak_claim": sent})
            continue
        weakest = min(vs, key=lambda v: v["max_entailment"])
        label = "contradiction" if any(v["label"] == "contradiction" for v in vs) else weakest["label"]
        out.append({"index": i, "text": sent, "label": label, "confidence": weakest["confidence"],
                    "probs": weakest["probs"], "max_entailment": weakest["max_entailment"],
                    "is_hallucination": any(v["is_hallucination"] for v in vs),
                    "best_chunk_id": weakest["best_chunk_id"], "best_chunk_text": weakest["best_chunk_text"],
                    "claims": [v["claim"] for v in vs], "weak_claim": weakest["claim"]})
    return out


_REF_WORDS = re.compile(r"\b(context|passages?|sources?|provided text|text provided|documents?)\b", re.I)
_META_VERBS = re.compile(r"\b(stated|provided|mentioned|according|based on|found|given|described|supported|discussed|"
                         r"does not (?:contain|provide|mention|include)|do not (?:contain|provide|mention|include)|"
                         r"no information|not (?:available|specified))\b", re.I)


_CTX_WORDS = re.compile(r"\b(context|passages?|provided (?:text|context|information|documents?)|text provided|"
                        r"retrieved (?:text|documents?|passages?))\b", re.I)  # no bare "source": too common in real facts
_NOT_IN_CTX = re.compile(r"\b(?:not|never|n't)\b[^.]{0,40}\b(?:derived|contain(?:s|ed)?|mention(?:s|ed)?|stated?|found|"
                         r"includ(?:e|es|ed)|cover(?:s|ed)?|address(?:es|ed)?|support(?:s|ed)?|specif(?:y|ies|ied)|"
                         r"available|provid(?:e|es|ed))\b|\bno (?:information|mention)\b", re.I)


def is_abstention(sentence: str) -> bool:
    """A sentence saying the answer is NOT in the provided context ('"2+2=4" is not derived from the context').

    It says something about the documents, not a fact the documents back up, so it is shown as neutral. Short
    digit-free ones are already filler; this catches the rest (e.g. the question's own numbers quoted back).
    """
    plain = re.sub(r"\[\d+\]", "", sentence)
    return bool(_CTX_WORDS.search(plain) and _NOT_IN_CTX.search(plain))


def is_filler(sentence: str) -> bool:
    """Meta sentences like "This information is directly stated in the context provided." make no checkable claim."""
    plain = re.sub(r"\[\d+\]", "", sentence)
    if re.search(r"\d", plain) or len(plain.split()) > 25:
        return False
    return bool(_REF_WORDS.search(plain) and _META_VERBS.search(plain))

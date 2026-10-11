import re

from app.core.trust_engine import _nlp

MAX_CLAIMS = 4
MIN_CLAIM_WORDS = 4
_PRONOUNS = {"it", "they", "this", "that", "these", "those", "he", "she"}


def _join(tokens) -> str:
    return "".join(t.text_with_ws for t in sorted(tokens, key=lambda t: t.i)).strip()


def _clean(text: str) -> str:
    text = re.sub(r"\s+([.,;!?])", r"\1", text).strip(" ,;")
    text = text[:1].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def extract_claims(sentence: str) -> list[str]:
    """Split a sentence into atomic claims at coordinated clauses ("X did A, and Y did B").

    A clause without its own subject, or whose subject is a pronoun, borrows the main clause's subject so each
    claim can be checked on its own. Returns [sentence] when it cannot split safely.
    """
    doc = _nlp()(sentence)
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if not roots:
        return [sentence]
    root = roots[0]
    conj_verbs = [t for t in root.children if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX")]
    if not conj_verbs:
        return [sentence]
    subj = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None)
    subj_text = _join(subj.subtree) if subj is not None else ""

    drop = set()
    claims_tail = []
    for cv in conj_verbs[:MAX_CLAIMS - 1]:
        toks = list(cv.subtree)
        drop.update(t.i for t in toks)
        cc = [c for c in root.children if c.dep_ == "cc" and c.i < cv.i]
        if cc:
            drop.add(cc[-1].i)
        own = next((c for c in cv.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        text = _join(toks)
        if own is None and subj_text:
            text = f"{subj_text} {text}"
        elif own is not None and own.text.lower() in _PRONOUNS and own.pos_ == "PRON" and subj_text:
            text = text.replace(own.text, subj_text, 1)
        claims_tail.append(text)
    head = _join([t for t in doc if t.i not in drop and not (t.is_punct and t.text in ",;")])
    claims = [_clean(head)] + [_clean(c) for c in claims_tail]
    if any(len(c.split()) < MIN_CLAIM_WORDS for c in claims):
        return [sentence]
    return claims

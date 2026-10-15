import random
import re
from collections import defaultdict

import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app import config
from app.core.nli_verifier import batch_verify, get_nli_model

_WORD_RE = re.compile(r"\w+(?:[.,]\w+)*")


def extract_attention(sentence: str, chunk_text: str) -> list[dict]:
    """Attention mass from hypothesis (sentence) tokens onto premise (chunk) words, merged to words.

    DeBERTa uses self-attention over the concatenated premise+hypothesis (not cross-attention): these are
    the source words the verifier attended to most, not proof of causation.
    """
    tok, model = get_nli_model()
    enc = tok(chunk_text, sentence, truncation="only_first", max_length=config.NLI_MAX_LENGTH,
              return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids(0)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    if not out.attentions:
        return []
    layers = out.attentions[-config.ATTENTION_LAST_LAYERS:]
    attn = torch.stack([a[0] for a in layers]).mean(dim=(0, 1))  # (seq, seq): avg last layers and all heads
    hyp_rows = [i for i, s in enumerate(seq_ids) if s == 1]
    prem_cols = [i for i, s in enumerate(seq_ids) if s == 0]
    if not hyp_rows or not prem_cols:
        return []
    col_weight = attn[hyp_rows][:, prem_cols].sum(dim=0).tolist()  # sum over hypothesis rows

    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(chunk_text)]
    word_w: dict[int, float] = defaultdict(float)
    for col, w in zip(prem_cols, col_weight):
        start, end = offsets[col]
        if end <= start:
            continue
        mid = (start + end - 1) // 1
        for wi, (ws, we) in enumerate(spans):
            if ws <= mid < we or ws <= start < we:
                word_w[wi] += w  # merge sub-word pieces into the containing word
                break
    results = []
    for wi, w in word_w.items():
        word = chunk_text[spans[wi][0]:spans[wi][1]]
        if word.lower() in ENGLISH_STOP_WORDS or not any(ch.isalnum() for ch in word):
            continue
        results.append({"token": word, "weight": w})
    return sorted(results, key=lambda r: r["weight"], reverse=True)


def top_contributing_tokens(sentence: str, chunk_text: str, n: int = config.ATTENTION_TOP_N) -> list[dict]:
    top = extract_attention(sentence, chunk_text)[:n]
    total = sum(t["weight"] for t in top) or 1.0
    return [{"token": t["token"], "weight": round(t["weight"] / total, 4)} for t in top]


def _delete_words(text: str, words: list[str]) -> str:
    for w in words:
        text = re.sub(r"(?<!\w)" + re.escape(w) + r"(?!\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tvd(p: dict, q: dict) -> float:
    """Total variation distance between two label distributions (0 = identical, 1 = completely different)."""
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def check_attribution(sentence: str, chunk_text: str, top_tokens: list[dict], n_random: int | None = None,
                      seed: int = 0) -> dict | None:
    """Mask-and-rerun check: do the top attention words really matter to the verifier?

    Deletes the top words from the source chunk and re-runs NLI; compares how far the label probabilities move
    with how far they move when the same number of random content words is deleted (control, averaged over
    n_random draws). supported = top shift beats the random shift by ATTENTION_CHECK_MARGIN.
    """
    words = [t["token"] for t in top_tokens]
    if not words:
        return None
    n_random = config.ATTENTION_CHECK_RANDOM if n_random is None else n_random
    pool = sorted({m.group(0) for m in _WORD_RE.finditer(chunk_text)
                   if m.group(0).lower() not in ENGLISH_STOP_WORDS and any(c.isalnum() for c in m.group(0))} - set(words))
    rng = random.Random(seed)
    random_sets = [rng.sample(pool, min(len(words), len(pool))) for _ in range(n_random)] if pool else []
    variants = [chunk_text, _delete_words(chunk_text, words)] + [_delete_words(chunk_text, ws) for ws in random_sets]
    res = batch_verify([(sentence, v) for v in variants])
    base, top = res[0]["probs"], res[1]["probs"]
    top_shift = _tvd(base, top)
    rand_shifts = [_tvd(base, r["probs"]) for r in res[2:]]
    rand_shift = sum(rand_shifts) / len(rand_shifts) if rand_shifts else 0.0
    return {"masked_words": words, "base_entailment": base["entailment"], "masked_entailment": top["entailment"],
            "top_shift": top_shift, "random_shift": rand_shift, "n_random": len(rand_shifts),
            "supported": top_shift > rand_shift + config.ATTENTION_CHECK_MARGIN}

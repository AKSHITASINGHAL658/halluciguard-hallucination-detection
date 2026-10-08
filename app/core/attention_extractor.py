import re
from collections import defaultdict

import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app import config
from app.core.nli_verifier import get_nli_model

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

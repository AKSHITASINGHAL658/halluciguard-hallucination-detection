import logging
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app import config

log = logging.getLogger("halluciguard.mi_pruning")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def approx_tokens(texts: list[str]) -> int:
    return int(sum(len(t.split()) for t in texts) * 1.3)


def compute_pmi_scores(query: str, chunks: list[dict]) -> list[float]:
    """PMI(q, c) = mean over query terms t of log((P(t|c)+eps)/(P(t)+eps)); P(t) over the retrieved set."""
    eps = config.PMI_EPS
    q_terms = list(dict.fromkeys(_terms(query)))
    chunk_terms = [_terms(c["text"]) for c in chunks]
    all_counts = Counter(t for ts in chunk_terms for t in ts)
    total = max(1, sum(all_counts.values()))
    vocab = max(1, len(all_counts))
    scores = []
    for ts in chunk_terms:
        counts = Counter(ts)
        n = max(1, len(ts))
        if not q_terms:
            scores.append(0.0)
            continue
        vals = []
        for t in q_terms:
            p_t_c = (counts[t] + 1) / (n + vocab)  # Laplace smoothing
            p_t = (all_counts[t] + 1) / (total + vocab)
            vals.append(math.log((p_t_c + eps) / (p_t + eps)))
        scores.append(sum(vals) / len(vals))
    return scores


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def prune_chunks(query: str, chunks: list[dict], keep_ratio: float = config.PRUNE_KEEP_RATIO,
                 min_keep: int = config.PRUNE_MIN_KEEP, redundancy: float = config.PRUNE_REDUNDANCY) -> list[dict]:
    """Keep chunks scoring >= keep_ratio * top score, drop near-duplicates, always keep >= min_keep."""
    if not chunks:
        return []
    scores = compute_pmi_scores(query, chunks)
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    top = scores[order[0]]
    threshold = keep_ratio * top if top > 0 else top / keep_ratio  # lenient when all scores are <= 0

    def redundant(i: int, kept_idx: list[int]) -> bool:
        emb = chunks[i].get("embedding")
        return emb is not None and any(
            chunks[j].get("embedding") is not None and _cosine(emb, chunks[j]["embedding"]) > redundancy
            for j in kept_idx)

    kept: list[int] = []
    for i in order:
        if scores[i] >= threshold and not redundant(i, kept):
            kept.append(i)

    # min_keep guarantee: top up with next-best chunks, non-redundant ones first
    need = min(min_keep, len(chunks))
    for strict in (True, False):
        for i in order:
            if len(kept) >= need:
                break
            if i not in kept and (not strict or not redundant(i, kept)):
                kept.append(i)

    result = []
    for i in kept:
        c = dict(chunks[i])
        c["pmi_score"] = scores[i]
        result.append(c)
    log.info("MI pruning: retrieved=%d kept=%d prompt_tokens %d -> %d", len(chunks), len(result),
             approx_tokens([c["text"] for c in chunks]), approx_tokens([c["text"] for c in result]))
    return result

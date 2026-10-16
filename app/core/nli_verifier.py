import logging
from functools import lru_cache

import torch

from app import config

log = logging.getLogger("halluciguard.nli")

LABELS = ("entailment", "neutral", "contradiction")


@lru_cache(maxsize=1)
def get_nli_model():
    """Cached singleton returning (tokenizer, model). The only place the NLI model is loaded."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for name in (config.NLI_MODEL, config.NLI_MODEL_FALLBACK):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            model.eval()
            log.info("loaded NLI model %s", name)
            return tok, model
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load %s: %s", name, e)
    raise RuntimeError("could not load any NLI model")


def _label_names(model) -> list[str]:
    id2label = model.config.id2label
    return [str(id2label[i]).lower() for i in range(len(id2label))]


def _probs_to_dict(row, names: list[str]) -> dict:
    probs = {n: float(p) for n, p in zip(names, row)}
    label = max(probs, key=probs.get)
    return {"label": label, "probs": {k: probs.get(k, 0.0) for k in LABELS}}


_pair_cache: dict[tuple[str, str], dict] = {}  # NLI is deterministic, so a pair never needs scoring twice


def _copy(r: dict) -> dict:
    return {"label": r["label"], "probs": dict(r["probs"])}


def batch_verify(pairs: list[tuple[str, str]], batch_size: int = config.NLI_BATCH_SIZE) -> list[dict]:
    """pairs: (sentence, chunk_text) = (hypothesis, premise). Repeated pairs are scored once."""
    out: list[dict | None] = [None] * len(pairs)
    todo: dict[tuple[str, str], list[int]] = {}  # unseen pair -> positions that need it
    for i, p in enumerate(pairs):
        hit = _pair_cache.get(p)
        if hit is not None:
            out[i] = _copy(hit)
        else:
            todo.setdefault(p, []).append(i)
    if todo:
        tok, model = get_nli_model()
        names = _label_names(model)
        unique = list(todo)
        for i in range(0, len(unique), batch_size):
            batch = unique[i:i + batch_size]
            enc = tok([p[1] for p in batch], [p[0] for p in batch], truncation="only_first",
                      max_length=config.NLI_MAX_LENGTH, padding=True, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)
            for p, row in zip(batch, probs.tolist()):
                r = _probs_to_dict(row, names)
                for pos in todo[p]:
                    out[pos] = _copy(r)
                if config.NLI_PAIR_CACHE_SIZE:
                    if len(_pair_cache) >= config.NLI_PAIR_CACHE_SIZE:
                        _pair_cache.clear()
                    _pair_cache[p] = r
    return out


def verify_sentence(sentence: str, chunk_text: str) -> dict:
    """-> {label, probs: {entailment, neutral, contradiction}}"""
    return batch_verify([(sentence, chunk_text)])[0]

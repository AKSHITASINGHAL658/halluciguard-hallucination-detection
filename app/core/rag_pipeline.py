import hashlib
import json
import logging
import re
import time

from app import config
from app.core import vector_store
from app.core.embeddings import embed_texts
from app.core.mi_pruning import prune_chunks

log = logging.getLogger("halluciguard.rag")

_client = None
_ollama_client = None
_api_calls = 0  # real LLM requests made by this process (either generator)
_gen_latencies: list[float] = []  # seconds per real (non-cached) generation
_NO_LOGPROBS: set[str] = set()  # models that rejected logprobs; skip the doomed first request next time


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def retrieve(query: str, k: int = config.RETRIEVE_K) -> list[dict]:
    """Top-k chunks from Chroma. Each dict has chunk_id, text, embedding, distance, document_id, title."""
    q_emb = embed_texts([query])[0]
    out = []
    for hit in vector_store.query(q_emb, k):
        meta = hit["metadata"]
        out.append({"chunk_id": int(meta["chunk_id"]), "document_id": int(meta["document_id"]),
                    "title": meta.get("title"), "text": hit["text"], "embedding": hit["embedding"],
                    "distance": hit["distance"]})
    return out


def _is_retryable_logprob_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "logprobs" in msg or "429" in msg or "quota" in msg or "resource_exhausted" in msg


def _extract_logprobs(response):
    """Order: per-token logprobs -> avg_logprobs -> None."""
    try:
        cand = response.candidates[0]
    except (IndexError, AttributeError, TypeError):
        return None, None, "none"
    lr = getattr(cand, "logprobs_result", None)
    chosen = getattr(lr, "chosen_candidates", None) if lr else None
    if chosen:
        toks = [c.log_probability for c in chosen if c.log_probability is not None]
        if toks:
            avg = sum(toks) / len(toks)
            return toks, avg, "tokens"
    avg = getattr(cand, "avg_logprobs", None)
    if avg is not None:
        return None, avg, "avg"
    return None, None, "none"


def build_prompt(query: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    return f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"


def get_api_call_count() -> int:
    return _api_calls


def get_generation_latencies() -> list[float]:
    return _gen_latencies


def active_model() -> str:
    return config.OLLAMA_MODEL if config.GENERATOR == "ollama" else config.GEMINI_MODEL


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama

        _ollama_client = ollama.Client(host=config.OLLAMA_HOST)
    return _ollama_client


def _parse_ollama_logprobs(resp) -> list[float] | None:
    lp = resp.get("logprobs") if isinstance(resp, dict) else getattr(resp, "logprobs", None)
    if not lp:
        return None
    vals = []
    for t in lp:
        v = t.get("logprob") if isinstance(t, dict) else getattr(t, "logprob", None)
        if v is not None:
            vals.append(float(v))
    return vals or None


def _generate_ollama(prompt: str) -> dict:
    global _api_calls
    _api_calls += 1
    resp = _get_ollama_client().chat(
        model=config.OLLAMA_MODEL, logprobs=True, options={"temperature": config.TEMPERATURE},
        messages=[{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    msg = resp["message"] if isinstance(resp, dict) else resp.message
    answer = (msg["content"] if isinstance(msg, dict) else msg.content) or ""
    toks = _parse_ollama_logprobs(resp)
    if toks:
        return {"answer": answer.strip(), "token_logprobs": toks, "avg_logprob": sum(toks) / len(toks),
                "logprobs_source": "tokens"}
    log.warning("Ollama returned no logprobs; token-prob component will be null")
    return {"answer": answer.strip(), "token_logprobs": None, "avg_logprob": None, "logprobs_source": "none"}


def _cache_path(prompt: str):
    key = hashlib.sha256(
        "\n".join([config.GENERATOR, active_model(), config.SYSTEM_PROMPT, prompt]).encode()).hexdigest()
    return config.LLM_CACHE_DIR / f"{key}.json"


def generate(query: str, chunks: list[dict]) -> dict:
    prompt = build_prompt(query, chunks)
    path = _cache_path(prompt) if config.LLM_CACHE else None
    if path is not None and path.exists():
        try:
            cached = json.loads(path.read_text())
            log.info("LLM cache hit (%s) | real API calls so far: %d", path.stem[:8], _api_calls)
            return cached
        except (OSError, ValueError):
            log.warning("unreadable cache entry %s; treating as miss", path.name)
    if path is not None:
        log.info("LLM cache miss (%s)", path.stem[:8])
    t0 = time.time()
    result = _generate_ollama(prompt) if config.GENERATOR == "ollama" else _generate_uncached(prompt)
    _gen_latencies.append(time.time() - t0)
    result["generator"] = config.GENERATOR
    result["model_name"] = active_model()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    return result


def _generate_uncached(prompt: str) -> dict:
    from google.genai import types

    def call(with_logprobs: bool):
        kwargs = dict(system_instruction=config.SYSTEM_PROMPT, temperature=config.TEMPERATURE,
                      thinking_config=types.ThinkingConfig(thinking_budget=0))
        if with_logprobs:
            kwargs.update(response_logprobs=True, logprobs=1)
        global _api_calls
        _api_calls += 1
        return _get_client().models.generate_content(
            model=config.GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig(**kwargs))

    def call_with_backoff(with_logprobs: bool):
        for attempt in range(4):
            try:
                return call(with_logprobs)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg) and attempt < 3 and "PerDay" not in msg:
                    m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"'retryDelay': '(\d+)s'", msg)
                    delay = min(float(m.group(1)) + 1 if m else 20.0, 70.0)
                    log.warning("rate limited; sleeping %.0fs", delay)
                    time.sleep(delay)
                    continue
                raise

    if not config.USE_LOGPROBS:
        resp = call_with_backoff(False)
        return {"answer": (resp.text or "").strip(), "token_logprobs": None, "avg_logprob": None,
                "logprobs_source": "none"}

    try:
        resp = call_with_backoff(config.GEMINI_MODEL not in _NO_LOGPROBS)
    except Exception as e:
        if not _is_retryable_logprob_error(e):
            raise
        if "logprobs" in str(e).lower():
            _NO_LOGPROBS.add(config.GEMINI_MODEL)
        log.warning("logprobs request failed (%s); retrying without logprobs", str(e)[:120])
        resp = call_with_backoff(False)

    token_logprobs, avg_logprob, source = _extract_logprobs(resp)
    return {"answer": (resp.text or "").strip(), "token_logprobs": token_logprobs,
            "avg_logprob": avg_logprob, "logprobs_source": source}


def run_pipeline(query: str, k: int = config.RETRIEVE_K) -> dict:
    retrieved = retrieve(query, k)
    used = prune_chunks(query, retrieved)
    gen = generate(query, used)
    return {"query": query, "retrieved": retrieved, "pruned": used, **gen}

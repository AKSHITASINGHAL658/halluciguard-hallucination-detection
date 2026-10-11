import hashlib
import json
import logging
import time

from app import config
from app.core import vector_store
from app.core.embeddings import embed_texts
from app.core.mi_pruning import prune_chunks

log = logging.getLogger("halluciguard.rag")

_ollama_client = None
_api_calls = 0  # real Ollama requests made by this process
_gen_latencies: list[float] = []  # seconds per real (non-cached) generation


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


def build_prompt(query: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    return f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"


def get_api_call_count() -> int:
    return _api_calls


def get_generation_latencies() -> list[float]:
    return _gen_latencies


def active_model() -> str:
    return config.OLLAMA_MODEL


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
        model=config.OLLAMA_MODEL, logprobs=True, keep_alive=config.OLLAMA_KEEP_ALIVE, options={"temperature": config.TEMPERATURE},
        messages=[{"role": "system", "content": config.get_system_prompt()}, {"role": "user", "content": prompt}])
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
        "\n".join(["ollama", active_model(), config.get_system_prompt(), prompt]).encode()).hexdigest()
    return config.LLM_CACHE_DIR / f"{key}.json"


def generate(query: str, chunks: list[dict]) -> dict:
    prompt = build_prompt(query, chunks)
    path = _cache_path(prompt) if config.LLM_CACHE else None
    if path is not None and path.exists():
        try:
            cached = json.loads(path.read_text())
            log.info("LLM cache hit (%s) | real LLM calls so far: %d", path.stem[:8], _api_calls)
            return cached
        except (OSError, ValueError):
            log.warning("unreadable cache entry %s; treating as miss", path.name)
    if path is not None:
        log.info("LLM cache miss (%s)", path.stem[:8])
    t0 = time.time()
    result = _generate_ollama(prompt)
    _gen_latencies.append(time.time() - t0)
    result["generator"] = "ollama"
    result["model_name"] = active_model()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    return result


def run_pipeline(query: str, k: int = config.RETRIEVE_K) -> dict:
    retrieved = retrieve(query, k)
    used = prune_chunks(query, retrieved)
    gen = generate(query, used)
    return {"query": query, "retrieved": retrieved, "pruned": used, **gen}

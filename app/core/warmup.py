import logging
import time

from app import config

log = logging.getLogger("halluciguard.warmup")


def warmup() -> dict:
    """Load the embedder, the NLI model and the Ollama model once so the first /query isn't slow."""
    timings = {}

    def step(name, fn):
        t = time.time()
        try:
            fn()
            timings[name] = round(time.time() - t, 1)
        except Exception as e:  # noqa: BLE001
            log.warning("warmup %s failed: %s", name, str(e)[:120])
            timings[name] = None

    def embed():
        from app.core.embeddings import embed_texts
        embed_texts(["warmup"])

    def nli():
        from app.core.nli_verifier import verify_sentence
        verify_sentence("It is warm.", "The room is warm.")

    def ollama():
        from app.core import rag_pipeline
        rag_pipeline._get_ollama_client().chat(
            model=config.OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1}, keep_alive=config.OLLAMA_KEEP_ALIVE)

    step("embedder", embed)
    step("nli", nli)
    step("ollama", ollama)
    log.info("warmup seconds: %s", timings)
    return timings

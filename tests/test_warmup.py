from types import SimpleNamespace

from app.core import rag_pipeline, warmup as warmup_mod


def test_warmup_loads_everything_and_survives_ollama_down(monkeypatch):
    seen = []

    class Down:
        def chat(self, **kw):
            seen.append(kw)
            raise ConnectionError("ollama not running")

    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: Down())
    t = warmup_mod.warmup()
    assert t["embedder"] is not None and t["nli"] is not None and t["ollama"] is None
    assert seen[0]["keep_alive"] == warmup_mod.config.OLLAMA_KEEP_ALIVE

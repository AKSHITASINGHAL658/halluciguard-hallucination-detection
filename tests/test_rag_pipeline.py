from types import SimpleNamespace

from app.core import rag_pipeline


def _resp(chosen=None, avg=None):
    lr = SimpleNamespace(chosen_candidates=[SimpleNamespace(log_probability=x) for x in chosen]) if chosen else None
    return SimpleNamespace(candidates=[SimpleNamespace(logprobs_result=lr, avg_logprobs=avg)], text="Answer.")


def test_extract_logprobs_order():
    toks, avg, src = rag_pipeline._extract_logprobs(_resp(chosen=[-0.1, -0.3], avg=-9))
    assert src == "tokens" and toks == [-0.1, -0.3] and abs(avg + 0.2) < 1e-9
    assert rag_pipeline._extract_logprobs(_resp(avg=-0.5)) == (None, -0.5, "avg")
    assert rag_pipeline._extract_logprobs(_resp()) == (None, None, "none")


def test_generate_falls_back_without_logprobs(monkeypatch):
    calls = []
    rag_pipeline._NO_LOGPROBS.clear()

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(config.response_logprobs)
            if config.response_logprobs:
                raise RuntimeError("400 INVALID_ARGUMENT Logprobs is not enabled for this model")
            return _resp()

    monkeypatch.setattr(rag_pipeline, "_get_client", lambda: SimpleNamespace(models=Models()))
    out = rag_pipeline.generate("q?", [{"text": "ctx"}])
    assert calls == [True, None] and out["logprobs_source"] == "none" and out["answer"] == "Answer."


def test_retrieve_and_query_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(rag_pipeline, "generate", lambda q, c: {"answer": "Stub.", "token_logprobs": None,
                                                              "avg_logprob": None, "logprobs_source": "none"})
    with TestClient(app) as c:
        r = c.post("/query", json={"text": "How much solar capacity was added in 2023?"})
        assert r.status_code == 200
        body = r.json()
        assert body["retrieved_chunk_ids"] and body["answer"] == "Stub."
        assert c.get(f"/response/{body['response_id']}").json()["answer"] == "Stub."


def test_query_populates_sentences(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(rag_pipeline, "generate", lambda q, c: {
        "answer": "About 345 gigawatts of new solar capacity were added in 2023. The moon is made of cheese.",
        "token_logprobs": None, "avg_logprob": None, "logprobs_source": "none"})
    with TestClient(app) as c:
        body = c.post("/query", json={"text": "How much solar capacity was added in 2023?"}).json()
        assert len(body["sentences"]) == 2
        assert all(s["nli_label"] and s["nli_confidence"] is not None for s in body["sentences"])
        assert body["sentences"][1]["is_hallucination"] is True
        assert body["sentences"][0]["is_hallucination"] is False


def test_generate_skips_logprobs_after_rejection(monkeypatch):
    rag_pipeline._NO_LOGPROBS.add(rag_pipeline.config.GEMINI_MODEL)
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(config.response_logprobs)
            return _resp()

    monkeypatch.setattr(rag_pipeline, "_get_client", lambda: SimpleNamespace(models=Models()))
    rag_pipeline.generate("q?", [{"text": "ctx"}])
    rag_pipeline._NO_LOGPROBS.clear()
    assert calls == [None]

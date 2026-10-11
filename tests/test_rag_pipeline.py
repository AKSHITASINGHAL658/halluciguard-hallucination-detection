from types import SimpleNamespace

import pytest

from app.core import rag_pipeline


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    # never touch the real cache dir or a real Ollama server from tests
    monkeypatch.setattr(rag_pipeline.config, "LLM_CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(rag_pipeline.config, "LLM_CACHE", False)


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


def _ollama_resp(logprobs, content="Ollama answer."):
    return SimpleNamespace(message=SimpleNamespace(content=content), logprobs=logprobs)


def _ollama_client(resp, seen):
    class Client:
        def chat(self, **kw):
            seen.append(kw)
            return resp

    return Client()


def test_ollama_parses_token_logprobs(monkeypatch):
    seen = []
    lp = [SimpleNamespace(token="a", logprob=-0.2), SimpleNamespace(token="b", logprob=-0.4)]
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _ollama_client(_ollama_resp(lp), seen))
    out = rag_pipeline.generate("q?", [{"text": "ctx"}])
    assert out["logprobs_source"] == "tokens" and out["token_logprobs"] == [-0.2, -0.4]
    assert abs(out["avg_logprob"] + 0.3) < 1e-9 and out["generator"] == "ollama"
    assert out["model_name"] == rag_pipeline.config.OLLAMA_MODEL
    assert seen[0]["logprobs"] is True and seen[0]["options"]["temperature"] == 0


def test_ollama_dict_logprobs_and_fallback_when_missing(monkeypatch):
    resp = {"message": {"content": "x"}, "logprobs": [{"token": "a", "logprob": -1.0}]}
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _ollama_client(resp, []))
    assert rag_pipeline.generate("q?", [{"text": "ctx"}])["logprobs_source"] == "tokens"
    for empty in (None, []):
        monkeypatch.setattr(rag_pipeline, "_get_ollama_client",
                            lambda e=empty: _ollama_client(_ollama_resp(e), []))
        out = rag_pipeline.generate("q?", [{"text": "ctx"}])
        assert out["logprobs_source"] == "none" and out["token_logprobs"] is None and out["avg_logprob"] is None


def test_query_inject_faults_flags_and_excludes_token_prob(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(rag_pipeline, "generate", lambda q, c: {
        "answer": "About 345 gigawatts of new solar capacity were added in 2023.",
        "token_logprobs": [-0.1], "avg_logprob": -0.1, "logprobs_source": "tokens"})
    with TestClient(app) as c:
        clean = c.post("/query", json={"text": "How much solar capacity was added in 2023?"}).json()
        faulty = c.post("/query", json={"text": "How much solar capacity was added in 2023?", "inject_faults": True}).json()
    assert clean["injected_faults"] == [] and clean["trust"]["token_prob_component"] is not None
    assert len(faulty["injected_faults"]) == 1 and faulty["answer"] != clean["answer"]
    assert faulty["injected_faults"][0]["corrupted"] in faulty["answer"]
    assert faulty["trust"]["token_prob_component"] is None
    assert faulty["trust_score_overall"] < clean["trust_score_overall"]


def test_wrong_numbers_flag_sentence_and_return_spans(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(rag_pipeline, "generate", lambda q, c: {
        "answer": "About 172 gigawatts of new solar capacity were added in 2023.",
        "token_logprobs": None, "avg_logprob": None, "logprobs_source": "none"})
    with TestClient(app) as c:
        body = c.post("/query", json={"text": "How much new solar capacity was added globally in 2023?"}).json()
    sent = body["sentences"][0]
    assert sent["is_hallucination"] is True
    assert "172" in [sp["text"] for sp in sent["wrong_spans"]]
    assert "345" in sent["supporting_chunk_text"]



def test_cache_second_identical_call_makes_no_llm_request(monkeypatch):
    monkeypatch.setattr(rag_pipeline.config, "LLM_CACHE", True)
    seen = []
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _ollama_client(_ollama_resp(None), seen))
    before = rag_pipeline.get_api_call_count()
    first = rag_pipeline.generate("q?", [{"text": "ctx"}])
    second = rag_pipeline.generate("q?", [{"text": "ctx"}])
    assert len(seen) == 1 and rag_pipeline.get_api_call_count() - before == 1
    assert first == second and first["logprobs_source"] == "none"
    rag_pipeline.generate("different?", [{"text": "ctx"}])  # different prompt -> miss
    assert len(seen) == 2


def test_cache_key_includes_model(monkeypatch):
    monkeypatch.setattr(rag_pipeline.config, "LLM_CACHE", True)
    seen = []
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _ollama_client(_ollama_resp(None), seen))
    rag_pipeline.generate("q?", [{"text": "ctx"}])
    monkeypatch.setattr(rag_pipeline.config, "OLLAMA_MODEL", "other:1b")
    rag_pipeline.generate("q?", [{"text": "ctx"}])
    assert len(seen) == 2 and seen[1]["model"] == "other:1b"


def test_loose_prompt_mode_changes_system_prompt_and_cache_key(monkeypatch):
    monkeypatch.setattr(rag_pipeline.config, "LLM_CACHE", True)
    seen = []
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _ollama_client(_ollama_resp(None), seen))
    monkeypatch.setattr(rag_pipeline.config, "PROMPT_MODE", "strict")
    rag_pipeline.generate("q?", [{"text": "ctx"}])
    monkeypatch.setattr(rag_pipeline.config, "PROMPT_MODE", "loose")
    rag_pipeline.generate("q?", [{"text": "ctx"}])  # same question, different prompt -> not a cache hit
    systems = [c["messages"][0]["content"] for c in seen]
    assert systems == [rag_pipeline.config.SYSTEM_PROMPT_STRICT, rag_pipeline.config.SYSTEM_PROMPT_LOOSE]

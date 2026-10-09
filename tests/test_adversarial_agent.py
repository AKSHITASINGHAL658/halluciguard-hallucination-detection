import random

from types import SimpleNamespace

from app.core import adversarial_agent, rag_pipeline
from app.core.adversarial_agent import (inject_llm_contradiction, inject_negation, inject_noise,
                                        inject_stale_fact)


def test_negation_roundtrip():
    s = "The dam is located in China."
    neg = inject_negation(s)
    assert "not" in neg and inject_negation(neg) == s


def test_stale_fact_changes_numbers():
    s = "The dam was completed in 2012 with 22,500 megawatts."
    out = inject_stale_fact(s, random.Random(1))
    assert out != s and "2012" not in out and "22,500" not in out


def test_noise_changes_sentence():
    assert inject_noise("China installed solar panels.", ["Brazil"], random.Random(0)) != "China installed solar panels."


def _client(content, seen):
    class C:
        def chat(self, **kw):
            seen.append(kw)
            if isinstance(content, Exception):
                raise content
            return SimpleNamespace(message=SimpleNamespace(content=content))

    return C()


def test_llm_contradiction_cleans_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(adversarial_agent.config, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(adversarial_agent.config, "LLM_CACHE", True)
    seen = []
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _client('"China added 90 gigawatts."\nextra', seen))
    s = "China added 217 gigawatts."
    assert inject_llm_contradiction(s) == "China added 90 gigawatts."
    assert inject_llm_contradiction(s) == "China added 90 gigawatts."
    assert len(seen) == 1 and "factually different" in seen[0]["messages"][0]["content"]


def test_llm_contradiction_failure_or_identical_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(adversarial_agent.config, "LLM_CACHE_DIR", tmp_path)
    s = "China added 217 gigawatts."
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _client(RuntimeError("down"), []))
    assert inject_llm_contradiction(s) is None
    monkeypatch.setattr(rag_pipeline, "_get_ollama_client", lambda: _client(s, []))
    assert inject_llm_contradiction(s) is None

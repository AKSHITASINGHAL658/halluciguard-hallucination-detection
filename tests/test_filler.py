from app.core import rag_pipeline
from app.core.sentence_localizer import is_filler


def test_filler_detection():
    for s in ["This information is directly stated in the context provided.",
              "This information is provided in the context under point [1].",
              "The context does not contain the answer.",
              "According to the provided passages, this is mentioned."]:
        assert is_filler(s), s
    for s in ["About 345 gigawatts of solar capacity were added in 2023.",
              "China is the largest solar market in the world.",
              "The context passage states that 345 gigawatts were added in 2023."]:
        assert not is_filler(s), s


def test_filler_is_never_flagged_and_excluded_from_scores(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    answers = ["About 345 gigawatts of new solar capacity were added in 2023.",
               "About 345 gigawatts of new solar capacity were added in 2023. "
               "This information is directly stated in the context provided."]
    out = []
    for a in answers:
        monkeypatch.setattr(rag_pipeline, "generate", lambda q, c, a=a: {
            "answer": a, "token_logprobs": None, "avg_logprob": None, "logprobs_source": "none"})
        with TestClient(app) as c:
            out.append(c.post("/query", json={"text": "How much new solar capacity was added globally in 2023?"}).json())
    filler = out[1]["sentences"][1]
    assert filler["is_filler"] is True and filler["is_hallucination"] is False and filler["wrong_spans"] == []
    assert out[1]["trust"]["sentence_level_score"] == out[0]["trust"]["sentence_level_score"]  # filler ignored

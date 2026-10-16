from app.core import rag_pipeline
from app.core.sentence_localizer import is_abstention, is_filler


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


def test_abstention_detection():
    for s in ['The statement "2+2=4" is not derived from the provided context passages.',
              "The provided context does not mention the 2024 figures at all, so I cannot say.",
              "No information about the year 2030 is found in the context."]:
        assert is_abstention(s), s
    for s in ["Solar is a source of power that does not emit carbon dioxide.",  # real facts: never swallowed
              "About 345 gigawatts of new solar capacity were added in 2023.",
              "The context passage states that 345 gigawatts were added in 2023.",
              "The report found that wind capacity grew 12 percent in 2023."]:
        assert not is_abstention(s), s


def test_abstention_is_neutral_not_supported_and_not_flagged():
    from app.core.verification import verify_sentences

    chunk = {"chunk_id": 1, "document_id": 1, "text": "Wind and hydropower capacity grew strongly in 2023, "
             "with offshore wind farms expanding across Europe and China."}
    s = verify_sentences(['The statement "2+2=4" is not derived from the provided context passages.'], [chunk])[0]
    assert s["is_abstention"] and not s["is_filler"]
    assert s["label"] == "neutral" and s["is_hallucination"] is False and s["wrong_spans"] == []

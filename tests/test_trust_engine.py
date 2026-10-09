import math

from app import config
from app.core.trust_engine import aggregate_trust, doc_score, entity_score, sentence_score, token_prob_component

SENTS = [
    {"max_entailment": 0.9, "is_hallucination": False, "best_chunk_id": 1},
    {"max_entailment": 0.1, "is_hallucination": True, "best_chunk_id": 2},
]


def test_entity_score():
    chunks = [{"text": "China installed 217 gigawatts of solar in 2023."}]
    score, ents = entity_score("China added solar in 2023, while Brazil added little.", chunks)
    texts = {e["text"]: e["supported"] for e in ents}
    assert texts["China"] is True and texts["Brazil"] is False and 0 < score < 1
    assert entity_score("Nothing special here.", chunks) == (1.0, [])


def test_sentence_and_doc_score():
    assert math.isclose(sentence_score(SENTS), 0.5 * 0.5)
    assert doc_score(SENTS, {1: 0.9, 2: 0.2}) == 0.9
    assert doc_score([SENTS[1]], {2: 0.2}) == 0.0


def test_token_prob():
    assert math.isclose(token_prob_component([-1.0, -1.0], None), math.exp(-1))
    assert math.isclose(token_prob_component(None, -0.5), math.exp(-0.5))
    assert token_prob_component(None, None) is None


def test_aggregate_with_and_without_token_prob():
    w = config.TRUST_WEIGHTS
    assert math.isclose(aggregate_trust(1, 0, 0, 0), w["sentence"])
    assert math.isclose(aggregate_trust(1, 1, 1, 1), 1.0)
    assert math.isclose(aggregate_trust(1, 1, 1, None), 1.0)  # renormalized
    assert math.isclose(aggregate_trust(1, 0, 0, None), w["sentence"] / (1 - w["token_prob"]))


def test_entity_modifiers_stripped_and_relative_dates_skipped():
    chunks = [{"text": "About 345 gigawatts of solar were added in 2023, reaching 1,600 gigawatts in total."}]
    score, ents = entity_score(
        "In 2023, approximately 345 gigawatts were added, for roughly 1,600 gigawatts by the end of that year.", chunks)
    texts = {e["text"]: e["supported"] for e in ents}
    assert "345" in texts and texts["345"] and texts["1,600"] and texts["2023"]
    assert not any("end of" in t for t in texts)
    assert score == 1.0

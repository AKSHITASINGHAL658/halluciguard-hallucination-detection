from app.core.claim_extractor import extract_claims
from app.core.sentence_localizer import localize_hallucinations

CHUNK = {"chunk_id": 1, "text": "The Three Gorges Dam is located on the Yangtze River in China and has an installed "
                                "capacity of 22,500 megawatts. It was fully commissioned in 2012."}


def test_splits_coordinated_clauses_and_borrows_subject():
    c = extract_claims("The Three Gorges Dam has a capacity of 22,500 megawatts and it typically generates "
                       "between 85 and 100 terawatt-hours per year.")
    assert c == ["The Three Gorges Dam has a capacity of 22,500 megawatts.",
                 "The Three Gorges Dam typically generates between 85 and 100 terawatt-hours per year."]
    assert extract_claims("The dam was completed in 2012 and has 32 main turbines.") == [
        "The dam was completed in 2012.", "The dam has 32 main turbines."]


def test_leaves_simple_sentences_alone():
    s = "Wind supplied about 7.8 percent of global electricity in 2023."
    assert extract_claims(s) == [s]
    s2 = "China installed 217 gigawatts, installing more than half of the total."
    assert extract_claims(s2) == [s2]


def test_weakest_claim_decides_the_sentence():
    s = "The Three Gorges Dam is located in China and it was designed by a famous Brazilian architect."
    whole = localize_hallucinations([s], [CHUNK], use_claims=False)[0]
    split = localize_hallucinations([s], [CHUNK], use_claims=True)[0]
    assert len(split["claims"]) == 2 and split["is_hallucination"] is True
    assert split["weak_claim"].lower().startswith("the three gorges dam was designed") or "architect" in split["weak_claim"]
    assert split["max_entailment"] <= whole["max_entailment"] + 1e-9 or whole["is_hallucination"]

from app.core.nli_verifier import get_nli_model, verify_sentence

PREMISE = "The Three Gorges Dam is located on the Yangtze River in China and has an installed capacity of 22,500 megawatts."


def test_entailment():
    r = verify_sentence("The Three Gorges Dam is in China.", PREMISE)
    assert r["label"] == "entailment" and r["probs"]["entailment"] > 0.5


def test_contradiction():
    r = verify_sentence("The Three Gorges Dam is located in Brazil.", PREMISE)
    assert r["label"] == "contradiction"


def test_neutral():
    r = verify_sentence("The dam was designed by a famous architect.", PREMISE)
    assert r["label"] == "neutral"


def test_singleton_and_labels_from_config():
    assert get_nli_model() is get_nli_model()
    names = {v.lower() for v in get_nli_model()[1].config.id2label.values()}
    assert names == {"entailment", "neutral", "contradiction"}

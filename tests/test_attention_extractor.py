from app.core.attention_extractor import top_contributing_tokens

PREMISE = "The Three Gorges Dam on the Yangtze River has an installed capacity of 22,500 megawatts and was completed in 2012."


def test_top_tokens_shape():
    top = top_contributing_tokens("The dam has a capacity of 5,000 megawatts.", PREMISE)
    assert 1 <= len(top) <= 5
    assert all(set(t) == {"token", "weight"} for t in top)
    assert abs(sum(t["weight"] for t in top) - 1.0) < 0.01
    assert all(t["token"].lower() not in {"the", "of", "and"} for t in top)
    assert all(t["token"] in PREMISE for t in top)


def test_check_attribution_arithmetic(monkeypatch):
    from app.core import attention_extractor as ae

    def probs(e, n, c):
        return {"label": "x", "probs": {"entailment": e, "neutral": n, "contradiction": c}}

    calls = []

    def fake(pairs):
        calls.append(pairs)
        # variant 0 = original chunk, 1 = top words deleted, 2.. = random controls
        return [probs(0.0, 0.1, 0.9), probs(0.0, 0.9, 0.1)] + [probs(0.0, 0.1, 0.9)] * (len(pairs) - 2)

    monkeypatch.setattr(ae, "batch_verify", fake)
    out = ae.check_attribution("s", "Solar added 345 GW in 2023 worldwide.", [{"token": "345", "weight": 1.0}], n_random=3)
    assert len(calls[0]) == 5 and out["n_random"] == 3
    assert abs(out["top_shift"] - 0.8) < 1e-9 and out["random_shift"] == 0.0 and out["supported"] is True
    assert "345" not in calls[0][1][1] and "345" in calls[0][0][1]
    assert ae.check_attribution("s", "text", []) is None


def test_check_attribution_real_model_wrong_number():
    from app.core.attention_extractor import check_attribution

    premise = ("The Three Gorges Dam on the Yangtze River has an installed capacity of 22,500 megawatts and was fully "
               "commissioned in 2012. It typically generates between 85 and 100 terawatt-hours per year.")
    sent = "The dam has an installed capacity of 5,000 megawatts."
    top = top_contributing_tokens(sent, premise)
    out = check_attribution(sent, premise, top)
    assert out["top_shift"] > out["random_shift"] and out["supported"] is True

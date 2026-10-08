from app.core.attention_extractor import top_contributing_tokens

PREMISE = "The Three Gorges Dam on the Yangtze River has an installed capacity of 22,500 megawatts and was completed in 2012."


def test_top_tokens_shape():
    top = top_contributing_tokens("The dam has a capacity of 5,000 megawatts.", PREMISE)
    assert 1 <= len(top) <= 5
    assert all(set(t) == {"token", "weight"} for t in top)
    assert abs(sum(t["weight"] for t in top) - 1.0) < 0.01
    assert all(t["token"].lower() not in {"the", "of", "and"} for t in top)
    assert all(t["token"] in PREMISE for t in top)

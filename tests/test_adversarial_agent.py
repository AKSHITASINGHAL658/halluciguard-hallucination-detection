import random

from app.core.adversarial_agent import inject_negation, inject_noise, inject_stale_fact


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

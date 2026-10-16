import json

from app import config
from app.core.evaluation import detector_report


def test_detector_report_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ABLATION_PATH", tmp_path / "nope.json")
    assert detector_report() == {"available": False}


def test_detector_report_reads_production_variant(monkeypatch, tmp_path):
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps({
        "labeled_by": "an AI", "n_sentences": 10, "n_unsupported": 4,
        "variants": {"full (claims + span + filler) = production": {"recall": 0.8, "precision": 0.6},
                     "NLI only (sentence level)": {"recall": 0.7}},
        "recall_by_error_kind": {"invented_specific": {"n": 5, "recall": 0.9}},
        "answer_level": {"auroc": {"final": 0.89}}}))
    monkeypatch.setattr(config, "ABLATION_PATH", path)
    r = detector_report()
    assert r["available"] and r["production"]["recall"] == 0.8 and r["nli_only"]["recall"] == 0.7
    assert r["labeled_by"] == "an AI" and r["auroc"]["final"] == 0.89
    assert r["recall_by_error_kind"]["invented_specific"]["n"] == 5

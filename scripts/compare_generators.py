import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.core import rag_pipeline  # noqa: E402
from app.core.evaluation import run_evaluation  # noqa: E402
from app.core.warmup import warmup  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402


def run_one(generator: str, limit: int | None, use_cache: bool) -> dict:
    config.GENERATOR = generator
    config.LLM_CACHE = use_cache  # cache off by default so latency is real
    db = SessionLocal()
    try:
        r = run_evaluation(db, limit=limit)
    finally:
        db.close()
    return {"model": rag_pipeline.active_model(), "n_questions": r["n_questions"], "n_failed": r["n_failed"],
            "n_sentences": r["n_sentences"], "hallucination_rate": r["hallucination_rate"],
            "avg_trust": r["avg_trust"], "avg_token_prob": r["avg_token_prob"],
            "avg_latency_s": r["avg_generation_latency_s"], "api_calls": r["api_calls"]}


def main():
    ap = argparse.ArgumentParser(description="Compare generators on the QA eval set")
    ap.add_argument("--gemini", action="store_true", help="also run Gemini (uses API quota)")
    ap.add_argument("--limit", type=int, default=None, help="first N questions (default: all)")
    ap.add_argument("--use-cache", action="store_true", help="allow the LLM cache (latency then not meaningful)")
    a = ap.parse_args()
    init_db()
    warmup()  # keep model-load time out of the latency numbers
    results = {"ollama": run_one("ollama", a.limit, a.use_cache)}
    if a.gemini:
        results["gemini"] = run_one("gemini", a.limit, a.use_cache)
    out = ROOT / "outputs" / "generator_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

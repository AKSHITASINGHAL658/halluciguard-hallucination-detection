import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.core.evaluation import run_evaluation  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Evaluate on data/eval/qa_eval_set.json")
    ap.add_argument("--k", type=int, default=config.RETRIEVE_K)
    ap.add_argument("--retrieval-only", action="store_true", help="retrieval + pruning metrics, no generation")
    ap.add_argument("--rescore", action="store_true", help="reuse stored answers; rerun NLI/localization/trust")
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    a = ap.parse_args()
    init_db()
    db = SessionLocal()
    print(json.dumps(run_evaluation(db, a.k, a.retrieval_only, a.rescore, a.limit), indent=2))
    db.close()


if __name__ == "__main__":
    main()

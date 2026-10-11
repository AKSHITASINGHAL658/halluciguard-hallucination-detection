"""Generate candidate answers (strict + loose prompt) for labeling. Output goes to outputs/labeling_candidates.json."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.core import rag_pipeline  # noqa: E402
from app.core.mi_pruning import prune_chunks  # noqa: E402
from app.core.query_service import fix_split_decimals  # noqa: E402
from app.core.sentence_localizer import split_sentences  # noqa: E402
from app.core.warmup import warmup  # noqa: E402


def main():
    qa = [q["question"] for q in json.loads(config.EVAL_SET_PATH.read_text())]
    extra = json.loads((ROOT / "data" / "eval" / "labeling_questions.json").read_text())
    jobs = [("strict", q) for q in qa + extra] + [("loose", q) for q in extra + qa[:10]]
    warmup()
    rows = []
    for mode, q in jobs:
        config.PROMPT_MODE = mode
        retrieved = rag_pipeline.retrieve(q)
        kept = prune_chunks(q, retrieved)
        g = rag_pipeline.generate(q, kept)
        answer = fix_split_decimals(g["answer"])
        for si, sent in enumerate(split_sentences(answer)):
            rows.append({"id": len(rows), "mode": mode, "question": q, "answer": answer, "sentence_index": si,
                         "sentence": sent, "avg_logprob": g["avg_logprob"],
                         "chunk_ids": [c["chunk_id"] for c in kept]})
    out = ROOT / "outputs" / "labeling_candidates.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"{len(jobs)} answers, {len(rows)} sentences -> {out}")


if __name__ == "__main__":
    main()

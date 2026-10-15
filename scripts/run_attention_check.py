"""How often does the mask-and-rerun check support the top attention words? Output: outputs/attention_check.json."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_ablation  # noqa: E402  (reuses the NLI memo so the verification step is fast)
from app.core.attention_extractor import check_attribution, top_contributing_tokens  # noqa: E402
from app.core.mi_pruning import prune_chunks  # noqa: E402
from app.core.rag_pipeline import retrieve  # noqa: E402
from app.core.verification import verify_sentences  # noqa: E402


def main():
    run_ablation.install_nli_memo()
    doc = json.loads(run_ablation.LABELED.read_text())
    items = [x for x in doc["items"] if x["label"] in ("supported", "unsupported")]
    by_q = defaultdict(list)
    for x in items:
        by_q[x["question"]].append(x)
    rows = []
    for q, xs in by_q.items():
        chunks = prune_chunks(q, retrieve(q))
        for x, r in zip(xs, verify_sentences([x["sentence"] for x in xs], chunks)):
            if not r["nli_flagged"] or not r["best_chunk_text"]:
                continue
            top = top_contributing_tokens(r["text"], r["best_chunk_text"])
            c = check_attribution(r["text"], r["best_chunk_text"], top)
            if c:
                rows.append({"label": x["label"], "sentence": x["sentence"], **{k: c[k] for k in (
                    "masked_words", "top_shift", "random_shift", "supported")}})
                print(f"{len(rows)} {x['label']:11} top={c['top_shift']:.2f} rand={c['random_shift']:.2f} {c['supported']}", flush=True)

    def summ(rs):
        n = len(rs)
        return {"n": n, "supported_rate": sum(r["supported"] for r in rs) / n if n else None,
                "mean_top_shift": sum(r["top_shift"] for r in rs) / n if n else None,
                "mean_random_shift": sum(r["random_shift"] for r in rs) / n if n else None}

    out = {"all_flagged": summ(rows), "truly_unsupported": summ([r for r in rows if r["label"] == "unsupported"]),
           "wrongly_flagged_supported": summ([r for r in rows if r["label"] == "supported"]), "rows": rows}
    (ROOT / "outputs" / "attention_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

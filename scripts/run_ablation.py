"""Precision/recall/F1 on the labeled sentence set + ablations. Output: outputs/ablation.json (+ markdown tables)."""
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from app import config  # noqa: E402
from app.core.mi_pruning import prune_chunks  # noqa: E402
from app.core.rag_pipeline import retrieve  # noqa: E402
from app.core.sentence_localizer import split_sentences  # noqa: E402
from app.core.trust_engine import (aggregate_trust, doc_score, entity_score, sentence_score,  # noqa: E402
                                   token_prob_component)
from app.core.verification import verify_sentences  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import Document  # noqa: E402

NLI_CACHE = ROOT / "cache" / "nli_ablation.pkl"


def install_nli_memo():
    """Memoize NLI results per (sentence, chunk) pair (also on disk), so configs share work and reruns are fast."""
    import app.core.sentence_localizer as loc

    real = loc.batch_verify
    memo = pickle.loads(NLI_CACHE.read_bytes()) if NLI_CACHE.exists() else {}
    stats = {"calls": 0, "t0": time.time()}

    def wrapped(pairs, *a, **k):
        todo = [p for p in dict.fromkeys(pairs) if p not in memo]
        if todo:
            for p, r in zip(todo, real(todo, *a, **k)):
                memo[p] = r
            stats["calls"] += len(todo)
            NLI_CACHE.parent.mkdir(exist_ok=True)
            NLI_CACHE.write_bytes(pickle.dumps(memo))
            print(f"  NLI pairs computed so far: {stats['calls']} ({time.time() - stats['t0']:.0f}s)", flush=True)
        return [memo[p] for p in pairs]

    loc.batch_verify = wrapped


LABELED = ROOT / "data" / "eval" / "labeled_sentences.json"


def prf(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = len(y_true) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": (tp + tn) / len(y_true),
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def bootstrap_f1(y_true, y_pred, n=1000, seed=0) -> list[float]:
    rng = random.Random(seed)
    idx = range(len(y_true))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(prf([y_true[i] for i in s], [y_pred[i] for i in s])["f1"])
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def main():
    install_nli_memo()
    doc = json.loads(LABELED.read_text())
    items = [x for x in doc["items"] if x["label"] in ("supported", "unsupported")]
    y = [1 if x["label"] == "unsupported" else 0 for x in items]
    print(f"{len(items)} evaluable sentences ({sum(y)} unsupported, {len(y) - sum(y)} supported)")

    # chunks per question (deterministic)
    retrieved, kept = {}, {}
    for q in sorted({x["question"] for x in items}):
        retrieved[q] = retrieve(q)
        kept[q] = prune_chunks(q, retrieved[q])

    def run(use_claims: bool, pruned: bool):
        out = {}
        by_q = defaultdict(list)
        for i, x in enumerate(items):
            by_q[x["question"]].append(i)
        for q, idxs in by_q.items():
            res = verify_sentences([items[i]["sentence"] for i in idxs], kept[q] if pruned else retrieved[q],
                                   use_claims=use_claims)
            for i, r in zip(idxs, res):
                out[i] = r
        return [out[i] for i in range(len(items))]

    print("config A: sentence-level NLI + spans", flush=True)
    A = run(False, True)
    print("config B: claims + spans (production)", flush=True)
    B = run(True, True)
    print("config C: production without MI pruning (6 chunks)", flush=True)
    C = run(True, False)

    variants = {
        "NLI only (sentence level)": [int(r["nli_flagged"]) for r in A],
        "span check only (numbers/names)": [int(bool(r["wrong_spans"])) for r in A],
        "NLI + span check": [int(r["is_hallucination"]) for r in A],
        "NLI + claim splitting": [int(r["nli_flagged"]) for r in B],
        "full (claims + span + filler) = production": [int(r["is_hallucination"]) for r in B],
        "full, without MI pruning (6 chunks)": [int(r["is_hallucination"]) for r in C],
    }
    results = {}
    for name, pred in variants.items():
        m = prf(y, pred)
        m["f1_ci95"] = bootstrap_f1(y, pred)
        results[name] = m
    sweep = {}
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pred = [int((r["max_entailment"] < thr and not r["is_filler"]) or bool(r["wrong_spans"])) for r in B]
        sweep[str(thr)] = prf(y, pred)
    by_kind = {}
    for kind in ("contradicts_or_miscopies_source", "invented_specific", "commentary_or_inference"):
        idx = [i for i, x in enumerate(items) if x["kind"] == kind]
        by_kind[kind] = {"n": len(idx), "recall": sum(variants["full (claims + span + filler) = production"][i]
                                                        for i in idx) / len(idx)}

    # ---- answer-level trust ablation (AUROC of each trust component for spotting unsupported answers)
    answers = {}
    for x in doc["items"]:
        key = (x["mode"], x["question"], x["answer"])
        a = answers.setdefault(key, {"labels": [], "logprob": x["avg_logprob"]})
        a["labels"].append(x["label"])
    print("answer-level trust ablation", flush=True)
    db = SessionLocal()
    authority = {d.id: d.authority for d in db.query(Document).all()}
    db.close()
    rows = []
    for (mode, q, answer), a in answers.items():
        if not ({"supported", "unsupported"} & set(a["labels"])):
            continue
        res = verify_sentences(split_sentences(answer), kept[q], use_claims=True)
        claims = [r for r in res if not r["is_filler"]]
        auth = {c["chunk_id"]: authority.get(c["document_id"], config.DEFAULT_AUTHORITY) for c in kept[q]}
        ent, _ = entity_score(answer, kept[q])
        sent, dsc = sentence_score(claims), doc_score(claims, auth)
        tok = token_prob_component(None, a["logprob"])
        rows.append({"y": int("unsupported" in a["labels"]), "sentence": sent, "entity": ent, "document": dsc,
                     "token_prob": tok, "final": aggregate_trust(sent, ent, dsc, tok)})
    ya = [r["y"] for r in rows]

    def auroc(key):
        scores = [1 - (r[key] if r[key] is not None else 0.0) for r in rows]
        return roc_auc_score(ya, scores) if len(set(ya)) > 1 else float("nan")

    trust_auc = {k: auroc(k) for k in ("sentence", "entity", "document", "token_prob", "final")}
    trust_auc["sentence + entity only"] = roc_auc_score(
        ya, [1 - (0.35 * r["sentence"] + 0.25 * r["entity"]) / 0.6 for r in rows])

    out = {"labeled_by": doc["labeled_by"], "n_sentences": len(items), "n_unsupported": sum(y),
           "variants": results, "threshold_sweep": sweep, "recall_by_error_kind": by_kind,
           "answer_level": {"n_answers": len(rows), "n_unsupported_answers": sum(ya), "auroc": trust_auc}}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "ablation.json").write_text(json.dumps(out, indent=2))

    print("\n| Variant | Precision | Recall | F1 (95% CI) | FPR |\n| --- | --- | --- | --- | --- |")
    for n, m in results.items():
        print(f"| {n} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} "
              f"({m['f1_ci95'][0]:.2f}-{m['f1_ci95'][1]:.2f}) | {m['false_positive_rate']:.2f} |")
    print("\nthreshold sweep (full): " + ", ".join(f"{t}: P{m['precision']:.2f}/R{m['recall']:.2f}/F1 {m['f1']:.2f}"
                                                   for t, m in sweep.items()))
    print("recall by error kind:", {k: f"{v['recall']:.2f} (n={v['n']})" for k, v in by_kind.items()})
    print("answer-level AUROC:", {k: round(v, 2) for k, v in trust_auc.items()}, f"({len(rows)} answers, {sum(ya)} unsupported)")


if __name__ == "__main__":
    main()

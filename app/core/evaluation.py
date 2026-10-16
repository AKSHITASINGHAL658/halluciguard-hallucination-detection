import json
import logging

from sqlalchemy.orm import Session

from app import config
from app.core.metrics import retrieval_precision_at_k
from app.core.mi_pruning import prune_chunks
from app.core.query_service import process_query, rescore_response
from app.core.rag_pipeline import get_api_call_count, get_generation_latencies, retrieve
from app.models import Query, Response

log = logging.getLogger("halluciguard.eval")


def _latest_response(db: Session, question: str) -> Response | None:
    return (db.query(Response).join(Query, Response.query_id == Query.id)
            .filter(Query.text == question, Response.faults_injected.is_(None)).order_by(Response.id.desc()).first())


def run_evaluation(db: Session, k: int = config.RETRIEVE_K, retrieval_only: bool = False,
                   rescore: bool = False, limit: int | None = None) -> dict:
    """retrieval_only: retrieval + pruning metrics, no generation.
    rescore: reuse stored answers from SQLite; rerun NLI, localization and trust only.
    limit: first N questions."""
    qa = json.loads(config.EVAL_SET_PATH.read_text())
    if limit:
        qa = qa[:limit]
    calls_before = get_api_call_count()
    lat_before = len(get_generation_latencies())
    token_probs = []
    precisions, pruned_prec, prune_ratios = [], [], []
    trusts, n_sent, n_halluc, failed, missing = [], 0, 0, 0, 0
    for item in qa:
        retrieved = retrieve(item["question"], k)
        kept = prune_chunks(item["question"], retrieved)
        precisions.append(retrieval_precision_at_k([c["title"] for c in retrieved], item["relevant_docs"], k))
        pruned_prec.append(retrieval_precision_at_k([c["title"] for c in kept], item["relevant_docs"]))
        prune_ratios.append(1 - len(kept) / len(retrieved) if retrieved else 0.0)
        if retrieval_only:
            continue
        try:
            if rescore:
                old = _latest_response(db, item["question"])
                if old is None:
                    missing += 1
                    continue
                resp = rescore_response(db, old)
            else:
                resp = process_query(db, item["question"], k)
        except Exception as e:  # noqa: BLE001
            log.warning("query failed during eval: %s", str(e)[:150])
            db.rollback()  # a failed flush would otherwise poison the session for every later question
            failed += 1
            continue
        trusts.append(resp.trust_score_overall)
        if resp.trust and resp.trust.token_prob_component is not None:
            token_probs.append(resp.trust.token_prob_component)
        claims = [s for s in resp.sentences if not s.is_filler]
        n_sent += len(claims)
        n_halluc += sum(1 for s in claims if s.is_hallucination)
    n = len(qa)
    out = {"n_questions": n, "n_failed": failed, "k": k,
           "mode": "retrieval_only" if retrieval_only else "rescore" if rescore else "full",
           "retrieval_precision_at_k": sum(precisions) / n,
           "precision_after_pruning": sum(pruned_prec) / n,
           "avg_prune_ratio": sum(prune_ratios) / n,
           "avg_trust": None, "hallucination_rate": None, "n_sentences": 0,
           "api_calls": get_api_call_count() - calls_before, "avg_token_prob": None,
           "avg_generation_latency_s": None}
    lats = get_generation_latencies()[lat_before:]
    if lats:
        out["avg_generation_latency_s"] = sum(lats) / len(lats)
    if not retrieval_only:
        out.update(avg_token_prob=sum(token_probs) / len(token_probs) if token_probs else None,
                   avg_trust=sum(trusts) / len(trusts) if trusts else None,
                   hallucination_rate=n_halluc / n_sent if n_sent else None, n_sentences=n_sent)
        if rescore:
            out["n_no_stored_answer"] = missing
    return out


def detector_report() -> dict:
    """Detector accuracy on the labeled sentence set, read from outputs/ablation.json (scripts/run_ablation.py)."""
    if not config.ABLATION_PATH.exists():
        return {"available": False}
    d = json.loads(config.ABLATION_PATH.read_text())
    variants = d.get("variants", {})
    return {"available": True, "labeled_by": d.get("labeled_by"), "n_sentences": d.get("n_sentences"),
            "n_unsupported": d.get("n_unsupported"),
            "production": variants.get("full (claims + span + filler) = production"),
            "nli_only": variants.get("NLI only (sentence level)"),
            "recall_by_error_kind": d.get("recall_by_error_kind", {}),
            "auroc": (d.get("answer_level") or {}).get("auroc")}

import json
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import config
from app.core.metrics import retrieval_precision_at_k
from app.core.mi_pruning import prune_chunks
from app.core.query_service import process_query
from app.core.rag_pipeline import retrieve
from app.db import get_db

router = APIRouter()
log = logging.getLogger("halluciguard.eval")


@router.post("/evaluate")
def evaluate(k: int = config.RETRIEVE_K, db: Session = Depends(get_db)):
    qa = json.loads(config.EVAL_SET_PATH.read_text())
    precisions, pruned_prec, prune_ratios = [], [], []
    trusts, n_sent, n_halluc, failed = [], 0, 0, 0
    for item in qa:
        retrieved = retrieve(item["question"], k)
        kept = prune_chunks(item["question"], retrieved)
        precisions.append(retrieval_precision_at_k([c["title"] for c in retrieved], item["relevant_docs"], k))
        pruned_prec.append(retrieval_precision_at_k([c["title"] for c in kept], item["relevant_docs"]))
        prune_ratios.append(1 - len(kept) / len(retrieved) if retrieved else 0.0)
        try:
            resp = process_query(db, item["question"], k)
        except Exception as e:  # noqa: BLE001
            log.warning("query failed during eval: %s", str(e)[:150])
            failed += 1
            continue
        trusts.append(resp.trust_score_overall)
        n_sent += len(resp.sentences)
        n_halluc += sum(1 for s in resp.sentences if s.is_hallucination)
    n = len(qa)
    return {"n_questions": n, "n_failed": failed, "k": k,
            "retrieval_precision_at_k": sum(precisions) / n,
            "precision_after_pruning": sum(pruned_prec) / n,
            "avg_prune_ratio": sum(prune_ratios) / n,
            "avg_trust": sum(trusts) / len(trusts) if trusts else None,
            "hallucination_rate": n_halluc / n_sent if n_sent else None,
            "n_sentences": n_sent}

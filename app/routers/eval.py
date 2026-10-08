import json

from fastapi import APIRouter

from app import config
from app.core.metrics import retrieval_precision_at_k
from app.core.mi_pruning import prune_chunks
from app.core.rag_pipeline import retrieve

router = APIRouter()


@router.post("/evaluate")
def evaluate(k: int = config.RETRIEVE_K):
    qa = json.loads(config.EVAL_SET_PATH.read_text())
    precisions, pruned_prec, prune_ratios = [], [], []
    for item in qa:
        retrieved = retrieve(item["question"], k)
        kept = prune_chunks(item["question"], retrieved)
        precisions.append(retrieval_precision_at_k([c["title"] for c in retrieved], item["relevant_docs"], k))
        pruned_prec.append(retrieval_precision_at_k([c["title"] for c in kept], item["relevant_docs"]))
        prune_ratios.append(1 - len(kept) / len(retrieved) if retrieved else 0.0)
    n = len(qa)
    return {"n_questions": n, "k": k, "retrieval_precision_at_k": sum(precisions) / n,
            "precision_after_pruning": sum(pruned_prec) / n,
            "avg_prune_ratio": sum(prune_ratios) / n}

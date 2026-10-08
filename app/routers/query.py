import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import config
from app.core.rag_pipeline import run_pipeline
from app.db import get_db
from app.models import Chunk, Query, Response
from app.schemas import QueryIn, ResponseOut

router = APIRouter()


def build_response_out(db: Session, resp: Response) -> ResponseOut:
    return ResponseOut(
        response_id=resp.id, query=resp.query.text, answer=resp.raw_text, model_name=resp.model_name,
        logprobs_source=resp.logprobs_source,
        retrieved_chunk_ids=json.loads(resp.retrieved_chunk_ids), pruned_chunk_ids=json.loads(resp.pruned_chunk_ids),
        trust_score_overall=resp.trust_score_overall)


@router.post("/query", response_model=ResponseOut)
def post_query(body: QueryIn, db: Session = Depends(get_db)):
    result = run_pipeline(body.text, body.k)
    q = Query(text=body.text)
    db.add(q)
    db.flush()
    resp = Response(query_id=q.id, raw_text=result["answer"], model_name=config.GEMINI_MODEL,
                    retrieved_chunk_ids=json.dumps([c["chunk_id"] for c in result["retrieved"]]),
                    pruned_chunk_ids=json.dumps([c["chunk_id"] for c in result["pruned"]]),
                    logprobs_source=result["logprobs_source"])
    db.add(resp)
    db.commit()
    return build_response_out(db, resp)


@router.get("/response/{response_id}", response_model=ResponseOut)
def get_response(response_id: int, db: Session = Depends(get_db)):
    resp = db.get(Response, response_id)
    if not resp:
        raise HTTPException(404, "response not found")
    return build_response_out(db, resp)

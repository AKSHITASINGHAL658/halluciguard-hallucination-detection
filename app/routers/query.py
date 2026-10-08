import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import config
from app.core.attention_extractor import top_contributing_tokens
from app.core.rag_pipeline import run_pipeline
from app.core.sentence_localizer import localize_hallucinations, split_sentences
from app.db import get_db
from app.models import Chunk, Query, Response, ResponseSentence
from app.schemas import QueryIn, ResponseOut, SentenceOut

router = APIRouter()


def build_response_out(db: Session, resp: Response) -> ResponseOut:
    sentences = []
    for s in sorted(resp.sentences, key=lambda x: x.sentence_index):
        chunk = db.get(Chunk, s.supporting_chunk_id) if s.supporting_chunk_id else None
        sentences.append(SentenceOut(
            index=s.sentence_index, text=s.text, nli_label=s.nli_label, nli_confidence=s.nli_confidence,
            is_hallucination=s.is_hallucination, supporting_chunk_id=s.supporting_chunk_id,
            supporting_chunk_text=chunk.text if chunk else None,
            top_attention_tokens=json.loads(s.top_attention_tokens) if s.top_attention_tokens else []))
    return ResponseOut(
        sentences=sentences,
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
    db.flush()
    loc = localize_hallucinations(split_sentences(result["answer"]), result["pruned"])
    for s in loc:
        db.add(ResponseSentence(
            response_id=resp.id, sentence_index=s["index"], text=s["text"], nli_label=s["label"],
            nli_confidence=s["confidence"], is_hallucination=s["is_hallucination"],
            supporting_chunk_id=s["best_chunk_id"],
            top_attention_tokens=json.dumps(top_contributing_tokens(s["text"], s["best_chunk_text"]))
            if s["is_hallucination"] and s["best_chunk_text"] else None))
    db.commit()
    db.refresh(resp)
    return build_response_out(db, resp)


@router.get("/response/{response_id}", response_model=ResponseOut)
def get_response(response_id: int, db: Session = Depends(get_db)):
    resp = db.get(Response, response_id)
    if not resp:
        raise HTTPException(404, "response not found")
    return build_response_out(db, resp)

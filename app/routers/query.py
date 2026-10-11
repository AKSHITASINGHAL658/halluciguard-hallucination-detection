import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.query_service import process_query
from app.db import get_db
from app.models import Chunk, Response
from app.schemas import EntityOut, QueryIn, ResponseOut, SentenceOut, TrustOut

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
    t = resp.trust
    trust = TrustOut(entity_level_score=t.entity_level_score, sentence_level_score=t.sentence_level_score,
                     document_level_score=t.document_level_score, token_prob_component=t.token_prob_component,
                     final_score=t.final_score) if t else None
    entities = [EntityOut(text=e.text, entity_type=e.entity_type, supported=e.supported,
                          entity_trust_score=e.entity_trust_score) for e in resp.entities]
    return ResponseOut(
        sentences=sentences, entities=entities, trust=trust,
        response_id=resp.id, query=resp.query.text, answer=resp.raw_text, model_name=resp.model_name, generator=resp.generator,
        logprobs_source=resp.logprobs_source,
        retrieved_chunk_ids=json.loads(resp.retrieved_chunk_ids), pruned_chunk_ids=json.loads(resp.pruned_chunk_ids),
        trust_score_overall=resp.trust_score_overall,
        injected_faults=json.loads(resp.faults_injected) if resp.faults_injected else [])


@router.post("/query", response_model=ResponseOut)
def post_query(body: QueryIn, db: Session = Depends(get_db)):
    return build_response_out(db, process_query(db, body.text, body.k, body.inject_faults))


@router.get("/response/{response_id}", response_model=ResponseOut)
def get_response(response_id: int, db: Session = Depends(get_db)):
    resp = db.get(Response, response_id)
    if not resp:
        raise HTTPException(404, "response not found")
    return build_response_out(db, resp)

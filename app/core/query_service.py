import json

from sqlalchemy.orm import Session

from app import config
from app.core.attention_extractor import top_contributing_tokens
from app.core.rag_pipeline import run_pipeline
from app.core.sentence_localizer import localize_hallucinations, split_sentences
from app.core.trust_engine import (aggregate_trust, doc_score, entity_score, sentence_score,
                                   token_prob_component)
from app.models import Document, Entity, Query, Response, ResponseSentence, TrustScore


def process_query(db: Session, text: str, k: int = config.RETRIEVE_K) -> Response:
    """Full pipeline: retrieve -> prune -> generate -> NLI -> localize -> attention -> trust; persists everything."""
    result = run_pipeline(text, k)
    q = Query(text=text)
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
        flagged = s["is_hallucination"] and s["best_chunk_text"]
        db.add(ResponseSentence(
            response_id=resp.id, sentence_index=s["index"], text=s["text"], nli_label=s["label"],
            nli_confidence=s["confidence"], is_hallucination=s["is_hallucination"],
            supporting_chunk_id=s["best_chunk_id"],
            top_attention_tokens=json.dumps(top_contributing_tokens(s["text"], s["best_chunk_text"]))
            if flagged else None))

    ent_s, entities = entity_score(result["answer"], result["pruned"])
    for e in entities:
        db.add(Entity(response_id=resp.id, text=e["text"], entity_type=e["entity_type"],
                      supported=e["supported"], entity_trust_score=e["score"]))
    authority = {c["chunk_id"]: (db.get(Document, c["document_id"]).authority) for c in result["pruned"]}
    sent_s = sentence_score(loc)
    doc_s = doc_score(loc, authority)
    tok_s = token_prob_component(result["token_logprobs"], result["avg_logprob"])
    final = aggregate_trust(sent_s, ent_s, doc_s, tok_s)
    db.add(TrustScore(response_id=resp.id, entity_level_score=ent_s, sentence_level_score=sent_s,
                      document_level_score=doc_s, token_prob_component=tok_s, final_score=final))
    resp.trust_score_overall = final
    db.commit()
    db.refresh(resp)
    return resp

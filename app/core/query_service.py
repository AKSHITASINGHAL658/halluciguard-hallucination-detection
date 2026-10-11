import json
import re

from sqlalchemy.orm import Session

from app import config
from app.core.attention_extractor import top_contributing_tokens
from app.core import adversarial_agent, rag_pipeline
from app.core.rag_pipeline import run_pipeline
from app.core.sentence_localizer import split_sentences
from app.core.verification import verify_sentences
from app.core.trust_engine import (aggregate_trust, doc_score, entity_score, sentence_score,
                                   token_prob_component)
from app.models import Chunk, Document, Entity, Query, Response, ResponseSentence, TrustScore


def process_query(db: Session, text: str, k: int = config.RETRIEVE_K, inject_faults: bool = False) -> Response:
    """Full pipeline: retrieve -> prune -> generate -> NLI -> localize -> attention -> trust; persists everything.

    inject_faults (demo): corrupt the generated answer first. Such responses are tagged via faults_injected and
    excluded from rescoring/eval. Logprobs describe the original text, so the token-prob component is dropped.
    """
    result = run_pipeline(text, k)
    if inject_faults:
        result["answer"], result["faults"] = adversarial_agent.inject_faults(result["answer"])
        result["token_prob"] = None
    return score_and_store(db, text, result)


def rescore_response(db: Session, old: Response) -> Response:
    """Reuse a stored answer and its chunks; rerun NLI, localization, attention and trust only (no LLM call)."""
    pruned_ids = json.loads(old.pruned_chunk_ids)
    retrieved_ids = json.loads(old.retrieved_chunk_ids)
    rows = {c.id: c for c in db.query(Chunk).filter(Chunk.id.in_(set(pruned_ids) | set(retrieved_ids))).all()}

    def to_dict(cid):
        c = rows[cid]
        return {"chunk_id": c.id, "document_id": c.document_id, "text": c.text}

    result = {"answer": old.raw_text, "retrieved": [to_dict(i) for i in retrieved_ids if i in rows],
              "pruned": [to_dict(i) for i in pruned_ids if i in rows], "token_logprobs": None, "avg_logprob": None,
              "logprobs_source": old.logprobs_source, "model_name": old.model_name, "generator": old.generator,
              "token_prob": old.trust.token_prob_component if old.trust else None}
    return score_and_store(db, old.query.text, result)


def fix_split_decimals(answer: str) -> str:
    """Small models sometimes write decimals as '42. 5'; rejoin them so sentence splitting and number checks work."""
    return re.sub(r"(?<=\d)\. (?=\d)", ".", answer)


def score_and_store(db: Session, text: str, result: dict) -> Response:
    result = {**result, "answer": fix_split_decimals(result["answer"])}
    q = Query(text=text)
    db.add(q)
    db.flush()
    resp = Response(query_id=q.id, raw_text=result["answer"], model_name=result.get("model_name", rag_pipeline.active_model()),
                    generator=result.get("generator", "ollama"),
                    faults_injected=json.dumps(result["faults"]) if result.get("faults") else None,
                    retrieved_chunk_ids=json.dumps([c["chunk_id"] for c in result["retrieved"]]),
                    pruned_chunk_ids=json.dumps([c["chunk_id"] for c in result["pruned"]]),
                    logprobs_source=result["logprobs_source"])
    db.add(resp)
    db.flush()

    loc = verify_sentences(split_sentences(result["answer"]), result["pruned"])
    for s in loc:
        spans, nli_flagged = s["wrong_spans"], s["nli_flagged"]
        flagged = nli_flagged and s["best_chunk_text"]  # attention explains NLI rejections only
        db.add(ResponseSentence(
            response_id=resp.id, sentence_index=s["index"], text=s["text"], nli_label=s["label"],
            nli_confidence=s["confidence"], is_hallucination=s["is_hallucination"],
            supporting_chunk_id=s["best_chunk_id"],
            top_attention_tokens=json.dumps(top_contributing_tokens(s["text"], s["best_chunk_text"]))
            if flagged else None,
            wrong_spans=json.dumps(spans) if spans else None, is_filler=s["is_filler"]))

    ent_s, entities = entity_score(result["answer"], result["pruned"])
    for e in entities:
        db.add(Entity(response_id=resp.id, text=e["text"], entity_type=e["entity_type"],
                      supported=e["supported"], entity_trust_score=e["score"]))
    authority = {c["chunk_id"]: (db.get(Document, c["document_id"]).authority) for c in result["pruned"]}
    claims = [s for s in loc if not s["is_filler"]]
    sent_s = sentence_score(claims)
    doc_s = doc_score(claims, authority)
    tok_s = result["token_prob"] if "token_prob" in result else token_prob_component(
        result["token_logprobs"], result["avg_logprob"])
    final = aggregate_trust(sent_s, ent_s, doc_s, tok_s)
    db.add(TrustScore(response_id=resp.id, entity_level_score=ent_s, sentence_level_score=sent_s,
                      document_level_score=doc_s, token_prob_component=tok_s, final_score=final))
    resp.trust_score_overall = final
    db.commit()
    db.refresh(resp)
    return resp

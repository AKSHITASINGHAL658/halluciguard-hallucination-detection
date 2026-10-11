from typing import Optional

from pydantic import BaseModel


class DocumentIn(BaseModel):
    title: str
    text: str
    source: Optional[str] = None
    authority: float = 0.8


class DocumentOut(BaseModel):
    id: int
    title: str
    source: Optional[str] = None
    authority: float
    n_chunks: int


class QueryIn(BaseModel):
    text: str
    k: int = 6
    inject_faults: bool = False  # demo: corrupt the answer before verification


class SentenceOut(BaseModel):
    index: int
    text: str
    nli_label: Optional[str] = None
    nli_confidence: Optional[float] = None
    is_hallucination: Optional[bool] = None
    supporting_chunk_id: Optional[int] = None
    supporting_chunk_text: Optional[str] = None
    top_attention_tokens: list = []


class EntityOut(BaseModel):
    text: str
    entity_type: str
    supported: bool
    entity_trust_score: float


class TrustOut(BaseModel):
    entity_level_score: Optional[float] = None
    sentence_level_score: Optional[float] = None
    document_level_score: Optional[float] = None
    token_prob_component: Optional[float] = None
    final_score: Optional[float] = None


class ResponseOut(BaseModel):
    response_id: int
    query: str
    answer: str
    model_name: str
    generator: Optional[str] = None
    injected_faults: list[dict] = []
    logprobs_source: Optional[str] = None
    retrieved_chunk_ids: list[int]
    pruned_chunk_ids: list[int]
    sentences: list[SentenceOut] = []
    entities: list[EntityOut] = []
    trust: Optional[TrustOut] = None
    trust_score_overall: Optional[float] = None

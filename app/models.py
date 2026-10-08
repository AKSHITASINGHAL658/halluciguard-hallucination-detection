from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db import Base


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    source = Column(String)
    authority = Column(Float, default=0.8)
    created_at = Column(DateTime, default=datetime.utcnow)
    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(Text)
    chunk_index = Column(Integer)
    embedding_id = Column(String)
    document = relationship("Document", back_populates="chunks")
    adversarial_tests = relationship("AdversarialTest", back_populates="chunk")


class Query(Base):
    __tablename__ = "queries"
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    responses = relationship("Response", back_populates="query")


class Response(Base):
    __tablename__ = "responses"
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("queries.id"))
    raw_text = Column(Text)
    model_name = Column(String)
    retrieved_chunk_ids = Column(Text)  # JSON list
    pruned_chunk_ids = Column(Text)  # JSON list
    trust_score_overall = Column(Float, nullable=True)
    logprobs_source = Column(String, nullable=True)  # tokens | avg | none
    created_at = Column(DateTime, default=datetime.utcnow)
    query = relationship("Query", back_populates="responses")
    sentences = relationship("ResponseSentence", back_populates="response")
    entities = relationship("Entity", back_populates="response")
    trust = relationship("TrustScore", back_populates="response", uselist=False)


class ResponseSentence(Base):
    __tablename__ = "response_sentences"
    id = Column(Integer, primary_key=True)
    response_id = Column(Integer, ForeignKey("responses.id"))
    sentence_index = Column(Integer)
    text = Column(Text)
    nli_label = Column(String, nullable=True)
    nli_confidence = Column(Float, nullable=True)
    is_hallucination = Column(Boolean, nullable=True)
    supporting_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=True)
    top_attention_tokens = Column(Text, nullable=True)  # JSON list
    response = relationship("Response", back_populates="sentences")


class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True)
    response_id = Column(Integer, ForeignKey("responses.id"))
    text = Column(String)
    entity_type = Column(String)
    supported = Column(Boolean)
    entity_trust_score = Column(Float)
    response = relationship("Response", back_populates="entities")


class TrustScore(Base):
    __tablename__ = "trust_scores"
    id = Column(Integer, primary_key=True)
    response_id = Column(Integer, ForeignKey("responses.id"), unique=True)
    entity_level_score = Column(Float)
    sentence_level_score = Column(Float)
    document_level_score = Column(Float)
    token_prob_component = Column(Float, nullable=True)
    final_score = Column(Float)
    response = relationship("Response", back_populates="trust")


class AdversarialTest(Base):
    __tablename__ = "adversarial_tests"
    id = Column(Integer, primary_key=True)
    original_chunk_id = Column(Integer, ForeignKey("chunks.id"))
    attack_type = Column(String)
    original_sentence = Column(Text)
    corrupted_text = Column(Text)
    nli_label = Column(String)
    detected = Column(Boolean)
    run_at = Column(DateTime, default=datetime.utcnow)
    chunk = relationship("Chunk", back_populates="adversarial_tests")

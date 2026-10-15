from app import config
from app.core.ingestion import authority_for
from app.db import SessionLocal
from app.models import Document


def test_authority_lookup_and_default():
    assert authority_for("renewable_policy_investment.txt") == 0.90
    assert authority_for("unknown_doc.txt") == config.DEFAULT_AUTHORITY


def test_seeded_documents_get_distinct_authority():
    db = SessionLocal()
    try:
        docs = {d.source: d.authority for d in db.query(Document).all()}
    finally:
        db.close()
    assert docs == {name: config.DOC_AUTHORITY[name] for name in config.DOC_AUTHORITY}
    assert len(set(docs.values())) > 1  # the document-level score now carries information

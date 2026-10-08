from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.ingestion import ingest_document
from app.db import get_db
from app.models import Document
from app.schemas import DocumentIn, DocumentOut

router = APIRouter()


def _out(d: Document) -> DocumentOut:
    return DocumentOut(id=d.id, title=d.title, source=d.source, authority=d.authority, n_chunks=len(d.chunks))


@router.post("/documents", response_model=DocumentOut)
def post_document(body: DocumentIn, db: Session = Depends(get_db)):
    doc = ingest_document(db, body.title, body.text, body.source or body.title, body.authority)
    return _out(doc)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return [_out(d) for d in db.query(Document).order_by(Document.id).all()]

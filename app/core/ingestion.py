from sqlalchemy.orm import Session

from app import config
from app.core.chunking import split_into_chunks
from app.core.embeddings import embed_texts
from app.core import vector_store
from app.models import Chunk, Document


def ingest_document(db: Session, title: str, text: str, source: str, authority: float = config.DEFAULT_AUTHORITY) -> Document:
    doc = Document(title=title, source=source, authority=authority)
    db.add(doc)
    db.flush()
    pieces = split_into_chunks(text)
    rows = []
    for i, piece in enumerate(pieces):
        row = Chunk(document_id=doc.id, text=piece, chunk_index=i)
        db.add(row)
        rows.append(row)
    db.flush()
    for row in rows:
        row.embedding_id = f"chunk-{row.id}"
    embeddings = embed_texts(pieces)
    vector_store.add([r.embedding_id for r in rows], pieces, embeddings,
                     [{"chunk_id": r.id, "document_id": doc.id, "title": title} for r in rows])
    db.commit()
    return doc

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.core import vector_store  # noqa: E402
from app.core.ingestion import ingest_document  # noqa: E402
from app.db import Base, SessionLocal, engine, init_db  # noqa: E402


def main():
    Base.metadata.drop_all(engine)
    vector_store.reset()
    init_db()
    db = SessionLocal()
    files = sorted(config.RAW_DOCS_DIR.glob("*.txt"))
    for f in files:
        title = f.stem.replace("_", " ").title()
        doc = ingest_document(db, title, f.read_text(), f.name)
        print(f"ingested {f.name} -> doc {doc.id}, {len(doc.chunks)} chunks")
    db.close()
    print(f"seeded {len(files)} documents")


if __name__ == "__main__":
    main()

"""Apply config.DOC_AUTHORITY to the documents already stored in the database (no re-seeding needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.ingestion import authority_for  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.models import Document  # noqa: E402


def main():
    init_db()
    db = SessionLocal()
    for d in db.query(Document).order_by(Document.id).all():
        new = authority_for(d.source)
        print(f"{d.source}: {d.authority} -> {new}")
        d.authority = new
    db.commit()
    db.close()


if __name__ == "__main__":
    main()

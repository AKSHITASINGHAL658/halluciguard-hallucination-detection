import atexit
import os
import shutil
import tempfile

import pytest

# Tests run against a throwaway SQLite DB and Chroma dir (seeded from data/raw_docs), never the real app.db/chroma.
# This must happen before any `app` module is imported, so it lives at conftest import time.
_TMP = tempfile.mkdtemp(prefix="halluciguard-tests-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
os.environ["DB_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["CHROMA_DIR"] = os.path.join(_TMP, "chroma")
os.environ.setdefault("WARMUP", "0")  # tests must not preload models or ping Ollama at API startup


@pytest.fixture(scope="session", autouse=True)
def _seed_test_corpus():
    from app import config
    from app.core.ingestion import ingest_document
    from app.db import SessionLocal, init_db

    assert "halluciguard-tests-" in config.DB_URL and "halluciguard-tests-" in config.CHROMA_DIR
    init_db()
    db = SessionLocal()
    try:
        for f in sorted(config.RAW_DOCS_DIR.glob("*.txt")):
            ingest_document(db, f.stem.replace("_", " ").title(), f.read_text(), f.name)
    finally:
        db.close()
    yield

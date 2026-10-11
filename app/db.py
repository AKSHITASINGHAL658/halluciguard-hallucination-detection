from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app import config

engine = create_engine(config.DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def init_db():
    from app import models  # noqa: F401

    Base.metadata.create_all(engine)
    # lightweight migration: add columns introduced after the first release
    from sqlalchemy import inspect, text

    cols = {c["name"] for c in inspect(engine).get_columns("responses")}
    sent_cols = {c["name"] for c in inspect(engine).get_columns("response_sentences")}
    for table, existing, col, ddl in (("responses", cols, "generator", "VARCHAR"),
                                      ("responses", cols, "faults_injected", "TEXT"),
                                      ("response_sentences", sent_cols, "wrong_spans", "TEXT"),
                                      ("response_sentences", sent_cols, "is_filler", "BOOLEAN")):
        if col not in existing:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

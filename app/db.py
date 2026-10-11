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
    for col, ddl in (("generator", "VARCHAR"), ("faults_injected", "TEXT")):
        if col not in cols:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE responses ADD COLUMN {col} {ddl}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

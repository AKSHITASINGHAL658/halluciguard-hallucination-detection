from fastapi import FastAPI

from app.db import init_db
from app.routers import ingest, query

app = FastAPI(title="HalluciGuard")
app.include_router(ingest.router)
app.include_router(query.router)


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}

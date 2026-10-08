from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import config
from app.core.adversarial_agent import run_attack_suite, summarize_results
from app.db import get_db

router = APIRouter()


@router.post("/adversarial/run")
def run(n_chunks: int = config.ADV_N_CHUNKS, db: Session = Depends(get_db)):
    return run_attack_suite(db, n_chunks)


@router.get("/adversarial/results")
def results(db: Session = Depends(get_db)):
    return summarize_results(db)

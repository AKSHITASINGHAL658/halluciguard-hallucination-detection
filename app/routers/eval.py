from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import config
from app.core.evaluation import detector_report, run_evaluation
from app.db import get_db

router = APIRouter()


@router.post("/evaluate")
def evaluate(k: int = config.RETRIEVE_K, retrieval_only: bool = False, rescore: bool = False,
             limit: int | None = None, db: Session = Depends(get_db)):
    return run_evaluation(db, k, retrieval_only, rescore, limit)


@router.get("/evaluate/detector")
def detector():
    return detector_report()

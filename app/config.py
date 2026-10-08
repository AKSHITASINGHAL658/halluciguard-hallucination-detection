import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
TEMPERATURE = 0.0

DB_URL = f"sqlite:///{BASE_DIR / 'app.db'}"
CHROMA_DIR = str(BASE_DIR / "chroma")
CHROMA_COLLECTION = "halluciguard_chunks"
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
EVAL_SET_PATH = BASE_DIR / "data" / "eval" / "qa_eval_set.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_MODEL_FALLBACK = "cross-encoder/nli-deberta-v3-base"
SPACY_MODEL = "en_core_web_sm"

CHUNK_WORDS = 250
CHUNK_OVERLAP = 40
RETRIEVE_K = 6
DEFAULT_AUTHORITY = 0.8

# MI pruning
PRUNE_KEEP_RATIO = 0.5
PRUNE_MIN_KEEP = 2
PRUNE_REDUNDANCY = 0.85
PMI_EPS = 1e-6

# NLI / localization
NLI_BATCH_SIZE = 8
NLI_MAX_LENGTH = 512
HALLUCINATION_ENTAIL_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.5
ATTENTION_LAST_LAYERS = 4
ATTENTION_TOP_N = 5

# Trust
TRUST_WEIGHTS = {"sentence": 0.35, "entity": 0.25, "document": 0.20, "token_prob": 0.20}

# Adversarial
ADV_N_CHUNKS = 20
ADV_SEED = 42

SYSTEM_PROMPT = (
    "Answer the question using only the numbered context passages below. "
    "Respond in 3-6 short sentences. If the context does not contain the answer, say so."
)

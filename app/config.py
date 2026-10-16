# config.py — Central configuration for HalluciGuard
import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root directory (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent
# Load environment variables from the .env file at project root
load_dotenv(BASE_DIR / ".env")

TEMPERATURE = 0.0


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "qwen2.5:3b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE") or "30m"  # keep the model loaded between queries
WARMUP = os.getenv("WARMUP", "1").strip().lower() in ("1", "true", "yes", "on")  # preload models at API startup
LLM_CACHE = _env_bool("LLM_CACHE", True)
LLM_CACHE_DIR = BASE_DIR / "cache" / "llm"

DB_URL = os.getenv("DB_URL") or f"sqlite:///{BASE_DIR / 'app.db'}"
CHROMA_DIR = os.getenv("CHROMA_DIR") or str(BASE_DIR / "chroma")
CHROMA_COLLECTION = "halluciguard_chunks"
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
EVAL_SET_PATH = BASE_DIR / "data" / "eval" / "qa_eval_set.json"
ABLATION_PATH = BASE_DIR / "outputs" / "ablation.json"  # detector precision/recall, written by scripts/run_ablation.py

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
NLI_PAIR_CACHE_SIZE = 5000  # in-memory (premise, hypothesis) -> result entries; 0 disables
HALLUCINATION_ENTAIL_THRESHOLD = 0.5
USE_CLAIMS = _env_bool("USE_CLAIMS", True)  # split sentences into atomic claims before NLI
CONTRADICTION_THRESHOLD = 0.5
ATTENTION_LAST_LAYERS = 4
ATTENTION_TOP_N = 5
ATTENTION_VALIDATE = _env_bool("ATTENTION_VALIDATE", True)  # mask-and-rerun check on the top attention words
ATTENTION_CHECK_RANDOM = 5       # random-word masks used as the control
ATTENTION_CHECK_MARGIN = 0.05    # top-word shift must beat the random-word shift by at least this much

# Span checking: a number found only in the NLI-preferred chunk still counts as supported if NLI is this sure
SPAN_STRONG_ENTAILMENT = 0.9

# Trust
TRUST_WEIGHTS = {"sentence": 0.35, "entity": 0.25, "document": 0.20, "token_prob": 0.20}

# Adversarial
ADV_N_CHUNKS = 20
ADV_SEED = 42

SYSTEM_PROMPT_STRICT = (
    "Answer the question using only the numbered context passages below. "
    "Respond in 3-6 short sentences. If the context does not contain the answer, say so."
)
# Demo mode: invites the model to go beyond the context so unsupported claims appear and can be detected.
SYSTEM_PROMPT_LOOSE = (
    "Answer the question thoroughly in 4-6 sentences. The context passages below are a starting point; "
    "enrich the answer with additional background, specific figures, dates, named examples and comparisons "
    "from your own knowledge. Do not mention the context or say when information is missing."
)
PROMPT_MODE = os.getenv("PROMPT_MODE", "strict").strip().lower()  # "strict" | "loose"


def get_system_prompt() -> str:
    return SYSTEM_PROMPT_LOOSE if PROMPT_MODE == "loose" else SYSTEM_PROMPT_STRICT

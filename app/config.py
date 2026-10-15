import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_MODEL_FALLBACK = "cross-encoder/nli-deberta-v3-base"
SPACY_MODEL = "en_core_web_sm"

CHUNK_WORDS = 250
CHUNK_OVERLAP = 40
RETRIEVE_K = 6
DEFAULT_AUTHORITY = 0.8
# Source authority (0-1) per document file, used by the document-level trust score. Assigned by source type:
# documents built mainly on official statistics and intergovernmental reports rank higher; documents that mix in
# vendor targets and forward-looking claims rank lower. These are judgement values: edit them, then run
# scripts/update_authority.py to apply them to an existing database.
DOC_AUTHORITY = {
    "renewable_policy_investment.txt": 0.90,  # IEA, IRENA, BloombergNEF, COP28, EU and US legislation
    "solar_power.txt": 0.85,                  # IRENA statistics plus company lab-efficiency claims
    "wind_power.txt": 0.85,                   # GWEC statistics plus manufacturer specifications
    "hydropower.txt": 0.85,                   # IEA statistics and well-documented plants
    "geothermal_energy.txt": 0.80,            # IRENA statistics plus pilot projects and DOE targets
    "energy_storage.txt": 0.75,               # market surveys plus vendor targets and forecasts
}

# MI pruning
PRUNE_KEEP_RATIO = 0.5
PRUNE_MIN_KEEP = 2
PRUNE_REDUNDANCY = 0.85
PMI_EPS = 1e-6

# NLI / localization
NLI_BATCH_SIZE = 8
NLI_MAX_LENGTH = 512
HALLUCINATION_ENTAIL_THRESHOLD = 0.5
USE_CLAIMS = _env_bool("USE_CLAIMS", True)  # split sentences into atomic claims before NLI
CONTRADICTION_THRESHOLD = 0.5
ATTENTION_LAST_LAYERS = 4
ATTENTION_TOP_N = 5

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

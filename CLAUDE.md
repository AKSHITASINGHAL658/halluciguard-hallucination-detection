# HalluciGuard
RAG + hallucination detection. Python 3.11, CPU only. Solo developer.
Full plan: docs/plan.md. Read ONLY the section named in the task.

## Architecture
- FastAPI backend (app/), React + Vite dashboard (frontend/), SQLite + ChromaDB.
- Logic lives in app/core/ as plain functions; routers only call core.
- Folder layout: docs/plan.md §3. DB schema: §4. API list: §5.

## Stack
Ollama qwen2.5:3b (local, logprobs=True; no cloud LLMs), ChromaDB (embedded),
sentence-transformers/all-MiniLM-L6-v2,
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, spaCy en_core_web_sm,
nltk, scikit-learn, SQLAlchemy, React + Vite, pytest.

## Rules
- Constants and model names only in app/config.py.
- NLI model loaded only via get_nli_model() singleton in nli_verifier.py.
  Read labels from model.config.id2label, never hardcode indices.
- /query is synchronous and returns the full result.
- Don't open data/, chroma/, app.db, .env or docs/*.pdf.
- Only edit files named in the task. No refactoring elsewhere.
- Write or update the matching test; run only that test file.
- Keep replies short: summarize changes, don't print whole files.
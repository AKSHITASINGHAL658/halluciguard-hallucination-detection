# HalluciGuard

RAG pipeline with hallucination detection: retrieval (ChromaDB) → MI pruning → Gemini generation → sentence-level NLI verification (DeBERTa) → attention attribution → hierarchical trust score. FastAPI backend, Streamlit dashboard, SQLite storage. CPU only, Python 3.11.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env     # set GEMINI_API_KEY (and optionally GEMINI_MODEL)
python scripts/seed_corpus.py
uvicorn app.main:app --port 8000
streamlit run dashboard/streamlit_app.py     # in a second terminal
python scripts/run_adversarial_eval.py       # optional: populate the Robustness tab
pytest
```

`GEMINI_MODEL` defaults to `gemini-3.1-flash-lite` (see `app/config.py`). Gemini model access and daily quotas vary per key; `gemini-2.0-flash` is retired.

## Final results

Corpus: 6 renewable-energy documents (12 chunks). Eval set: 22 QA pairs (`data/eval/qa_eval_set.json`). Model: `gemini-3.1-flash-lite`.

| Metric | Value |
| --- | --- |
| Retrieval precision@6 | 0.33 (capped: each doc has only 2 chunks, so at most 2 of 6 can be relevant) |
| Precision after MI pruning | 0.74 |
| Avg prune ratio (chunks dropped) | 62% |
| Sentence hallucination rate | 13.0% (54 sentences, 22 questions, 0 failed) |
| Avg trust score | 0.79 |

Adversarial robustness (NLI premise = original chunk, hypothesis = corrupted sentence, 96 tests):

| Attack | Detection rate |
| --- | --- |
| Negation | 100% |
| Stale fact | 95.8% |
| Noise | 100% |
| Control false-positive rate | 12.5% |

## Notes and limitations

- **No token-probability component**: the Gemini models available to this key reject `response_logprobs` (HTTP 400 "Logprobs is not enabled"). `generate()` falls back to no logprobs, `token_prob_component` is null, and the trust weights renormalize over sentence/entity/document.
- Attention output is described as "source words the verifier attended to most" (self-attention over premise+hypothesis), not proof of causation.
- Sentences that rely on a pronoun ("It was completed in 2012") may be flagged because the chunk lacks the antecedent.
- Numbers vary slightly between runs because LLM output is not fully deterministic.
- Endpoints: `POST /documents`, `GET /documents`, `POST /query`, `GET /response/{id}`, `POST /evaluate`, `POST /adversarial/run`, `GET /adversarial/results`, `GET /health`.

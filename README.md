# HalluciGuard

RAG pipeline with hallucination detection: retrieval (ChromaDB) → MI pruning → local Ollama generation → sentence-level NLI verification (DeBERTa) → attention attribution → hierarchical trust score. FastAPI backend, Streamlit dashboard, SQLite storage. CPU only, Python 3.11.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama serve &            # local LLM server
ollama pull qwen2.5:3b
cp .env.example .env      # optional overrides (OLLAMA_MODEL, OLLAMA_HOST)
python scripts/seed_corpus.py
uvicorn app.main:app --port 8000
streamlit run dashboard/streamlit_app.py     # in a second terminal
python scripts/run_adversarial_eval.py       # optional: populate the Robustness tab
pytest
```

Generation runs locally through Ollama (`OLLAMA_MODEL=qwen2.5:3b`, `OLLAMA_HOST=http://localhost:11434`). No cloud LLM or API key is used.

## LLM cache and logprobs

- Ollama returns per-token logprobs, so `logprobs_source` is `tokens` and the token-probability trust component is used. If a model returns none, the component is null and the weights renormalize.
- `LLM_CACHE` (default `true`, env override): `generate()` caches responses in `cache/llm/<sha256(model+prompt)>.json` and logs cache hit/miss plus the count of real API calls.
- `POST /evaluate` and `python scripts/run_eval.py` accept:
  - `retrieval_only` / `--retrieval-only`: retrieval and pruning metrics only, no generation.
  - `rescore` / `--rescore`: reuse the latest stored answer per question from SQLite and rerun NLI, localization and trust only.
  - `limit` / `--limit N`: first N questions.
  The response includes `api_calls` (real Ollama requests made).

## Marking the wrong parts

Besides the sentence-level NLI label, each sentence is checked word-by-word: numbers (and years) that are not in the passage the sentence is about, and names that appear in no retrieved passage, are marked bright red inside the sentence, and the sentence counts as flagged. Caveat: a legitimate number that comes from a *different* retrieved passage than the sentence's main topic can be marked.

## Speed

The API preloads the embedder, NLI model and Ollama model at startup (`WARMUP=0` disables it) and asks Ollama to keep the model loaded (`OLLAMA_KEEP_ALIVE`, default 30m). Expect roughly 8 seconds per `/query` with `qwen2.5:3b` on CPU (about 5s generation, 3s NLI).

## Demo: provoking hallucinations

Set `PROMPT_MODE=loose` (default `strict`) to let the generator add outside knowledge, so unsupported sentences show up red in the dashboard. Do not mix loose-mode runs with the strict-mode evaluation numbers below.

Faulty mode: tick "Faulty mode" in the Query tab (or send `"inject_faults": true` to `/query`) to deliberately corrupt 1-2 sentences of the real answer and watch the checker catch them. These responses are tagged and excluded from evaluation.

## Final results

Corpus: 6 renewable-energy documents (12 chunks). Eval set: 22 QA pairs (`data/eval/qa_eval_set.json`). Strict prompt, no faults injected.

Benchmark (`python scripts/compare_generators.py`, saved in `outputs/generator_comparison.json`; cache off, models pre-warmed):

| Model | Hallucination rate | Avg trust | Avg token-prob | Avg generation latency |
| --- | --- | --- | --- | --- |
| Ollama `qwen2.5:3b` | 8.7% (2 of 23 claim sentences) | 0.90 | 0.94 | 2.5 s |

The Ollama row uses the current checker (wrong-span flagging, filler sentences excluded from the counts).

Retrieval: precision@6 0.33 (capped: each doc has only 2 chunks, so at most 2 of 6 can be relevant), precision after MI pruning 0.74, average prune ratio 62%.

Adversarial robustness (NLI premise = original chunk, hypothesis = corrupted sentence, 120 tests):

| Attack | Detection rate |
| --- | --- |
| Negation | 100% |
| Stale fact | 100% |
| Noise | 100% |
| LLM contradiction (`qwen2.5:3b` rewrites) | 20.8% |
| Control false-positive rate | 16.7% |

The low LLM-contradiction rate is partly a generator weakness: the 3B model often returns near-paraphrases instead of a factual change. Separately, NLI is weak on exact numbers: a corrupted sentence with a wrong figure can still be labelled "entailment" (seen in faulty mode).

## Notes and limitations

- **Logprobs**: Ollama (`qwen2.5:3b`) returns per-token logprobs, so the token-probability component is used.
- Attention output is described as "source words the verifier attended to most" (self-attention over premise+hypothesis), not proof of causation.
- Sentences that rely on a pronoun ("It was completed in 2012") may be flagged because the chunk lacks the antecedent.
- Numbers vary slightly between runs because LLM output is not fully deterministic.
- Endpoints: `POST /documents`, `GET /documents`, `POST /query`, `GET /response/{id}`, `POST /evaluate`, `POST /adversarial/run`, `GET /adversarial/results`, `GET /health`.

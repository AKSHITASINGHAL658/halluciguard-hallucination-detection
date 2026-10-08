# HalluciGuard — 3-Week Implementation Plan

## §1. Tech Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Backend API | Python 3.11 + FastAPI | Async, minimal boilerplate, auto docs at /docs |
| Frontend / Dashboard | Streamlit | Charts + highlighted text with almost no frontend code |
| Database | SQLite (via SQLAlchemy) | Zero setup, file-based |
| Vector store | ChromaDB (embedded, local, no server) | Simplest vector DB, persists to disk |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Small, fast, runs on CPU |
| Generation LLM | Gemini (Flash model, name set in config.py) via `google-genai` SDK, `response_logprobs=True` when supported | No GPU needed; logprobs give token-level confidence when available (see note) |
| NLI model | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (fallback: cross-encoder/nli-deberta-v3-base) | Pretrained DeBERTa-v3 NLI, runs on CPU |
| NER (entity-level trust) | spaCy en_core_web_sm | Fast, no training needed |
| Sentence splitting | nltk.sent_tokenize | One line |
| Chunk relevance / MI pruning | scikit-learn TfidfVectorizer + manual PMI calc | Closed-form PMI approximation, cheap |
| Adversarial agent | Rule-based Python module (regex + templates) | A script with a policy, not a trained RL agent |
| Env | venv + requirements.txt, .env for API keys | No Docker needed |

Not used: React/Next.js, Postgres, Kubernetes, fine-tuned local LLM, GPU training, trained MI estimator (e.g. MINE), Airflow.

Notes:
- NLI label order differs between models. Always read labels from `model.config.id2label`; never hardcode indices.
- Gemini logprobs support is inconsistent: some models return "Logprobs is not supported/enabled" (HTTP 400), and there are per-day quota limits on logprob requests. generate() must handle this (see §6 Week 1). Test logprobs with your chosen model on day 1.
- DeBERTa max input is 512 tokens; keep chunks ~250 words so chunk + sentence fits.

## §2. System Architecture

```
Streamlit UI (Trust Dashboard)
        |  HTTP (localhost:8000)
FastAPI Backend (app/main.py + routers)
        |
  +-------------+--------------+---------------+
  | RAG Engine  | NLI Verifier | Trust Engine  |
  | retrieve +  | + Attention  | (entity/sent/ |
  | generate    |              |  doc scores)  |
  +------+------+------+-------+-------+-------+
         |             |               |
     ChromaDB     MI Pruning +         |
     (vectors)    Adversarial module   |
                                       |
            SQLite (app.db): documents, chunks, queries,
            responses, response_sentences, entities,
            trust_scores, adversarial_tests
```

Data flow per query:
1. User submits query in Streamlit -> POST /query
2. Backend retrieves top-k chunks from ChromaDB
3. MI-pruning module drops low-value chunks
4. LLM generates answer (with logprobs) from surviving chunks
5. Response is split into sentences
6. Each sentence run through NLI vs. surviving chunks -> entailment/neutral/contradiction
7. For unsupported sentences, attention weights from the NLI model are extracted to show which source words the verifier attended to most
8. Trust Engine aggregates entity/sentence/document/token-prob scores into one hierarchical trust score
9. Everything persisted to SQLite; /query returns the full result synchronously (no polling); GET /response/{id} re-fetches a stored result
10. Adversarial agent runs offline (batch script) against a sample of chunks, writes results to adversarial_tests; dashboard "Robustness" tab reads from it

## §3. Repository Structure

```
halluciguard/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/
│   └── plan.md                  # this file
├── data/
│   ├── raw_docs/                # 5-10 sample .txt source documents
│   └── eval/
│       ├── qa_eval_set.json     # 20-30 QA pairs w/ gold answers + relevant doc titles
│       └── adversarial_seed.json # optional: hand-picked sentences to attack
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint, mounts routers
│   ├── config.py                # env vars, model names, thresholds, weights
│   ├── db.py                    # SQLAlchemy engine/session, init_db()
│   ├── models.py                # SQLAlchemy ORM models (§4)
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── routers/
│   │   ├── ingest.py            # POST /documents, GET /documents
│   │   ├── query.py             # POST /query, GET /response/{id}
│   │   ├── eval.py              # POST /evaluate
│   │   └── adversarial.py       # POST /adversarial/run, GET /adversarial/results
│   └── core/
│       ├── chunking.py          # split_into_chunks()
│       ├── embeddings.py        # get_embedder(), embed_texts()
│       ├── vector_store.py      # ChromaDB wrapper: add(), query()
│       ├── mi_pruning.py        # compute_pmi_scores(), prune_chunks()
│       ├── rag_pipeline.py      # retrieve(), generate(), run_pipeline()
│       ├── nli_verifier.py      # get_nli_model(), verify_sentence(), batch_verify()
│       ├── attention_extractor.py # extract_attention(), top_contributing_tokens()
│       ├── sentence_localizer.py  # split_sentences(), localize_hallucinations()
│       ├── trust_engine.py      # entity_score(), sentence_score(), doc_score(), token_prob_component(), aggregate_trust()
│       ├── adversarial_agent.py # inject_negation(), inject_stale_fact(), inject_noise(), run_attack_suite()
│       └── metrics.py           # retrieval_precision_at_k(), detection_rate(), false_positive_rate()
├── dashboard/
│   └── streamlit_app.py         # single-file Streamlit UI, calls FastAPI
├── scripts/
│   ├── seed_corpus.py           # loads data/raw_docs into Chroma + SQLite
│   └── run_adversarial_eval.py  # CLI batch runner for adversarial agent
└── tests/
    ├── test_rag_pipeline.py
    ├── test_nli_verifier.py
    ├── test_mi_pruning.py
    └── test_trust_engine.py
```

## §4. Database Schema (SQLite via SQLAlchemy)

documents
- id PK int
- title str
- source str (filename or URL)
- authority float (static source-authority weight 0-1, used in doc-level trust)
- created_at datetime

chunks
- id PK int
- document_id FK -> documents.id
- text text
- chunk_index int
- embedding_id str (id in ChromaDB collection)

queries
- id PK int
- text text
- created_at datetime

responses
- id PK int
- query_id FK -> queries.id
- raw_text text (full LLM output)
- model_name str
- retrieved_chunk_ids text (JSON list, before pruning)
- pruned_chunk_ids text (JSON list of chunk ids actually used after MI pruning)
- trust_score_overall float
- created_at datetime

response_sentences
- id PK int
- response_id FK -> responses.id
- sentence_index int
- text text
- nli_label str (entailment | neutral | contradiction)
- nli_confidence float
- is_hallucination bool
- supporting_chunk_id FK -> chunks.id, nullable
- top_attention_tokens text (JSON list of {token, weight})

entities
- id PK int
- response_id FK -> responses.id
- text str
- entity_type str (spaCy label)
- supported bool
- entity_trust_score float

trust_scores
- id PK int
- response_id FK -> responses.id (1:1)
- entity_level_score float
- sentence_level_score float
- document_level_score float
- token_prob_component float, nullable (from Gemini logprobs; null if the model returned none)
- final_score float (weighted aggregate)

adversarial_tests
- id PK int
- original_chunk_id FK -> chunks.id
- attack_type str (negation | stale_fact | noise | control)
- original_sentence text
- corrupted_text text (same as original for control rows)
- nli_label str
- detected bool (label != entailment)
- run_at datetime

Relationships: documents 1-N chunks, queries 1-N responses, responses 1-N response_sentences, responses 1-N entities, responses 1-1 trust_scores, chunks 1-N adversarial_tests.

## §5. API List (FastAPI)

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | /documents | Ingest a document -> chunk, embed, store in Chroma + documents/chunks |
| GET | /documents | List ingested documents |
| POST | /query | Full pipeline: retrieve -> MI-prune -> generate -> NLI verify -> localize -> attention -> trust. Returns full result (synchronous) |
| GET | /response/{id} | Stored response: text, per-sentence NLI labels, hallucination flags, attention tokens, trust breakdown |
| POST | /evaluate | Batch eval on data/eval/qa_eval_set.json: retrieval precision@k, avg trust, hallucination rate, avg prune ratio |
| POST | /adversarial/run | Run adversarial agent on a sample of chunks, store results |
| GET | /adversarial/results | Detection rate per attack type + control false-positive rate |
| GET | /health | Liveness check |

## §6. Implementation Tasks

### Week 1 — Foundation + RAG pipeline + MI pruning

- Init repo, requirements.txt, .env.example, FastAPI skeleton (main.py, config.py, db.py, models.py) — creates all tables
- core/chunking.py — fixed-size word chunking (~250 words, 40 overlap)
- core/embeddings.py, core/vector_store.py — Chroma wrapper add()/query()
- scripts/seed_corpus.py — load 5-10 docs from data/raw_docs/, populate DB + Chroma
- core/rag_pipeline.py — retrieve(query, k=6) and generate(query, chunks) using Gemini via google-genai, temperature 0; system prompt: answer only from the numbered context in 3-6 short sentences
  - Logprobs: request with `response_logprobs=True`; on an error saying logprobs are unsupported/not enabled or a 429 quota error, retry once WITHOUT logprobs. Then use, in order: per-token logprobs from `candidate.logprobs_result` -> `candidate.avg_logprobs` -> None
  - Return {answer, token_logprobs (list or None), avg_logprob (float or None), logprobs_source: "tokens" | "avg" | "none"}; store logprobs_source so the dashboard can show it
- routers/ingest.py and routers/query.py (no NLI yet — raw generation stored)
- core/mi_pruning.py:
  - compute_pmi_scores(query, chunks): for each chunk, PMI(q, c) = mean over query terms t of log((P(t|c) + eps) / (P(t) + eps)), using smoothed term frequencies over the retrieved set
  - prune_chunks(query, chunks, keep_ratio=0.5, min_keep=2, redundancy=0.85): keep chunks scoring >= 50% of top score; drop a chunk if cosine similarity to an already-kept chunk > 0.85; always keep at least 2
  - log retrieved count, kept count, and prompt token count before/after
- data/eval/qa_eval_set.json — 20-30 QA pairs against the corpus
- core/metrics.py — retrieval_precision_at_k()
- routers/eval.py (POST /evaluate)
- Smoke test: ingest -> query -> raw response end-to-end via /docs

### Week 2 — Hallucination detection, localization, attention, adversarial agent

- core/nli_verifier.py — get_nli_model() cached singleton returning (tokenizer, model); verify_sentence(sentence, chunk_text) -> {label, probs: {entailment, neutral, contradiction}}; batch_verify(pairs) in batches of 8; labels read from model.config.id2label
- core/sentence_localizer.py — split_sentences(); localize_hallucinations(): run every sentence against every surviving chunk, keep the chunk with highest entailment prob; is_hallucination = max entailment < 0.5; label = contradiction if contradiction > 0.5 against any chunk, else best chunk's label
- Wire NLI + localization into /query — populate response_sentences
- core/attention_extractor.py — for flagged sentences only, against their best chunk: forward pass with output_attentions=True; average last 4 layers and all heads; take rows for hypothesis tokens and columns for premise tokens; sum over rows; merge sub-word pieces into words; drop special tokens, stopwords, punctuation; return top 5 normalized {token, weight}; store in top_attention_tokens
  - Wording: DeBERTa uses self-attention over the concatenated premise+hypothesis (not cross-attention). Describe output as "source words the verifier attended to most", not proof of causation.
  - Optional check: mask the top-5 words in the premise and re-run NLI; a large drop in entailment prob supports the attribution.
- core/adversarial_agent.py:
  - inject_negation(s): insert/remove not/never, flip is/was
  - inject_stale_fact(s): shift years by 3-10, scale numbers by 0.5-2x (regex)
  - inject_noise(s): swap an entity or append an unrelated clause
  - run_attack_suite(n_chunks=20): from each chunk pick 1-2 sentences containing a number, date or verb; corrupt them; run NLI with premise = ORIGINAL CLEAN CHUNK and hypothesis = CORRUPTED SENTENCE; detected = label != entailment
  - Control set: also run the unchanged sentences (attack_type = control); their non-entailment rate = false-positive rate
  - Why: corrupting chunks and then querying lets the LLM faithfully repeat the corrupted text, which NLI marks as supported — that measures nothing.
- scripts/run_adversarial_eval.py + routers/adversarial.py — detection_rate per attack type + false_positive_rate
- Tests: test_nli_verifier.py (3 hand-written pairs, one per label), test_mi_pruning.py
- Checkpoint: /query returns full payload with per-sentence labels and attention tokens; /adversarial/results works

### Week 3 — Hierarchical trust engine, dashboard, write-up

- core/trust_engine.py:
  - entity_score(): spaCy entities in the answer; score = fraction found (case-insensitive) in surviving chunk text; no entities -> 1.0
  - sentence_score(): mean max-entailment prob x (1 - hallucination ratio)
  - doc_score(): mean documents.authority of the best chunks of supported sentences; 0 if none supported
  - token_prob_component(): exp(mean token logprob) (or exp(avg_logprob)), i.e. geometric-mean token probability, 0-1; returns None if no logprobs
  - aggregate_trust(): final = 0.35*sentence + 0.25*entity + 0.20*document + 0.20*token_prob (weights in config.py); if token_prob is None, drop it and renormalize the other three weights to sum to 1; store in trust_scores
- Wire trust engine into /query — populate entities + trust_scores
- dashboard/streamlit_app.py:
  - Query tab: input box -> POST /query; response text with sentences colored green (supported), amber (neutral), red (hallucination); expander per red sentence showing best source chunk + top-5 attention words
  - Entity list with supported/unsupported badges
  - Trust score: overall + bar chart of entity/sentence/document/token-prob components
  - Robustness tab: GET /adversarial/results — detection rate per attack type + FP rate as bar chart
  - Evaluation tab: POST /evaluate — precision@k, hallucination rate, avg prune ratio
- End-to-end run, bug fixing, README with setup + final eval numbers

## §7. Solo Build Order

1. Skeleton, models.py, db.py (§3, §4)
2. chunking, embeddings, vector_store, seed_corpus.py
3. rag_pipeline.py, routers/ingest.py, routers/query.py, main.py
4. mi_pruning.py + test, wired into rag_pipeline
5. qa_eval_set.json, metrics.py, routers/eval.py
6. nli_verifier.py + test
7. sentence_localizer.py, wired into /query
8. attention_extractor.py
9. adversarial_agent.py, run_adversarial_eval.py, routers/adversarial.py
10. trust_engine.py + test, wired into /query
11. streamlit_app.py (all tabs)
12. README, run all tests, final numbers

## §8. Dependencies

Must happen first, in order:
1. models.py / db.py — everything writes to these tables
2. chunking.py + embeddings.py + vector_store.py — nothing can be retrieved without them
3. seed_corpus.py run once — provides data for every later step

Then:
- nli_verifier.py and mi_pruning.py / adversarial_agent.py are independent of each other
- attention_extractor.py and adversarial_agent.py reuse get_nli_model() from nli_verifier.py — never load the model twice
- trust_engine.py needs NLI results, entities and the pruned chunk list — build it after those
- streamlit_app.py only needs the API contracts in schemas.py

## §9. Definition of Done

- [ ] RAG pipeline: /documents ingests a doc, /query returns a grounded answer; /evaluate reports precision@k
- [ ] NLI verification: every response_sentences row has nli_label and nli_confidence
- [ ] Sentence-level localization: dashboard highlights unsupported sentences
- [ ] Trust dashboard: overall score + entity/sentence/document/token-prob breakdown per response
- [ ] Attention attribution: flagged sentences have top_attention_tokens shown in the UI
- [ ] MI pruning: runs on every query; pruned_chunk_ids shorter than retrieved_chunk_ids on most queries; before/after counts logged
- [ ] Adversarial robustness: adversarial_tests populated; /adversarial/results returns detection rate per attack type + control FP rate
- [ ] Hierarchical trust: trust_scores has non-null entity/sentence/document components (+ token-prob when Gemini returns logprobs) AND a distinct final_score
- [ ] All 4 test files pass (pytest)
- [ ] README: setup steps (env vars, pip install -r requirements.txt, python -m spacy download en_core_web_sm, python scripts/seed_corpus.py, uvicorn app.main:app, streamlit run dashboard/streamlit_app.py) and final eval numbers

### requirements.txt

```
fastapi
uvicorn
sqlalchemy
chromadb
sentence-transformers
google-genai
python-dotenv
transformers
torch
spacy
nltk
scikit-learn
streamlit
plotly
requests
pytest
```

### .env.example

```
GEMINI_API_KEY=
GEMINI_MODEL=
```
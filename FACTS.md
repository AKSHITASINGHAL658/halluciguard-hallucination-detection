# FACTS.md — what the code actually does

Source of truth: the code at HEAD (32 commits). Written 2026-10-11 by reading the code. Where a fact comes from a past run, the run date is given. `pytest` at last run: 43 passed.

Things that do NOT exist (checked with grep): a "proxy scorer"; a pronoun-retry step in verification; any Gemini/cloud-LLM code.

---

## 1. Config (`app/config.py`) — current values

| Name | Value | Env override |
| --- | --- | --- |
| TEMPERATURE | 0.0 | no |
| OLLAMA_MODEL | `qwen2.5:3b` | OLLAMA_MODEL |
| OLLAMA_HOST | `http://localhost:11434` | OLLAMA_HOST |
| OLLAMA_KEEP_ALIVE | `30m` | OLLAMA_KEEP_ALIVE |
| WARMUP | True (preload models at API startup) | WARMUP |
| LLM_CACHE | True | LLM_CACHE |
| LLM_CACHE_DIR | `<repo>/cache/llm` | no |
| DB_URL | `sqlite:///<repo>/app.db` | DB_URL |
| CHROMA_DIR / COLLECTION | `<repo>/chroma` / `halluciguard_chunks` | CHROMA_DIR |
| RAW_DOCS_DIR / EVAL_SET_PATH | `data/raw_docs` / `data/eval/qa_eval_set.json` | no |
| EMBEDDING_MODEL | `sentence-transformers/all-MiniLM-L6-v2` | no |
| NLI_MODEL / fallback | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` / `cross-encoder/nli-deberta-v3-base` | no |
| SPACY_MODEL | `en_core_web_sm` | no |
| CHUNK_WORDS / CHUNK_OVERLAP | 250 / 40 | no |
| RETRIEVE_K | 6 | no |
| DEFAULT_AUTHORITY | 0.8 | no |
| PRUNE_KEEP_RATIO / MIN_KEEP / REDUNDANCY | 0.5 / 2 / 0.85 | no |
| PMI_EPS | 1e-6 | no |
| NLI_BATCH_SIZE / NLI_MAX_LENGTH | 8 / 512 | no |
| HALLUCINATION_ENTAIL_THRESHOLD | 0.5 | no |
| CONTRADICTION_THRESHOLD | 0.5 | no |
| USE_CLAIMS | True | USE_CLAIMS |
| SPAN_STRONG_ENTAILMENT | 0.9 | no |
| ATTENTION_LAST_LAYERS / TOP_N | 4 / 5 | no |
| ATTENTION_VALIDATE | True (mask-and-rerun check) | ATTENTION_VALIDATE |
| ATTENTION_CHECK_RANDOM / MARGIN | 5 / 0.05 | no |
| TRUST_WEIGHTS | sentence 0.35, entity 0.25, document 0.20, token_prob 0.20 | no |
| ADV_N_CHUNKS / ADV_SEED | 20 / 42 | no |
| PROMPT_MODE | `strict` | PROMPT_MODE (`strict` or `loose`) |

Constants that live OUTSIDE config.py (contradicts the CLAUDE.md rule "constants only in config.py"):
`claim_extractor.py` (MAX_CLAIMS=4, MIN_CLAIM_WORDS=4, `_PRONOUNS`), `span_checker.py` (NAME_TYPES, the 0.03 frequency boost), `sentence_localizer.py` (filler regexes, 25-word limit), `trust_engine.py` (`_MODIFIERS`, `_RELATIVE_DATE` regexes), `adversarial_agent.py` (attack prompt, UNRELATED_CLAUSES, stale-fact factors), `dashboard/streamlit_app.py` (colors).

## 2. Chunking and retrieval

- `chunking.split_into_chunks`: word-based, not sentence-aware. `text.split()` on whitespace, window 250 words, step 210 (overlap 40), words rejoined with single spaces (original newlines are lost). Stops after the window that reaches the end.
- Embeddings: MiniLM, normalized, CPU. Chroma collection uses cosine space.
- `rag_pipeline.retrieve`: embeds the query, returns top k=6 by cosine distance; each result carries chunk_id, document_id, title, text, embedding, distance.
- Corpus: 6 documents, 2 chunks each (12 chunks).

## 3. MI pruning (`mi_pruning.py`)

- Terms: lowercase `[a-z0-9]+` tokens minus sklearn English stop words. Query terms are de-duplicated.
- For chunk c and query term t: `P(t|c) = (count_c(t)+1) / (n_c + V)`, `P(t) = (count_all(t)+1) / (N_total + V)`, V = number of distinct terms in the retrieved set.
- `PMI(q,c) = mean over query terms of ln((P(t|c)+1e-6) / (P(t)+1e-6))`; 0.0 if the query has no terms.
- Selection, in score order: keep chunk if `score >= threshold` AND cosine(embedding, any already-kept) is not > 0.85. `threshold = 0.5 × top` if top > 0, else `top / 0.5` (more lenient when all scores are ≤ 0).
- Then top up to `min_keep = 2` (or the number of chunks if fewer): non-redundant chunks first, then redundant ones.
- Logs retrieved count, kept count, approx prompt tokens (words × 1.3) before/after. Each kept chunk gets `pmi_score`.

## 4. Generation (`rag_pipeline.py`)

- Generator: Ollama only, model `qwen2.5:3b`, `chat(logprobs=True, keep_alive="30m", options={"temperature": 0.0})`, no output-length cap.
- Strict prompt (system, verbatim): "Answer the question using only the numbered context passages below. Respond in 3-6 short sentences. If the context does not contain the answer, say so."
- Loose prompt (system, verbatim): "Answer the question thoroughly in 4-6 sentences. The context passages below are a starting point; enrich the answer with additional background, specific figures, dates, named examples and comparisons from your own knowledge. Do not mention the context or say when information is missing."
- User message: `Context:\n[1] <chunk>\n\n[2] <chunk>...\n\nQuestion: <q>\nAnswer:` (pruned chunks only).
- Logprobs: per-token logprobs parsed from the response → `logprobs_source="tokens"`, `avg_logprob` = mean. If none returned → `"none"` and a warning (no avg-only path).
- Cache: key = sha256("ollama", model, system prompt, user prompt); file `cache/llm/<key>.json` stores the whole result. Counters: real call count and per-call latencies (cache hits excluded).
- Proxy scorer: does not exist.
- `fix_split_decimals` (in `query_service`): regex `(?<=\d)\. (?=\d)` → `.` applied to every answer before scoring (the model writes "42. 5").

## 5. Verification (`verification.verify_sentences`, `sentence_localizer.py`)

- Splitting: `nltk.sent_tokenize` (punkt), after `fix_split_decimals`.
- **Filler sentence** (`is_filler`): after removing `[n]` citations, NO digit, ≤ 25 words, AND matches a reference word (`context|passage(s)|source(s)|provided text|text provided|document(s)`) AND a meta verb (`stated|provided|mentioned|according|based on|found|given|described|supported|discussed|does/do not contain/provide/mention/include|no information|not available/specified`). Filler is never flagged and is excluded from scoring/eval counts. A **claim sentence** = any non-filler sentence.
- NLI: premise = chunk, hypothesis = sentence (or claim); `truncation="only_first"`, max 512 tokens; softmax; labels read from `model.config.id2label`.
- Claim splitting (`claim_extractor.extract_claims`, when USE_CLAIMS): spaCy; splits at coordinated verbs of the root (max 4 claims, each ≥ 4 words, otherwise the sentence is kept whole); a clause with no subject, or a pronoun subject (it/they/this/that/these/those/he/she), gets the main clause's subject. Not a retry step.
- Per claim: best chunk = highest entailment over the surviving chunks. `max_entailment` = that value. If any chunk's contradiction prob > 0.5: label = contradiction (best chunk stays the entailment chunk if max_ent ≥ 0.5, else the max-contradiction chunk); else label = best chunk's label.
- Per sentence: takes the **weakest claim** (lowest max_entailment); label = contradiction if any claim is; `is_hallucination` = any claim with `max_entailment < 0.5`. Then filler → False; `nli_flagged` records the NLI-only verdict.
- Span override: any wrong span (section 6) sets `is_hallucination = True` and replaces the displayed source chunk with the reference chunk.
- **Dashboard colors** (`streamlit_app.py`): filler → grey italic; `is_hallucination` with no spans → light red; flagged with spans → amber base with bright red (#ff5c5c) words; otherwise green if label = entailment, else amber (neutral, or contradiction label with max_ent ≥ 0.5).
- Expander per flagged non-filler sentence: NLI label/confidence, span reasons, source chunk, attention words.

## 6. Number/name checker (`span_checker.py`)

- `find_wrong_spans(sentence, chunks, nli_best_id, nli_entailment)`; skipped for filler and when `use_spans=False`.
- Reference chunk (`pick_reference_chunk`): MiniLM cosine of the number-stripped sentence vs each chunk + 0.03 × mean log(1+frequency of the sentence's content words in the chunk); ties → the NLI best chunk; one chunk → itself.
- Numbers: regex `(?<![\w.\[])\d+(?:,\d{3})*(?:\.\d+)?(?!\w)`. Normalisation = strip commas only. A number is wrong if it is not among the reference chunk's numbers. Exception: if NLI entailment ≥ 0.9, the NLI best chunk's numbers also count.
- Names: spaCy entities of type PERSON, ORG, GPE, NORP, LOC, FAC, PRODUCT, EVENT, WORK_OF_ART, LAW; leading modifiers stripped; wrong if the lowercase text is not a substring of ANY surviving chunk.
- Skipped: numbers inside words (COP28), after `.` or `[` (citations), followed by a letter/digit (2030s, 5th); number words; DATE/CARDINAL/etc. entity types; no unit, magnitude or rounding handling (42.5 ≠ 42.50; 345 GW ≠ 0.345 TW).

## 7. Attention (`attention_extractor.py`)

- Only for sentences with `nli_flagged` and a best chunk (not for sentences flagged only by the span check). Forward pass on (chunk, sentence) with `output_attentions=True`, truncation 512.
- Last 4 layers, averaged over layers and ALL heads. Rows = hypothesis tokens (sequence id 1), columns = premise tokens (id 0); sum over rows. Sub-word pieces merged into words via a `\w+(?:[.,]\w+)*` regex on the chunk text (offset mapping).
- Dropped: stop words (sklearn list), tokens with no alphanumeric char. Top 5, weights normalised to sum to 1 over the top 5 (4 decimals).
- Note: if the span check had replaced the best chunk, attention runs against that reference chunk.
- **Mask-and-rerun check** (`attention_extractor.check_attribution`, on when `ATTENTION_VALIDATE`): deletes the top-5 words from the source chunk and re-runs NLI; the shift is the total-variation distance between the three label probabilities before and after. The control deletes the same number of random content words not in the top 5 (5 draws, seed 0). `supported` = top shift > mean random shift + 0.05 (`ATTENTION_CHECK_MARGIN`). Stored in `response_sentences.attention_check` and shown in the dashboard expander.
- Latest result (2026-10-16, `scripts/run_attention_check.py`, 69 NLI-flagged labelled sentences): `supported` for 35% (24/69); mean shift 0.24 for top words vs 0.11 for random words (45 truly unsupported: 33%; 24 wrongly flagged: 38%).

## 8. Trust (`trust_engine.py`, `aggregate_trust`)

- **entity_score**: all spaCy entities in the answer (any type); DATE/TIME with relative wording (that/this/last/next/…/"end of"/today/…) skipped; leading modifiers (approximately, roughly, about, around, nearly, almost, over, under, more than, less than, up to, some, the, a, an) stripped; de-duplicated by (text, label); supported = lowercase substring of the joined surviving chunks. Score = fraction supported; **no entities → 1.0**.
- **sentence_score** (non-filler sentences): `mean(max_entailment) × (1 − fraction flagged)`; empty → 0.0. (`max_entailment` is the NLI value, not changed by span overrides.)
- **doc_score**: mean `documents.authority` of the best chunks of non-flagged non-filler sentences; none → 0.0. All six documents have authority 0.8.
- **token_prob_component**: `exp(mean token logprob)` of the whole original answer; None if no logprobs. In faulty mode it is forced to None; rescoring reuses the stored value.
- **Final**: `0.35·sentence + 0.25·entity + 0.20·document + 0.20·token_prob`; if token_prob is None it is dropped and the other three weights are divided by their sum (0.80). Not clipped.
- Edge: an answer made only of filler sentences gets sentence = 0.0 and document = 0.0.

## 9. Faulty mode and loose mode

- **Faulty mode**: `POST /query {"inject_faults": true}` (default false); the dashboard checkbox sends it. `process_query` calls `adversarial_agent.inject_faults(answer)` after generation: sentences with digits are picked first (random order), then others; 2 sentences if the answer has ≥ 4 sentences, else 1. Digit sentence → `inject_stale_fact` (see section 10, changes every number in it); else `inject_negation`; if the text did not change → negation. Sentences re-joined with spaces. Stored in `responses.faults_injected`; token-prob dropped; such responses are excluded from `--rescore` (lookup requires `faults_injected IS NULL`).
- **Loose mode**: env var `PROMPT_MODE=loose`, read at import into `config.PROMPT_MODE`; swaps the system prompt only. There is no API or dashboard switch. The mode is NOT recorded in the database (only in the cache key via the prompt text).

## 10. Adversarial (`adversarial_agent.py`)

- `run_attack_suite(db, n_chunks=20, seed=42)`: all chunks shuffled with `Random(42)`, first 20 taken (the corpus has 12, so all 12). Candidate sentences = nltk sentences with ≥ 5 words that contain a digit or a spaCy VERB/AUX; up to 2 per chunk, numeric ones first. Rows with corrupted text equal to the original are dropped (except control). Previous rows are deleted at each run.
- Per sentence: `control` (unchanged); `negation`; `stale_fact` (only if the sentence has a digit); `noise`; `llm_contradiction`.
  - negation: if "not/never" present remove the first; else insert "not" after the first of is/was/are/were/has/have/had/can/will/does/did; else prefix "It is not true that ".
  - stale_fact: one random shift 3–10 (random sign) applied to every 4-digit year (1500–2099); every other number × a random choice of [0.5, 0.6, 0.75, 1.5, 2.0] (decimals keep their places, ints rounded).
  - noise: with 50% chance (if the sentence has an entity and the pool has another) swap one entity (GPE/ORG/PERSON/NORP/LOC/FAC) for a random pool entity; otherwise append one of 4 fixed unrelated clauses.
  - llm_contradiction: Ollama, temperature 0, prompt "Rewrite this sentence so it remains fluent and plausible but states something factually different. Output only the sentence."; first non-empty line, quotes stripped; skipped if the call fails or the output equals the input; cached as `cache/llm/adv-<sha>.json`.
- Scoring (two measurements per row): (a) **NLI alone**: plain `batch_verify`, premise = ORIGINAL chunk, hypothesis = corrupted text; `detected` = argmax label ≠ entailment (the 0.5 threshold is not used). (b) **Full pipeline** (`detected_full`): `verify_sentences([corrupted], [clean chunk])` = claim splitting, number/name check against the clean chunk, filler logic; detected = `is_hallucination`. Control false-positive rate = share of unchanged sentences flagged, per measurement.

## 11. Evaluation

- `POST /evaluate` and `scripts/run_eval.py` (`run_evaluation`, `app/core/evaluation.py`) over the 22 questions in `qa_eval_set.json`. Metrics: `retrieval_precision_at_k` (share of the 6 retrieved chunks whose document title is in the question's relevant_docs), `precision_after_pruning` (same on kept chunks), `avg_prune_ratio` (mean of 1 − kept/retrieved), `avg_trust`, `hallucination_rate` (flagged / non-filler sentences over all answers), `n_sentences`, `avg_token_prob`, `avg_generation_latency_s` (real calls only), `api_calls`, `n_failed` (rolls back the DB session after a failed question).
- `retrieval_only`: retrieval + pruning metrics only, no generation. `rescore`: for each question takes the latest stored non-faulty response, re-runs verification and trust on its stored answer and chunk ids (reuses the stored token-prob), no LLM call; writes new rows; questions with no stored answer are counted in `n_no_stored_answer`. `limit`: first N questions.
- Labeled-set ablation: `scripts/run_ablation.py` (176 sentences, 55 unsupported, labelled by an AI assistant, not a human); outputs P/R/F1 with 1000-sample bootstrap CI, an entailment-threshold sweep, recall by error kind, and answer-level AUROC.
- **Latest numbers and dates**
  - 22-question eval, 2026-10-11 20:00 (`outputs/generator_comparison.json`): hallucination rate 8.7% (2/23 claim sentences), avg trust 0.899, avg token-prob 0.937, 2.5 s/answer, 0 failed. **This run predates claim splitting; the 22-question eval has not been re-run with USE_CLAIMS on.**
  - Retrieval (deterministic; last measured 2026-10-11 via `--rescore`): precision@6 0.333 (capped: 2 chunks per doc), after pruning 0.741, prune ratio 0.621.
  - Adversarial (120 tests): 2026-10-11 NLI alone: negation 100%, stale_fact 100%, noise 100%, llm_contradiction 20.8%, control FP 16.7%. 2026-10-16 full pipeline: negation 100%, stale_fact 100%, noise 100%, llm_contradiction 29.2%, control FP 20.8%.
  - Ablation, 2026-10-11 21:32 (current code): NLI only P 0.68 / R 0.82 / F1 0.74; full pipeline P 0.62 / R 0.84 / F1 0.71; without MI pruning F1 0.65; answer-level AUROC token-prob 0.98, sentence 0.88, final 0.89, entity 0.65, document 0.46.

## 12. API and database

- Endpoints: `POST /documents`, `GET /documents`, `POST /query` (body: text, k=6, inject_faults=false), `GET /response/{id}`, `POST /evaluate` (query params k, retrieval_only, rescore, limit), `POST /adversarial/run`, `GET /adversarial/results`, `GET /health`. Startup: `init_db()` (create tables + add missing columns) and, if WARMUP, `warmup()` (embedder, NLI, Ollama).
- Tables: `documents`, `chunks`, `queries`, `responses`, `response_sentences`, `entities`, `trust_scores`, `adversarial_tests`, as in the plan plus added columns: `responses.generator`, `responses.faults_injected`, `responses.logprobs_source`; `response_sentences.wrong_spans`, `response_sentences.is_filler`, `response_sentences.attention_check`; `adversarial_tests.detected_full`. Columns added after the first release are added by an `ALTER TABLE` migration in `init_db`.
- Response payload extras beyond the plan: `generator`, `injected_faults`, per-sentence `wrong_spans`, `is_filler`, `supporting_chunk_text`.
- Scripts: `seed_corpus.py`, `run_adversarial_eval.py`, `run_eval.py`, `compare_generators.py` (Ollama only), `probe_ollama_logprobs.py`, `build_labeled_set.py`, `run_ablation.py`.

## 13. Versus `docs/plan.md`

**Built that was not in the plan:** the attention mask-and-rerun check with a random-word control and `run_attention_check.py`; full-pipeline scoring of the adversarial suite (`detected_full`); Ollama generator (replaces Gemini); LLM response cache; startup warmup and keep-alive; `PROMPT_MODE` loose mode; faulty mode; span/number checker (`span_checker.py`); filler detection; claim splitting (`claim_extractor.py`); shared `verification.py`; `query_service.py`, `evaluation.py`, `ingestion.py`, `warmup.py`; `llm_contradiction` attack; `retrieval_only` / `rescore` / `limit` eval options and `run_eval.py`; `compare_generators.py`; labeled set and `run_ablation.py`; `fix_split_decimals`; temp-DB test fixtures (`tests/conftest.py`) and `DB_URL` / `CHROMA_DIR` overrides; extra test files (adversarial, attention, claims, filler, spans, warmup); `precision_after_pruning` metric.

**In the plan but NOT built:** Gemini/`google-genai` generation and the Gemini logprob fallback chain (tokens → avg → none; only tokens → none remains); `data/eval/adversarial_seed.json` (marked optional); `.env.example` GEMINI keys (replaced by OLLAMA keys).

**Differences in behaviour:** `is_hallucination` is no longer only "max entailment < 0.5" (also claim-splitting, span override, filler exclusion); `docs/plan.md` and `CLAUDE.md` (partially updated) still describe the original design; the plan's 5–10 documents / 20–30 QA pairs are 6 documents / 22 pairs.

## 14. End-to-end request lifecycle

This section gives a high-level map of the documented HalluciGuard
request path. It is not intended to replace the implementation as the
source of truth.

1. A user submits a question through the dashboard or query API.
2. The backend receives the request and passes the question into the
   query-processing workflow.
3. The retrieval component searches the ChromaDB collection for
   relevant document chunks.
4. The MI-pruning component scores retrieved chunks and selects the
   context passed to later stages.
5. The generation component constructs a prompt from the selected
   context and the user's question.
6. Ollama runs the configured local language model and returns the
   generated answer.
7. The answer is prepared for verification, including the
   decimal-format correction documented earlier.
8. The verification stage splits the answer into sentences and, when
   enabled, smaller claims.
9. NLI compares each claim or sentence with available evidence chunks.
10. The span checker looks for unsupported numbers and named entities
    according to its documented rules.
11. Attention attribution is computed for eligible NLI-flagged
    sentences.
12. The trust engine combines its configured components into an
    answer-level score.
13. The backend stores the resulting records and returns a response for
    display.
14. The dashboard presents the answer, sentence-level status, supporting
    context, and available diagnostics.

The exact ordering and conditional behavior should be checked against
the current implementation before treating this list as a line-by-line
execution trace. Some stages are conditional: attention analysis is not
run for every sentence, and cache hits can avoid a real Ollama request.

### 14.1 Important distinctions

- **Retrieval relevance is not claim entailment.** A passage can be
  topically relevant without proving a specific claim.
- **Generation and verification are separate.** A fluent answer is not
  necessarily supported by evidence.
- **Sentence verdicts and answer-level trust are different outputs.**
  The trust score aggregates multiple signals; it is not a calibrated
  probability that the entire answer is true.
- **Span warnings and NLI verdicts are different signals.** The span
  checker targets particular number/name mismatches, while NLI
  evaluates a broader textual relationship.
- **Dashboard colors are presentation.** They communicate the
  backend's classifications; they are not an independent verification
  method.

  ## 15. Component responsibilities and boundaries

The system is easier to understand when each component is considered
separately.

### 15.1 Retrieval

- **Responsibility:** Find candidate passages that may contain
  information relevant to the question.
- **Input:** The user's query and the indexed document collection.
- **Output:** A ranked set of candidate chunks with metadata.
- **Boundary:** Retrieval does not establish that a generated claim is
  true. It supplies candidate evidence for later stages.

### 15.2 MI pruning

- **Responsibility:** Reduce retrieved context by ranking chunks using
  the project's information-based scoring and redundancy rules.
- **Input:** Retrieved chunks, query terms, and chunk embeddings.
- **Output:** A smaller set of chunks selected for downstream
  processing.
- **Boundary:** A high pruning score is a selection signal, not a
  guarantee that a passage is authoritative or factually correct.
  Minimum-keep and redundancy rules can also affect selection.

### 15.3 Local generation

- **Responsibility:** Produce an answer using the configured prompt
  and selected context.
- **Input:** The question, selected chunks, and active system prompt.
- **Output:** Generated text and, when supplied by the runtime, token
  log-probability information.
- **Boundary:** Model confidence signals do not independently prove
  factual correctness. Cache behavior can affect whether a request
  invokes Ollama again.

### 15.4 NLI verification

- **Responsibility:** Estimate the relationship between a claim and a
  candidate evidence passage.
- **Input:** A premise passage and a sentence or extracted claim.
- **Output:** Entailment, contradiction, and neutral probabilities or
  labels, as represented by the configured model.
- **Boundary:** NLI is a model-based judgment and can be wrong. The
  project notes identify exact-number errors and some factual
  contradictions as difficult cases.

### 15.5 Span checking

- **Responsibility:** Detect selected unsupported number and
  named-entity spans using the implementation's documented rules.
- **Input:** The sentence, candidate chunks, and NLI-related context.
- **Output:** Detected wrong spans and their reasons.
- **Boundary:** The current rules are limited. They do not perform
  general mathematical reasoning, comprehensive unit conversion, or
  complete semantic fact checking.

### 15.6 Attention attribution

- **Responsibility:** Identify source words receiving higher attention
  under the project's extraction procedure and optionally test their
  effect through masking and rerunning NLI.
- **Input:** An eligible source passage and sentence, plus the
  verifier model.
- **Output:** Ranked source words and, when validation is enabled, a
  mask-and-rerun diagnostic.
- **Boundary:** Attention is not proof of causation or a guaranteed
  explanation of a model decision. The measured results documented
  earlier show that the top words do not beat the random control for
  every sentence.

### 15.7 Trust aggregation

- **Responsibility:** Combine configured sentence, entity, document,
  and token-probability components when available.
- **Input:** Component scores produced by earlier stages.
- **Output:** An aggregate trust score.
- **Boundary:** The score depends on component definitions and
  weights. It is not a universal probability of truth and should not
  be compared across configurations without checking that the scoring
  setup is the same.

### 15.8 Persistence and dashboard

- **Responsibility:** Store records and present the resulting answer
  and diagnostics.
- **Boundary:** Successful storage and readable presentation do not
  validate the underlying factual judgments. Database records should
  be interpreted alongside the evaluation method and its limitations.

  ## 16. Error cases and interpretation notes

This section records scenarios readers and testers should keep in mind.
It does not claim that every scenario has a dedicated automated test.

### 16.1 Retrieval and evidence

- **No useful passage is retrieved:** The available context may not
  answer the question. The strict prompt asks the generator to say
  when context does not contain the answer, but prompt wording is not
  a guarantee.
- **Relevant but insufficient passage:** A passage may discuss the
  right topic without supporting the specific detail in the answer.
- **Redundant passages:** Multiple chunks can repeat similar
  information. The pruning stage includes a redundancy rule, but that
  rule is not a complete measure of semantic diversity.
- **Context split across chunks:** A fact may require information from
  more than one chunk. A checker that chooses a single reference chunk
  for a span can flag a detail that is supported elsewhere.
- **Small corpus:** Results measured on six renewable-energy documents
  may not generalize to a larger or more diverse collection.

### 16.2 Generation and caching

- **Model output varies:** Even with a fixed temperature, runtime
  behavior and model implementation can affect output. Existing
  project notes report that some measured values vary between runs.
- **Cache hit:** A repeated prompt may return a cached result. This
  can improve speed, but a run with caching enabled may make fewer
  real model calls.
- **Model unavailable:** If the local Ollama service or requested
  model is unavailable, generation may fail. Check the actual
  exception handling and logs before documenting the precise
  user-visible behavior.
- **Prompt mode matters:** Strict and loose prompts intentionally
  encourage different response behavior. Results from the two modes
  should not be mixed without recording the mode.

### 16.3 Verification

- **Pronoun without an antecedent:** A sentence can rely on context
  that is not present in the selected chunk, causing an unsupported
  verdict.
- **Wrong exact figure:** NLI can consider a sentence supported even
  when a number has changed. The span checker targets some of these
  cases but has its own limits.
- **Legitimate detail in another chunk:** Reference-chunk selection
  may not choose the passage that contains a valid number.
- **Name variation:** Entity extraction and lowercase substring
  matching can miss legitimate variants or flag a name expressed
  differently.
- **Neutral is not synonymous with false:** A neutral NLI result means
  the premise does not establish entailment or contradiction under the
  model's classification. It does not independently prove the claim
  false.
- **Filler exclusion:** Filler sentences are excluded from some
  scoring and evaluation counts according to the project's rules.
  Evaluation totals should be interpreted using the documented
  denominator.

### 16.4 Trust and interpretation

- **High token probability:** A model can produce a confidently worded
  unsupported statement. Token probability is one signal, not a
  factuality guarantee.
- **Equal document authority:** If all documents have the same
  authority value, the document component may have limited ability to
  distinguish answers.
- **Small evaluation set:** Results on 22 questions or 176 labeled
  sentences describe that test setup, not universal performance.
- **AI-created labels:** The existing labeled-set notes say labels
  were created by an AI assistant rather than a human. This introduces
  possible label errors and should be disclosed when reporting
  results.
- **Correlated examples:** Multiple sentences from the same answer are
  not necessarily independent samples. Confidence intervals can be
  optimistic if this dependence is ignored.

  ## 17. Testing and evaluation workflow

The repository contains scripts for different kinds of checks. Use the
existing script names and verify each script's current options before
relying on them.

### 17.1 Before running an evaluation

- Confirm the Python environment is activated.
- Confirm required dependencies are installed.
- Confirm the local Ollama service is available for tests that require
  generation.
- Confirm the configured model is present if a script calls Ollama.
- Confirm the corpus has been seeded if the test depends on indexed
  documents.
- Read the script's options and defaults before launching a
  potentially long run.
- Record whether caching is enabled and whether models were warmed up.
- Keep strict-mode and loose-mode results separate.
- Do not mix deliberately corrupted faulty-mode responses with clean
  evaluation results.
- Record the date, command, dataset, and relevant environment settings
  alongside results.

### 17.2 Existing evaluation commands

- `pytest` runs the automated test suite.
- `python scripts/run_eval.py` runs the question-set evaluation
  workflow.
- `python scripts/run_eval.py --retrieval-only` runs retrieval and
  pruning without generation, if the current command-line interface
  exposes that option as documented.
- `python scripts/run_eval.py --rescore` reuses stored non-faulty
  answers for verification and trust rescoring, according to the
  documented behavior.
- `python scripts/run_eval.py --limit N` limits the evaluation to the
  first N questions, if supported by the current script version.
- `python scripts/run_adversarial_eval.py` populates robustness
  results.
- `python scripts/run_ablation.py` evaluates the labeled-set variants.
- `python scripts/run_attention_check.py` runs the attention
  attribution check.
- `python scripts/compare_generators.py` runs the documented generator
  comparison workflow.

Check each script's current argument parser before relying on an option
copied from documentation. The command-line interface can change as the
project evolves.

### 17.3 Interpreting common metrics

- **Precision:** Of the items classified as positive, the fraction
  that are positive according to the evaluation labels.
- **Recall:** Of the positive items in the evaluation labels, the
  fraction detected by the system.
- **F1:** The harmonic mean of precision and recall. It balances the
  two measures but does not show their trade-off by itself.
- **False-positive rate:** The fraction of negative/control examples
  incorrectly flagged.
- **AUROC:** A measure of how well a score ranks positive examples
  above negative examples across thresholds. It does not establish
  that a score is calibrated.
- **Retrieval precision at k:** The fraction of the retrieved top-k
  chunks counted as relevant under the evaluation's relevance
  definition.
- **Prune ratio:** The fraction of retrieved chunks removed by
  pruning, averaged according to the evaluation implementation.
- **Generation latency:** Timing should distinguish real model calls
  from cache hits, as the project does in its documented metrics.

### 17.4 Reporting results responsibly

When reporting a metric, include the test set, number of examples, model
and prompt mode, and evaluation date. State whether labels were
human-annotated or AI-generated. Mention known limitations and avoid
presenting a small benchmark as proof that the system will behave
equally well on unrelated datasets.
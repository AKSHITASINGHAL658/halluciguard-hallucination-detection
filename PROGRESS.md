# PROGRESS

## Completed steps
- Step 1: skeleton, config, db, models, main
- Step 2: chunking, embeddings, vector_store, ingestion helper, seed_corpus, 6 renewable-energy docs
- Step 3: rag_pipeline, ingest/query routers, test_rag_pipeline.py (3 pass)
- Step 4: mi_pruning.py + test (3 pass), wired into run_pipeline
- Step 5: qa_eval_set.json (22 QA), metrics.py, /evaluate (retrieval-only so far; trust/hallucination metrics added in step 10)
- Step 6: nli_verifier.py + test (4 pass)
- Step 7: sentence_localizer.py, wired into /query (test added to test_rag_pipeline.py, 4 pass)
- Step 8: attention_extractor.py + test (1 pass), wired into /query for flagged sentences. Weights normalized over the top-5. Optional mask-and-rerun check skipped.
- Step 9: adversarial_agent.py, router, run_adversarial_eval.py, test_adversarial_agent.py. First run: negation 1.00, stale_fact 0.96, noise 1.00, control FP 0.125 (12 chunks available, 24 sentences). Each run replaces previous adversarial_tests rows.
- Step 10: trust_engine.py + test (4 pass), new app/core/query_service.py (process_query: full pipeline + persistence; used by /query and /evaluate). /evaluate now also returns avg_trust and hallucination_rate. generate() got 429 backoff and remembers models that reject logprobs (skips the doomed request).
- Step 11: dashboard/streamlit_app.py (Query, Robustness, Evaluation tabs); smoke-tested with streamlit AppTest + TestClient, no exceptions.
- Step 12: README with setup + final numbers; full suite 20 passed. Gemini 2.5-flash hit its 20/day free quota, so default model switched to gemini-3.1-flash-lite (same: no logprobs). Added 503 retry.

- Usage-reduction task: USE_LOGPROBS=False and LLM_CACHE (config, env override); disk cache in generate(); /evaluate + scripts/run_eval.py got retrieval_only / rescore / limit; cache tests added (22 pass). Real Gemini not called.

- Ollama task step 1: GENERATOR/OLLAMA_MODEL/OLLAMA_HOST in config (default GENERATOR=ollama since Gemini quota is tight; set GENERATOR=gemini to switch back), Ollama path in generate(), cache key includes generator+model, responses.generator column (auto-migrated in init_db), ollama added to requirements.
- Ollama task step 2: scripts/probe_ollama_logprobs.py. RESULT: qwen2.5:3b via ollama SDK 0.6.3 chat(logprobs=True) DOES return per-token logprobs (logprobs_source="tokens"); token_prob_component is now non-null. One real /query: ~24s cold (includes model loads), trust 0.91, token_prob 0.89.

- Ollama task step 3: inject_llm_contradiction (qwen2.5:3b, cached as cache/llm/adv-<sha>.json), new attack_type "llm_contradiction" in the suite (Robustness tab iterates detection_rate keys, so it appears automatically). New run (120 tests): negation 1.00, stale_fact 0.96, noise 1.00, llm_contradiction 0.29, control FP 0.125. Spot checks show the 3B model often returns near-paraphrases ("approximately", year-only tweaks), so the low rate partly reflects weak rewrites, not only NLI misses.

- Prompt mode: PROMPT_MODE=strict|loose (env; default strict). Loose lets the model add outside knowledge so unsupported sentences appear for the demo. Cache key includes the system prompt, so modes don't collide. Loose-mode spot check (qwen2.5:3b): 2 queries, 1 of 7 sentences flagged red. Tag loose-mode runs so they are not mixed with strict-mode eval numbers.
- Observed: the Ollama server was not running at one point (ConnectionError); `ollama serve` had to be started manually. This may explain the 189s average latency in the uncommitted comparison run; re-run before trusting it.

- Faulty mode (demo): POST /query {inject_faults: true} and a dashboard checkbox. adversarial_agent.inject_faults corrupts 1 sentence (2 if the answer has 4+) with wrong numbers (else negation) AFTER generation, then the normal NLI/entity/trust checks run. Tagged in responses.faults_injected (auto-migrated), excluded from rescore/eval, token-prob dropped (logprobs describe the original text). Live check on Three Gorges: 22,500 -> 33,750 MW, flagged contradiction, entity "33,750 megawatts" unsupported, attention showed "22,500", trust 0.16. Full suite 31 passed. evaluation.py also carries the (uncommitted-step-4) latency/token-prob fields.

- Latency profiling (Ollama qwen2.5:3b, warm): ~1.8-5s generation, ~3s NLI, attention ~1.5s per flagged sentence, ~8s per /query total. Cold first query was 18-36s (embedder ~8s, NLI load, Ollama model load). Fix: startup warmup (app/core/warmup.py, WARMUP=1) + OLLAMA_KEEP_ALIVE=30m; first query now ~8s. The earlier 189s/answer in the comparison run was NOT reproducible; re-run compare_generators before trusting any latency.

- Ollama task steps 4-5 DONE. compare_generators (Ollama only; Gemini needs --gemini): hallucination 20.0% (6/30 sentences), avg trust 0.879, avg token-prob 0.937, 4.4 s/answer generation (warm). Adversarial re-run after year-regex fix: negation 1.00, stale_fact 1.00, noise 1.00, llm_contradiction 0.208, control FP 0.167. Mocked-Ollama tests (logprob parsing, dict/object shapes, fallback when none, cache key) are in tests/test_rag_pipeline.py. Full suite: 33 passed. README results table updated.

- Span-level marking: app/core/span_checker.py finds the wrong parts of a sentence: numbers (incl. years) missing from the reference chunk (chosen by lexical topic overlap, not NLI) and named entities missing from every surviving chunk. [n] citations and "2030s"-style tokens are ignored. Wired into score_and_store: any wrong span sets is_hallucination=True and switches the shown source to the reference chunk; spans stored in response_sentences.wrong_spans (auto-migrated) and returned as sentences[].wrong_spans; dashboard marks only those words bright red inside the sentence. Also fix_split_decimals (qwen writes "42. 5"). Rescore of 22 stored answers: hallucination rate 20% (5/25), no false span flags on real answers. Faulty-mode check: "2029" and "207" both marked, trust 0.34 (was 0.69 with NLI alone). Full suite 40 passed.
- Span checker fix (after a screenshot showed "2030" unmarked): reference chunk is now picked by embedding similarity + word frequency (set-overlap ties were falling back to the NLI chunk), and a number found only in the NLI-preferred chunk is accepted when NLI entailment >= SPAN_STRONG_ENTAILMENT (0.9), which removed a false alarm on "hydropower ... 14 percent". Live faulty checks: 3/3 corruptions marked (e.g. 2026+207, 11,250, 1425+22); rescore of real answers: 20% hallucination, no false span flags. Suite: 42 passed.
- Polish: flagged sentences with wrong parts now render amber (not green); filler/meta sentences ("This information is directly stated in the context...", refusals) are detected by sentence_localizer.is_filler, shown grey italic, never flagged, and excluded from sentence/doc scores and eval counts (responses column is_filler, auto-migrated). Attention is computed only for NLI rejections. Rescore of stored real answers: hallucination rate 13.0% (3/23 claim sentences; was 20% of 25 before filler exclusion), avg trust 0.86. Suite: 44 passed. README table (20.0%) predates this.
- FIXED test pollution: tests/conftest.py points DB_URL/CHROMA_DIR (new env overrides in config) at a temp dir and seeds it from data/raw_docs, so tests never touch app.db/chroma. Deleted the 54 stub responses earlier tests had left in app.db. Rescore on clean data: hallucination 8.7% (2/23 claim sentences), avg trust 0.90. Suite 44 passed; real-DB row count unchanged by pytest.
- Known: filler sentences ("This information is directly stated in the context") are still marked supported by NLI.

## Assumptions
- CLAUDE.md says OpenAI gpt-4o-mini; plan.md (declared the spec) says Gemini via google-genai. Followed plan.md/user prompt (Gemini).
- Added `logprobs_source` column to responses (plan §6 says to store it).
- Added httpx to requirements (needed by FastAPI TestClient).
- GEMINI_MODEL defaults to gemini-2.0-flash if unset in .env.
- .env is never opened by me; it is loaded by app/config.py via python-dotenv.

- Gemini logprobs test (step 3): response_logprobs=True returns HTTP 400 "Logprobs is not enabled" on gemini-2.5-flash, gemini-3.8-flash, gemini-3.1-flash-lite (gemini-2.0-flash and 2.5-flash-lite are retired, 404). Default model set to gemini-2.5-flash (GEMINI_MODEL env overrides). Fallback works: logprobs_source="none", token_prob_component will be null and weights renormalize.
- Added app/core/ingestion.py (shared ingest helper used by router + seed script); not in §3 layout.
- Chunks are ~250 words, so each ~400-word doc yields 2 chunks.

- PMI threshold: if top PMI score <= 0, threshold = top/keep_ratio (lenient). Prompt tokens approximated as words*1.3. Redundancy check always applies; min_keep top-up prefers non-redundant chunks.

## Known issues
- Eval logic moved to app/core/evaluation.py and rescoring to query_service.rescore_response (rescore writes new response rows copying the stored answer). scripts/run_eval.py is new ("its script" did not exist). The README results table predates this change and was produced with logprobs attempted (outcome identical: none).
- Gemini free-tier rate limits (429) can make /evaluate slow or fail some questions (n_failed reported).
- Pronoun sentences ("It was completed in 2012") can be flagged by NLI because the premise chunk lacks the antecedent; kept per spec.
- precision@6 is structurally capped at ~0.33: each doc has only 2 chunks (~400 words) so at most 2 of 6 retrieved can be relevant. /evaluate also reports precision_after_pruning.
(none)

## Next step
Done (optional: numeric guard for NLI number blindness)

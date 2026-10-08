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
- Gemini free-tier rate limits (429) can make /evaluate slow or fail some questions (n_failed reported).
- Pronoun sentences ("It was completed in 2012") can be flagged by NLI because the premise chunk lacks the antecedent; kept per spec.
- precision@6 is structurally capped at ~0.33: each doc has only 2 chunks (~400 words) so at most 2 of 6 retrieved can be relevant. /evaluate also reports precision_after_pruning.
(none)

## Next step
Step 12

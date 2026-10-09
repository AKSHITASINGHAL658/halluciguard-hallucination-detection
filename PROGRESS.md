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
Done

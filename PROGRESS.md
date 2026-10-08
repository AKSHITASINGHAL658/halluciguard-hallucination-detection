# PROGRESS

## Completed steps
- Step 1: skeleton, config, db, models, main
- Step 2: chunking, embeddings, vector_store, ingestion helper, seed_corpus, 6 renewable-energy docs
- Step 3: rag_pipeline, ingest/query routers, test_rag_pipeline.py (3 pass)

## Assumptions
- CLAUDE.md says OpenAI gpt-4o-mini; plan.md (declared the spec) says Gemini via google-genai. Followed plan.md/user prompt (Gemini).
- Added `logprobs_source` column to responses (plan §6 says to store it).
- Added httpx to requirements (needed by FastAPI TestClient).
- GEMINI_MODEL defaults to gemini-2.0-flash if unset in .env.
- .env is never opened by me; it is loaded by app/config.py via python-dotenv.

- Gemini logprobs test (step 3): response_logprobs=True returns HTTP 400 "Logprobs is not enabled" on gemini-2.5-flash, gemini-3.8-flash, gemini-3.1-flash-lite (gemini-2.0-flash and 2.5-flash-lite are retired, 404). Default model set to gemini-2.5-flash (GEMINI_MODEL env overrides). Fallback works: logprobs_source="none", token_prob_component will be null and weights renormalize.
- Added app/core/ingestion.py (shared ingest helper used by router + seed script); not in §3 layout.
- Chunks are ~250 words, so each ~400-word doc yields 2 chunks.

## Known issues
(none)

## Next step
Step 4

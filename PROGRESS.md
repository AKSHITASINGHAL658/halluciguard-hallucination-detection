# PROGRESS

## Completed steps
- Step 1: skeleton, config, db, models, main

## Assumptions
- CLAUDE.md says OpenAI gpt-4o-mini; plan.md (declared the spec) says Gemini via google-genai. Followed plan.md/user prompt (Gemini).
- Added `logprobs_source` column to responses (plan §6 says to store it).
- Added httpx to requirements (needed by FastAPI TestClient).
- GEMINI_MODEL defaults to gemini-2.0-flash if unset in .env.
- .env is never opened by me; it is loaded by app/config.py via python-dotenv.

## Known issues
(none)

## Next step
Step 2

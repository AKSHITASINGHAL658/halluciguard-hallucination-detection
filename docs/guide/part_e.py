from guide_lib import *


def build():
    s = []
    s.append(part_page("Part E", "Running the demo and finding your way around the code"))
    s.append(h1("24. Running the demo", nobreak=True))
    s.append(h2("24.1 Before presenting"))
    s.append(ul(["Start Ollama: <code>ollama serve</code> (and once: <code>ollama pull qwen2.5:3b</code>).",
                 "Start the backend: <code>.venv/bin/python -m uvicorn app.main:app --port 8000</code>. It preloads the models at startup (about 12 seconds).",
                 "Start the dashboard: <code>.venv/bin/python -m streamlit run dashboard/streamlit_app.py</code> and open http://localhost:8501.",
                 "Using the <code>.venv/bin/python -m ...</code> form avoids starting a global Python that lacks the packages.",
                 "Run one test question so everything is warm.",
                 "Test your questions the day before, and keep screenshots or a screen recording as a backup.",
                 "For a loose-mode demo, start the backend with <code>PROMPT_MODE=loose</code> and restart it without afterwards."], ordered=True))
    s.append(h2("24.2 Demo script (about 3 minutes)"))
    s.append(ul(["Ask a normal question. Show the green sentences, the grey filler sentence, the entity list and the trust score. Point out &#8220;6 retrieved &#8594; 2 used&#8221;.",
                 "Tick <b>Faulty mode</b> and ask again. Show the bright red wrong numbers inside the amber sentence, the struck-through original, the &#8220;Caught&#8221; line, the lower trust score, and the source passage in the expander.",
                 "Open the <b>Robustness</b> tab: detection per attack type and the false-positive rate.",
                 "Open the <b>Evaluation</b> tab for the headline numbers (this calls the generator for all 22 questions, so run it beforehand or rely on <code>README.md</code>).",
                 "Mention the ablation table (section 18.3) and be ready to say the labels were made by an AI assistant."], ordered=True))
    s.append(box("say", "Tip", "Good demo questions: one the documents answer fully (all green); one they answer only partly; and one completely off-topic (&#8220;Who won the 2022 football World Cup?&#8221;), where the model should say the context has no answer and the sentence appears as grey filler."))
    s.append(box("care", "Be careful", "In faulty mode a one-claim answer can show a trust score of 0.00. That is the formula working as designed (section 13.9), but be ready to explain it."))
    s.append(h2("24.3 Troubleshooting"))
    s.append(table(["Problem", "Likely cause", "Fix"], [
        ["Dashboard shows a connection error", "Backend not running", "Start uvicorn"],
        ["ModuleNotFoundError (e.g. nltk) when starting", "Started with a Python outside the project's <code>.venv</code>", "Use <code>.venv/bin/python -m uvicorn ...</code>"],
        ["Query fails with a connection error to Ollama", "Ollama server not running", "<code>ollama serve</code>"],
        ["First query very slow", "Models not preloaded (<code>WARMUP=0</code>) or Ollama unloaded the model", "Run a test query; keep-alive is 30 minutes"],
        ["&#8220;database is locked&#8221;", "Another process holds <code>app.db</code>", "Wait (SQLite waits up to 30 s) or close the other process"],
        ["Everything slow", "Memory full (swapping)", "Close other programs; restart"],
        ["Answer all green in faulty mode", "Checkbox not ticked", "Tick it and ask again"]], [32, 38, 30]))

    s.append(h1("25. Where things are in the code"))
    s.append(p("For an exact, line-by-line list of every threshold, rule and number, see <b>FACTS.md</b> in the project root."))
    s.append(table(["File", "What it does"], [
        ["app/main.py", "Starts the FastAPI backend, connects the routes, preloads models"],
        ["app/config.py", "Settings: model names, thresholds, weights, prompts (a few constants still live in other files; see FACTS.md)"],
        ["app/db.py, app/models.py", "The SQLite database and its tables (and the small migration that adds new columns)"],
        ["app/routers/", "API endpoints: query, ingest, evaluate, adversarial"],
        ["app/core/chunking.py, embeddings.py, vector_store.py, ingestion.py", "Cut documents into chunks, embed them, store and search them"],
        ["app/core/mi_pruning.py", "PMI scoring and pruning"],
        ["app/core/rag_pipeline.py", "Retrieve &#8594; prune &#8594; generate (with the cache and logprobs)"],
        ["app/core/query_service.py", "The full /query flow: runs the pipeline, verification, attention, trust, and saves everything; also rescoring"],
        ["app/core/verification.py", "The sentence check used by the app and the ablation: NLI, claims, filler, wrong parts"],
        ["app/core/nli_verifier.py", "Loads DeBERTa once (<code>get_nli_model</code>) and runs NLI"],
        ["app/core/sentence_localizer.py", "Sentence splitting, claim-level NLI verdicts, filler detection"],
        ["app/core/claim_extractor.py", "Splits compound sentences into claims"],
        ["app/core/span_checker.py", "Finds numbers and names the sources do not contain"],
        ["app/core/attention_extractor.py", "Top attention words for NLI-rejected sentences"],
        ["app/core/trust_engine.py", "Entity check and the four trust components"],
        ["app/core/adversarial_agent.py", "Attacks, the attack suite, and faulty-mode corruption"],
        ["app/core/evaluation.py, metrics.py", "Evaluation metrics; retrieval-only and rescore modes"],
        ["app/core/warmup.py", "Preloads models at startup"],
        ["dashboard/streamlit_app.py", "The dashboard"],
        ["scripts/seed_corpus.py", "Loads the documents (run once)"],
        ["scripts/run_eval.py, run_adversarial_eval.py", "Evaluation and the adversarial suite from the command line"],
        ["scripts/build_labeled_set.py, run_ablation.py", "Generate candidate answers; run the labelled-set evaluation and ablation (about 25 minutes the first time)"],
        ["scripts/run_attention_check.py", "Measures how often the masking check supports the attention words (about 15 minutes)"],
        ["scripts/compare_generators.py", "Benchmarks the generator (hallucination rate, trust, latency)"],
        ["data/raw_docs/, data/eval/", "The 6 documents; the 22 questions; the labelled sentences"],
        ["outputs/", "Saved results: ablation.json, generator_comparison.json"],
        ["tests/", "43 automated tests (pytest, on a temporary database)"],
        ["FACTS.md, PROGRESS.md, README.md", "Exact rules and numbers; build log and assumptions; setup and results"]], [38, 62]))
    s.append(h2("25.1 Suggested reading order for the code"))
    s.append(ul(["<b>query_service.py</b> &#8212; the whole flow in one place.",
                 "<b>verification.py</b> &#8212; how a sentence gets its colour.",
                 "<b>trust_engine.py</b> &#8212; how the score is built.",
                 "<b>FACTS.md</b> &#8212; the exact rules; <b>PROGRESS.md</b> &#8212; the assumptions made during the build (professors often ask about these)."], ordered=True))

    s.append(h1("26. Glossary"))
    G = [
        ("Ablation", "Switching one component off to measure what it contributes"),
        ("Adversarial test", "Deliberately corrupting sentences to test whether the checker notices"),
        ("AI-labelled", "Labels made by an AI assistant instead of a human"),
        ("API / endpoint", "An address a program sends requests to (e.g. POST /query)"),
        ("Attention", "How strongly a transformer connects each word to the other words"),
        ("Masking check", "Deleting the top attention words and re-running NLI, compared with deleting random words"),
        ("Authority score", "A 0-1 trust value for each source document (all 0.8 here)"),
        ("AUROC", "Probability that a bad example scores worse than a good one; 0.5 = chance"),
        ("Bootstrap confidence interval", "A range for a number from many random re-samplings of the data"),
        ("Cache", "Saved answers reused for identical prompts"),
        ("ChromaDB", "A database that stores embeddings and finds the nearest ones"),
        ("Chunk", "A ~250-word piece of a document"),
        ("Claim (atomic)", "One checkable fact inside a sentence"),
        ("Claim sentence", "A sentence that is not filler"),
        ("Control set", "Unchanged true sentences used to measure false positives"),
        ("Corpus", "The collection of source documents (6 here)"),
        ("Cosine similarity", "How similar two embeddings are"),
        ("DeBERTa", "The transformer that performs NLI"),
        ("Detection rate", "Fraction of corrupted sentences caught"),
        ("Embedding", "A list of numbers representing a text's meaning"),
        ("Entailment / neutral / contradiction", "The premise supports / doesn't address / opposes the hypothesis"),
        ("Entity", "A specific name, number or date in a sentence"),
        ("F1", "The balance of precision and recall"),
        ("Faithfulness / groundedness", "Whether a claim is supported by the sources"),
        ("False positive", "A true sentence wrongly flagged"),
        ("Faulty mode", "Test mode that corrupts an answer on purpose"),
        ("Filler sentence", "A meta sentence with no claim (&#8220;this is stated in the context&#8221;)"),
        ("Generator", "The AI model that writes the answer (qwen2.5:3b)"),
        ("Hallucination", "In this project: a claim not supported by the sources"),
        ("Hypothesis / premise", "In NLI: the claim being checked / the source text"),
        ("Keep-alive", "Ollama's setting for how long a model stays in memory (30 minutes)"),
        ("Labelled set", "176 answer sentences marked supported or unsupported, used to measure P/R/F1"),
        ("Localisation", "Finding exactly which sentence (or word) is unsupported"),
        ("Logprob", "The logarithm of the probability the model gave a word"),
        ("Loose mode", "Test mode with a prompt that invites outside knowledge"),
        ("MiniLM", "The small model that creates embeddings"),
        ("NLI", "Natural language inference: does text A support text B?"),
        ("Ollama", "A program that runs AI models locally"),
        ("PMI", "Pointwise mutual information: is a word more common in this chunk than on average?"),
        ("Precision / recall", "Of the items flagged, the fraction that were right / of the true items, the fraction found"),
        ("Pruning", "Removing chunks that aren't useful"),
        ("RAG", "Retrieval-Augmented Generation: look up relevant text, then answer from it"),
        ("Reference chunk", "The chunk a sentence is about, chosen by meaning; used by the number check"),
        ("Renormalisation", "Rescaling weights to add up to 1 when one is missing"),
        ("Softmax", "Turns raw model scores into probabilities"),
        ("Span check", "The number and name check that marks wrong words"),
        ("spaCy", "A library that finds entities and grammar in text"),
        ("SQLite", "A simple database stored in one file"),
        ("Streamlit", "The tool used to build the dashboard"),
        ("Temperature", "How random the generator is; 0 = always the most likely word"),
        ("Threshold", "The cut-off (0.5) below which a sentence is flagged"),
        ("Token", "A word or piece of a word, the unit models read"),
        ("Transformer", "The neural-network type behind modern language models"),
        ("Trust score", "The 0-1 summary of an answer's reliability"),
        ("Warm-up", "Loading all models at startup so the first query is fast"),
    ]
    s.append(table(["Term", "Meaning"], [[f"<b>{a}</b>", b] for a, b in G], [30, 70]))

    s.append(h1("Appendix: the running example, end to end"))
    s.append(table(["Step", "What happened to &#8220;How much new solar capacity was added globally in 2023?&#8221;"], [
        ["Retrieval", "6 chunks closest in meaning were found, including some about wind and policy"],
        ["MI pruning", "Chunks that don't concentrate the question's words were dropped: 6 &#8594; 2 kept (a solar chunk and a policy chunk)"],
        ["Generation", "qwen2.5:3b wrote: <i>&#8220;In 2023, about 345 gigawatts of new solar capacity was added globally. This information is directly stated in the context provided.&#8221;</i> (about 2.5 s)"],
        ["Filler detection", "The second sentence has no number and mentions &#8220;context&#8221; and &#8220;stated&#8221;: filler, grey, ignored in scoring"],
        ["Claims + NLI", "The first sentence is one claim; its best chunk is the solar chunk, entailment well above 0.5: green"],
        ["Number/name check", "345 and 2023 are in the solar chunk: nothing marked"],
        ["Attention", "Not needed: no sentence rejected by NLI"],
        ["Trust", "Sentence about 0.9, entity 1.0, document 0.8, token probability about 0.9: trust about 0.9"],
        ["Faulty mode (same question)", "The answer becomes &#8220;In 2013, about 690 gigawatts...&#8221;; NLI still says entailment (0.80), but 2013 and 690 are not in the solar chunk: both marked bright red, sentence amber, trust 0.00 (one claim)"]], [22, 78]))
    s.append(box("chk", "Final self-check", "Can you explain, in your own words: (1) why the answer is checked sentence by sentence and claim by claim; (2) why a true sentence can be red; (3) why the adversarial test uses the clean chunk and what it does not cover; (4) why model confidence is only 20% of the trust score, and what the ablation says about that; (5) who labelled the evaluation data? If yes, you understand the project."))
    return "".join(s)

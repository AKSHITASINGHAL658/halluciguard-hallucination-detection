from guide_lib import *


def flow_fig():
    G, W = "#dcfce7", "#ffffff"
    boxes = [
        (8, 8, 124, 52, "1. Question", "dashboard -> POST /query", W),
        (150, 8, 124, 52, "2. Retrieve", "ChromaDB: 6 nearest\nchunks (MiniLM)", W),
        (292, 8, 124, 52, "3. MI pruning", "PMI + duplicates\n6 -> about 2 chunks", W),
        (434, 8, 124, 52, "4. Generate", "qwen2.5:3b (Ollama)\nanswer + logprobs", W),
        (434, 84, 124, 52, "5. Verify", "split sentences + claims\nNLI + number/name check", G),
        (292, 84, 124, 52, "6. Attention", "NLI-rejected sentences:\ntop-5 source words", G),
        (150, 84, 124, 52, "7. Trust score", "sentence, entity,\ndocument, token prob", G),
        (8, 84, 124, 52, "8. Save + show", "SQLite + dashboard\ncolours, red words", W),
    ]
    arrows = [(132, 34, 150, 34), (274, 34, 292, 34), (416, 34, 434, 34), (496, 60, 496, 84), (434, 110, 416, 110),
              (292, 110, 274, 110), (150, 110, 132, 110)]
    return svg_boxes(boxes, arrows, 566, 150, "Green boxes = the checking part (the main contribution)")


def prune_fig():
    items = [("A: solar\n2023", 2.4, "#16a34a"), ("B: solar\ncosts", 1.5, "#16a34a"), ("C: wind\n2023", 0.7, "#dc2626"),
             ("D: energy\ngeneral", 0.4, "#dc2626"), ("E: hydro", 0.2, "#dc2626"), ("F: copy\nof A", 2.35, "#d97706")]
    base = svg_bars(items, "Illustration: green = kept, red = below cut-off, amber = duplicate of a kept chunk", 520, 220, max_v=3.0, fmt="{:.1f}")
    return base.replace("</svg>", '<line x1="34" y1="102.8" x2="510" y2="102.8" stroke="#111" stroke-dasharray="4 3"/><text x="500" y="98" text-anchor="end" font-size="8" fill="#111">cut-off = half the best score</text></svg>')


def heat_table():
    def c(v, best=False):
        shade = int(255 - v * 160)
        col = f"rgb({shade},{min(255, shade + 40)},255)"
        return f'<td style="background:{col};text-align:center;{"font-weight:700;" if best else ""}">{v:.2f}{" (best)" if best else ""}</td>'
    rows = [("Sentence 1 &#8594; <span class='g'>green</span>", 0.94, 0.10, 0), ("Sentence 2 &#8594; <span class='g'>green</span>", 0.20, 0.85, 1),
            ("Sentence 3 &#8594; <span class='lr'>red</span> (best 0.30 &lt; 0.5)", 0.15, 0.30, 1)]
    out = "<table style='width:70%;margin:6pt auto'><tr><th></th><th>Chunk 1</th><th>Chunk 2</th></tr>"
    for lab, a, b, best in rows:
        out += f"<tr><td>{lab}</td>{c(a, best == 0)}{c(b, best == 1)}</tr>"
    return out + "</table>"


def attn_fig():
    words = [("In", .05), ("2023", 1.0), (",", .02), ("the", .06), ("world", .25), ("added", .45), ("about", .1), ("345", .8), ("GW", .35), ("of", .05), ("new", .1), ("solar", .4), ("capacity", .1)]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 70" width="560" height="70" font-family="Helvetica, Arial, sans-serif">']
    x = 6
    for w, a in words:
        wd = 9 * len(w) + 14
        r, g, b = 255, int(245 - a * 150), int(235 - a * 200)
        out.append(f'<rect x="{x}" y="14" width="{wd}" height="24" rx="4" fill="rgb({r},{g},{b})"/><text x="{x+wd/2}" y="30" text-anchor="middle" font-size="10">{w}</text>')
        x += wd + 4
    out.append('<text x="6" y="58" font-size="8.5" fill="#6b7280">Flagged sentence: &#8220;In 2017, the world added about 345 GW of new solar capacity.&#8221;  Darker = more attention (illustrative).</text></svg>')
    return "".join(out)


def trust_fig():
    items = [("Sentence\n0.90", 0.90, "#2563eb"), ("Entity\n1.00", 1.0, "#16a34a"), ("Document\n0.80", 0.8, "#d97706"), ("Token prob\n0.94", 0.94, "#7c3aed")]
    return svg_bars(items, "Example answer: component scores (illustrative)", 360, 190, fmt="{:.2f}")


def renorm_fig():
    def seg(x, w, color, lab):
        return f'<rect x="{x}" y="{{y}}" width="{w}" height="34" fill="{color}"/><text x="{x+w/2}" y="{{ty}}" text-anchor="middle" font-size="9" fill="#fff">{lab}</text>'
    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 120" width="540" height="120" font-family="Helvetica, Arial, sans-serif">']
    rows = [("All 4 available", 20, [(0.35, "#2563eb", "Sentence 0.35"), (0.25, "#16a34a", "Entity 0.25"), (0.20, "#d97706", "Document 0.20"), (0.20, "#7c3aed", "Token 0.20")]),
            ("Token prob missing", 70, [(0.4375, "#2563eb", "Sentence 0.4375"), (0.3125, "#16a34a", "Entity 0.3125"), (0.25, "#d97706", "Document 0.25")])]
    for lab, y, segs in rows:
        o.append(f'<text x="4" y="{y+21}" font-size="9" fill="#374151">{lab}</text>')
        x = 110
        for wgt, col, t in segs:
            w = 420 * wgt
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="34" fill="{col}"/><text x="{x+w/2}" y="{y+21}" text-anchor="middle" font-size="9" fill="#fff">{t}</text>')
            x += w
    o.append("</svg>")
    return "".join(o)


def build():
    s = []
    s.append(part_page("Part B", "How the system works, step by step",
                       box("ex", "The running example", "<b>&#8220;How much new solar capacity was added globally in 2023?&#8221;</b>")))
    s.append(h1("5. The system at a glance", nobreak=True))
    s.append(fig(flow_fig(), "Figure 2. How one question flows through HalluciGuard. Steps 5-7 are the checking part."))
    s.append(p("Inside step 5, in order: fix split decimals (&#8220;42. 5&#8221; &#8594; &#8220;42.5&#8221;), split the answer into sentences, mark <b>filler</b> sentences, split compound sentences into <b>claims</b>, run <b>NLI</b> on every claim against every kept chunk, then run the <b>number and name check</b>, and decide each sentence's colour."))
    s.append(h2("5.1 The main parts of the software"))
    s.append(table(["Part", "Technology", "Role"], [
        ["Backend (API)", "Python + FastAPI", "Receives requests, runs the pipeline, returns results; preloads models at startup"],
        ["Dashboard", "Streamlit", "Input box, coloured answer, charts, three tabs"],
        ["Vector database", "ChromaDB", "Stores chunk embeddings and finds the nearest ones"],
        ["Relational database", "SQLite (app.db)", "Documents, chunks, questions, answers, sentence labels, scores, test results"],
        ["Embedding model", "all-MiniLM-L6-v2", "384-number embeddings for search, and for choosing which chunk a sentence is about"],
        ["Generator", "qwen2.5:3b via Ollama", "Writes the answer, returns token logprobs"],
        ["Verifier", "DeBERTa-v3-base NLI", "Checks each claim against the sources"],
        ["Language tools", "spaCy (en_core_web_sm), NLTK", "spaCy finds entities and splits compound sentences; NLTK splits sentences"]], [22, 26, 52]))
    s.append(p("The dashboard does no work itself; it calls the backend through the API. Logic lives in <code>app/core/</code> as plain functions; the API routes only call them."))
    s.append(h2("5.2 The API endpoints"))
    s.append(table(["Method", "Endpoint", "What it does"], [
        ["POST", "/documents", "Adds a document: chunks it, embeds it and stores it"],
        ["GET", "/documents", "Lists the stored documents"],
        ["POST", "/query", "Full pipeline for a question; body: <code>text</code>, <code>k</code> (default 6), <code>inject_faults</code> (default false)"],
        ["GET", "/response/{id}", "Fetches a stored result again"],
        ["POST", "/evaluate", "Runs the 22 evaluation questions; options <code>retrieval_only</code>, <code>rescore</code>, <code>limit</code>"],
        ["POST", "/adversarial/run", "Runs the adversarial test suite"],
        ["GET", "/adversarial/results", "Detection rate per attack type and the false-positive rate"],
        ["GET", "/health", "Checks that the backend is running"]], [12, 24, 64]))

    s.append(h1("6. Step 0: Preparing the documents (ingestion)"))
    s.append(p("Before anyone can ask a question, the documents are prepared once by running <code>scripts/seed_corpus.py</code>."))
    s.append(h2("6.1 The corpus"))
    s.append(p("The corpus is <b>6 documents (about 400 words each) on renewable energy</b>: solar, wind, hydropower, geothermal, energy storage, and policy and investment. They contain specific numbers and dates, which gives the checker something concrete to verify and the adversarial test something to corrupt. They were written for the project, which keeps the experiment controlled but means the results are not a public benchmark."))
    s.append(h2("6.2 Chunking"))
    s.append(p("Each document is cut into <b>word-based</b> windows of <b>250 words</b> that advance 210 words at a time, so neighbouring chunks <b>overlap by 40 words</b>. The cutting is not sentence-aware (it counts words, and rejoins them with single spaces). The 6 documents produce <b>12 chunks</b> (2 per document)."))
    s.append(ul(["<b>Precision:</b> finding one fact in one page is easier than in a whole book.",
                 "<b>The checker's reading limit:</b> DeBERTa reads at most 512 tokens, chunk and sentence together; about 250 words of chunk leaves room for the sentence.",
                 "<b>Shorter prompts:</b> the generator only gets the few chunks that matter.",
                 "<b>Why overlap?</b> A fact at the boundary of two chunks could be cut in half; with 40 words of overlap it appears complete in at least one."]))
    s.append(box("care", "Be careful", "Chunk size is a trade-off. Smaller chunks suit the checker better but can separate a sentence from the context it depends on (for example a pronoun from the name it refers to). 250 words is a reasonable middle; testing other sizes is future work."))
    s.append(h2("6.3 Embedding and storing"))
    s.append(p("Each chunk is embedded by MiniLM and stored in ChromaDB (cosine space); the chunk text, position and document details go to SQLite. Each document has an <b>authority</b> score between 0 and 1, used in the trust score. <b>In the current corpus all six documents have the same authority, 0.8</b> (the default), so this component carries no information yet (see section 13.3)."))
    s.append(h2("6.4 The database tables"))
    s.append(table(["Table", "What it stores"], [
        ["documents", "Title, source, authority score"],
        ["chunks", "Chunk text, its document, its position, its ID in ChromaDB"],
        ["queries", "Each question asked"],
        ["responses", "The full answer, model, <b>generator</b>, chunks retrieved and kept, overall trust, <b>logprobs_source</b>, <b>faults_injected</b> (set only in faulty mode)"],
        ["response_sentences", "Each sentence: NLI label and confidence, flagged or not, best chunk, top attention words, <b>wrong_spans</b> (the red words), <b>is_filler</b>"],
        ["entities", "Each name/number/date in an answer, and whether it was found in the sources"],
        ["trust_scores", "The four trust components and the final score"],
        ["adversarial_tests", "Each test: attack type, original and corrupted sentence, NLI label, detected or not"]], [26, 74]))

    s.append(h1("7. Step 1: Retrieval &#8212; finding the right chunks"))
    s.append(p("The question is turned into an embedding with the same MiniLM model. ChromaDB then returns the <b>6 chunks closest to the question</b> (cosine similarity). These are the <b>retrieved</b> chunks."))
    s.append(h2("7.1 Why embeddings instead of word search?"))
    s.append(p("Meaning matters more than exact words. If the user asks &#8220;How much solar power was installed?&#8221; and a document says &#8220;new photovoltaic capacity added&#8221;, a word search finds almost nothing in common, but the embeddings are close."))
    s.append(h2("7.2 The weakness of retrieval"))
    s.append(p("Embeddings are deliberately &#8220;loose&#8221;. A chunk about &#8220;wind capacity added globally in 2023&#8221; is also close to our solar question. So the top 6 always contain related but useless chunks, and irrelevant context can confuse the generator. That is why the next step exists."))
    s.append(box("chk", "Check yourself", "Why is a wind chunk a problem when the question is about solar?<br><i>Answer:</i> the generator might mix wind facts into the solar answer, a hallucination caused by bad context."))

    s.append(h1("8. Step 2: MI pruning &#8212; removing useless chunks"))
    s.append(p("Pruning looks at the 6 retrieved chunks and keeps only those that genuinely match the question. 6 chunks usually become about 2."))
    s.append(h2("8.1 The question pruning asks"))
    s.append(box("key", "Key idea", "&#8220;Does this chunk use the question's words <b>more than the other retrieved chunks do</b>?&#8221;"))
    s.append(p("A word that appears everywhere (&#8220;energy&#8221;) says nothing about which chunk is relevant; a word concentrated in one chunk (&#8220;solar&#8221; in a solar chunk) is a strong signal. This is measured with PMI."))
    s.append(h2("8.2 PMI in plain words"))
    s.append(p("<b>PMI (pointwise mutual information)</b> compares, for each word of the question, how often it appears <i>in this chunk</i> with how often it appears <i>across all the retrieved chunks</i>:"))
    s.append("<pre>PMI(question, chunk) = average over question words t of\n    ln( (P(t | chunk) + 0.000001) / (P(t | all retrieved chunks) + 0.000001) )\n(counts are smoothed by adding 1 to every word count)</pre>")
    s.append(p("Question words are lower-cased, stop words (&#8220;the&#8221;, &#8220;was&#8221;...) are removed and repeated words counted once. Equal frequency scores 0, more common in the chunk is positive, less common is negative."))
    s.append(box("ex", "Example", "Question words: &#8220;solar&#8221;, &#8220;capacity&#8221;, &#8220;added&#8221;, &#8220;2023&#8221;. If &#8220;solar&#8221; is 6 times more common in Chunk A than on average, it adds ln 6 &#8776; 1.8 to Chunk A's average. In a wind chunk &#8220;solar&#8221; never appears, giving a negative contribution. Words like &#8220;capacity&#8221; appear everywhere and contribute about 0."))
    s.append(h2("8.3 The pruning rules"))
    s.append(ul(["<b>Relevance cut-off:</b> go through the chunks from best to worst score and keep a chunk only if its score is at least <b>half of the best score</b>. (If every score is zero or negative, the cut-off is relaxed to twice the best score.)",
                 "<b>Duplicate removal:</b> also skip a chunk if its embedding similarity to a chunk already kept is above <b>0.85</b>.",
                 "<b>Safety net:</b> always keep at least <b>2</b> chunks; if the rules left fewer, top up with the next-best chunks, preferring non-duplicates."], ordered=True))
    s.append(fig(prune_fig(), "Figure 3. Illustration of pruning six retrieved chunks."))
    s.append(h2("8.4 Why prune?"))
    s.append(ul(["<b>Fewer context-induced hallucinations:</b> the generator is not distracted by off-topic chunks.", "<b>Shorter prompts:</b> less to process.", "<b>Cleaner verification:</b> fewer chunks to compare each sentence against."]))
    s.append(h2("8.5 Why PMI and not a trained model?"))
    s.append(p("Mutual information can also be estimated with neural networks, but those need training data and time. The PMI formula is closed-form, fast, and every score can be explained."))
    s.append(h2("8.6 The result"))
    s.append(p("Pruning drops about <b>62%</b> of retrieved chunks (6 &#8594; about 2) and precision rises from <b>0.33 to 0.74</b>. The ablation (section 18.3) also shows that <b>pruning improves the checker itself</b>: feeding all 6 chunks to NLI lowered recall from 0.84 to 0.67 and F1 from 0.71 to 0.65."))
    s.append(box("chk", "Check yourself", "A wind chunk mentions &#8220;2023&#8221;, which is in the question. Why is it still dropped?<br><i>Answer:</i> &#8220;2023&#8221; appears in most retrieved chunks, so it adds little; and the chunk never mentions &#8220;solar&#8221;, which pulls its score below half of the best."))

    s.append(h1("9. Step 3: Generation &#8212; the AI writes the answer"))
    s.append(h2("9.1 What the generator receives"))
    s.append(p("The kept chunks are numbered and placed in a prompt with the question. A system instruction tells the model how to answer. The model runs at <b>temperature 0</b> (always the most likely next word), so the same prompt gives the same answer."))
    s.append(table(["Mode", "System instruction (verbatim)"], [
        ["<b>Strict</b> (default)", "&#8220;Answer the question using only the numbered context passages below. Respond in 3-6 short sentences. If the context does not contain the answer, say so.&#8221;"],
        ["<b>Loose</b> (test mode)", "&#8220;Answer the question thoroughly in 4-6 sentences. The context passages below are a starting point; enrich the answer with additional background, specific figures, dates, named examples and comparisons from your own knowledge. Do not mention the context or say when information is missing.&#8221;"]], [20, 80]))
    s.append(box("ex", "Example answer", "&#8220;In 2023, about 345 gigawatts of new solar capacity was added globally. This information is directly stated in the context provided.&#8221;<br>(Second sentence is <i>filler</i>: it makes no claim, see section 10.7.)"))
    s.append(box("care", "Be careful", "The model is <i>told</i> to use only the context, but it doesn't always obey. This is exactly why verification is needed."))
    s.append(h2("9.2 Logprobs: the model's confidence"))
    s.append(p("The generator returns a logprob for every word it writes. Averaging them (and converting back with <code>exp</code>) gives the model's typical per-word probability, which becomes the <b>token-probability</b> part of the trust score. If a model returned no logprobs, this component is simply missing and the other weights are scaled up (section 13.6)."))
    s.append(h2("9.3 The cache"))
    s.append(p("Every answer is saved on disk, keyed by the model, the system instruction and the exact prompt. If the same prompt is sent again, the saved answer is reused. This makes re-running evaluations fast. (The prompt is part of the key, so strict and loose answers never mix.)"))
    s.append(h2("9.4 Why a local model (Ollama) and not an online API?"))
    s.append(p("The project first used Google's Gemini API, then <b>removed it completely</b>. Two problems appeared: the Gemini models available rejected logprobs (&#8220;Logprobs is not enabled&#8221;), and the free tier allowed only a small number of requests per day, which made repeated evaluation impossible. With Ollama running <b>qwen2.5:3b</b> (a 3-billion-parameter model):"))
    s.append(ul(["there are no usage limits and nothing leaves the computer;", "per-token logprobs are returned, so all four trust components work;", "generation takes about <b>2.5 seconds</b>; a whole <code>/query</code> (retrieval, generation, NLI) takes about <b>8 seconds</b>, of which NLI is about 3.",
                 "the API preloads the models at startup and asks Ollama to keep the model in memory (30 minutes), so even the first query is fast."]))
    s.append(box("key", "Key idea", "A side effect worth knowing: small local models hallucinate somewhat more than large online ones. For a project about <i>detecting</i> hallucinations that is useful, because the checker has real errors to catch. In strict mode though, this model is mostly faithful (section 18), which is why the test modes exist."))

    s.append(h1("10. Step 4: NLI verification &#8212; checking each sentence"))
    s.append(p("This is the heart of the project. The answer is split into sentences (NLTK), compound sentences are split into smaller claims, and each claim is checked against each kept chunk using <b>NLI</b>."))
    s.append(h2("10.1 What NLI is"))
    s.append(p("<b>NLI (natural language inference)</b> answers one question about two texts: <b>&#8220;If Text A is true, is Text B also true?&#8221;</b> <b>Premise</b> (Text A) = a chunk from the documents. <b>Hypothesis</b> (Text B) = the claim being checked."))
    s.append(table(["Label", "Meaning", "Example (premise: &#8220;In 2023 the world added about 345 GW of new solar capacity.&#8221;)"], [
        ["<b>Entailment</b>", "The premise supports the hypothesis", "&#8220;More than 300 GW of solar was added in 2023.&#8221;"],
        ["<b>Neutral</b>", "The premise doesn't say either way", "&#8220;Solar panels are popular in India.&#8221;"],
        ["<b>Contradiction</b>", "The premise says the opposite", "&#8220;Only 100 GW of solar was added in 2023.&#8221;"]], [18, 28, 54]))
    s.append(p("&#8220;More than 300&#8221; is not written in the premise, but 345 is more than 300, so it is supported. <b>NLI judges meaning, not copied words.</b>"))
    s.append(h2("10.2 Why the chunk is the premise and the sentence is the hypothesis"))
    s.append(p("We want to know whether the <i>sources</i> support the <i>answer</i>, so the source plays &#8220;what is known to be true&#8221; and the answer is the claim under test. Swapping them would ask whether the answer supports the whole chunk, which it almost never does."))
    s.append(h2("10.3 The model: DeBERTa"))
    s.append(p("The checker is <b>DeBERTa-v3-base</b>, specifically <code>MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli</code>, a Microsoft transformer further trained for NLI on MNLI (many kinds of text), FEVER (claims checked against Wikipedia) and ANLI (hard pairs written to fool models). It learned patterns such as &#8220;not&#8221; flipping meaning, or different numbers for the same thing signalling contradiction. Nobody wrote these as rules. The model is used <b>pretrained, not fine-tuned</b> (section 20)."))
    s.append(h2("10.4 What happens inside, step by step"))
    s.append("<pre>[CLS] premise text [SEP] hypothesis text [SEP]      (at most 512 tokens; the chunk is cut first if too long)</pre>")
    s.append(p("The model reads both texts at the same time, so every word can be related to every other word (this is how it notices that &#8220;345 GW&#8221; in the chunk matches &#8220;345 gigawatts&#8221; in the sentence). It outputs three raw scores, and softmax turns them into probabilities, for example entailment 0.94, neutral 0.05, contradiction 0.01 (illustrative)."))
    s.append(box("care", "Be careful", "Different NLI models order their labels differently. The code reads label names from the model's configuration (<code>model.config.id2label</code>) instead of assuming. For this model the configuration lists 0 = entailment, 1 = neutral, 2 = contradiction."))
    s.append(h2("10.5 Checking a whole answer: the sentence &#215; chunk table"))
    s.append(p("An answer has several sentences and there are usually 2 kept chunks, so every claim is checked against every kept chunk. The chunk with the highest entailment probability is the claim's <b>best chunk</b>. A fact only needs to be supported by <i>one</i> source."))
    s.append(fig(heat_table(), "Figure 4. Entailment probability for every sentence-chunk pair (illustrative). The best chunk decides."))
    s.append(h2("10.6 Atomic claims: splitting compound sentences"))
    s.append(p("Many answer sentences contain two facts joined by &#8220;and&#8221;. If one is wrong and one is right, a whole-sentence check can be fooled. So compound sentences are split, using spaCy's grammar analysis, at coordinated verbs. A clause that has no subject, or only a pronoun like &#8220;it&#8221;, borrows the main clause's subject so it can stand alone:"))
    s.append(box("ex", "Example", "&#8220;The Three Gorges Dam has a capacity of 22,500 megawatts <i>and it</i> typically generates between 85 and 100 terawatt-hours per year.&#8221;<br>&#8594; claim 1: &#8220;The Three Gorges Dam has a capacity of 22,500 megawatts.&#8221;<br>&#8594; claim 2: &#8220;The Three Gorges Dam typically generates between 85 and 100 terawatt-hours per year.&#8221;"))
    s.append(p("The split only happens when it is safe: at most 4 claims, each at least 4 words, otherwise the sentence stays whole. A sentence takes the verdict of its <b>weakest claim</b> (the lowest best-entailment); it is flagged if any claim is flagged. This does <b>not</b> fix pronouns that refer to a previous sentence (&#8220;It was completed in 2012.&#8221;)."))
    s.append(box("care", "Be careful", "In the ablation (section 18.3) claim splitting did <b>not</b> measurably improve F1 (0.73 vs 0.74 for sentence-level NLI; the intervals overlap). Present it as a reasonable design that was measured and gave no clear gain."))
    s.append(h2("10.7 Filler sentences"))
    s.append(p("Some sentences make no claim: &#8220;This information is directly stated in the context provided.&#8221; or &#8220;The context does not contain the answer.&#8221; A sentence is <b>filler</b> when, after removing [1]-style citations, it has <b>no digit</b>, is <b>at most 25 words</b>, and mentions a reference word (context, passage, source, document...) together with a meta word (stated, provided, mentioned, according, found, does not contain...). Filler is shown in grey italics, is <b>never flagged</b>, and is excluded from the sentence score, the document score and the evaluation counts. A <b>claim sentence</b> is any sentence that is not filler."))
    s.append(h2("10.8 The decision rule (exactly how a sentence gets its colour)"))
    s.append(table(["Condition (checked in this order)", "Result"], [
        ["Sentence is filler", "<span class='gr'>grey</span>: never flagged"],
        ["The weakest claim's best entailment is <b>below 0.5</b>", "<span class='lr'>red</span> (flagged; shown with a source chunk and attention words)"],
        ["The number/name check found wrong parts (section 11)", "<span class='a'>amber</span> sentence with <span class='red'>bright red</span> words (flagged)"],
        ["Best entailment &#8805; 0.5 and the label is entailment", "<span class='g'>green</span> (supported)"],
        ["Best entailment &#8805; 0.5 but the label is neutral or contradiction", "<span class='a'>amber</span>"]], [56, 44]))
    s.append(p("Three details: (0) if a sentence is rejected by NLI <i>and</i> has wrong parts, the dashboard shows it amber with the bright red words; (1) the label is &#8220;contradiction&#8221; whenever <i>any</i> chunk gives contradiction above 0.5, but a sentence only turns red if its <i>best</i> entailment is below 0.5; (2) the 0.5 threshold is a design choice, not learned. The ablation sweep (section 18.3) found that thresholds from 0.3 to 0.9 all give F1 of about 0.71-0.72, so the result is not sensitive to it."))
    s.append(h2("10.9 Why sentence by sentence?"))
    s.append(p("If the whole answer were checked at once, five correct sentences and one wrong one would still mostly &#8220;agree&#8221; with the sources and the error would hide. Checking each sentence separately finds <b>exactly which claim</b> is unsupported (sentence-level hallucination localisation)."))
    s.append(h2("10.10 Similar words are not the same meaning"))
    s.append(box("ex", "A classic trap", "Premise: &#8220;<b>Wind</b> capacity grew in 2023.&#8221; Hypothesis: &#8220;<b>Solar</b> capacity grew in 2023.&#8221; Five of six words match, but the premise says nothing about solar. The correct label is <b>neutral</b>. A word-matching checker would wrongly say &#8220;supported&#8221;; this is why NLI is used."))
    s.append(h2("10.11 Where NLI makes mistakes"))
    s.append(table(["Weak spot", "Example", "Why it happens"], [
        ["Pronouns", "&#8220;It was completed in 2012.&#8221;", "What &#8220;it&#8221; refers to is in another sentence"],
        ["Long premises", "One relevant line inside a 250-word chunk", "Trained mostly on short premises; long text dilutes the evidence"],
        ["Exact numbers", "A wrong figure got entailment 0.97; &#8220;In 2030, about 172 GW...&#8221; got 0.68", "It learned meaning, not arithmetic; the chunk is full of similar numbers"],
        ["Loosely related chunks", "A policy chunk 'entails' a sentence about solar because it also mentions 2030 and gigawatts", "The best chunk can be a weakly related one"],
        ["Subtle rewrites", "A fluent sentence with one detail quietly changed", "Obvious negations are easy; small shifts are hard"]], [18, 40, 42]))
    s.append(p("These weaknesses explain most of the errors in the results, and the number and name check (next section) was added to address the numeric ones. The ablation also shows that <b>contradictions and miscopied facts</b> are the hardest category (recall 0.44 on 9 examples)."))
    s.append(box("chk", "Check yourself", "Premise: &#8220;Wind capacity grew 10% in 2022.&#8221; Hypothesis: &#8220;Wind capacity did not grow in 2022.&#8221; Which label?<br><i>Answer:</i> contradiction."))

    s.append(h1("11. Step 5: The number and name checker (span check)"))
    s.append(p("Testing showed that NLI is weak with exact numbers: a sentence with a wrong figure once got entailment 0.97. Wrong numbers are among the most common and most harmful hallucinations, so a second, simpler check was added. It also lets the dashboard mark <b>exactly which words</b> are wrong instead of colouring the whole sentence."))
    s.append(h2("11.1 What it checks"))
    s.append(ul(["<b>Numbers (including years):</b> every standalone number in the sentence must appear among the numbers of the <b>source passage the sentence is about</b>. A number that does not is marked bright red.",
                 "<b>Names:</b> every person, organisation, place, nationality, facility, product, event or law name (found by spaCy) must appear somewhere in <b>any</b> kept chunk. A name that does not is marked bright red."]))
    s.append(h2("11.2 Exactly how matching works"))
    s.append(table(["Step", "Rule"], [
        ["Which number counts", "Standalone numbers like 345, 22,500, 7.5, 2023. <b>Skipped:</b> numbers inside words (COP28), citations [1], and numbers followed by letters (2030s, 5th); number words (&#8220;three&#8221;)"],
        ["Normalisation", "Only commas are removed: 22,500 = 22500. <b>No</b> unit or magnitude conversion (1.6 thousand, 0.345 TW) and no rounding (42.5 &#8800; 42.50)"],
        ["Which passage (the &#8220;reference chunk&#8221;)", "The kept chunk the sentence is <i>about</i>: highest embedding similarity to the number-free sentence, plus a small boost for how often the sentence's content words occur in the chunk. Chosen by meaning, <b>not</b> by NLI, because NLI can prefer a loosely related chunk (the 2030 example)"],
        ["Rescue rule", "A number found only in the NLI-preferred chunk is still accepted if NLI entailment there is at least 0.9"],
        ["Names", "Hedge words (about, roughly, the...) are stripped; the name must appear (lower-cased substring) in any kept chunk"],
        ["Skipped entirely", "Filler sentences; dates and quantities as entities (their numbers are covered by the number rule); relative dates like &#8220;the end of that year&#8221;"]], [26, 74]))
    s.append(p("A sentence with any wrong part becomes <b>flagged</b>, is shown amber with the wrong words in bright red, and its displayed source becomes the reference chunk."))
    s.append(box("ex", "Example (faulty mode)", "Original: &#8220;In 2023, about 345 gigawatts of new solar capacity was added globally.&#8221;<br>Corrupted: &#8220;<span class='a'>In <span class='red'>2013</span>, about <span class='red'>690</span> gigawatts of new solar capacity was added globally.</span>&#8221;<br>NLI said <i>entailment</i> (0.80), but neither 2013 nor 690 is in the Solar chunk, so both are marked."))
    s.append(h2("11.3 Its limits, and why it is still useful"))
    s.append(p("The check confirms that a number <b>exists</b> in the source passage, not that it is attached to the right fact. The two checks cover each other:"))
    s.append(table(["Check", "Good at", "Weak at"], [["NLI", "Meaning of the whole sentence", "Exact numbers"], ["Number and name checker", "Exact numbers and names", "Context and meaning; units and rounding"]], [24, 38, 38]))
    s.append(box("care", "Be careful", "The ablation (section 18.3) shows the span check alone has precision 0.57 but low recall (0.22), and adding it to NLI did <b>not</b> raise overall F1 on the labelled set, which consists mostly of invented details and commentary rather than changed numbers. Its value is explanation (marking the exact words) and catching changed numbers, which the adversarial/faulty tests show."))
    s.append(box("say", "What to say", "&#8220;NLI judges the meaning of the sentence; the number checker makes sure the specific numbers and names exist in the sources. Each covers the other's weakness, and the second one tells the user exactly which words to distrust.&#8221;"))

    s.append(h1("12. Step 6: Attention attribution &#8212; where was the checker looking?"))
    s.append(p("For sentences that <b>NLI itself rejected</b> (best entailment below 0.5), the system shows the <b>top 5 words in the best source chunk</b> that the checker paid most attention to. (Sentences flagged only by the number check do not get this view, because NLI accepted them.)"))
    s.append(fig(attn_fig(), "Figure 5. Illustrative attention view: darker source words received more attention from the flagged sentence."))
    s.append(h2("12.1 How it is computed"))
    s.append(ul(["Run DeBERTa on (best chunk + flagged sentence) and ask it to also return its attention weights.",
                 "Average the weights over the <b>last 4 layers</b> and <b>all 12 heads</b>.",
                 "Keep only attention <b>from the sentence's tokens to the chunk's tokens</b>, and add it up for each chunk token.",
                 "Merge word pieces back into whole words (&#8220;giga&#8221; + &#8220;watts&#8221; &#8594; &#8220;gigawatts&#8221;).",
                 "Remove special tokens, punctuation and common words (&#8220;the&#8221;, &#8220;of&#8221;).",
                 "Return the top 5 words, with weights scaled to add up to 1 over those five."], ordered=True))
    s.append(h2("12.2 The correct way to describe it"))
    s.append(box("care", "Be careful", "Attention shows <b>where the checker focused</b>. It does <b>not</b> prove what <b>caused</b> its decision; research shows attention weights and the true reasons don't always match. Say &#8220;the source words the checker attended to most&#8221;. DeBERTa uses <b>self-attention</b> over the joined pair, not cross-attention."))
    s.append(h2("12.3 The masking check: do those words really matter?"))
    s.append(p("To test the attention words instead of just trusting them, the system runs a <b>mask-and-rerun check</b> on every NLI-rejected sentence: it <b>deletes the top 5 words</b> from the source chunk, runs NLI again, and measures how far the three label probabilities move (total variation distance, 0 = no change, 1 = completely different). As a control, it does the same after deleting <b>5 random content words</b> (averaged over 5 draws). The attention words are <b>supported</b> when their shift beats the random shift by at least 0.05. The dashboard shows this line in the sentence's expander."))
    s.append(box("ex", "Example", "Sentence: &#8220;The dam has an installed capacity of 5,000 megawatts.&#8221; Top attention words include <b>22,500</b>, the true figure in the source. Deleting them moves the probabilities by <b>0.85</b>; deleting random words moves them by <b>0.01</b>: the attention words clearly mattered."))
    s.append(box("care", "Result on the labelled set", "For the 69 sentences NLI flagged, deleting the top words moved the probabilities by <b>0.24</b> on average versus <b>0.11</b> for random words, but the top words beat the random control in only <b>35%</b> of sentences (24 of 69). So attention words are a useful hint on average, <b>not a reliable per-sentence explanation</b>. Say this plainly."))

    s.append(h1("13. Step 7: The hierarchical trust score"))
    s.append(p("The trust score summarises how reliable an answer is as one number from 0 to 1. It combines four signals:"))
    s.append(table(["Component", "Weight", "Question it answers", "Level"], [
        ["Sentence", "0.35", "Do the sources support the claims? (from NLI and the wrong-part check)", "Sentence"],
        ["Entity", "0.25", "Do the names, numbers and dates in the answer appear in the sources?", "Entity"],
        ["Document", "0.20", "How trustworthy are the documents behind the supported sentences?", "Document"],
        ["Token probability", "0.20", "How confident was the generator while writing?", "Token"]], [18, 10, 56, 16]))
    s.append("<pre>trust = 0.35 x sentence + 0.25 x entity + 0.20 x document + 0.20 x token probability</pre>")
    s.append(h2("13.1 Sentence score"))
    s.append(p("Over the <b>claim sentences</b> (filler excluded): the average best-entailment probability &#215; (1 &#8722; fraction of sentences flagged). A flagged sentence lowers both factors. If there are no claim sentences the score is 0."))
    s.append(h2("13.2 Entity score"))
    s.append(p("spaCy finds every entity in the whole answer (names, numbers, dates, percentages...). Relative dates (&#8220;the end of that year&#8221;) are skipped, hedge words (&#8220;approximately&#8221;, &#8220;roughly&#8221;, &#8220;the&#8221;) are removed, duplicates counted once. An entity is <b>supported</b> if its lower-cased text appears (as a substring) in the kept chunks. The score is the fraction supported; <b>no entities means 1.0</b>."))
    s.append(h2("13.3 Document score"))
    s.append(p("The average authority of the documents behind the non-flagged claim sentences; 0 if none. <b>All six documents currently have authority 0.8</b>, so this component only varies with how many sentences are supported, and carries no extra information about source quality (the AUROC is 0.46, section 18.3)."))
    s.append(h2("13.4 Token-probability score"))
    s.append(p("<code>exp(average logprob)</code> over the generator's words: its typical per-word probability. It describes the <i>original</i> answer, so in faulty mode it is dropped."))
    s.append(h2("13.5 A worked example"))
    s.append(fig(trust_fig(), "Figure 6. Component scores for an example answer. Weighted sum: 0.35x0.90 + 0.25x1.00 + 0.20x0.80 + 0.20x0.94 = 0.913."))
    s.append(h2("13.6 When a component is missing: renormalisation"))
    s.append(p("If a component has no data, its weight is removed and the others are scaled up to add to 1. Without token probability the other weights add to 0.80, so they become 0.35/0.80 = 0.4375, 0.25/0.80 = 0.3125 and 0.20/0.80 = 0.25."))
    s.append(fig(renorm_fig(), "Figure 7. Renormalisation keeps the total weight at 1."))
    s.append(box("ex", "Example (faulty mode, no token probability)", "A corrupted answer had sentence 0.76, entity 0.50 and document 0.80: 0.4375&#215;0.76 + 0.3125&#215;0.50 + 0.25&#215;0.80 = 0.333 + 0.156 + 0.200 &#8776; <b>0.69</b>."))
    s.append(h2("13.7 Why &#8220;hierarchical&#8221;, and why combine four signals?"))
    s.append(p("Each signal works at a different level: tokens (words), entities (specific facts), sentences (whole claims) and documents (source quality). Each alone has blind spots: NLI alone gave a wrong number 0.97 entailment; entity matching alone cannot tell whether a number is attached to the right fact; document authority says nothing about whether a claim matches."))
    s.append(box("key", "Key idea", "<b>Confidence is not correctness</b>, but it is informative. On our labelled answers the generator's token probability separated good from bad answers best (AUROC 0.98, section 18.3). Caveat: loose-prompt answers are both less confident <i>and</i> more often unsupported, so part of that signal is the prompt mode."))
    s.append(h2("13.8 How the weights were chosen"))
    s.append(p("The weights were <b>set by design</b>, not learned. Sentence support gets the most weight as the most direct evidence. The ablation shows token probability would deserve more weight than 0.20 on this data, and the document component deserves less; learning the weights from labelled data is future work."))
    s.append(box("say", "What to say", "&#8220;The weights are designed, not learned. Our own analysis suggests token probability is stronger than its weight implies, and the document component is uninformative while all authorities are equal. I would learn the weights from more labelled data.&#8221;"))
    s.append(h2("13.9 Known edge cases"))
    s.append(ul(["<b>One claim, flagged:</b> the sentence and document scores become 0 and the trust score drops to about 0 (we saw 0.00 for a faulty-mode answer with one claim). A smoother formula would help.",
                 "<b>Only filler:</b> sentence and document scores are 0, so trust is low."]))

    s.append(h1("14. Step 8: Saving and showing the result"))
    s.append(p("Everything is saved in SQLite: the question, the answer, retrieved and kept chunks, each sentence's label, confidence, best chunk, attention words and wrong parts, the entities, and all trust components. <code>GET /response/{id}</code> fetches it again."))
    s.append(h2("14.1 The dashboard"))
    s.append(table(["Tab", "What it shows"], [
        ["Query", "Input box and a <b>Faulty mode</b> checkbox. After a question: model used, chunks retrieved and used, the coloured answer, in faulty mode a &#8220;Caught / Missed&#8221; line per injected error, an expander per flagged sentence (NLI label, wrong parts, source chunk, attention words and the masking-check line), the entity list with &#10003;/&#10007;, the trust score and a bar chart of its components"],
        ["Robustness", "Adversarial results for NLI alone and for the full pipeline: detection rate per attack type and the false-positive rate; a button to run the suite"],
        ["Evaluation", "Evaluation results: precision, hallucination rate, average prune ratio, average trust"]], [18, 82]))
    s.append(table(["What you see", "Meaning"], [
        ["<span class='g'>green sentence</span>", "Supported: best entailment at least 0.5, label entailment"],
        ["<span class='a'>amber sentence</span>", "Neutral, or flagged because it contains wrong parts"],
        ["<span class='red'>bright red words</span>", "A number or name the sources do not contain (hover shows the reason)"],
        ["<span class='lr'>light red sentence</span>", "Rejected by NLI (best entailment below 0.5)"],
        ["<span class='gr'>grey italic sentence</span>", "Filler: no claim to check"]], [32, 68]))
    return "".join(s)

from guide_lib import *


def adv_design_fig():
    G, R, B, W = "#dcfce7", "#fee2e2", "#dbeafe", "#ffffff"
    boxes = [
        (8, 10, 130, 54, "Clean chunk", "'...Solar added 345 GW\nin 2023...'", G),
        (8, 96, 130, 54, "Pick a true sentence", "'Solar added 345 GW\nin 2023.'", W),
        (170, 96, 130, 54, "Break it", "'Solar added 345 GW\nin 2017.'", R),
        (340, 50, 110, 60, "NLI", "premise = CLEAN chunk\nhypothesis = broken", B),
        (480, 18, 80, 40, "Detected", "not entailment", G),
        (480, 98, 80, 40, "Missed", "entailment", R),
    ]
    arrows = [(73, 64, 73, 96), (138, 123, 170, 123), (300, 123, 340, 95), (138, 37, 340, 70), (450, 70, 480, 40), (450, 92, 480, 115)]
    return svg_boxes(boxes, arrows, 568, 170, "Control: the same sentence UNCHANGED should get entailment. If not, that is a false positive.")


def results_fig():
    items = [("Negation", 1.0, "#16a34a"), ("Stale\nfact", 1.0, "#16a34a"), ("Noise", 1.0, "#16a34a"), ("LLM\ncontradiction", 0.208, "#d97706"), ("False positives\n(control)", 0.167, "#dc2626")]
    return svg_bars(items, "Adversarial results (NLI alone, 120 tests): detection rate (higher = better), false-positive rate (lower = better)", 540, 220, unit_pct=True)


def prec_fig():
    return svg_bars([("Precision@6\n(before pruning)", 0.33, "#9ca3af"), ("Precision\n(after pruning)", 0.74, "#2563eb")], "Retrieval precision before and after pruning (ceiling before pruning = 2/6)", 360, 200)


def f1_fig():
    items = [("NLI only", 0.74, "#2563eb"), ("Span\nonly", 0.32, "#9ca3af"), ("NLI +\nspan", 0.73, "#2563eb"), ("NLI +\nclaims", 0.73, "#2563eb"), ("Full\n(production)", 0.71, "#16a34a"), ("Full, no\npruning", 0.65, "#dc2626")]
    return svg_bars(items, "F1 on the labelled set (176 sentences; positive = unsupported)", 540, 210, label_size=8.5)


def auc_fig():
    items = [("Token\nprobability", 0.98, "#7c3aed"), ("Sentence", 0.88, "#2563eb"), ("Final\ntrust", 0.89, "#16a34a"), ("Entity", 0.65, "#d97706"), ("Document", 0.46, "#9ca3af")]
    return svg_bars(items, "Answer-level AUROC: how well each part separates bad answers from good ones (0.5 = chance)", 520, 210)


def build():
    s = []
    s.append(part_page("Part C", "Testing the system and understanding the results"))
    s.append(h1("15. The adversarial test: does the checker actually work?", nobreak=True))
    s.append(p("To know whether a checker is good, you must give it sentences <b>you already know are wrong</b> and see whether it catches them. That is what the adversarial test does (&#8220;adversarial&#8221; = deliberately trying to break the system)."))
    s.append(h2("15.1 Normal use vs testing"))
    s.append(table(["", "Normal use (every question)", "Adversarial test (separate script)"], [
        ["Input", "The generator's real answer", "Sentences we broke on purpose"],
        ["Do we know the right answer?", "No: that is what we're trying to find out", "Yes: we know exactly which sentences are wrong"],
        ["Purpose", "Find hallucinations in answers", "Measure how good the checker is"]], [26, 37, 37]))
    s.append(box("key", "Key idea", "In normal use, nothing fake is added. The adversarial test is a separate measurement of the checker itself, like giving a teacher an answer sheet with known mistakes to see how many they find."))
    s.append(h2("15.2 How it works"))
    s.append(fig(adv_design_fig(), "Figure 8. The adversarial test design."))
    s.append(ul(["<b>Which sentences:</b> chunks are shuffled with a fixed seed (42) and up to 20 are taken; the corpus has only 12, so all 12 are used. From each chunk up to 2 sentences are picked (sentences with a number first), each at least 5 words and containing a number or a verb: 24 sentences.",
                 "<b>Corrupt it</b> with each attack below; the unchanged sentence is the control.",
                 "<b>Run NLI</b> with the <b>original clean chunk</b> as premise and the corrupted sentence as hypothesis.",
                 "<b>Detected</b> means NLI's top label is anything other than entailment. Each run replaces the previous results: 24 sentences &#215; 5 variants = 120 tests."], ordered=True))
    s.append(h2("15.3 The attack types"))
    s.append(table(["Attack", "Exact rule", "Example"], [
        ["Negation", "If &#8220;not&#8221; or &#8220;never&#8221; is present, remove the first; otherwise insert &#8220;not&#8221; after the first is/was/are/were/has/have/had/can/will/does/did; otherwise prefix &#8220;It is not true that&#8221;", "&#8220;Solar added 345 GW.&#8221; &#8594; &#8220;Solar did not add 345 GW.&#8221;"],
        ["Stale fact", "Every year (1500-2099) shifts by the same random 3 to 10 years (earlier or later); every other number is multiplied by a random choice of 0.5, 0.6, 0.75, 1.5 or 2", "&#8220;...in 2023&#8221; &#8594; &#8220;...in 2017&#8221;; &#8220;345 GW&#8221; &#8594; &#8220;172 GW&#8221;"],
        ["Noise", "Half the time (if the sentence has a name and another is available) swap one name for a random name from the corpus; otherwise append one of four unrelated clauses", "&#8220;...in China&#8221; &#8594; &#8220;...in Brazil&#8221;"],
        ["LLM contradiction", "Ask qwen2.5:3b (temperature 0): &#8220;Rewrite this sentence so it remains fluent and plausible but states something factually different. Output only the sentence.&#8221;; results are cached", "A fluent sentence with one detail changed"]], [17, 55, 28]))
    s.append(p("The first three are <b>rule-based</b>; the fourth uses the generator for subtler attacks. The &#8220;automated adversarial agent&#8221; in the proposal is this script: a program with a policy for choosing sentences and attacks, not a trained reinforcement-learning agent."))
    s.append(h2("15.4 The control set: why it is essential"))
    s.append(p("The unchanged true sentences are also checked; they should pass. When the checker flags an unchanged sentence, that is a <b>false positive</b>."))
    s.append(box("key", "Key idea", "Without a control set, a checker that flags <i>everything</i> would score 100% detection and look perfect. The control set shows whether the checker is accurate or just suspicious of everything."))
    s.append(h2("15.5 Why not corrupt the documents and ask normal questions?"))
    s.append(p("That measures nothing: the generator would faithfully repeat the corrupted text, and NLI would correctly say the (corrupted) source supports it. Comparing corrupted sentences against the <b>clean</b> source is what makes the test meaningful."))
    s.append(h2("15.6 Interpreting the results"))
    s.append(p("<b>Important:</b> this test runs <b>NLI alone</b>. It does not use claim splitting or the number and name check, so it measures the base checker, not the whole pipeline (the labelled-set evaluation in section 18.3 measures the full pipeline)."))
    s.append(p("LLM-written contradictions were detected only <b>21%</b> of the time. Two explanations, and both may be true:"))
    s.append(ul(["<b>The attacks are weak:</b> the small 3B model often wrote near-paraphrases (&#8220;approximately&#8221;, a small year change) instead of real contradictions; a &#8220;contradiction&#8221; that does not contradict is correctly passed by the checker.",
                 "<b>The checker is weak on subtle changes:</b> fluent rewrites with a small shift in meaning are genuinely harder for NLI than obvious negations."]))
    s.append(box("care", "Be careful", "Until someone reads the attack sentences and marks which are real contradictions, you cannot say how much of the 21% is due to each cause. Say: &#8220;Detection on LLM-written attacks was 21%; many rewrites were near-paraphrases, so this test is weak.&#8221;"))

    s.append(h1("16. Test modes: faulty mode and loose mode"))
    s.append(p("The checker's value is only visible when an answer contains errors. Two clearly labelled test modes create errors on purpose:"))
    s.append(table(["Mode", "Exactly what it does", "How to switch it on"], [
        ["<b>Faulty mode</b>", "After the generator answers, corrupts <b>1 sentence (2 if the answer has 4 or more)</b>: sentences with numbers first (all their numbers are changed, using the stale-fact rule), otherwise a negation. Then the normal checks run. The response is tagged, its token probability is dropped (the logprobs describe the original text), and it is excluded from evaluation", "<b>Dashboard:</b> the &#8220;Faulty mode&#8221; checkbox on the Query tab (API: <code>inject_faults: true</code>)"],
        ["<b>Loose mode</b>", "Replaces the &#8220;answer only from the context&#8221; instruction with one that invites the model to add outside background, figures and examples (verbatim in section 9.1)", "<b>Environment variable</b> <code>PROMPT_MODE=loose</code> before starting the backend. There is no dashboard switch, and the mode is not stored with the answer"]], [16, 52, 32]))
    s.append(box("key", "Why test modes are legitimate", "This is not cheating. The project's purpose is <i>detecting</i> hallucinations, and a detector can only be tested on answers that contain some. The modes are labelled openly, and normal mode stays strict, as real RAG systems are. In the report: &#8220;We used a loose prompt to create more hallucinations for testing.&#8221;"))
    s.append(p("In loose mode many red sentences may be <i>true</i> facts from the model's general knowledge, or plain commentary. By the project's definition (section 3) they are still unsupported, so flagging them is correct."))

    s.append(h1("17. How the system is measured: the metrics"))
    s.append(h2("17.1 The evaluation set"))
    s.append(p("<b>22 questions</b> about the documents, each with a reference answer and the list of documents that contain it. The evaluation (<code>POST /evaluate</code> or <code>scripts/run_eval.py</code>) runs them all through the system. Cheaper modes: <code>retrieval_only</code> (retrieval and pruning metrics, no generation), <code>rescore</code> (re-check the stored answers, no new generation) and <code>limit</code> (first N questions)."))
    s.append(h2("17.2 Metric definitions"))
    s.append(table(["Metric", "Plain meaning", "Better is"], [
        ["Precision@6", "Of the 6 retrieved chunks, the fraction from a relevant document", "Higher"],
        ["Precision after pruning", "Of the chunks kept after pruning, the fraction that are relevant", "Higher"],
        ["Prune ratio", "The fraction of retrieved chunks that pruning removed", "&#8212;"],
        ["Hallucination rate", "Of all <b>claim sentences</b> (filler excluded), the fraction flagged", "Lower (for the generator)"],
        ["Average trust", "Mean trust score across the answers", "Higher"],
        ["Detection rate", "Of the corrupted sentences of one attack type, the fraction the checker caught", "Higher"],
        ["False-positive rate", "Of unchanged control sentences, the fraction wrongly flagged", "Lower"],
        ["Precision / recall / F1", "On the labelled set, treating &#8220;unsupported&#8221; as the target: precision = of flagged sentences, how many were really unsupported; recall = of the really unsupported, how many were flagged; F1 = their balance", "Higher"],
        ["95% confidence interval", "A range for F1 from 1000 random re-samplings (bootstrap): how much the number could move by luck", "&#8212;"],
        ["AUROC", "Chance that a randomly chosen bad answer scores worse than a randomly chosen good one (0.5 = guessing, 1 = perfect)", "Higher"],
        ["Latency", "Time to generate one answer (about 2.5 s); a full query is about 8 s", "Lower"]], [22, 58, 20]))
    s.append(h2("17.3 Why precision@6 can never exceed 0.33 here"))
    s.append(p("Each document produces only 2 chunks. If the answer is in one document, at most 2 of the 6 retrieved chunks can come from it: the best possible precision is 2/6 = 0.33. The measured 0.33 is a <b>ceiling of the setup, not a failure of retrieval</b>. Recall@k and hit rate would suit this setup better."))

    s.append(h1("18. The results"))
    s.append(box("care", "About the numbers", "All figures below were measured on 2026-10-11 with the code described in this guide, except where marked. The 22-question evaluation (8.7% hallucination rate) was run <b>before claim splitting was added</b> and has not been re-run with it; re-run <code>python scripts/run_eval.py</code> if you want the number for the final code."))
    s.append(h2("18.1 Results table"))
    s.append(table(["What", "Result", "What it means"], [
        ["Precision@6 (before pruning)", "0.33", "At the ceiling of the setup (2 of 6)"],
        ["Precision after pruning", "0.74", "About 3 in 4 kept chunks are relevant"],
        ["Chunks removed by pruning", "62%", "6 chunks &#8594; about 2"],
        ["Hallucination rate (22 questions, strict)", "8.7% (2 of 23 claim sentences)", "Very few flagged; the sample is small, say &#8220;2 of 23&#8221;"],
        ["Average trust", "0.90", "Most answers well supported"],
        ["Average token probability", "0.94", "The generator is confident"],
        ["Time per answer", "about 2.5 s (generation); about 8 s end to end", "Fast enough for live use"],
        ["Negation / stale fact / noise detected (NLI alone)", "100% / 100% / 100%", "Clear errors are caught"],
        ["LLM-written contradictions detected", "21%", "Weak test: many near-paraphrases"],
        ["False-positive rate (control, NLI alone)", "16.7%", "About 1 in 6 true sentences wrongly flagged"],
        ["Automated tests", "43 passing", "pytest, run on a temporary database"]], [34, 30, 36]))
    s.append(fig(prec_fig(), "Figure 9. Retrieval precision before and after pruning."))
    s.append(fig(results_fig(), "Figure 10. Adversarial results."))
    s.append(h2("18.2 What the results show"))
    s.append(ul(["<b>Pruning works:</b> it removes most chunks while sharply raising the share of relevant ones, and it improves the checker (18.3).",
                 "<b>The checker catches clear errors reliably:</b> negations, changed numbers and noise were all detected.",
                 "<b>Subtle errors are harder:</b> LLM-written rewrites (weak test) and contradictions in the labelled set (recall 0.44).",
                 "<b>The checker is somewhat over-cautious:</b> it flags about 1 in 6 unchanged true sentences (adversarial control) and about 1 in 5 supported sentences in the labelled set (false-positive rate 0.23).",
                 "<b>The local generator is mostly faithful in strict mode</b> (2 of 23 sentences flagged), which is why the test modes are needed for demonstrations."]))
    s.append(h2("18.3 Labelled-set evaluation and ablation"))
    s.append(p("To measure the full pipeline on <i>natural</i> answers, 92 real answers from the generator (strict and loose prompt) were split into 215 sentences, and each sentence was labelled <b>supported</b> (121), <b>unsupported</b> (55), <b>filler</b> (30, excluded) or <b>ambiguous</b> (9, excluded). The 176 usable sentences were run through variants of the pipeline."))
    s.append(box("care", "Be careful: who made the labels", "The labels were made by an <b>AI assistant</b> reading each sentence against the source documents, <b>not by a human</b>. Spot-check <code>data/eval/labeled_sentences.json</code> before presenting these numbers as ground truth, and say so if asked. Sentences from the same answer are correlated, so the confidence intervals are optimistic."))
    s.append(table(["Variant", "Precision", "Recall", "F1 (95% CI)", "False-positive rate"], [
        ["NLI only (sentence level)", "0.68", "0.82", "<b>0.74</b> (0.65-0.83)", "0.17"],
        ["Number/name check only", "0.57", "0.22", "0.32 (0.17-0.46)", "0.07"],
        ["NLI + number/name check", "0.65", "0.84", "0.73 (0.64-0.81)", "0.21"],
        ["NLI + claim splitting", "0.65", "0.82", "0.73 (0.63-0.81)", "0.20"],
        ["Full pipeline (claims + check + filler) &#8212; the app", "0.62", "0.84", "0.71 (0.62-0.80)", "0.23"],
        ["Full pipeline <b>without</b> MI pruning (6 chunks)", "0.63", "0.67", "0.65 (0.54-0.74)", "0.18"]], [36, 12, 12, 24, 16]))
    s.append(fig(f1_fig(), "Figure 11. F1 by pipeline variant."))
    s.append(ul(["<b>MI pruning clearly helps:</b> recall 0.84 vs 0.67 (F1 0.71 vs 0.65). Giving NLI all 6 chunks dilutes the evidence.",
                 "<b>Claim splitting and the number/name check did not measurably improve F1.</b> All NLI variants lie within each other's intervals (0.71-0.74). They trade a little precision for a little recall. They remain useful for explanation (marking the exact wrong words) and for catching changed numbers, which this labelled set (mostly invented details and commentary) does not reward. Say this honestly.",
                 "<b>Entailment threshold:</b> from 0.3 to 0.9, F1 stays at 0.71-0.72 (precision 0.63 &#8594; 0.57, recall 0.82 &#8594; 0.95), so 0.5 is not critical; higher thresholds trade precision for recall.",
                 "<b>Recall by error type</b> (full pipeline): unsupported commentary 0.93 (27 sentences), invented specifics 0.89 (19), <b>contradictions or miscopied source facts 0.44 (only 9)</b>: the hardest case."]))
    s.append(p("<b>Answer-level trust analysis</b> (83 answers, 33 containing an unsupported sentence): AUROC of each trust component for spotting bad answers:"))
    s.append(fig(auc_fig(), "Figure 12. AUROC per trust component."))
    s.append(box("care", "Be careful", "Token probability (0.98) looks best, but loose-prompt answers are both less confident and more often unsupported, so part of that is the prompt mode. The document component (0.46, chance level) is uninformative because all documents have the same authority."))
    s.append(h2("18.4 The pronoun problem in detail"))
    s.append(p("Many false positives come from sentences that start with a pronoun. Example: the chunk says &#8220;The Three Gorges Dam ... It was fully commissioned in 2012&#8221; and the sentence &#8220;It was completed in 2012&#8221; is checked on its own. Without the name it refers to, the model is unsure what is claimed and gives low entailment. Claim splitting only helps <i>inside</i> a sentence. A &#8220;pronoun retry&#8221; (re-checking a flagged pronoun sentence together with the previous sentence) was <b>not built</b>; checking against smaller pieces of each chunk is the other fix."))
    s.append(h2("18.5 Things to double-check before presenting"))
    s.append(ul(["The numbers in the tables match the latest <code>README.md</code> and <code>FACTS.md</code>. (The adversarial false-positive rate was 12.5% on an earlier run and 16.7% on the latest; sampling changed slightly when a year-handling bug was fixed.)",
                 "The hallucination rate of 8.7% is 2 of 23 <b>claim sentences</b> (filler excluded) over 22 answers, and was measured before claim splitting.",
                 "The labelled-set numbers rest on <b>AI-made labels</b> (18.3).",
                 "Faulty mode works in the dashboard; loose mode needs the <code>PROMPT_MODE=loose</code> environment variable.",
                 "The pronoun retry is <b>not</b> implemented; the threshold sweep <b>is</b> done (18.3)."]))
    return "".join(s)

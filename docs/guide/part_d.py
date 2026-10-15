from guide_lib import *


def qa(q, a):
    return f'<div class="box say"><div class="t">Q: {q}</div><p>{a}</p></div>'


def build():
    s = []
    s.append(part_page("Part D", "Defending the project: limitations, decisions and questions"))
    s.append(h1("19. Limitations", nobreak=True))
    s.append(p("Every system has limits. Naming them before a professor does shows real understanding, and professors reward it."))
    s.append(table(["Limitation", "Explanation", "What could be done"], [
        ["Supported, not true", "If a source document is wrong, a sentence copying it is marked supported", "Combine with trusted, curated sources; inherent to the faithfulness approach"],
        ["Weak on exact numbers (NLI)", "NLI gave a wrong figure 0.97 entailment; a policy chunk 'entailed' a wrong solar figure", "Number/name check (added); units, rounding and magnitude handling; a numeric reasoning step"],
        ["Contradictions are the weakest case", "Recall 0.44 on 9 labelled examples of facts miscopied from the source", "A second verifier; sentence-level evidence; more labelled data"],
        ["False positives", "Adversarial control 16.7%; labelled set 23% of supported sentences flagged. Causes: pronouns, long chunks, number formats", "Check against 2-3 sentence snippets; pronoun retry"],
        ["Claim splitting and span check show no F1 gain", "On the labelled set all NLI variants are within the same interval (0.71-0.74)", "Evaluate on a set with more changed-number errors"],
        ["Labels made by an AI assistant", "The labelled set was not checked by a human; same-answer sentences are correlated", "Human re-labelling of a sample; more annotators"],
        ["Trust weights designed, not learned", "Token probability looks stronger and documents weaker than their weights imply", "Learn weights from labelled answers"],
        ["Document authority is constant", "All six documents have authority 0.8, so the document score carries no information", "Give documents real, differing authority values"],
        ["Single-claim answers", "One flagged claim drives trust to about 0", "A smoother formula"],
        ["Small, self-written corpus", "6 documents, 12 chunks, 22 questions; results indicative only", "A larger public dataset"],
        ["Weak attack generator", "The 3B model writes near-paraphrases instead of contradictions", "Manual review of attacks; a larger model for attacks"],
        ["Attack set is small and simple", "24 sentences, 12 chunks; three attack types are rule-based and easy; LLM rewrites are weak", "A larger attack set; manual review of LLM attacks"],
        ["Number/name matching is simple", "Substring matching, no units or rounding; a number can match in the wrong context within the reference chunk", "Structured claim extraction (who, what, how much)"],
        ["Attention is a weak per-sentence explanation", "Masking the top words beat the random control in only 35% of 69 flagged sentences (average shift 0.24 vs 0.11)", "Other attribution methods; show the masking result to users"],
        ["Loose mode is an environment variable", "Not selectable in the dashboard and not stored with the answer", "Add a toggle and record the mode"]], [24, 40, 36]))

    s.append(h1("20. Design decisions and the alternatives considered"))
    s.append(p("Professors often ask &#8220;why did you choose X instead of Y?&#8221;. Each decision has a short, honest reason."))
    s.append(table(["Decision", "Alternative", "Why this choice"], [
        ["NLI classifier as the checker", "Ask another LLM to judge", "Runs locally, fast, same result every time, gives probabilities for a threshold; an LLM judge is slower, less predictable and can hallucinate itself"],
        ["Pretrained DeBERTa", "Fine-tune the NLI model", "Fine-tuning needs a large labelled dataset, a GPU and a strict train/test split; the proposal specified a pretrained model. (Nothing in this project is fine-tuned.)"],
        ["Local model via Ollama (Gemini removed)", "Gemini or another online API", "No daily limits, returns logprobs, private, reproducible"],
        ["PMI formula for pruning", "A trained mutual-information estimator", "No training, fast, every score is explainable; the ablation shows pruning helps"],
        ["Sentence-level checking, split into claims", "Check the whole answer at once", "Finds exactly which claim is wrong (claim splitting itself gave no measurable F1 gain)"],
        ["Number/name check as a second verifier", "Trust NLI alone", "NLI is blind to exact numbers; the check also gives the exact words to mark"],
        ["Reference chunk chosen by meaning", "Use the chunk NLI likes best", "NLI can prefer a loosely related chunk that happens to contain the same year"],
        ["Filler sentences excluded", "Score every sentence", "&#8220;This is stated in the context&#8221; makes no claim and would inflate or distort the scores"],
        ["Best chunk per sentence", "Require support from all chunks", "A fact only needs to be in one source"],
        ["Rule-based + LLM attacks", "A trained reinforcement-learning attacker", "Feasible in the time available; transparent rules"],
        ["ChromaDB + SQLite", "Larger databases (Postgres)", "Zero setup, file-based, enough for this scale"],
        ["Streamlit dashboard", "A custom web app (e.g. React)", "Charts and highlighted text with very little code; the plan lists React as not used"],
        ["Temperature 0, cached answers", "Random sampling", "Reproducible answers for evaluation"]], [26, 24, 50]))

    s.append(h1("21. Future work"))
    s.append(ul(["<b>Sentence-level evidence:</b> check each claim against the 2-3 most relevant source sentences instead of a whole 250-word chunk (likely the biggest single improvement for precision and for explanation).",
                 "<b>Human-verified labels:</b> have a person re-label at least a sample of the labelled set, and extend it.",
                 "<b>Learn the trust weights</b> from labelled answers; give documents real authority values; smooth the single-claim case.",
                 "<b>A second verifier</b> or numeric reasoning (units, magnitude, rounding) to raise recall on contradictions (0.44).",
                 "<b>Pronoun retry:</b> re-check flagged pronoun sentences with the previous sentence attached.",
                 "<b>Citations:</b> make the generator cite which chunk supports each sentence, then verify the citation.",
                 "<b>Manual review of LLM attacks</b> and a larger attack set.",
                 "<b>Retrieval metrics</b> that suit 2-chunk documents (recall@k, hit rate); a chunk-size experiment; a larger corpus and more questions.",
                 "<b>Better attribution:</b> the masking check shows attention words are not reliable per sentence; try other attribution methods; a larger NLI model (more accurate, slower).",
                 "<b>Done already (no longer future work):</b> threshold sweep, labelled-set precision/recall/F1 (AI-labelled), ablation of each component, the attention masking check, and scoring the attacks through the full pipeline."]))

    s.append(h1("22. Question bank: likely questions and answers"))
    s.append(p("Every answer follows the same pattern: <b>what we did, why, and its limit</b>."))
    s.append(h2("22.1 The project as a whole"))
    s.append(qa("What problem does your project solve?", "AI models can state unsupported things confidently. HalluciGuard checks each sentence of a RAG answer against the sources, marks the unsupported parts, and gives a trust score."))
    s.append(qa("What is new compared to a normal RAG system?", "Normal RAG stops after generating. We add sentence- and claim-level verification with NLI, a number and name checker that marks the exact wrong words, attention-based explanations, a hierarchical trust score, MI pruning, and an evaluation of the checker (adversarial test, labelled set and ablation)."))
    s.append(qa("Do you detect false information?", "No: we detect claims not supported by the sources. A true fact not in the documents is still unverified, and a wrong fact copied from a wrong document is marked supported."))
    s.append(qa("Did you train or fine-tune any model?", "No. All models are used pretrained: DeBERTa NLI, MiniLM embeddings, spaCy and the qwen2.5:3b generator. The weights of the trust score and the thresholds are set by design."))
    s.append(h2("22.2 Retrieval and pruning"))
    s.append(qa("Why split documents into chunks?", "Smaller pieces are easier to search precisely, and the checker can only read about 512 tokens at once."))
    s.append(qa("Why embeddings and not keyword search?", "Embeddings match meaning, so different words with the same meaning are still found."))
    s.append(qa("Why is precision only 0.33?", "Each document has about 2 chunks, so at most 2 of 6 retrieved chunks can be relevant: a ceiling of the setup. After pruning, precision is 0.74."))
    s.append(qa("What does PMI measure, and does pruning really help?", "Whether the question's words are more common in a chunk than across all retrieved chunks. Yes: in the ablation, feeding NLI all 6 chunks lowered recall from 0.84 to 0.67 and F1 from 0.71 to 0.65."))
    s.append(qa("Why keep at least 2 chunks?", "A safety net so pruning never leaves the generator without enough evidence."))
    s.append(h2("22.3 Verification"))
    s.append(qa("Why NLI and not an LLM judge?", "It is local, fast, gives the same result every time, and outputs probabilities we can threshold. An LLM judge is slower and can hallucinate too."))
    s.append(qa("Why is the chunk the premise?", "We are asking whether the source supports the answer, so the source is what we treat as known."))
    s.append(qa("Why 0.5? Did you tune it?", "It is the midpoint where support becomes more likely than not, a design choice. We swept it from 0.3 to 0.9 on the labelled set: F1 stayed at 0.71-0.72, so the result is not sensitive to it; higher thresholds trade precision for recall."))
    s.append(qa("Why sentence by sentence, and what are claims?", "Checking the whole answer hides a single wrong sentence. Compound sentences are further split into claims so one wrong fact is not hidden by a right one; the sentence takes its weakest claim's verdict."))
    s.append(qa("Your false-positive rate is 16.7%. Isn't that bad?", "It is a main weakness (23% of supported sentences in the labelled set). Known causes: pronouns, long chunks, number formats. Smaller evidence snippets and a pronoun retry are the next fixes."))
    s.append(qa("Why a number checker? And did it help?", "NLI is weak with exact numbers (a wrong figure once got 0.97 entailment). The checker confirms numbers and names exist in the source and marks the exact words. On our labelled set, which is mostly invented details and commentary, it did not raise overall F1, but it catches changed numbers and makes the explanation precise."))
    s.append(qa("Couldn't a number match in the wrong context?", "Yes. We choose the passage the sentence is about by meaning, not by NLI, to reduce this, and NLI still judges the sentence; but a number that exists in the right passage attached to a different fact would pass."))
    s.append(qa("Why not fine-tune DeBERTa?", "It needs a large labelled dataset and a GPU, and a strict split between training and test data; the proposal specified a pretrained model."))
    s.append(h2("22.4 Attention and trust"))
    s.append(qa("What does the attention view show?", "The source words the checker focused on most for a sentence NLI rejected. It is an explanation aid, not proof of what caused the decision."))
    s.append(qa("How do you know the attention words are meaningful?", "We test them: delete the top 5 words from the source, re-run NLI, and compare the change with deleting 5 random words. On average the top words move the probabilities more (0.24 vs 0.11), but they beat the random control in only 35% of flagged sentences, so we present them as a hint, not as proof."))
    s.append(qa("Is this cross-attention?", "No: DeBERTa uses self-attention over the joined chunk and sentence. We read the part from sentence tokens to chunk tokens."))
    s.append(qa("Why these trust weights?", "Sentence support is the most direct evidence, so it has the largest weight. They are designed, not learned. Our analysis suggests token probability is stronger than its 0.20 weight implies and the document component is uninformative while all authorities are equal; learning the weights is future work."))
    s.append(qa("Why combine four signals?", "Each has blind spots. NLI missed a wrong number that the entity check caught; the number check cannot judge meaning; model confidence alone does not show which sentence is wrong."))
    s.append(qa("What if a model gives no logprobs?", "The token component's weight is dropped and the other weights are scaled to sum to 1."))
    s.append(h2("22.5 Testing"))
    s.append(qa("How do you know the checker works?", "Three ways: (1) we corrupt true sentences and check detection (100% for negation, changed numbers and noise; 21% for weak LLM rewrites) with a control set for false positives; (2) a labelled set of 176 real answer sentences with precision, recall and F1; (3) an ablation of each component."))
    s.append(qa("Who labelled the data?", "An AI assistant, reading each sentence against the source documents; it was not human-verified. We state that openly; a human re-label of a sample is the next step."))
    s.append(qa("Why compare against the clean chunk?", "If we corrupted the documents, the generator would repeat the corruption and NLI would correctly call it supported, so nothing would be measured."))
    s.append(qa("Why a control set?", "Without it, a checker that flags everything would look perfect."))
    s.append(qa("Why only 21% on LLM attacks?", "The 3B model often wrote near-paraphrases rather than real contradictions, so the test is weak; subtle rewrites are also genuinely harder for NLI. The full pipeline improves it only a little, to 29%."))
    s.append(qa("Isn't faulty or loose mode cheating?", "No: a detector can only be shown on answers with errors. The modes are labelled, and normal mode stays strict."))
    s.append(qa("Why Ollama instead of Gemini?", "The Gemini models we could use refused logprobs and had a small daily limit. Ollama has no limits, returns logprobs, and is private and reproducible. Gemini was removed from the project."))
    s.append(qa("How big is your evaluation?", "22 questions, 6 documents and 176 labelled sentences: indicative, not a benchmark."))

    s.append(h1("23. Wording guide: say this, not that"))
    s.append(table(["Say", "Don't say", "Why"], [
        ["&#8220;Detects claims not supported by the sources&#8221;", "&#8220;Detects false information&#8221;", "The system checks support, not truth"],
        ["&#8220;Source words the checker attended to most&#8221;", "&#8220;The words that caused the hallucination&#8221;", "Attention isn't causal"],
        ["&#8220;Attention words matter on average but not reliably for every sentence&#8221;", "&#8220;The attention words explain the decision&#8221;", "The masking check supported them in only 35% of flagged sentences"],
        ["&#8220;Self-attention over the joined pair&#8221;", "&#8220;Cross-attention&#8221;", "DeBERTa is a single encoder"],
        ["&#8220;A probability for each label&#8221;", "&#8220;Calibrated probabilities&#8221;", "Calibration was never tested"],
        ["&#8220;2 of 23 claim sentences&#8221;", "Stressing &#8220;8.7%&#8221; alone", "The sample is small"],
        ["&#8220;F1 0.71 on a labelled set of 176 sentences, labelled by an AI assistant&#8221;", "&#8220;Our system achieves 0.71 F1&#8221; without context", "The labels are not human-verified; the intervals overlap with simpler variants"],
        ["&#8220;Claim splitting and the number check didn't measurably change F1&#8221;", "&#8220;Each component improved accuracy&#8221;", "The ablation didn't show that"],
        ["&#8220;The weights are designed, not learned&#8221;", "Inventing a justification", "Honesty is safer and respected"],
        ["&#8220;100% on negation, stale facts and noise; 21% on weak LLM rewrites&#8221;", "&#8220;Our system achieves 100% detection&#8221;", "100% is only for simple attacks"],
        ["&#8220;0.5 is a design choice; F1 is flat from 0.3 to 0.9&#8221;", "&#8220;0.5 is the optimal threshold&#8221;", "It is not tuned to an optimum"],
        ["&#8220;Nothing is fine-tuned&#8221;", "&#8220;We trained the model&#8221;", "All models are pretrained"]], [36, 34, 30]))
    return "".join(s)

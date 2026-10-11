from guide_lib import *


def embed_map():
    pts = {
        "Solar": ("#2563eb", [(120, 150), (140, 165), (105, 185)]),
        "Wind": ("#16a34a", [(220, 80), (250, 100), (275, 70)]),
        "Hydro": ("#ea580c", [(400, 180), (420, 170), (385, 195)]),
        "Policy": ("#7c3aed", [(90, 235), (110, 245), (75, 250)]),
    }
    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 280" width="520" height="280" font-family="Helvetica, Arial, sans-serif">',
         '<rect x="1" y="1" width="518" height="278" fill="#fff" stroke="#e5e7eb"/>']
    q = (170, 135)
    near = [(120, 150), (140, 165), (105, 185), (220, 80), (250, 100), (90, 235)]
    for (x, y) in near:
        o.append(f'<line x1="{q[0]}" y1="{q[1]}" x2="{x}" y2="{y}" stroke="#9ca3af" stroke-dasharray="3 3"/>')
    for name, (c, lst) in pts.items():
        for (x, y) in lst:
            o.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{c}"/>')
    o.append(f'<polygon points="{q[0]},{q[1]-9} {q[0]+3},{q[1]-3} {q[0]+9},{q[1]-3} {q[0]+4},{q[1]+2} {q[0]+6},{q[1]+9} {q[0]},{q[1]+5} {q[0]-6},{q[1]+9} {q[0]-4},{q[1]+2} {q[0]-9},{q[1]-3} {q[0]-3},{q[1]-3}" fill="#dc2626"/>')
    o.append('<text x="190" y="128" font-size="9" fill="#374151">Question: &#8220;How much new solar capacity was added in 2023?&#8221;</text>')
    for i, (name, (c, _)) in enumerate(pts.items()):
        o.append(f'<circle cx="400" cy="{35+i*14}" r="4" fill="{c}"/><text x="410" y="{38+i*14}" font-size="9" fill="#374151">{name} chunks</text>')
    o.append("</svg>")
    return "".join(o)


def build():
    s = []
    s.append(part_page("Part A", "Foundations: the problem and the key ideas"))
    s.append(h1("1. The problem: AI hallucinations", nobreak=True))
    s.append(h2("1.1 The one-minute summary"))
    s.append(p("AI language models (like ChatGPT or the model used here) sometimes state things that are not true, and they say them just as confidently as things that are true. This is called a <b>hallucination</b>."))
    s.append(p("<b>HalluciGuard</b> answers questions using a small set of documents, then <b>checks every sentence of its own answer against those documents</b>. It colours each sentence (green = supported, amber = neutral or containing wrong parts, red = unsupported, grey = filler with no claim), marks the <i>exact wrong words</i> (numbers and names the sources don't contain) in bright red, shows which source words the checker focused on, and gives the whole answer a <b>trust score</b> between 0 and 1."))
    s.append(box("say", "The one-paragraph explanation (memorise this)",
                 "&#8220;HalluciGuard answers questions from 6 renewable-energy documents. It finds the relevant chunks, removes useless ones (pruning), and a local AI model (qwen2.5:3b running on Ollama) writes an answer. Then a checker model (DeBERTa NLI) tests each sentence, or each smaller claim inside it, against the sources, a second check looks for numbers and names the sources do not contain, unsupported parts are marked, and a trust score is computed. We test the checker with sentences we broke on purpose and with a labelled set of real answers.&#8221;"))
    s.append(h2("1.2 What exactly is a hallucination?"))
    s.append(p("A language model writes text by repeatedly predicting the next word. It learned which words usually follow which from huge amounts of text. It does not look facts up in a database; it produces text that <i>sounds</i> right. Usually that text is also correct, but sometimes it is fluent and wrong. Typical hallucinations:"))
    s.append(ul(["<b>Wrong numbers or dates:</b> &#8220;345 GW&#8221; becomes &#8220;120 GW&#8221;, or &#8220;2023&#8221; becomes &#8220;2019&#8221;.",
                 "<b>Invented details:</b> a plausible-sounding name, statistic or quotation that appears nowhere in the sources.",
                 "<b>Mixed-up facts:</b> a fact about wind power stated as if it were about solar power.",
                 "<b>Over-generalisation or commentary:</b> &#8220;some countries&#8221; becomes &#8220;all countries&#8221;, or opinions such as &#8220;this underscores the country's commitment&#8221;."]))
    s.append(p("The danger is that the user cannot tell the difference from the way the answer is written. A wrong sentence looks exactly as confident as a right one."))
    s.append(h2("1.3 Why it matters"))
    s.append(p("If people use AI answers for studying, business, medicine or law, a confident wrong answer can cause real harm. Organisations that use AI on their own documents need to know <b>which parts of an answer actually come from their documents</b> and which parts the AI added on its own. HalluciGuard addresses exactly this need."))

    s.append(h1("2. The big idea: an open-book exam with a teacher"))
    s.append(p("Imagine a student taking an open-book exam:"))
    s.append(ul(["The student gets a question.", "They look up the relevant pages in the book.", "They write an answer using those pages.",
                 "<b>A teacher then checks the answer line by line</b>, asking for each line: &#8220;Does the book actually say this?&#8221;"], ordered=True))
    s.append(table(["In the exam", "In HalluciGuard"], [
        ["The book", "6 documents about renewable energy (the <b>corpus</b>)"],
        ["Looking up the relevant pages", "<b>Retrieval</b>: finding the most relevant pieces of the documents"],
        ["Throwing away pages that aren't useful", "<b>MI pruning</b>: removing chunks that don't really match the question"],
        ["The student writing the answer", "The <b>generator</b>: qwen2.5:3b, running locally through Ollama"],
        ["The teacher checking each line", "The <b>verifier</b>: DeBERTa (NLI) plus a number-and-name checker"],
        ["The teacher's overall grade", "The <b>trust score</b>"]], [34, 66]))
    s.append(p("Steps 1 to 3 are a standard technique called <b>RAG</b> (Retrieval-Augmented Generation): &#8220;retrieval&#8221; means looking things up, &#8220;augmented&#8221; means the AI gets extra information, and &#8220;generation&#8221; means writing the answer."))
    s.append(box("key", "Key idea", "Most RAG systems stop after the student writes the answer. <b>HalluciGuard adds the teacher.</b> The checking part (verification, wrong-part marking, attention, trust score) is the main contribution of the project. The RAG part only needs to work properly; it is the setting in which the checker operates."))
    s.append(h2("2.1 Prevent, detect, explain"))
    s.append(table(["Job", "How", "Components"], [
        ["Prevent hallucinations a little", "Give the AI only relevant, non-repetitive context", "MI pruning"],
        ["Detect hallucinations (the main job)", "Check every sentence (and each claim inside it) against the sources", "NLI verifier, claim splitting, number and name checker"],
        ["Explain them", "Show where and why, and summarise reliability", "Red words, attention words, trust score, dashboard"]], [28, 40, 32]))

    s.append(h1("3. The most important idea: &#8220;supported&#8221; is not the same as &#8220;true&#8221;"))
    s.append(p("HalluciGuard checks whether a sentence is <b>supported by the documents</b>. It does <b>not</b> check whether the sentence is true in the real world. Professors very often ask about this."))
    s.append(table(["", "Question", "Can we check it?"], [
        ["<b>True</b> (factual)", "Is it correct in the real world?", "Very hard: you would need all the world's knowledge, always up to date"],
        ["<b>Supported</b> (grounded / faithful)", "Do <i>our documents</i> back it up?", "Yes: this is what HalluciGuard does"]], [24, 34, 42]))
    s.append(box("ex", "Example", "Our documents say nothing about the capital of Australia. If the AI writes &#8220;Australia has Canberra as its capital&#8221;, the sentence is <b>true</b>, but the documents don't say it, so it is marked unsupported. (This exact sentence appeared in our test data.) The opposite can also happen: if a document contained a mistake and the AI copied it, the sentence would be marked <b>supported</b> even though it is false."))
    s.append(p("So in this project, red does not mean &#8220;false&#8221;. It means: <b>&#8220;this cannot be verified from the sources, so don't trust it blindly.&#8221;</b> That includes true facts from the model's own knowledge <i>and</i> unsupported commentary such as &#8220;this achievement underscores the country's commitment to renewable energy&#8221;."))
    s.append(box("key", "Key idea", "In this project: <b>Hallucination = a claim that is not supported by the source documents.</b> Researchers call this property <i>faithfulness</i> or <i>groundedness</i>."))
    s.append(box("say", "What to say", "Question: &#8220;Your system marked a true sentence red. Isn't that a mistake?&#8221;<br>Answer: &#8220;No. We detect unsupported claims, not false ones. The sentence may be true, but the sources don't back it, so the user can't verify it.&#8221;"))

    s.append(h1("4. Key AI concepts you need"))
    s.append(p("Each concept is explained without maths first; where a formula helps, it is explained in words."))
    s.append(h2("4.1 Tokens"))
    s.append(p("AI models do not read whole words; they read <b>tokens</b>, which are words or pieces of words. &#8220;solar&#8221; might be one token, while &#8220;gigawatts&#8221; might be split into &#8220;giga&#8221; + &#8220;watts&#8221;. A rough rule: 100 tokens &#8776; 75 English words. Models can only read a limited number of tokens at once. The checker model (DeBERTa) reads at most <b>512 tokens</b> (the code truncates the chunk, never the sentence). This limit shapes the chunk size."))
    s.append(h2("4.2 Embeddings: turning meaning into numbers"))
    s.append(p("An <b>embedding</b> is a list of numbers that represents the meaning of a piece of text. HalluciGuard uses a small model called <b>MiniLM</b> (all-MiniLM-L6-v2), which turns any text into a list of <b>384 numbers</b>. Texts with similar meaning get similar numbers: picture each text as a point on a map, where solar texts cluster together, wind texts cluster elsewhere, and so on."))
    s.append(fig(embed_map(), "Figure 1. Illustration of embeddings as a meaning map (real embeddings have 384 dimensions, not 2). Dashed lines = the 6 nearest chunks."))
    s.append(p("<b>Cosine similarity</b> measures how close two embeddings are: think of each embedding as an arrow, and ask how much two arrows point the same way (1 = same direction, same meaning)."))
    s.append(h2("4.3 Probabilities, logprobs and softmax"))
    s.append(ul(["When the generator writes a word it has a probability for every possible next word. The probability of the word it chose tells us how <b>confident</b> it was. A <b>logprob</b> is the logarithm of that probability: 0 means certain, more negative means less confident.",
                 "The checker outputs three probabilities (entailment, neutral, contradiction) that add up to 1."]))
    s.append(p("Models first produce raw scores; <b>softmax</b> turns them into probabilities that are positive and add up to 1."))
    s.append(h2("4.4 Transformers and attention"))
    s.append(p("MiniLM and DeBERTa are <b>transformer</b> models, the same family as GPT. The key mechanism is <b>attention</b>: when the model processes a word, it decides how much every other word in the input matters for understanding it."))
    s.append(ul(["A transformer has stacked <b>layers</b> (DeBERTa-base has 12), each with several <b>attention heads</b> (12). Different heads can focus on different relationships, e.g. matching numbers.",
                 "<b>Self-attention</b> means the words of one input attend to each other. DeBERTa receives the chunk and the answer sentence joined into one input, so words in the sentence can attend to words in the chunk."]))
    s.append(h2("4.5 Classifiers vs generators"))
    s.append(table(["Kind", "What it does", "In HalluciGuard"], [
        ["<b>Generator</b>", "Writes new text, word by word", "qwen2.5:3b (writes the answer; also writes one kind of test attack)"],
        ["<b>Classifier</b>", "Reads text and picks a label", "DeBERTa (entailment / neutral / contradiction)"],
        ["<b>Encoder</b> (embedding model)", "Reads text and outputs numbers", "MiniLM (embeddings for search and for picking a sentence's source)"]], [24, 30, 46]))
    s.append(p("Using a classifier as the checker, instead of another generator, is deliberate: classifiers are fast, give the same answer every time, cannot invent text, and give probabilities we can put a threshold on."))
    s.append(box("chk", "Check yourself", "Why does the system turn text into embeddings before it can find relevant chunks?<br><i>Answer:</i> computers compare numbers precisely. Embeddings let the system measure &#8220;closeness of meaning&#8221; as a number, so it finds chunks that mean the same thing as the question even when the words differ."))
    return "".join(s)

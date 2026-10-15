"""Build HalluciGuard_Complete_Guide.pdf from the part_*.py files (HTML -> PDF through headless Chrome).

Usage: python3 docs/guide/build_guide.py
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import guide_lib as L  # noqa: E402
import part_a, part_b, part_c, part_d, part_e  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    body = [part_a.build(), part_b.build(), part_c.build(), part_d.build(), part_e.build()]  # fills L.TOC
    cover = (f'<div class="cover"><h1 style="color:#1d4ed8">HalluciGuard</h1><div class="sub">Complete Project Guide</div>'
             '<p>Detecting, locating and explaining hallucinations in a Retrieval-Augmented Generation (RAG) system</p><br>'
             '<p>Everything you need to understand the project from scratch: the ideas behind it, how every component works, '
             'what the results mean, its limitations, and how to answer questions about it. No coding knowledge is required.</p>'
             f'<p style="margin-top:30mm;font-size:9pt;color:#6b7280">Updated edition, matching the code at commit {git_rev()} ({date.today().isoformat()}). '
             'Exact rules and numbers: FACTS.md.</p></div>')
    how = L.h1("How to use this guide", nobreak=False).replace('<h1 id=', '<h1 data-x="1" id=')
    # front matter headings are not listed in the TOC
    L.TOC[:] = [t for t in L.TOC if not t[2].startswith(("How to use", "What changed"))]
    how = ('<h1>How to use this guide</h1><p>This guide is written for someone who has never seen the project before. It starts with the basic ideas, '
           'follows one real question through every step of the system, explains how the system is tested and what the results mean, and how to defend it.</p>'
           + L.table(["If you want to...", "Read"], [
               ["Get the idea in one minute", "Section 1.1"], ["Understand the concepts (embeddings, NLI, attention...)", "Part A, sections 1-4"],
               ["Understand how a question flows through the system", "Part B, sections 5-14"], ["Understand the testing and the numbers", "Part C, sections 15-18"],
               ["Prepare for questions from a professor", "Part D, sections 19-23"], ["Run the demo or find a file in the code", "Part E, sections 24-26"],
               ["See what changed from the earlier edition", "The next page"]], [60, 40])
           + '<p>Coloured boxes highlight the most important points:</p>'
           + L.box("key", "Key idea", "The core concepts you must understand.") + L.box("say", "What to say", "Ready-made sentences for explaining or answering.")
           + L.box("care", "Be careful", "Common mistakes, wrong wording, or things that could get you into trouble.") + L.box("ex", "Example", "Worked examples with concrete numbers.")
           + L.box("chk", "Check yourself", "A short question to test your understanding; the answer follows.")
           + L.box("care", "About the numbers", "Numbers come from runs on 2026-10-11 and are dated where they appear. Some figures are marked <b>illustrative</b>: made-up but realistic numbers that explain an idea. Before presenting, check the tables in section 18 against README.md and FACTS.md, and read section 18.5."))
    changed = ('<h1>What changed in this edition</h1><p>The earlier edition described the project as it was part-way through the build. This edition matches the final code. The main differences:</p>'
               + L.table(["Topic", "Earlier edition said", "Now (the code)"], [
                   ["Gemini", "&#8220;The system can still be switched back to Gemini with a setting&#8221;", "Gemini was <b>removed completely</b>; Ollama (qwen2.5:3b) is the only generator"],
                   ["When a sentence is red", "Red if any chunk gives contradiction above 0.5, or best entailment below 0.5", "Red only if the best entailment is below 0.5. A contradiction label alone gives amber. Wrong numbers or names make a sentence amber with bright red words (section 10.8)"],
                   ["Number checker", "Normalises &#8220;1,600&#8221; = &#8220;1.6 thousand&#8221;, matches dates by year", "Only commas are removed; no unit or magnitude handling. Numbers are checked against the passage the sentence is about, chosen by meaning (section 11.2)"],
                   ["Claims and filler", "Not present", "Compound sentences are split into claims; filler sentences are grey and excluded from scoring (10.6, 10.7)"],
                   ["Document authority", "Hand-set per document (e.g. 0.9 and 0.6)", "All six documents are 0.8, so the document score is uninformative (6.3, 13.3)"],
                   ["Loose mode", "Switched on in the dashboard", "An environment variable (<code>PROMPT_MODE=loose</code>); only faulty mode is a dashboard checkbox (16)"],
                   ["Attention", "For every red sentence", "Only for sentences NLI rejected (12)"],
                   ["Threshold 0.5", "&#8220;Not tuned; a sweep would tune it&#8221;", "Swept from 0.3 to 0.9: F1 flat at 0.71-0.72 (18.3)"],
                   ["Evaluation", "22 questions and the adversarial test", "Plus a labelled set of 176 sentences (AI-labelled) with precision, recall, F1, confidence intervals, an ablation and AUROC for the trust parts (18.3)"],
                   ["Speed", "About 2.5 s per answer", "2.5 s generation; about 8 s for a whole query; models preloaded at startup"],
                   ["Code map", "rag_pipeline.py runs the whole flow", "rag_pipeline.py only retrieves, prunes and generates; query_service.py and verification.py do the rest (25)"],
                   ["Trust example", "Used a Gemini answer (0.75)", "Uses a faulty-mode answer (0.69); known edge cases include 0.00 for one flagged claim (13.9)"],
                   ["Attention check", "The masking check was listed as not built", "Built: top words vs random words; supported in only 35% of flagged sentences (12.3)"],
                   ["Adversarial test", "NLI alone", "Scored twice: NLI alone and the full pipeline (15.6, 18.1)"],
                   ["Not built", "Pronoun retry was listed as an idea", "Still not built; stated as such (10.6, 18.4)"]], [16, 36, 48]))
    toc = '<h1>Contents</h1>' + L.build_toc()
    html = L.page([cover, how, changed, toc] + body, "")
    html = html.replace("<body>" + cover + how + changed + toc, "<body>" + cover + how + changed + toc)
    out_html = HERE / "HalluciGuard_Guide.html"
    pdf = HERE / "HalluciGuard_Complete_Guide.pdf"

    def render(h):
        out_html.write_text(h)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out_html.as_uri()],
                       check=True, capture_output=True, timeout=240)

    render(html)
    try:  # second pass: fill in contents page numbers (needs PyMuPDF; skipped if it is not installed)
        import html as _h
        import re
        import pymupdf
        doc = pymupdf.open(str(pdf))
        pages = [d.get_text().splitlines() for d in doc]
        start = next(i for i, lines in enumerate(pages) if any(l.startswith("AI language models (like ChatGPT") for l in lines))
        for lvl, sid, title in L.TOC:
            plain = re.sub(r"<[^>]+>", "", _h.unescape(title)).strip()
            is_part = plain.startswith("Part ")
            key = plain.split(":")[0] if is_part else plain[:34]
            for i in range(start - 1 if title.startswith("Part A") else start, len(pages)):
                lines = [l.strip() for l in pages[i]]
                if (is_part and key in lines) or (not is_part and any(l.startswith(key) for l in lines)):
                    L.PAGES[sid] = i + 1
                    break
        toc2 = '<h1>Contents</h1>' + L.build_toc()
        render(L.page([cover, how, changed, toc2] + body, ""))
    except ImportError:
        print("PyMuPDF not installed: contents page numbers skipped")
    print("wrote", pdf, pdf.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()

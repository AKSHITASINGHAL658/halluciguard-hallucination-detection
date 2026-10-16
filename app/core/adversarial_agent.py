import hashlib
import json
import logging
import random
import re
from functools import lru_cache

from sqlalchemy.orm import Session

from app import config
from app.core import rag_pipeline
from app.core.metrics import detection_rate, false_positive_rate
from app.core.nli_verifier import batch_verify
from app.core.sentence_localizer import split_sentences
from app.core.verification import verify_sentences
from app.models import AdversarialTest, Chunk

log = logging.getLogger("halluciguard.adversarial")
ATTACKS = ("negation", "stale_fact", "near_number", "noise", "llm_contradiction")
LLM_CONTRADICTION_PROMPT = (
    "Rewrite this sentence so it remains fluent and plausible but states something factually different. "
    "Output only the sentence.\n\nSentence: {sentence}"
)
UNRELATED_CLAUSES = [
    "and the Eiffel Tower was repainted in 1889",
    "while the city of Lisbon banned all bicycles last winter",
    "and a famous chess tournament was held on Mars",
    "although penguins were reclassified as mammals by the committee",
]
_AUX = r"(is|was|are|were|has|have|had|can|will|does|did)"


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load(config.SPACY_MODEL)


def inject_negation(s: str) -> str:
    """Remove not/never if present, else insert 'not' after the first auxiliary verb, else prefix a denial."""
    if re.search(r"\b(not|never)\b", s):
        return re.sub(r"\s*\b(not|never)\b", "", s, count=1)
    new, n = re.subn(rf"\b{_AUX}\b", r"\1 not", s, count=1)
    if n:
        return new
    return "It is not true that " + s[0].lower() + s[1:]


def inject_stale_fact(s: str, rng: random.Random | None = None) -> str:
    """Shift years by 3-10 and scale other numbers by 0.5-2x (never ~1x). Returns s unchanged if no numbers."""
    rng = rng or random
    shift = rng.randint(3, 10) * rng.choice([-1, 1])

    def repl(m):
        raw = m.group(0)
        val = float(raw.replace(",", ""))
        if re.fullmatch(r"(1[5-9]|20)\d\d", raw):
            return str(int(val) + shift)
        factor = rng.choice([0.5, 0.6, 0.75, 1.5, 2.0])
        new = val * factor
        if "." in raw:
            return f"{new:.{len(raw.split('.')[1])}f}"
        return f"{round(new):,}" if "," in raw else str(round(new))

    return re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", repl, s)


def inject_near_number(s: str, rng: random.Random | None = None) -> str:
    """Change ONE number slightly (year +-1, other numbers by 3-5%): wrong, but close enough that NLI may accept it."""
    rng = rng or random
    matches = list(re.finditer(r"\d+(?:,\d{3})*(?:\.\d+)?", s))
    if not matches:
        return s
    m = rng.choice(matches)
    raw = m.group(0)
    val = float(raw.replace(",", ""))
    if re.fullmatch(r"(1[5-9]|20)\d\d", raw):
        new = str(int(val) + rng.choice([-1, 1]))
    else:
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out = val * rng.choice([0.95, 0.97, 1.03, 1.05])
        if round(out, decimals) == val:  # tiny numbers: nudge by the smallest visible step
            out = val + rng.choice([-1, 1]) * 10 ** -decimals
        new = f"{out:.{decimals}f}"
        if "," in raw:
            new = f"{round(out):,}"
    return s[:m.start()] + new + s[m.end():]


def inject_noise(s: str, entity_pool: list[str] | None = None, rng: random.Random | None = None) -> str:
    """Swap one named entity for a different one from the pool, or append an unrelated clause."""
    rng = rng or random
    doc = _nlp()(s)
    ents = [e for e in doc.ents if e.label_ in {"GPE", "ORG", "PERSON", "NORP", "LOC", "FAC"}]
    pool = [p for p in (entity_pool or []) if ents and p.lower() != ents[0].text.lower()]
    if ents and pool and rng.random() < 0.5:
        e = rng.choice(ents)
        alt = [p for p in pool if p.lower() != e.text.lower()]
        if alt:
            return s[:e.start_char] + rng.choice(alt) + s[e.end_char:]
    clause = rng.choice(UNRELATED_CLAUSES)
    return s.rstrip(" .") + ", " + clause + "."


def inject_llm_contradiction(s: str) -> str | None:
    """Ask the local Ollama model for a fluent but factually different rewrite. Cached on disk.

    Returns None if Ollama is unavailable or the rewrite is empty/identical.
    """
    key = hashlib.sha256("\n".join(["llm_contradiction", config.OLLAMA_MODEL, s]).encode()).hexdigest()
    path = config.LLM_CACHE_DIR / f"adv-{key}.json"
    if config.LLM_CACHE and path.exists():
        try:
            return json.loads(path.read_text())["text"]
        except (OSError, ValueError, KeyError):
            pass
    try:
        resp = rag_pipeline._get_ollama_client().chat(
            model=config.OLLAMA_MODEL, options={"temperature": 0},
            messages=[{"role": "user", "content": LLM_CONTRADICTION_PROMPT.format(sentence=s)}])
        msg = resp["message"] if isinstance(resp, dict) else resp.message
        raw = msg["content"] if isinstance(msg, dict) else msg.content
    except Exception as e:  # noqa: BLE001
        log.warning("llm_contradiction failed: %s", str(e)[:120])
        return None
    lines = [ln.strip().strip('"\'') for ln in (raw or "").strip().splitlines() if ln.strip()]
    out = lines[0] if lines else ""
    if not out or out == s:
        return None
    if config.LLM_CACHE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": out}))
    return out


def _candidate_sentences(chunk_text: str) -> list[str]:
    out = []
    for s in split_sentences(chunk_text):
        if len(s.split()) < 5:
            continue
        if re.search(r"\d", s) or any(t.pos_ in ("VERB", "AUX") for t in _nlp()(s)):
            out.append(s)
    return out


def run_attack_suite(db: Session, n_chunks: int = config.ADV_N_CHUNKS, seed: int = config.ADV_SEED) -> dict:
    """Corrupt sentences, then run NLI with premise = ORIGINAL clean chunk, hypothesis = corrupted sentence."""
    rng = random.Random(seed)
    near_rng = random.Random(seed + 1)  # separate stream so adding near_number leaves the other attacks unchanged
    chunks = db.query(Chunk).order_by(Chunk.id).all()
    rng.shuffle(chunks)
    chunks = chunks[:n_chunks]
    pool = sorted({e.text for c in chunks for e in _nlp()(c.text).ents if e.label_ in {"GPE", "ORG", "PERSON"}})

    items = []  # (chunk, attack_type, original, corrupted)
    for c in chunks:
        sents = _candidate_sentences(c.text)
        numeric = [s for s in sents if re.search(r"\d", s)]
        picks = rng.sample(numeric, min(2, len(numeric))) if numeric else []
        rest = [s for s in sents if s not in picks]
        if len(picks) < 2 and rest:
            picks += rng.sample(rest, min(2 - len(picks), len(rest)))
        for s in picks:
            items.append((c, "control", s, s))
            items.append((c, "negation", s, inject_negation(s)))
            if re.search(r"\d", s):
                items.append((c, "stale_fact", s, inject_stale_fact(s, rng)))
                items.append((c, "near_number", s, inject_near_number(s, near_rng)))
            items.append((c, "noise", s, inject_noise(s, pool, rng)))
            llm = inject_llm_contradiction(s)
            if llm:
                items.append((c, "llm_contradiction", s, llm))
    items = [it for it in items if it[1] == "control" or it[3] != it[2]]

    results = batch_verify([(corrupt, c.text) for c, _, _, corrupt in items])  # NLI alone
    full = _full_pipeline_flags(items)  # claims + number/name check + filler, evidence = the clean chunk
    db.query(AdversarialTest).delete()
    for k, ((c, attack, orig, corrupt), r) in enumerate(zip(items, results)):
        db.add(AdversarialTest(original_chunk_id=c.id, attack_type=attack, original_sentence=orig,
                               corrupted_text=corrupt, nli_label=r["label"], detected=r["label"] != "entailment",
                               detected_full=full[k]))
    db.commit()
    return summarize_results(db)


def _full_pipeline_flags(items) -> list[bool]:
    """Run each corrupted sentence through the app's full sentence check against its clean chunk."""
    flags: list[bool | None] = [None] * len(items)
    by_chunk: dict[int, list[int]] = {}
    for k, (c, _, _, _) in enumerate(items):
        by_chunk.setdefault(c.id, []).append(k)
    for ks in by_chunk.values():
        c = items[ks[0]][0]
        chunk = {"chunk_id": c.id, "document_id": c.document_id, "text": c.text}
        for k, r in zip(ks, verify_sentences([items[k][3] for k in ks], [chunk])):
            flags[k] = bool(r["is_hallucination"])
    return flags


def summarize_results(db: Session) -> dict:
    rows = db.query(AdversarialTest).all()
    by_type = {a: [r.detected for r in rows if r.attack_type == a] for a in ATTACKS}
    control = [r.detected for r in rows if r.attack_type == "control"]
    full_rows = [r for r in rows if r.detected_full is not None]
    full_by_type = {a: [r.detected_full for r in full_rows if r.attack_type == a] for a in ATTACKS}
    full_control = [r.detected_full for r in full_rows if r.attack_type == "control"]
    return {"detection_rate": {a: detection_rate(v) for a, v in by_type.items()},
            "counts": {**{a: len(v) for a, v in by_type.items()}, "control": len(control)},
            "false_positive_rate": false_positive_rate(control), "n_tests": len(rows),
            "full_pipeline": {"detection_rate": {a: detection_rate(v) for a, v in full_by_type.items()},
                              "false_positive_rate": false_positive_rate(full_control), "n_tests": len(full_rows)}}


def inject_faults(answer: str, rng: random.Random | None = None) -> tuple[str, list[dict]]:
    """Demo "faulty RAG": corrupt 1 sentence (2 if the answer has 4+) of a real answer.

    Prefers a wrong-number attack on sentences with digits, otherwise negation. Returns the corrupted answer and
    the list of {index, attack_type, original, corrupted}.
    """
    rng = rng or random.Random()
    sents = split_sentences(answer)
    if not sents:
        return answer, []
    with_digits = [i for i, t in enumerate(sents) if re.search(r"\d", t)]
    others = [i for i in range(len(sents)) if i not in with_digits]
    n = 2 if len(sents) >= 4 else 1
    order = rng.sample(with_digits, len(with_digits)) + rng.sample(others, len(others))
    faults = []
    for i in order[:n]:
        orig = sents[i]
        attack = "stale_fact" if i in with_digits else "negation"
        new = inject_stale_fact(orig, rng) if attack == "stale_fact" else inject_negation(orig)
        if new == orig:
            attack, new = "negation", inject_negation(orig)
        sents[i] = new
        faults.append({"index": i, "attack_type": attack, "original": orig, "corrupted": new})
    return " ".join(sents), sorted(faults, key=lambda f: f["index"])

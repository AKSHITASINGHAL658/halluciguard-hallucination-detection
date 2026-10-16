import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Card, ErrorBox, Spinner, Stat } from "./ui.jsx";

const pct = (v, d = 0) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`);
const KIND_LABEL = {
  contradicts_or_miscopies_source: "Contradicts or miscopies the source",
  invented_specific: "Invents a specific fact",
  commentary_or_inference: "Commentary beyond the source",
};
const KIND_DESC = {
  contradicts_or_miscopies_source:
    "The sentence reuses the source's wording but changes or flips a fact. The verifier sees heavy word overlap and often accepts it, so this is the hardest kind to catch.",
  invented_specific:
    "The sentence adds a specific number, name or date that the source never states. The number and name check looks for exactly this.",
  commentary_or_inference:
    "The sentence adds opinion or reasoning that goes beyond what the source says. Nothing in the source backs it, so the verifier usually rejects it.",
};

function DetectorAccuracy({ d }) {
  const p = d.production;
  if (!p) return null;
  const bad = p.tp + p.fn, good = p.fp + p.tn, flagged = p.tp + p.fp;
  return (
    <>
      <div className="stats">
        <Stat label="Recall" value={pct(p.recall)} hint={`${p.tp} of ${bad} unsupported sentences caught`}
          why={`Of the ${bad} sentences that really are unsupported, how many the detector flagged (${p.tp} ÷ ${bad}). It shows how many bad sentences slip through: ${bad - p.tp} did.`} />
        <Stat label="Precision" value={pct(p.precision)} hint={`${p.tp} of ${flagged} flags were right`}
          why={`Of the ${flagged} sentences the detector flagged, how many really were unsupported (${p.tp} ÷ ${flagged}). It shows how trustworthy a flag is: ${p.fp} flags were false alarms.`} />
        <Stat label="False-positive rate" value={pct(p.false_positive_rate)} hint={`${p.fp} of ${good} supported sentences wrongly flagged`}
          why={`Of the ${good} sentences that really are fine, how many were flagged anyway (${p.fp} ÷ ${good}). Lower is better. It is the price of catching more bad sentences.`} />
        <Stat label="F1 score" value={p.f1.toFixed(2)} hint={p.f1_ci95 ? `95% range ${p.f1_ci95[0].toFixed(2)}–${p.f1_ci95[1].toFixed(2)}` : undefined}
          why="One score that balances recall and precision, so the detector can't look good by flagging everything or by flagging almost nothing. 1.0 is perfect. The range shows how much it could shift with a different sample of sentences." />
        {d.auroc?.final != null && (
          <Stat label="Trust-score AUROC" value={d.auroc.final.toFixed(2)} hint="separating good from bad answers"
            why="Measured on whole answers, not sentences. Pick one good and one bad answer at random: this is how often the bad one gets the lower trust score. 0.5 is a coin flip and 1.0 is perfect." />
        )}
      </div>
      {Object.keys(d.recall_by_error_kind || {}).length > 0 && (
        <Card title="Recall by kind of error">
          <p className="muted" style={{ marginBottom: 16 }}>
            The same recall measure, split by what kind of mistake the unsupported sentence contains. n is how many sentences of that kind
            are in the labelled set, so a small n means a rough number.
          </p>
          {Object.entries(d.recall_by_error_kind).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 18 }}>
              <div className="bar-row">
                <div className="bar-label"><span>{KIND_LABEL[k] || k} <small>(n={v.n})</small></span><b>{pct(v.recall)}</b></div>
                <div className="bar-track"><div className="bar-fill" style={{ width: `${v.recall * 100}%`, background: v.recall >= 0.75 ? "var(--green)" : v.recall >= 0.5 ? "var(--amber)" : "var(--red)" }} /></div>
              </div>
              {KIND_DESC[k] && <p className="kind-desc">{KIND_DESC[k]} Caught {Math.round(v.recall * v.n)} of {v.n}.</p>}
            </div>
          ))}
        </Card>
      )}
    </>
  );
}

export default function EvaluationTab({ result: ev, setResult }) {
  const [det, setDet] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api("GET", "/evaluate/detector").then(setDet).catch((e) => setError(`API error on /evaluate/detector: ${e.message}`));
  }, []);

  async function run() {
    setBusy(true);
    setError("");
    try { setResult(await api("POST", "/evaluate", null, 1800000)); }
    catch (e) { setError(`API error on /evaluate: ${e.message}`); }
    finally { setBusy(false); }
  }

  return (
    <div className="stack">
      <div className="hero">
        <h2>Detector accuracy</h2>
        <p className="muted">How well the hallucination detector agrees with hand-labelled sentences (supported vs. unsupported by the source).</p>
      </div>
      <ErrorBox error={error} />
      {det && det.available && <DetectorAccuracy d={det} />}
      {det && !det.available && <Card><p className="muted">No labelled-set results yet. Run <code>python scripts/run_ablation.py</code>.</p></Card>}

      <div className="hero">
        <h2>Run on the test questions</h2>
        <p className="muted">Answers every test question and reports what the detector made of them. These two numbers are the detector's own verdict, not a check against the truth. This calls the LLM for each question.</p>
        <button className="primary" onClick={run} disabled={busy}>{busy ? "Running…" : "Run evaluation"}</button>
      </div>
      {busy && <Spinner label="Answering and checking the test questions…" />}
      {ev && !busy && (
        <>
          <div className="stats">
            <Stat label="Flagged as unsupported" value={pct(ev.hallucination_rate, 1)} hint={`${ev.n_sentences} sentences checked`}
              why={`The share of the answers' sentences the detector flagged${ev.hallucination_rate != null ? ` (about ${Math.round(ev.hallucination_rate * ev.n_sentences)} of ${ev.n_sentences})` : ""}. Filler sentences like "this is stated in the context" are not counted. It is the detector's own verdict, not checked against the truth.`} />
            <Stat label="Average trust" value={ev.avg_trust == null ? "n/a" : ev.avg_trust.toFixed(2)}
              why="The mean trust score (0 to 1) over all the answers. Each answer's score blends how well its sentences, entities and sources check out with how confident the model was. Higher is better." />
          </div>
          <Card><p className="muted">{ev.n_questions} questions · {ev.n_failed} failed</p></Card>
        </>
      )}
    </div>
  );
}

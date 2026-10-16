import { useState } from "react";
import { api } from "../api.js";
import { Bar, Card, Collapse, ErrorBox, Gauge, Spinner } from "./ui.jsx";

const sentenceClass = (s) => (s.is_hallucination ? "hallucination" : s.nli_label === "entailment" ? "supported" : "neutral");

function Sentence({ s }) {
  if (s.is_filler) return <span className="sent filler" title="No checkable claim">{s.text} </span>;
  const spans = [...(s.wrong_spans || [])].sort((a, b) => a.start - b.start);
  if (!spans.length) {
    return <span className={`sent ${sentenceClass(s)}`} title={`NLI: ${s.nli_label} (${s.nli_confidence?.toFixed(2)})`}>{s.text} </span>;
  }
  const parts = [];
  let pos = 0;
  spans.forEach((sp, i) => {
    parts.push(s.text.slice(pos, sp.start));
    parts.push(<mark key={i} title={sp.reason}>{s.text.slice(sp.start, sp.end)}</mark>);
    pos = sp.end;
  });
  parts.push(s.text.slice(pos));
  return <span className="sent neutral">{parts} </span>;
}

function Faults({ res }) {
  const byText = Object.fromEntries(res.sentences.map((x) => [x.text, x]));
  return (
    <Card title="Faulty mode — did the checker catch it?" className="warn">
      {res.injected_faults.map((f, i) => {
        const hit = byText[f.corrupted] || res.sentences.find((x) => x.index === f.index);
        const caught = !!(hit && (hit.is_hallucination || hit.wrong_spans?.length || hit.nli_label !== "entailment"));
        return (
          <div className="fault" key={i}>
            <span className={caught ? "pill ok" : "pill bad"}>{caught ? "Caught" : "Missed"}</span>
            <b>{f.attack_type}</b>
            <del>{f.original}</del> → <span>{f.corrupted}</span>
          </div>
        );
      })}
      <p className="muted">Token-probability is omitted here: logprobs describe the original answer, not the edited one.</p>
    </Card>
  );
}

function Unsupported({ s }) {
  const chk = s.attention_check;
  return (
    <Collapse head={<><span className="dot red" /> Unsupported: {s.text.slice(0, 90)}{s.text.length > 90 ? "…" : ""}</>}>
      <p>NLI: <b>{s.nli_label}</b> (confidence {s.nli_confidence.toFixed(2)})</p>
      {(s.wrong_spans || []).map((sp, i) => (
        <p key={i}><span className="dot red" /> <b>{sp.text}</b>: {sp.reason}</p>
      ))}
      <div className="muted">Best source chunk</div>
      <blockquote>{s.supporting_chunk_text || "n/a"}</blockquote>
      {chk && (
        <p className="muted">
          Masking check: hiding the top words moved NLI probabilities by {chk.top_shift.toFixed(2)} (hiding {chk.n_random}× random
          words: {chk.random_shift.toFixed(2)}) → {chk.supported ? "✅ the words matter" : "⚠️ not clearly more important than random words"}
        </p>
      )}
      {s.top_attention_tokens?.length > 0 && (
        <>
          <div className="muted">Source words the verifier attended to most</div>
          <div className="chips">
            {s.top_attention_tokens.map((t, i) => (
              <span className="chip" key={i}>{t.token}
                <small>{(t.weight * 100).toFixed(0)}%</small>
              </span>
            ))}
          </div>
        </>
      )}
    </Collapse>
  );
}

export default function QueryTab({ result: res, setResult }) {
  const [q, setQ] = useState("How much new solar capacity was added globally in 2023?");
  const [faulty, setFaulty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run(e) {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setError("");
    try {
      setResult(await api("POST", "/query", { text: q, inject_faults: faulty }));
    } catch (err) {
      setError(`API error on /query: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  const t = res?.trust;
  return (
    <div className="stack">
      <form className="hero" onSubmit={run}>
        <h2>Ask a question about the corpus</h2>
        <div className="ask">
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Type your question…" />
          <button className="primary" disabled={busy || !q.trim()}>{busy ? "Working…" : "Run query"}</button>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={faulty} onChange={(e) => setFaulty(e.target.checked)} />
          <span className="switch" />
          <span>Faulty mode (demo): corrupt the answer before verification</span>
        </label>
      </form>

      <ErrorBox error={error} />
      {busy && <Spinner label="Retrieving, generating and verifying…" />}

      {res && !busy && (
        <>
          <div className="meta">
            <span className="chip">🧠 {res.model_name}</span>
            <span className="chip">logprobs: {res.logprobs_source}</span>
            <span className="chip">chunks {res.retrieved_chunk_ids.length} → {res.pruned_chunk_ids.length}</span>
          </div>

          <div className="grid-2">
            <Card title="Answer" className="wide">
              <p className="answer">{res.sentences.map((s) => <Sentence key={s.index ?? s.text} s={s} />)}</p>
              <div className="legend">
                <span><i className="sw supported" />Supported</span>
                <span><i className="sw neutral" />Neutral / wrong parts</span>
                <span><i className="sw hallucination" />Unsupported</span>
                <span><i className="sw filler" />Filler</span>
                <span><i className="sw mark" />Number/name not in source</span>
              </div>
            </Card>
            <Card title="Trust" className="trust">
              <Gauge value={res.trust_score_overall} />
              <Bar label="Entity" value={t.entity_level_score} />
              <Bar label="Sentence" value={t.sentence_level_score} />
              <Bar label="Document" value={t.document_level_score} />
              <Bar label="Token-prob" value={t.token_prob_component} />
              {t.token_prob_component == null && (
                <p className="muted">Token-probability unavailable (no logprobs); weights renormalized.</p>
              )}
            </Card>
          </div>

          {res.injected_faults?.length > 0 && <Faults res={res} />}

          {res.sentences.some((s) => s.is_hallucination && !s.is_filler) && (
            <Card title="Flagged sentences">
              {res.sentences.filter((s) => s.is_hallucination && !s.is_filler).map((s) => <Unsupported key={s.index ?? s.text} s={s} />)}
            </Card>
          )}

          <Card title="Entities">
            {res.entities.length ? (
              <div className="chips">
                {res.entities.map((e, i) => (
                  <span key={i} className={e.supported ? "chip good" : "chip badc"}>
                    {e.supported ? "✓" : "✕"} {e.text} <small>{e.entity_type}</small>
                  </span>
                ))}
              </div>
            ) : <p className="muted">No entities found.</p>}
          </Card>
        </>
      )}
    </div>
  );
}

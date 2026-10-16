import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { Card, ErrorBox, Spinner } from "./ui.jsx";

function GroupedBars({ rows }) {
  const metrics = [...new Set(rows.map((r) => r.metric))];
  const pipelines = [...new Set(rows.map((r) => r.pipeline))];
  const colors = { "NLI alone": "var(--accent)", "Full pipeline": "var(--accent2)" };
  return (
    <div className="gbars">
      {metrics.map((m) => (
        <div className="gb-group" key={m}>
          <div className="gb-bars">
            {pipelines.map((p) => {
              const row = rows.find((r) => r.metric === m && r.pipeline === p);
              if (!row) return null;
              return (
                <div className="gb-col" key={p} title={`${p}: ${(row.value * 100).toFixed(0)}%`}>
                  <small>{(row.value * 100).toFixed(0)}%</small>
                  <div className="gb-bar" style={{ height: `${row.value * 100}%`, background: colors[p] }} />
                </div>
              );
            })}
          </div>
          <span className="gb-label">{m}</span>
        </div>
      ))}
      <div className="gb-legend">
        {pipelines.map((p) => <span key={p}><i className="sw" style={{ background: colors[p] }} />{p}</span>)}
      </div>
    </div>
  );
}

export default function RobustnessTab() {
  const [r, setR] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try { setR(await api("GET", "/adversarial/results")); setError(""); }
    catch (e) { setError(`API error on /adversarial/results: ${e.message}`); }
  }, []);
  useEffect(() => { load(); }, [load]);

  async function run() {
    setBusy(true);
    try { await api("POST", "/adversarial/run", null, 1800000); await load(); }
    catch (e) { setError(`API error on /adversarial/run: ${e.message}`); }
    finally { setBusy(false); }
  }

  let rows = [];
  if (r?.n_tests) {
    const full = r.full_pipeline || {};
    rows = Object.entries(r.detection_rate).map(([k, v]) => ({ metric: `${k} detection`, pipeline: "NLI alone", value: v }));
    rows.push({ metric: "false positives", pipeline: "NLI alone", value: r.false_positive_rate });
    if (full.n_tests) {
      rows.push(...Object.entries(full.detection_rate).map(([k, v]) => ({ metric: `${k} detection`, pipeline: "Full pipeline", value: v })));
      rows.push({ metric: "false positives", pipeline: "Full pipeline", value: full.false_positive_rate });
    }
  }

  return (
    <div className="stack">
      <div className="hero">
        <h2>Adversarial robustness</h2>
        <p className="muted">The NLI verifier is tested against corrupted sentences. Higher detection is better; lower false positives is better.</p>
        <button className="primary" onClick={run} disabled={busy}>{busy ? "Running…" : "Run adversarial suite"}</button>
      </div>
      <ErrorBox error={error} />
      {busy && <Spinner label="Running attack suite…" />}
      {!busy && r && !r.n_tests && <Card><p className="muted">No adversarial results yet. Click the button above.</p></Card>}
      {!busy && rows.length > 0 && (
        <>
          <Card title="Detection by attack type"><GroupedBars rows={rows} /></Card>
          <Card title="Raw counts"><pre className="json">{JSON.stringify(r.counts, null, 2)}</pre></Card>
        </>
      )}
    </div>
  );
}

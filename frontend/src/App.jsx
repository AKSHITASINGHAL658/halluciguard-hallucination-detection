import { useState } from "react";
import QueryTab from "./components/QueryTab.jsx";
import RobustnessTab from "./components/RobustnessTab.jsx";
import EvaluationTab from "./components/EvaluationTab.jsx";

const TABS = [
  { id: "query", label: "Query", icon: "✦" },
  { id: "robust", label: "Robustness", icon: "⚔" },
  { id: "eval", label: "Evaluation", icon: "◎" },
];

export default function App() {
  const [tab, setTab] = useState("query");
  // Results live here so switching tabs doesn't throw them away.
  const [state, setState] = useState({ result: null, evalResult: null });
  const patch = (p) => setState((s) => ({ ...s, ...p }));

  return (
    <div className="app">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />
      <header className="topbar">
        <div className="brand">
          <span className="logo">🛡️</span>
          <div>
            <h1>HalluciGuard</h1>
            <p>Trust dashboard for grounded answers</p>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
              <span>{t.icon}</span> {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "query" && <QueryTab result={state.result} setResult={(result) => patch({ result })} />}
        {tab === "robust" && <RobustnessTab />}
        {tab === "eval" && <EvaluationTab result={state.evalResult} setResult={(evalResult) => patch({ evalResult })} />}
      </main>
    </div>
  );
}

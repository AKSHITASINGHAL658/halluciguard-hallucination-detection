import { useState } from "react";

export const scoreColor = (v) => (v >= 0.75 ? "var(--green)" : v >= 0.5 ? "var(--amber)" : "var(--red)");

export function Spinner({ label }) {
  return (
    <div className="loading">
      <div className="spinner" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorBox({ error }) {
  return error ? <div className="error">⚠ {error}</div> : null;
}

export function Card({ title, children, right, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <div className="card-head">
          <h2>{title}</h2>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Gauge({ value }) {
  const r = 70, c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value ?? 0));
  const color = scoreColor(v);
  return (
    <div className="gauge">
      <svg viewBox="0 0 180 180">
        <circle cx="90" cy="90" r={r} className="gauge-bg" />
        <circle
          cx="90" cy="90" r={r} className="gauge-fg" stroke={color}
          strokeDasharray={c} strokeDashoffset={c * (1 - v)} transform="rotate(-90 90 90)"
        />
      </svg>
      <div className="gauge-text">
        <b style={{ color }}>{Math.round(v * 100)}</b>
        <small>trust score</small>
      </div>
    </div>
  );
}

export function Bar({ label, value, color }) {
  const na = value == null;
  return (
    <div className="bar-row">
      <div className="bar-label"><span>{label}</span><b>{na ? "n/a" : value.toFixed(2)}</b></div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${(na ? 0 : value) * 100}%`, background: color || scoreColor(value) }} />
      </div>
    </div>
  );
}

export function Stat({ label, value, hint, why }) {
  return (
    <div className="stat">
      <small>{label}</small>
      <b>{value}</b>
      {hint && <em>{hint}</em>}
      {why && <p className="why">{why}</p>}
    </div>
  );
}

export function Collapse({ head, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={open ? "collapse open" : "collapse"}>
      <button className="collapse-head" onClick={() => setOpen(!open)}>
        <span>{head}</span>
        <i>{open ? "−" : "+"}</i>
      </button>
      {open && <div className="collapse-body">{children}</div>}
    </div>
  );
}

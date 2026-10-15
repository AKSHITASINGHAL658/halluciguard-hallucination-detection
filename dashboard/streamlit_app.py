import html
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.getenv("HALLUCIGUARD_API", "http://localhost:8000")
COLORS = {"supported": "#c8f7c5", "neutral": "#ffe9a8", "hallucination": "#ffc2c2", "filler": "#e3e3e8"}

st.set_page_config(page_title="HalluciGuard", layout="wide")
st.title("HalluciGuard — Trust Dashboard")
tab_query, tab_robust, tab_eval = st.tabs(["Query", "Robustness", "Evaluation"])


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 300), **kw)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        st.error(f"API error on {path}: {e}")
        return None


def sentence_html(s: dict) -> str:
    """Sentence with the wrong parts (numbers/names the source doesn't back up) marked red."""
    text, spans = s["text"], sorted(s.get("wrong_spans") or [], key=lambda x: x["start"])
    if s.get("is_filler"):
        return (f"<span title='No checkable claim' style='background:{COLORS['filler']};padding:2px 4px;margin-right:4px;"
                f"border-radius:4px;color:#555;font-style:italic'>{html.escape(text)}</span>")
    if not spans:
        return (f"<span style='background:{COLORS[sentence_class(s)]};padding:2px 4px;margin-right:4px;"
                f"border-radius:4px;color:#111'>{html.escape(text)}</span>")
    base = COLORS["neutral"]  # flagged by the wrong-parts check: amber, never green
    out, pos = [], 0
    for sp in spans:
        out.append(html.escape(text[pos:sp["start"]]))
        out.append(f"<span title='{html.escape(sp['reason'])}' style='background:#ff5c5c;color:#111;"
                   f"font-weight:700;padding:0 3px;border-radius:3px'>{html.escape(text[sp['start']:sp['end']])}</span>")
        pos = sp["end"]
    out.append(html.escape(text[pos:]))
    return (f"<span style='background:{base};padding:2px 4px;margin-right:4px;border-radius:4px;color:#111'>"
            f"{''.join(out)}</span>")


def sentence_class(s: dict) -> str:
    if s["is_hallucination"]:
        return "hallucination"
    return "supported" if s["nli_label"] == "entailment" else "neutral"


with tab_query:
    q = st.text_input("Ask a question about the corpus", "How much new solar capacity was added globally in 2023?")
    faulty = st.checkbox("Faulty mode (demo): corrupt the answer before verification",
                         help="Injects wrong numbers / negations into 1-2 sentences, then runs the normal checks.")
    if st.button("Run query") and q.strip():
        with st.spinner("Retrieving, generating and verifying..."):
            st.session_state["result"] = api("POST", "/query", json={"text": q, "inject_faults": faulty})
    res = st.session_state.get("result")
    if res:
        st.caption(f"Model: {res['model_name']} · logprobs source: {res['logprobs_source']} · "
                   f"chunks retrieved {len(res['retrieved_chunk_ids'])} → used {len(res['pruned_chunk_ids'])}")
        st.subheader("Answer")
        spans = "".join(sentence_html(s) for s in res["sentences"])
        st.markdown(spans, unsafe_allow_html=True)
        st.caption("Green = supported · Amber = neutral or has wrong parts · Grey italic = filler (no claim) · Red sentence = unsupported by NLI · Bright red words = numbers/names the source does not contain")

        if res.get("injected_faults"):
            st.warning("Faulty mode: the answer was deliberately corrupted. Did the checker catch it?")
            by_text = {x["text"]: x for x in res["sentences"]}
            for f in res["injected_faults"]:
                hit = by_text.get(f["corrupted"]) or next((x for x in res["sentences"] if x["index"] == f["index"]), None)
                caught = bool(hit and (hit["is_hallucination"] or hit.get("wrong_spans") or hit["nli_label"] != "entailment"))
                st.markdown(f"{'✅ Caught' if caught else '❌ Missed'} · **{f['attack_type']}** · "
                            f"~~{f['original']}~~ → {f['corrupted']}")
            st.caption("Token-probability is omitted here: logprobs describe the original answer, not the edited one.")

        for s in res["sentences"]:
            if s["is_hallucination"] and not s.get("is_filler"):
                with st.expander(f"🔴 Unsupported: {s['text'][:90]}"):
                    st.write(f"NLI: **{s['nli_label']}** (confidence {s['nli_confidence']:.2f})")
                    for sp in s.get("wrong_spans") or []:
                        st.write(f"🔴 **{sp['text']}**: {sp['reason']}")
                    st.write("Best source chunk:")
                    st.info(s["supporting_chunk_text"] or "n/a")
                    chk = s.get("attention_check")
                    if chk:
                        verdict = "✅ the words matter" if chk["supported"] else "⚠️ not clearly more important than random words"
                        st.caption(f"Masking check: hiding the top words moved the NLI probabilities by {chk['top_shift']:.2f} "
                                   f"(hiding {chk['n_random']}x random words: {chk['random_shift']:.2f}) → {verdict}")
                    if s["top_attention_tokens"]:
                        st.write("Source words the verifier attended to most:")
                        st.dataframe(pd.DataFrame(s["top_attention_tokens"]), hide_index=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Entities")
            for e in res["entities"]:
                badge = "✅" if e["supported"] else "❌"
                st.write(f"{badge} {e['text']} _({e['entity_type']})_")
            if not res["entities"]:
                st.write("No entities found.")
        with right:
            t = res["trust"]
            st.subheader(f"Trust score: {res['trust_score_overall']:.2f}")
            comp = {"Entity": t["entity_level_score"], "Sentence": t["sentence_level_score"],
                    "Document": t["document_level_score"], "Token-prob": t["token_prob_component"]}
            df = pd.DataFrame({"component": list(comp), "score": [v if v is not None else 0 for v in comp.values()]})
            st.plotly_chart(px.bar(df, x="component", y="score", range_y=[0, 1]), width="stretch")
            if t["token_prob_component"] is None:
                st.caption("Token-probability component unavailable (model returned no logprobs); weights renormalized.")

with tab_robust:
    st.write("Adversarial robustness: the NLI verifier is tested against corrupted sentences.")
    if st.button("Run adversarial suite"):
        with st.spinner("Running..."):
            api("POST", "/adversarial/run")
    r = api("GET", "/adversarial/results")
    if r and r["n_tests"]:
        full = r.get("full_pipeline") or {}
        rows = [{"metric": f"{k} detection", "pipeline": "NLI alone", "value": v} for k, v in r["detection_rate"].items()]
        rows.append({"metric": "false positives (control)", "pipeline": "NLI alone", "value": r["false_positive_rate"]})
        if full.get("n_tests"):
            rows += [{"metric": f"{k} detection", "pipeline": "Full pipeline", "value": v} for k, v in full["detection_rate"].items()]
            rows.append({"metric": "false positives (control)", "pipeline": "Full pipeline", "value": full["false_positive_rate"]})
        st.plotly_chart(px.bar(pd.DataFrame(rows), x="metric", y="value", color="pipeline", barmode="group", range_y=[0, 1]),
                        width="stretch")
        st.json(r["counts"])
    elif r:
        st.info("No adversarial results yet. Click the button above.")

with tab_eval:
    if st.button("Run evaluation"):
        with st.spinner("Evaluating on the QA set (this calls the LLM for each question)..."):
            st.session_state["eval"] = api("POST", "/evaluate", timeout=1800)
    ev = st.session_state.get("eval")
    if ev:
        c = st.columns(4)
        c[0].metric(f"Precision@{ev['k']}", f"{ev['retrieval_precision_at_k']:.2f}")
        c[1].metric("Precision after pruning", f"{ev['precision_after_pruning']:.2f}")
        c[2].metric("Hallucination rate", "n/a" if ev["hallucination_rate"] is None else f"{ev['hallucination_rate']:.2%}")
        c[3].metric("Avg prune ratio", f"{ev['avg_prune_ratio']:.2%}")
        st.metric("Average trust", "n/a" if ev["avg_trust"] is None else f"{ev['avg_trust']:.2f}")
        st.caption(f"{ev['n_questions']} questions, {ev['n_failed']} failed, {ev['n_sentences']} sentences checked")

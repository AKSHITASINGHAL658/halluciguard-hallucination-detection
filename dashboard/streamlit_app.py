import html
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.getenv("HALLUCIGUARD_API", "http://localhost:8000")
COLORS = {"supported": "#c8f7c5", "neutral": "#ffe9a8", "hallucination": "#ffc2c2"}

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
        spans = "".join(
            f"<span style='background:{COLORS[sentence_class(s)]};padding:2px 4px;margin-right:4px;border-radius:4px;"
            f"color:#111'>{html.escape(s['text'])}</span>" for s in res["sentences"])
        st.markdown(spans, unsafe_allow_html=True)
        st.caption("Green = supported · Amber = neutral · Red = hallucination")

        if res.get("injected_faults"):
            st.warning("Faulty mode: the answer was deliberately corrupted. Did the checker catch it?")
            by_text = {x["text"]: x for x in res["sentences"]}
            for f in res["injected_faults"]:
                hit = by_text.get(f["corrupted"]) or next((x for x in res["sentences"] if x["index"] == f["index"]), None)
                caught = bool(hit and (hit["is_hallucination"] or hit["nli_label"] != "entailment"))
                st.markdown(f"{'✅ Caught' if caught else '❌ Missed'} · **{f['attack_type']}** · "
                            f"~~{f['original']}~~ → {f['corrupted']}")
            st.caption("Token-probability is omitted here: logprobs describe the original answer, not the edited one.")

        for s in res["sentences"]:
            if s["is_hallucination"]:
                with st.expander(f"🔴 Unsupported: {s['text'][:90]}"):
                    st.write(f"NLI: **{s['nli_label']}** (confidence {s['nli_confidence']:.2f})")
                    st.write("Best source chunk:")
                    st.info(s["supporting_chunk_text"] or "n/a")
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
        rows = [{"metric": f"{k} detection rate", "value": v} for k, v in r["detection_rate"].items()]
        rows.append({"metric": "control false-positive rate", "value": r["false_positive_rate"]})
        st.plotly_chart(px.bar(pd.DataFrame(rows), x="metric", y="value", range_y=[0, 1]), width="stretch")
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

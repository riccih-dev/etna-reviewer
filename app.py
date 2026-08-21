"""Claim-Evidence Reviewer — Streamlit prototype.

Run with:
    streamlit run app.py

Uses MockLLMClient by default (rule-based, no API key). See
reviewer/llm_client.py to wire in a real model.
"""

from __future__ import annotations

import os

import streamlit as st

from reviewer.llm_client import GeminiLLMClient, MockLLMClient
from reviewer.pipeline import run_pipeline
from reviewer.pipeline.graph_builder import build_graph_data, render_combined_html
from reviewer.pipeline.pdf_ingest import extract_pdf_pages, parse_pdf_pages

SAMPLE_ADAPTER = """Abstract
We propose a lightweight adapter module for cross-lingual transfer. Across five benchmark tasks, accuracy improves significantly over the baseline, and the model requires 40% fewer trainable parameters than full fine-tuning.

Results
On CIFAR-10, our model achieves state-of-the-art accuracy, reaching 91.7% compared to 91.2% for the strongest baseline, averaged over five runs. For the low-resource transfer setting, the difference between the adapter and baseline conditions was not statistically significant. The classifier obtained a precision of 92.0% with TP = 110, FP = 20 on the held-out set.

Discussion
Beyond the tasks evaluated here, we believe our method generalizes across domains. All experiments in this section were conducted using only a single benchmark dataset."""

SAMPLE_SEIZURE = """Abstract
We introduce SeizureNet, a video-based architecture for detecting neonatal seizures without contact sensors. Our method significantly improves detection accuracy over existing approaches and generalizes across clinical domains. The attention module causes the observed robustness to occlusion. We report a precision of 0.95 on held-out recordings.

Introduction
Automated neonatal seizure detection would reduce the burden on clinical staff. Previous studies consistently demonstrate that video-based methods outperform contact-based monitoring. Existing systems remain limited to single-site data.

Methods
We train a spatiotemporal encoder on video clips of 10 seconds sampled at 25 frames per second. An attention module weights spatial regions before temporal pooling. Models were trained for 60 epochs with the Adam optimizer. Evaluation uses the Helsinki cohort, a set of continuous cot-side recordings annotated by two clinical experts.

Results
On the Helsinki cohort our model reached an accuracy of 0.91 compared with 0.90 for the baseline encoder. The difference between the two systems was not statistically significant. The confusion matrix contains TP = 110, FP = 20 and FN = 10. We observed a precision of 0.95 for the proposed system.

Discussion
The results show that SeizureNet generalizes across clinical domains and is robust to occlusion by blankets and caregivers. Because the attention module suppresses background motion, false positives due to caregiver movement are reduced. Our method is also computationally efficient and runs in real time on a single GPU.

Conclusion
SeizureNet substantially improves neonatal seizure detection and can be deployed across neonatal intensive care units."""

SAMPLES = {
    "Adapter / CIFAR-10 (ML)": SAMPLE_ADAPTER,
    "Neonatal seizure detection (clinical)": SAMPLE_SEIZURE,
}

SEVERITY_ORDER = {"major": 0, "minor": 1, "info": 2}

VERDICT_SEVERITY = {
    "CONTRADICTS": "major",
    "INSUFFICIENT_INFORMATION": "major",
    "PARTIALLY_SUPPORTS": "minor",
    "SUPPORTS": "info",
}
VERDICT_LABEL = {
    "CONTRADICTS": "Contradicts evidence",
    "INSUFFICIENT_INFORMATION": "No supporting evidence found",
    "PARTIALLY_SUPPORTS": "Partially supported",
    "SUPPORTS": "Supported",
}

st.set_page_config(page_title="ETNA", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    "<style>.block-container{padding-top:1rem;padding-bottom:0.5rem}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 style="color:#e8562a; margin-bottom:0.2rem">'
    "<span style=\"font-family: Georgia, 'Times New Roman', serif\">ETNA</span>"
    '<span style="font-family: inherit; font-weight: 400; font-size: 0.55em"> - Every Thesis Needs Anchoring</span>'
    "</h1>",
    unsafe_allow_html=True,
)


def _info(message: str) -> None:
    st.markdown(
        f'<div style="background:rgba(232,86,42,0.12);border:1px solid rgba(232,86,42,0.4);'
        f'border-radius:8px;padding:0.75rem 1rem;color:#f2a08a">{message}</div>',
        unsafe_allow_html=True,
    )


def _apply_sample():
    st.session_state["manuscript_text"] = SAMPLES[st.session_state["sample_choice"]]
    # file_uploader ignores a plain session_state pop; remounting it under a new key is the
    # reliable way to actually clear a previously uploaded file.
    st.session_state["uploader_key"] += 1


def _clear_text_on_upload():
    st.session_state["manuscript_text"] = ""


if "manuscript_text" not in st.session_state:
    st.session_state["manuscript_text"] = SAMPLE_ADAPTER
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

with st.sidebar:
    st.subheader("Input")
    st.selectbox(
        "Load a sample (or paste/upload your own below)",
        list(SAMPLES),
        key="sample_choice",
        on_change=_apply_sample,
    )
    text = st.text_area("Manuscript text", key="manuscript_text", height=220)
    uploaded_pdf = st.file_uploader(
        "...or upload a PDF",
        type=["pdf"],
        key=f"pdf_upload_{st.session_state['uploader_key']}",
        on_change=_clear_text_on_upload,
    )
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    backend = st.radio(
        "Model backend",
        ["Offline", "Gemini API"],
        index=1 if has_key else 0,
        help="Defaults to Gemini API automatically when GEMINI_API_KEY is set in the "
        "environment. Gemini API falls back to offline mode per call if the key is "
        "missing, rate-limited, or the request fails.",
    )
    if not has_key:
        st.caption(
            "No GEMINI_API_KEY found in the environment — defaulting to Offline."
        )

    st.subheader("Review settings")
    max_claims = st.slider(
        "Max claims reviewed",
        min_value=5,
        max_value=50,
        value=20,
        help="Caps how many extracted claims get processed. Keeps long documents fast and "
        "the findings list manageable; doesn't affect the deterministic checks.",
    )
    evidence_top_k = st.slider(
        "Evidence candidates per claim",
        min_value=1,
        max_value=8,
        value=4,
        help="How many candidate sentences are retrieved and scored per claim. The graph "
        "shows all of them as edges; only the top-ranked one drives the finding card.",
    )
    supports_threshold = st.slider(
        "Supported threshold",
        min_value=0.05,
        max_value=0.6,
        value=0.35,
        step=0.01,
        help="Minimum entailment score (word-overlap offline, model-judged score with "
        "Gemini) to mark a finding 'Supported'. Lower this if too few claims ever "
        "register as supported.",
    )
    partial_threshold = st.slider(
        "Partially-supported threshold",
        min_value=0.02,
        max_value=0.3,
        value=0.15,
        step=0.01,
        help="Minimum entailment score to mark a finding 'Partially supported' instead of "
        "'No supporting evidence found'.",
    )
    run = st.button("Run pipeline", type="primary")

display_llm = MockLLMClient()  # used for the "generic comparison" text only


if run:
    llm = MockLLMClient(supports_threshold, partial_threshold)
    if backend.startswith("Gemini"):
        try:
            llm = GeminiLLMClient(
                supports_threshold=supports_threshold,
                partial_threshold=partial_threshold,
            )
        except Exception as e:
            st.warning(
                f"Couldn't start the Gemini client ({e}); falling back to the mock model for this run."
            )
    if uploaded_pdf is not None:
        pages = extract_pdf_pages(uploaded_pdf)
        sentences = parse_pdf_pages(pages)
        st.session_state["result"] = run_pipeline(
            None,
            llm,
            sentences=sentences,
            max_claims=max_claims,
            evidence_top_k=evidence_top_k,
        )
        st.session_state["source_label"] = f"PDF: {uploaded_pdf.name}"
    else:
        st.session_state["result"] = run_pipeline(
            text, llm, max_claims=max_claims, evidence_top_k=evidence_top_k
        )
        st.session_state["source_label"] = "pasted text"
    if isinstance(llm, GeminiLLMClient):
        st.session_state["backend_label"] = (
            f"Gemini API — {llm.call_count - llm.fallback_count}/{llm.call_count} calls used the "
            f"real model, {llm.fallback_count} fell back to offline"
        )
    else:
        st.session_state["backend_label"] = "Offline"

if "result" in st.session_state:
    result = st.session_state["result"]
    findings = result["findings"]
    checks = result["checks"]

    with st.sidebar:
        st.subheader("Run info")
        st.caption(f"Source: {st.session_state.get('source_label', 'pasted text')}")
        st.caption(st.session_state.get("backend_label", "Offline"))

        st.subheader("All extracted sentences")
        with st.expander(
            f"{len(result['sentences'])} sentences with section/page provenance"
        ):
            for s in result["sentences"]:
                st.caption(f"`{s.id}` [{s.section}, p.{s.page}] {s.text}")

    graph_data = build_graph_data(
        result["claims"], result["sentences"], result["graph_edges"]
    )
    ASSUMPTION_TYPES = {"generalization", "causal"}

    cards = []
    for f in findings:
        no_evidence = f.verification.label == "INSUFFICIENT_INFORMATION"
        node_ids = [f.claim.id, f.evidence.id]
        if f.claim.type in ASSUMPTION_TYPES:
            node_ids.append(f"A_{f.claim.id}")
        evidence_text = f.evidence.text
        if no_evidence:
            evidence_text = f"Closest candidate found, but overlap is too low to count as support: “{evidence_text}”"
        cards.append(
            {
                "severity": VERDICT_SEVERITY[f.verification.label],
                "kind": f.verification.label.lower(),
                "badge": VERDICT_LABEL[f.verification.label],
                "locations": [f"{f.claim.section}, p.{f.claim.page}"],
                "claim_text": f.claim.text,
                "evidence_text": evidence_text,
                "missing": f.missing,
                "question": f.question,
                "action": f.action,
                "rationale": f.rationale,
                "confidence": f.confidence,
                "node_ids": node_ids,
            }
        )
    graph_node_ids = {n["id"] for n in graph_data["nodes"]}
    for c in checks:
        # Only link to the graph when the check names one specific claim (claim_id set);
        # `locations` is a coarse "section, p.page" string shared by many unrelated
        # sentences on that page, so matching by location alone falsely pulls in every
        # other claim/evidence node that happens to sit on the same page.
        node_ids = []
        if c.claim_id:
            node_ids.append(c.claim_id)
            if f"A_{c.claim_id}" in graph_node_ids:
                node_ids.append(f"A_{c.claim_id}")
        cards.append(
            {
                "severity": c.severity,
                "kind": c.kind,
                "badge": c.kind.replace("_", " ").title(),
                "locations": c.locations,
                "claim_text": None,
                "evidence_text": None,
                "missing": c.missing,
                "question": c.question,
                "action": c.action,
                "rationale": c.rationale or c.detail,
                "confidence": None,
                "node_ids": node_ids,
            }
        )
    cards.sort(key=lambda c: SEVERITY_ORDER.get(c["severity"], 9))

    counts = {
        sev: sum(1 for c in cards if c["severity"] == sev)
        for sev in ("major", "minor", "info")
    }
    st.markdown(
        f'<h3 style="color:#e8562a">Findings ({len(cards)})</h3>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"🚩 {counts['major']} major · ⚠️ {counts['minor']} minor · ✅ {counts['info']} supported/confirmed "
        f"· {len(graph_data['nodes'])} graph nodes, {len(graph_data['edges'])} edges"
    )

    if not cards:
        _info(
            "No claim-cue phrases matched, or no evidence could be retrieved. Try a longer excerpt."
        )
    else:
        st.iframe(render_combined_html(cards, graph_data, height=680), height=700)
else:
    _info("Paste text, pick a sample, or upload a PDF, then click <b>Run pipeline</b>.")

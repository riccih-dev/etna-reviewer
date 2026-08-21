"""Wires the seven stages together: parse -> extract -> retrieve -> verify ->
graph -> checks -> review. Each stage is a plain function/module so any one
of them can be swapped independently.
"""

from __future__ import annotations

from ..llm_client import LLMClient
from ..models import Claim, Evidence, Finding, Sentence, Verification
from .checkers import run_checkers
from .confidence import score_finding
from .parse import parse_manuscript
from .retrieval import retrieve_evidence
from .stats_checker import run_stats_checks


def _review_fields(claim: Claim, evidence: Evidence, verification: Verification) -> dict:
    """What a reviewer needs per finding: what's missing, what to ask, what to do, and why."""
    if verification.label == "SUPPORTS":
        return {
            "missing": "", "question": "", "action": "",
            "rationale": verification.rationale,
        }
    if verification.label == "PARTIALLY_SUPPORTS":
        return {
            "missing": "Evidence covering the full scope of the claim, not just part of it.",
            "question": f"Does the retrieved evidence fully establish “{claim.text}”, or only a narrower version of it?",
            "action": "Narrow the claim's wording to match what was actually shown, or provide evidence for the broader scope.",
            "rationale": verification.rationale,
        }
    if verification.label == "CONTRADICTS":
        return {
            "missing": "A reconciliation between the claim and the contradicting evidence.",
            "question": f"How do the authors reconcile “{claim.text}” with “{evidence.text}”?",
            "action": "Resolve the inconsistency: correct the claim, correct the evidence description, or explain the discrepancy.",
            "rationale": verification.rationale,
        }
    return {  # INSUFFICIENT_INFORMATION
        "missing": "A manuscript passage that directly supports this claim.",
        "question": f"What evidence in the manuscript supports “{claim.text}”?",
        "action": "Add an explicit result, citation, or analysis that supports the claim, or soften/remove it.",
        "rationale": verification.rationale,
    }


def run_pipeline(raw_text: str | None, llm: LLMClient, sentences: list[Sentence] | None = None,
                  max_claims: int | None = None, evidence_top_k: int = 4) -> dict:
    if sentences is None:
        sentences = parse_manuscript(raw_text or "")

    raw_claims = llm.extract_claims(sentences)
    if max_claims is not None:
        raw_claims = raw_claims[:max_claims]
    claims: list[Claim] = []
    for i, rc in enumerate(raw_claims, start=1):
        source = next(s for s in sentences if s.id == rc["sentence_id"])
        claims.append(Claim(
            id=f"C{i}", text=rc["text"], type=rc["type"], sentence_id=source.id,
            section=source.section, page=source.page,
        ))

    findings: list[Finding] = []
    graph_edges: list[tuple[Claim, object, Verification]] = []  # every scored (claim, evidence) pair, for the graph

    for claim in claims:
        evidence_list = retrieve_evidence(claim, sentences, top_k=evidence_top_k)
        if not evidence_list:
            continue

        # One batched judgment call per claim (all its candidates at once) rather than one
        # call per candidate — keeps a real API backend's request count down to ~N instead
        # of ~N*evidence_top_k, which matters a lot for free-tier rate limits.
        verdicts = llm.classify_verification_batch(claim.text, claim.type, [e.text for e in evidence_list])
        for i, (evidence, (label, entailment_score, rationale)) in enumerate(zip(evidence_list, verdicts)):
            verification = Verification(
                claim_id=claim.id, evidence_id=evidence.id, label=label,
                entailment_score=entailment_score, rationale=rationale,
            )
            graph_edges.append((claim, evidence, verification))

            if i == 0:  # the strongest match drives the finding shown in the review panel
                confidence, breakdown = score_finding(claim, evidence, verification)
                review_text = llm.write_review_sentence(claim, evidence, verification)
                findings.append(Finding(
                    claim=claim, evidence=evidence, verification=verification,
                    confidence=confidence, confidence_breakdown=breakdown, review_text=review_text,
                    **_review_fields(claim, evidence, verification),
                ))

    checks = run_stats_checks(sentences) + run_checkers(sentences, claims)

    return {
        "sentences": sentences,
        "claims": claims,
        "findings": findings,
        "graph_edges": graph_edges,
        "checks": checks,
    }

"""Confidence scoring — deterministic, not the LLM's self-reported number.

confidence = w1*retrieval + w2*entailment + w3*provenance + w4*agreement

`agreement` is a proxy for "would independent signals agree": how close the
retrieval score and the entailment score are to each other. Weights are
illustrative — in a real system you'd fit/calibrate them against
human-labeled validation data rather than hand-set them.
"""

from __future__ import annotations

from ..models import Claim, Evidence, Verification

WEIGHTS = {"retrieval": 0.30, "entailment": 0.35, "provenance": 0.15, "agreement": 0.20}


def score_finding(claim: Claim, evidence: Evidence, verification: Verification) -> tuple[float, dict]:
    retrieval = min(evidence.retrieval_score * 1.6, 1.0)  # TF-IDF cosine sims run low; rescale for readability
    entailment = verification.entailment_score
    provenance = 1.0 if claim.section and claim.page and evidence.section and evidence.page else 0.5
    agreement = 1.0 - abs(retrieval - entailment)

    components = {
        "Retrieval": round(retrieval, 2),
        "Entailment": round(entailment, 2),
        "Provenance": round(provenance, 2),
        "Agreement": round(agreement, 2),
    }
    total = sum(WEIGHTS[k.lower()] * v for k, v in components.items())
    return round(total, 2), components

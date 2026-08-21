"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sentence:
    id: str
    text: str
    section: str       # most specific heading, e.g. "4.1 Classification Performance" — used for citations
    page: int
    top_section: str = ""  # nearest top-level heading, e.g. "Results" — used by section-type checks

    def __post_init__(self):
        if not self.top_section:
            self.top_section = self.section


@dataclass
class Claim:
    id: str
    text: str
    type: str  # performance | generalization | causal | efficiency | citation | contribution
    sentence_id: str
    section: str
    page: int


@dataclass
class Evidence:
    id: str
    text: str
    sentence_id: str
    section: str
    page: int
    retrieval_score: float


@dataclass
class Verification:
    claim_id: str
    evidence_id: str
    label: str  # SUPPORTS | PARTIALLY_SUPPORTS | CONTRADICTS | INSUFFICIENT_INFORMATION
    entailment_score: float
    rationale: str


@dataclass
class CheckFinding:
    kind: str
    severity: str  # major | minor | info
    detail: str
    locations: list[str]
    missing: str = ""    # what evidence the system expected but couldn't find
    question: str = ""   # the concrete question a reviewer should ask
    action: str = ""     # what the authors could do to resolve it
    rationale: str = ""  # why the system raised this — lets a reviewer push back
    claim_id: str | None = None  # set only when the check concerns one specific claim node


@dataclass
class Finding:
    claim: Claim
    evidence: Evidence
    verification: Verification
    confidence: float
    confidence_breakdown: dict = field(default_factory=dict)
    review_text: str = ""
    missing: str = ""
    question: str = ""
    action: str = ""
    rationale: str = ""

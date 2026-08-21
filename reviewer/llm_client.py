"""Pluggable LLM interface.

MockLLMClient runs the whole pipeline with zero dependencies and no API key —
it uses cue-phrase rules and lexical overlap instead of a real model, so the
quality is intentionally limited. Swap in AnthropicLLMClient (or write your
own) to get real claim extraction, entailment and review prose; every other
module only talks to this interface, so nothing else has to change.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .models import Sentence

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "that",
    "this",
    "these",
    "those",
    "we",
    "our",
    "as",
    "by",
    "at",
    "from",
    "it",
    "its",
    "than",
    "over",
    "across",
    "than",
    "than",
    "not",
    "no",
    "which",
    "than",
}

CLAIM_CUES = [
    "we propose",
    "we introduce",
    "our method",
    "our model",
    "our approach",
    "achieves",
    "achieve",
    "outperform",
    "outperforms",
    "state-of-the-art",
    "state of the art",
    "generalizes",
    "generalize",
    "significantly",
    "demonstrate",
    "demonstrates",
    "show that",
    "shows that",
    "novel",
    "efficient",
    "fewer parameters",
    "faster than",
    "improves",
    "improve",
    "causes",
    "leads to",
    "results in",
    "due to the",
]

TYPE_RULES = [
    ("generalization", ["generaliz"]),
    (
        "performance",
        ["outperform", "accuracy", "state-of-the-art", "state of the art", "sota"],
    ),
    (
        "causal",
        [
            "because",
            "due to",
            "causes",
            "leads to",
            "caused by",
            "causal",
            "is responsible for",
            "causal mechanism",
            "causal driver",
        ],
    ),
    ("efficiency", ["efficient", "faster", "fewer parameters", "runtime", "latency"]),
    ("citation", ["previous work", "prior work", "prior studies", "as shown in"]),
    ("contribution", ["we propose", "we introduce", "novel"]),
]

NEGATION_CUES = [
    "not significant",
    "no significant",
    "did not",
    "does not",
    "failed to",
    "no evidence",
    "not statistically",
]
POSITIVE_ASSERTIONS = [
    "significant",
    "outperform",
    "improves",
    "improve",
    "generalizes",
    "generalize",
    "better",
    "state-of-the-art",
    "sota",
]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def classify_claim_type(text: str) -> str:
    low = text.lower()
    for claim_type, keywords in TYPE_RULES:
        if any(k in low for k in keywords):
            return claim_type
    return "empirical"


class LLMClient(ABC):
    """Interface every pipeline stage that needs "understanding" talks to."""

    @abstractmethod
    def extract_claims(self, sentences: list[Sentence]) -> list[dict]:
        """Return [{sentence_id, text, type}] for sentences worth reviewing."""

    @abstractmethod
    def classify_verification(
        self, claim_text: str, claim_type: str, evidence_text: str
    ) -> tuple[str, float, str]:
        """Return (label, entailment_score, rationale) for a claim/evidence pair."""

    def classify_verification_batch(
        self,
        claim_text: str,
        claim_type: str,
        evidence_texts: list[str],
    ) -> list[tuple[str, float, str]]:
        """Judge every retrieved evidence candidate for one claim.

        Default: one classify_verification call per candidate — fine for Mock (instant,
        local). Real API backends should override this to make one request per claim
        instead of one per candidate; see GeminiLLMClient.
        """
        return [
            self.classify_verification(claim_text, claim_type, t)
            for t in evidence_texts
        ]

    @abstractmethod
    def write_review_sentence(self, claim, evidence, verification) -> str:
        """Turn a verified finding into reviewer-facing prose."""

    @abstractmethod
    def write_generic_review_sentence(self, claim) -> str:
        """For comparison only: what an ungrounded, claim-blind reviewer might say."""


class MockLLMClient(LLMClient):
    """Rule-based stand-in. No network calls, no API key, deterministic."""

    def __init__(
        self, supports_threshold: float = 0.35, partial_threshold: float = 0.15
    ):
        self.supports_threshold = supports_threshold
        self.partial_threshold = partial_threshold

    def extract_claims(self, sentences: list[Sentence]) -> list[dict]:
        claims = []
        for s in sentences:
            low = s.text.lower()
            if any(cue in low for cue in CLAIM_CUES):
                claims.append(
                    {
                        "sentence_id": s.id,
                        "text": s.text.strip(),
                        "type": classify_claim_type(s.text),
                    }
                )
        return claims

    def classify_verification(
        self, claim_text: str, claim_type: str, evidence_text: str
    ) -> tuple[str, float, str]:
        claim_tokens = _tokens(claim_text)
        evid_tokens = _tokens(evidence_text)
        if not claim_tokens or not evid_tokens:
            return (
                "INSUFFICIENT_INFORMATION",
                0.0,
                "No usable overlap between claim and evidence text.",
            )

        overlap = len(claim_tokens & evid_tokens) / len(claim_tokens | evid_tokens)
        low_claim, low_evid = claim_text.lower(), evidence_text.lower()

        negated = any(cue in low_evid for cue in NEGATION_CUES)
        asserts_positive = any(cue in low_claim for cue in POSITIVE_ASSERTIONS)
        if negated and asserts_positive:
            rationale = (
                f"Evidence contains a negation cue ('{next(c for c in NEGATION_CUES if c in low_evid)}') "
                f"while the claim asserts a positive result. Lexical overlap = {overlap:.2f}."
            )
            return "CONTRADICTS", round(0.75 + 0.2 * overlap, 2), rationale

        if claim_type == "generalization" and any(
            phrase in low_evid
            for phrase in ["one dataset", "single domain", "a single", "only evaluated"]
        ):
            rationale = f"Evidence describes a single-dataset/domain result; claim asserts broader generalization. Overlap = {overlap:.2f}."
            return "PARTIALLY_SUPPORTS", round(0.6 + 0.3 * overlap, 2), rationale

        if overlap >= self.supports_threshold:
            label = "SUPPORTS"
        elif overlap >= self.partial_threshold:
            label = "PARTIALLY_SUPPORTS"
        else:
            label = "INSUFFICIENT_INFORMATION"

        rationale = f"Lexical overlap between claim and evidence = {overlap:.2f} (rule-based heuristic, not a trained entailment model)."
        return label, round(overlap, 2), rationale

    def write_review_sentence(self, claim, evidence, verification) -> str:
        loc = f"{claim.section}, p.{claim.page}"
        eloc = f"{evidence.section}, p.{evidence.page}"
        quote = evidence.text.strip()
        if verification.label == "SUPPORTS":
            return f"Supported. The claim in {loc} is backed by {eloc}: “{quote}”."
        if verification.label == "PARTIALLY_SUPPORTS":
            return (
                f"Broader than the evidence shown. The claim in {loc} is only supported by {eloc}: "
                f"“{quote}” — consider narrowing the claim or adding further evidence."
            )
        if verification.label == "CONTRADICTS":
            return (
                f"Possible inconsistency: the claim in {loc} appears contradicted by {eloc}: "
                f"“{quote}”. Recommend reconciling the two statements."
            )
        return (
            f"No sentence in the manuscript clearly supports or refutes the claim in {loc}. "
            f"Closest candidate ({eloc}, overlap={verification.entailment_score:.2f}): “{quote}”. "
            "Consider adding explicit supporting evidence."
        )

    def write_generic_review_sentence(self, claim) -> str:
        generic_by_type = {
            "performance": "The experimental results look solid and support the paper's claims.",
            "generalization": "More experiments are needed to establish robustness.",
            "causal": "The causal mechanism could be explained in more depth.",
            "efficiency": "The efficiency gains are interesting but could be validated further.",
            "citation": "The related work section could engage more with prior literature.",
            "contribution": "The contribution is reasonable but the novelty could be clarified.",
        }
        return generic_by_type.get(
            claim.type, "This claim could be strengthened with additional evidence."
        )


class AnthropicLLMClient(LLMClient):
    """Real implementation using the Anthropic API. Requires ANTHROPIC_API_KEY.

    Not used by default. Wire it in via app.py's sidebar once you want real
    claim extraction / entailment / review prose instead of the mock rules.
    """

    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic  # imported lazily so MockLLMClient never needs this installed

        self.client = anthropic.Anthropic()
        self.model = model

    def _ask(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if hasattr(block, "text"))

    def extract_claims(self, sentences: list[Sentence]) -> list[dict]:
        raise NotImplementedError(
            "Prototype scaffold: implement a structured-output prompt over `sentences` "
            "that returns [{sentence_id, text, type}] and parse the JSON response."
        )

    def classify_verification(
        self, claim_text: str, claim_type: str, evidence_text: str
    ) -> tuple[str, float, str]:
        raise NotImplementedError(
            "Prototype scaffold: prompt the model to label (claim_text, evidence_text) as "
            "SUPPORTS / PARTIALLY_SUPPORTS / CONTRADICTS / INSUFFICIENT_INFORMATION with a rationale."
        )

    def write_review_sentence(self, claim, evidence, verification) -> str:
        raise NotImplementedError(
            "Prototype scaffold: prompt the model with the verified finding only."
        )

    def write_generic_review_sentence(self, claim) -> str:
        raise NotImplementedError("Prototype scaffold: for comparison only.")


class GeminiLLMClient(LLMClient):
    """Real implementation using the Google Gemini API. Requires GEMINI_API_KEY.

    Every method falls back to MockLLMClient's rule-based logic on any failure —
    missing/invalid key, rate limiting, network error, or a malformed response —
    so a free-tier hiccup degrades the review quality instead of crashing the run.
    """

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        supports_threshold: float = 0.35,
        partial_threshold: float = 0.15,
    ):
        from google import (
            genai,
        )  # imported lazily so MockLLMClient never needs this installed

        self._client = genai.Client()  # reads GEMINI_API_KEY from the environment
        self._model = model
        self.supports_threshold = supports_threshold
        self.partial_threshold = partial_threshold
        self._fallback = MockLLMClient(supports_threshold, partial_threshold)
        # Every call counts here so the UI can show whether Gemini actually ran or
        # silently fell back on every single call (e.g. a bad key or rate limit),
        # which otherwise looks identical to the offline results with no explanation.
        self.call_count = 0
        self.fallback_count = 0

    def _generate_json(self, prompt: str):
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)

    def _generate_text(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model, contents=prompt
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("empty response from Gemini")
        return text

    def extract_claims(self, sentences: list[Sentence]) -> list[dict]:
        self.call_count += 1
        try:
            numbered = "\n".join(f"{s.id}: {s.text}" for s in sentences)
            prompt = (
                "You are helping a peer reviewer scan a manuscript sentence by sentence. "
                "Identify sentences that make a checkable claim (a result, contribution, "
                "generalization, causal statement, efficiency claim, or citation-based "
                "assertion) worth a reviewer's scrutiny. Ignore purely descriptive sentences.\n\n"
                "Return ONLY a JSON array of objects, each: "
                '{"sentence_id": <id>, "text": <the sentence>, "type": one of '
                '"performance", "generalization", "causal", "efficiency", "citation", '
                '"contribution", "empirical"}.\n\n'
                f"Sentences:\n{numbered}"
            )
            claims = self._generate_json(prompt)
            if not isinstance(claims, list):
                raise ValueError("expected a JSON array of claims")
            return claims
        except Exception:
            self.fallback_count += 1
            return self._fallback.extract_claims(sentences)

    def classify_verification(
        self, claim_text: str, claim_type: str, evidence_text: str
    ) -> tuple[str, float, str]:
        return self.classify_verification_batch(
            claim_text, claim_type, [evidence_text]
        )[0]

    def classify_verification_batch(
        self,
        claim_text: str,
        claim_type: str,
        evidence_texts: list[str],
    ) -> list[tuple[str, float, str]]:
        """Judge every retrieved evidence candidate for this claim in one request instead of
        one request per candidate — the free tier's per-minute rate limit is hit fast
        otherwise (N claims x K candidates each adds up quickly)."""
        self.call_count += len(evidence_texts)
        try:
            numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(evidence_texts))
            prompt = (
                "Judge whether each numbered EVIDENCE sentence from a manuscript supports the "
                "CLAIM sentence. Return ONLY a JSON array, same length and order as the "
                'evidence list, each element: {"contradicts": true/false (does the evidence '
                "actively contradict the claim, e.g. reports a null/negative result where the "
                'claim asserts a positive one?), "score": <number 0 to 1, how strongly the '
                "evidence semantically entails the claim, independent of word overlap>, "
                '"rationale": <one sentence>}.\n\n'
                f"Claim type: {claim_type}\nClaim: {claim_text}\n\nEvidence:\n{numbered}"
            )
            results = self._generate_json(prompt)
            if not isinstance(results, list) or len(results) != len(evidence_texts):
                raise ValueError(
                    f"expected {len(evidence_texts)} JSON results, got {results!r}"
                )
            out = []
            for r in results:
                score = float(r["score"])
                rationale = str(r["rationale"])
                # The threshold sliders decide the SUPPORTS/PARTIALLY/INSUFFICIENT boundary for
                # both backends alike; only contradiction is left to the model's own judgment,
                # since it isn't a point on the same support-strength scale.
                if r.get("contradicts"):
                    label = "CONTRADICTS"
                elif score >= self.supports_threshold:
                    label = "SUPPORTS"
                elif score >= self.partial_threshold:
                    label = "PARTIALLY_SUPPORTS"
                else:
                    label = "INSUFFICIENT_INFORMATION"
                out.append((label, round(score, 2), rationale))
            return out
        except Exception:
            self.fallback_count += len(evidence_texts)
            return [
                self._fallback.classify_verification(claim_text, claim_type, t)
                for t in evidence_texts
            ]

    def write_review_sentence(self, claim, evidence, verification) -> str:
        self.call_count += 1
        try:
            prompt = (
                "Write one concise, concrete peer-review sentence about this claim/evidence "
                "pair, in the voice of a careful reviewer. Reference the verdict directly. "
                "Reply with only the sentence, no preamble.\n\n"
                f"Claim ({claim.section}, p.{claim.page}): {claim.text}\n"
                f"Evidence ({evidence.section}, p.{evidence.page}): {evidence.text}\n"
                f"Verdict: {verification.label} (score {verification.entailment_score:.2f})"
            )
            return self._generate_text(prompt)
        except Exception:
            self.fallback_count += 1
            return self._fallback.write_review_sentence(claim, evidence, verification)

    def write_generic_review_sentence(self, claim) -> str:
        self.call_count += 1
        try:
            prompt = (
                "In one sentence, write the kind of generic, ungrounded comment a rushed "
                "reviewer who hasn't checked the evidence might make about this claim. "
                "Reply with only the sentence, no preamble.\n\n"
                f"Claim: {claim.text}"
            )
            return self._generate_text(prompt)
        except Exception:
            self.fallback_count += 1
            return self._fallback.write_generic_review_sentence(claim)

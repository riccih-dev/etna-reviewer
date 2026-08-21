"""Stage 6a — deterministic specialist checker (statistics).

No model involved on purpose: confusion counts (TP/FP/FN/TN) are collected
from anywhere in the manuscript, precision/recall/F1/accuracy are recomputed
from them, and every restatement of those metrics — as "92.0%" or as a bare
fraction "0.95", possibly in a different sentence or section than the counts
— is checked against the recomputed value. This is the "don't ask an LLM to
do arithmetic" half of the pipeline.
"""

from __future__ import annotations

import re

from ..models import CheckFinding, Sentence

COUNT_PATTERN = re.compile(r"\b(TP|FP|FN|TN)\s*=\s*(\d+)", re.IGNORECASE)
METRIC_PATTERN = re.compile(
    r"\b(precision|recall|f1(?:-score)?|accuracy)\D{0,20}?(\d+(?:\.\d+)?)\s*(%)?",
    re.IGNORECASE,
)


def _loc(s: Sentence) -> str:
    return f"{s.section}, p.{s.page}"


def _derive_metrics(counts: dict) -> dict:
    tp, fp, fn, tn = counts.get("TP"), counts.get("FP"), counts.get("FN"), counts.get("TN")
    derived = {}
    if tp is not None and fp is not None and (tp + fp) > 0:
        derived["precision"] = 100 * tp / (tp + fp)
    if tp is not None and fn is not None and (tp + fn) > 0:
        derived["recall"] = 100 * tp / (tp + fn)
    if "precision" in derived and "recall" in derived and (derived["precision"] + derived["recall"]) > 0:
        p, r = derived["precision"], derived["recall"]
        f1 = 2 * p * r / (p + r)
        derived["f1"] = f1
        derived["f1-score"] = f1
    if None not in (tp, fp, fn, tn):
        total = tp + fp + fn + tn
        if total > 0:
            derived["accuracy"] = 100 * (tp + tn) / total
    return derived


def _normalize(raw: float, has_percent_sign: bool) -> float:
    """Reported values show up either as "92.0%" or as a bare fraction "0.95"."""
    if has_percent_sign or raw > 1.5:
        return raw
    return raw * 100


def run_stats_checks(sentences: list[Sentence], tolerance: float = 1.0) -> list[CheckFinding]:
    counts: dict[str, int] = {}
    count_hits: list[Sentence] = []
    for s in sentences:
        for name, val in COUNT_PATTERN.findall(s.text):
            counts[name.upper()] = int(val)
            if s not in count_hits:
                count_hits.append(s)
    if not counts:
        return []

    derived = _derive_metrics(counts)
    if not derived:
        return []

    mismatches: dict[tuple, list[Sentence]] = {}
    matches: dict[tuple, list[Sentence]] = {}
    for s in sentences:
        for metric, raw_value, pct_sign in METRIC_PATTERN.findall(s.text):
            key = metric.lower().replace("-score", "")
            if key not in derived:
                continue
            reported = _normalize(float(raw_value), bool(pct_sign))
            bucket = mismatches if abs(derived[key] - reported) > tolerance else matches
            bucket.setdefault((key, round(reported, 1)), []).append(s)

    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    findings: list[CheckFinding] = []
    for (metric, reported), hits in mismatches.items():
        locs = list(dict.fromkeys(_loc(s) for s in hits + count_hits))
        findings.append(CheckFinding(
            kind="statistics_mismatch", severity="major", locations=locs,
            detail=(
                f"Reported {metric.title()} = {reported:.1f}% but {counts_str} implies "
                f"{derived[metric]:.1f}%. Restated at: " + "; ".join(_loc(s) for s in hits) + "."
            ),
            missing="Consistent arithmetic between the reported metric and the reported confusion counts.",
            question=f"Which is correct — the reported {metric.title()} of {reported:.1f}%, or the {derived[metric]:.1f}% implied by the reported counts ({counts_str})?",
            action=f"Recheck and correct the discrepancy between the reported {metric.title()} and the underlying counts.",
            rationale="The reported value and the reported counts are mutually inconsistent; recomputing precision/recall/F1 from confusion counts is deterministic arithmetic, not a judgment call.",
        ))
    for (metric, reported), hits in matches.items():
        locs = list(dict.fromkeys(_loc(s) for s in hits + count_hits))
        findings.append(CheckFinding(
            kind="statistics_confirmed", severity="info", locations=locs,
            detail=f"Reported {metric.title()} = {reported:.1f}% matches recomputed value from {counts_str}.",
            rationale="Recomputed independently from the reported confusion counts; no discrepancy found.",
        ))
    return findings

"""Stage 6b — deterministic specialist checkers beyond arithmetic.

Same philosophy as stats_checker.py: these run over the parsed sentences
directly, no LLM. Adapted from a more complete "ceg_reviewer" prototype's
checkers.py — the arithmetic and contradiction ideas stay, the retrieval-
graph machinery around them doesn't (that lived in modules this project
doesn't have; these versions work straight off sentences and claims).

Every finding fills in the four fields a reviewer actually acts on:
missing (what evidence the system expected but couldn't find), question
(what to ask the authors), action (what would resolve it), and rationale
(why the system raised it, so the reviewer can push back on the system).
"""

from __future__ import annotations

import re

from ..models import CheckFinding, Claim, Sentence

PVAL = re.compile(r"\bp\s*[<=>]\s*0?\.\d+", re.I)
CI = re.compile(r"\b(95\s*%\s*CI|confidence interval|±|\bIQR\b)", re.I)
NSIZE = re.compile(r"\bn\s*=\s*\d+|\b\d+\s+(patients|subjects|participants|recordings|samples|images)\b", re.I)
POSITIVE_CLAIM = re.compile(r"significantly|substantially|clearly|consistently|markedly", re.I)
ABLATION = re.compile(
    r"\bablation\b|\bwe remove\b|\bintervention\b|\brandomi[sz]e[ds]?\b|"
    r"\bcontrolled (experiment|for)\b|\bcounterfactual\b",
    re.I,
)
# "we do not conduct an ablation" contains the word "ablation" but is not evidence of one —
# a sentence-level negation check keeps that from being read as a positive signal.
NEGATION_CUE = re.compile(r"\b(no|not|n't|nor|without|lack(?:ing)?)\b", re.I)
CAUSAL_SUBJECT = re.compile(
    r"\bthe ([a-z][\w\- ]{2,40}?)\s+(?:causes|is the causal (?:mechanism|driver)|"
    r"is responsible for|suppresses)",
    re.I,
)
KNOWN_DATASET_NAMES = re.compile(
    r"\b(CIFAR-?\d*|ImageNet|MNIST|COCO|MIMIC[- ]?[A-Z0-9]*|PhysioNet|UCI|SQuAD|GLUE)\b"
)
DATASET_PAT = re.compile(  # "<Name> dataset/cohort/..." — a proper noun followed by the generic word
    r"\b([A-Z][A-Za-z0-9-]+)\s+(?:data\s?set|dataset|corpus|cohort|benchmark|split)\b"
)
DATASET_PAT_PREFIX = re.compile(  # "dataset/cohort ... <Name>" — the generic word first
    r"\b(?:dataset|cohort|corpus|benchmark)\s+(?:called\s+|named\s+)?([A-Z][A-Za-z0-9-]+)\b"
)
DATASET_STOP = {
    "the", "our", "this", "these", "each", "both", "two", "three", "all", "same", "other",
    "independent", "external", "internal", "public", "private", "large", "small", "full",
    "whole", "entire", "training", "test", "validation", "held", "first", "second", "new",
}
LEAKAGE_PAT = re.compile(
    r"combined training and test|training and test (images|data|set)s?\b.{0,40}"
    r"(normali[sz]ation|preprocess|statistic|mean and (std|standard))",
    re.I,
)
TEST_SELECTION_PAT = re.compile(
    r"(selected|chosen|choose|choosing).{0,40}(test.set accuracy|test.set performance|"
    r"test accuracy|test performance)",
    re.I,
)
VALIDATION_MENTION = re.compile(r"\bvalidation\b|\bdev(elopment)? set\b|\bheld.out validation\b", re.I)

SUMMARY_SECTIONS = {"Abstract", "Results", "Discussion", "Conclusion", "Conclusions"}


def _loc(s: Sentence) -> str:
    return f"{s.section}, p.{s.page}"


def _causal_subject(text: str) -> str:
    m = CAUSAL_SUBJECT.search(text)
    return m.group(1).strip() if m else "the proposed mechanism"


def datasets_mentioned(sentences: list[Sentence]) -> list[str]:
    found: list[str] = []

    def add(name: str):
        name = name.strip()
        if name.lower() in DATASET_STOP or len(name) <= 2:
            return
        if name.lower() not in {n.lower() for n in found}:
            found.append(name)

    for s in sentences:
        for name in KNOWN_DATASET_NAMES.findall(s.text):
            add(name)
        for name in DATASET_PAT.findall(s.text):
            add(name)
        for name in DATASET_PAT_PREFIX.findall(s.text):
            add(name)
    return found


def check_uncertainty_reporting(sentences: list[Sentence]) -> list[CheckFinding]:
    """Significance language with no p-value, CI, or sample size anywhere."""
    hits = [s for s in sentences if POSITIVE_CLAIM.search(s.text) and s.top_section in SUMMARY_SECTIONS]
    if not hits:
        return []

    out = []
    # "no confidence interval is reported" contains the phrase "confidence interval" but
    # is explicitly disclosing the absence of one — the same negation trap as ABLATION above.
    has_p = any(PVAL.search(s.text) for s in sentences)  # "p < 0.05" itself can't be sensibly negated
    has_ci = any(CI.search(s.text) and not NEGATION_CUE.search(s.text) for s in sentences)
    has_n = any(NSIZE.search(s.text) for s in sentences)
    if not (has_p or has_ci):
        out.append(CheckFinding(
            kind="missing_uncertainty", severity="major",
            locations=[_loc(s) for s in hits[:3]],
            detail=(
                "Uses significance/robustness language (\"" + hits[0].text[:60] + "...\") "
                "but no p-value or confidence interval appears anywhere in the text."
            ),
            missing="A p-value, confidence interval, or variance/dispersion measure for the key comparison.",
            question="What is the statistical significance (p-value or 95% CI) of the reported difference, and over how many runs?",
            action="Report a confidence interval or significance test for the key comparison, ideally averaged across multiple random seeds.",
            rationale="Significance/robustness language without any uncertainty quantification cannot be distinguished from run-to-run noise.",
        ))
    if not has_n:
        out.append(CheckFinding(
            kind="missing_sample_size", severity="minor",
            locations=[_loc(s) for s in hits[:2]],
            detail="No sample size (n = ...) is reported for the evaluated data.",
            missing="An explicit sample size for the evaluated data.",
            question="How many examples/subjects/recordings were in the evaluation set?",
            action="State the evaluation set size explicitly wherever performance is reported.",
            rationale="Effect sizes cannot be interpreted without knowing the evaluation set size.",
        ))
    return out


def check_causal_claims_need_ablation(sentences: list[Sentence], claims: list[Claim]) -> list[CheckFinding]:
    """A causal claim needs an ablation/intervention/controlled comparison somewhere.

    A sentence that mentions "ablation" but is itself negated ("we do not conduct
    an ablation...") is not evidence one exists — it's often evidence of the
    opposite, and papers regularly disclose exactly that.
    """
    has_ablation = any(
        ABLATION.search(s.text) and not NEGATION_CUE.search(s.text) for s in sentences
    )
    if has_ablation:
        return []
    out = []
    for c in claims:
        if c.type != "causal":
            continue
        subject = _causal_subject(c.text)
        out.append(CheckFinding(
            kind="causal_jump", severity="major",
            locations=[f"{c.section}, p.{c.page}"], claim_id=c.id,
            detail=(
                f"“{c.text}” asserts a causal mechanism, but the manuscript contains no "
                "ablation, intervention, or controlled comparison anywhere."
            ),
            missing="A controlled ablation or intervention isolating the causal mechanism.",
            question=f"Is there an ablation experiment isolating {subject} while holding all other training choices fixed?",
            action=f"Run an ablation that removes or replaces {subject} while holding every other training choice fixed, and report the resulting change.",
            rationale="A causal claim requires evidence that isolates the mechanism; an accuracy comparison between two different model variants shows association, not causation.",
        ))
    return out


def check_generalization_scope(sentences: list[Sentence], claims: list[Claim]) -> list[CheckFinding]:
    """A generalization claim needs more than one evaluated dataset/cohort."""
    datasets = datasets_mentioned(sentences)
    if len(datasets) > 1:
        return []
    out = []
    for c in claims:
        if c.type != "generalization":
            continue
        scope = f" ({datasets[0]})" if datasets else ""
        out.append(CheckFinding(
            kind="evidence_gap_scope", severity="major",
            locations=[f"{c.section}, p.{c.page}"], claim_id=c.id,
            detail=(
                f"“{c.text}” is stated at cross-domain scope, but the manuscript evaluates on "
                f"only {len(datasets)} distinct dataset/cohort{scope}."
            ),
            missing="Evaluation on more than one distinct dataset or domain.",
            question=f"Was the method evaluated on any dataset or domain beyond{scope or ' the one reported'}? If not, what supports the cross-domain claim?",
            action="Evaluate on at least one additional dataset/domain, or narrow the claim's wording to the dataset actually tested.",
            rationale="A single evaluation setting cannot separate a domain-specific effect from a domain-general one.",
        ))
    return out


def check_data_leakage(sentences: list[Sentence]) -> list[CheckFinding]:
    """Preprocessing statistics (normalization etc.) fit on combined train+test data."""
    hits = [s for s in sentences if LEAKAGE_PAT.search(s.text)]
    if not hits:
        return []
    return [CheckFinding(
        kind="data_leakage", severity="major",
        locations=[_loc(s) for s in hits],
        detail=(
            "Preprocessing statistics appear to be computed using both training and test data: “"
            + hits[0].text[:110] + "...”"
        ),
        missing="Preprocessing/normalization statistics fit on the training split only.",
        question="Were normalization or preprocessing statistics computed using only the training split, or did they include test data?",
        action="Recompute preprocessing statistics using only the training set, then re-evaluate on the untouched test set.",
        rationale="Using test data to compute preprocessing statistics leaks test-set information into training, which can optimistically bias reported performance.",
    )]


def check_test_set_model_selection(sentences: list[Sentence]) -> list[CheckFinding]:
    """Hyperparameters/model chosen by test-set performance, with no validation split mentioned."""
    hits = [s for s in sentences if TEST_SELECTION_PAT.search(s.text)]
    if not hits:
        return []
    has_validation = any(VALIDATION_MENTION.search(s.text) for s in sentences)
    if has_validation:
        return []
    return [CheckFinding(
        kind="test_set_model_selection", severity="major",
        locations=[_loc(s) for s in hits],
        detail=(
            "Model/hyperparameter selection appears to use test-set performance directly, with no "
            "validation split mentioned anywhere: “" + hits[0].text[:110] + "...”"
        ),
        missing="A held-out validation split separate from the test set used for model/hyperparameter selection.",
        question="Was hyperparameter selection performed on a validation split distinct from the test set, or on the test set itself?",
        action="Select hyperparameters using a validation split, and report test-set performance only once, on the final chosen configuration.",
        rationale="Selecting hyperparameters by test-set performance turns the test set into a de facto validation set, which optimistically biases the reported result.",
    )]


def run_checkers(sentences: list[Sentence], claims: list[Claim]) -> list[CheckFinding]:
    out: list[CheckFinding] = []
    out += check_uncertainty_reporting(sentences)
    out += check_causal_claims_need_ablation(sentences, claims)
    out += check_generalization_scope(sentences, claims)
    out += check_data_leakage(sentences)
    out += check_test_set_model_selection(sentences)
    return out

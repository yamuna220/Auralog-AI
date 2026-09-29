"""
Deterministic privacy exposure risk engine.

Pure logic — NO Gemini calls, no Streamlit, no Plotly, no network requests
here. This is the ONLY module that computes a numeric privacy-exposure
score for AURAlog, and it computes it entirely from validated
PrivacySignal / CrossPostCorrelation / InferenceChain objects (already
produced and safety-validated upstream — see core/models.py and
services/gemini_service.py). Gemini NEVER determines the final score: it
only supplies the raw signals this formula runs on, plus (separately, for
narrative display) its own candidate correlations/inference chains, which
this module accepts as an optional INPUT (inference_chains) but never
generates itself.

Given the same signals/correlations/inference_chains, this always produces
the exact same score — no randomness, no external calls, no wall-clock
dependence.

WHAT THE SCORE MEANS (read this before wiring it into any UI copy):
A score of, say, 80 means "high MODELED PRIVACY EXPOSURE under AURAlog's
own scoring methodology" — a measure of how much a determined reader could
plausibly piece together from the supplied text, per the deterministic
rules below. It does NOT mean an 80% probability of being attacked,
doxxed, identified, or otherwise harmed. This distinction is deliberately
repeated in the generated `explanation` text (see _build_explanation)
so it travels with the score wherever it's displayed.

--------------------------------------------------------------------------
THE FORMULA, IN ONE PICTURE

    Signal Exposure   (per category, diminishing returns across signals)
          +
    Correlation Amplification   (diminishing returns across correlations)
          +
    Inference Amplification     (diminishing returns across chains)
          =
    Raw Exposure  -->  normalized, bounded weighted average  -->  0-100

Every constant referenced below (category weights, sensitivity weights,
the diminishing-returns factor, correlation/inference base points, the
three top-level component weights, the risk-level boundaries) lives in
config.py, documented there — nothing here is a magic number.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from core.models import (
    CategoryScore,
    CrossPostCorrelation,
    InferenceChain,
    PrivacySignal,
    RiskAssessment,
    RiskLevel,
)


# ===========================================================================
# A + B + C. Signal exposure (per category, with diminishing returns)
# ===========================================================================


def _signal_points(signal: PrivacySignal) -> float:
    """
    A single signal's raw point contribution, BEFORE any cross-signal
    diminishing-returns aggregation:

        signal_points = confidence * sensitivity_weight * category_weight
                         * SIGNAL_POINT_SCALE

    - confidence: how strongly the source TEXT supports this signal
      (supplied by the validated AI extraction — never treated as the
      score itself, only as one multiplicative input to it).
    - sensitivity_weight: how revealing this specific detail is
      (config.SENSITIVITY_WEIGHTS), independent of its category.
    - category_weight: how privacy-sensitive this category of information
      is in general (config.CATEGORY_WEIGHTS).

    Capped at SIGNAL_POINT_SCALE (100) since a category_weight above 1.0
    combined with full confidence/sensitivity could otherwise exceed it.
    """
    sensitivity_weight = config.SENSITIVITY_WEIGHTS.get(signal.sensitivity.value, 0.25)
    category_weight = config.CATEGORY_WEIGHTS.get(signal.category.value, 1.0)
    raw = signal.confidence * sensitivity_weight * category_weight * config.SIGNAL_POINT_SCALE
    return min(config.SIGNAL_POINT_SCALE, raw)


def _diminishing_sum(points: list[float]) -> float:
    """
    Combine several independent point values into one bounded total using
    diminishing returns: sorted descending, the i-th ranked value counts
    for DIMINISHING_RETURNS_FACTOR ** i of itself (see config.py's
    documentation of this constant for the reasoning). Shared by signal
    aggregation, correlation aggregation, and inference-chain aggregation
    below, so "more of the same kind of evidence helps less each time" is
    expressed identically everywhere in the formula.
    """
    if not points:
        return 0.0
    factor = config.DIMINISHING_RETURNS_FACTOR
    ranked = sorted(points, reverse=True)
    return sum(value * (factor**i) for i, value in enumerate(ranked))


def _category_scores(signals: list[PrivacySignal]) -> list[CategoryScore]:
    """
    Group signals by category, compute each category's 0-100 score via
    _signal_points + _diminishing_sum (capped at 100), and return one
    CategoryScore per category that actually has at least one signal —
    per the project brief's own example output, categories with no
    supporting signal simply don't appear.
    """
    by_category: dict[str, list[PrivacySignal]] = {}
    for signal in signals:
        by_category.setdefault(signal.category.value, []).append(signal)

    results: list[CategoryScore] = []
    for category_key, category_signals in by_category.items():
        points = [_signal_points(s) for s in category_signals]
        score = min(config.MAX_CATEGORY_SCORE, _diminishing_sum(points))
        display_name = config.CATEGORY_DISPLAY_NAMES.get(category_key, category_key.title())
        results.append(
            CategoryScore(
                category=display_name,
                score=round(score, 1),
                contributing_signal_ids=[s.id for s in category_signals],
            )
        )
    # Deterministic, judge-friendly ordering: highest-scoring category first;
    # ties broken alphabetically by display name for full reproducibility.
    results.sort(key=lambda c: (-c.score, c.category))
    return results


def _signal_contribution(
    category_scores: list[CategoryScore], signals: list[PrivacySignal]
) -> float:
    """
    Roll the per-category scores up into ONE 0-100 signal_contribution: a
    weighted average across the categories actually present, weighted by
    each category's own config.CATEGORY_WEIGHTS entry (so, at equal
    category scores, a Location-heavy audit still contributes more than a
    Lifestyle-heavy one). A weighted average of values already in [0, 100]
    is itself automatically bounded in [0, 100] — no separate cap needed.
    """
    if not category_scores or not signals:
        return 0.0

    # Map back from display name to the raw category key to look up its
    # weight (display names are presentational only; weights are keyed by
    # the raw SignalCategory value).
    display_to_key = {
        config.CATEGORY_DISPLAY_NAMES.get(key, key.title()): key
        for key in config.CATEGORY_WEIGHTS
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for cs in category_scores:
        raw_key = display_to_key.get(cs.category, cs.category.lower())
        weight = config.CATEGORY_WEIGHTS.get(raw_key, 1.0)
        weighted_sum += cs.score * weight
        weight_total += weight

    if weight_total <= 0:
        return 0.0
    return min(100.0, weighted_sum / weight_total)


# ===========================================================================
# D. Cross-post correlation amplification
# ===========================================================================


def _correlation_points(correlation: CrossPostCorrelation) -> float:
    """
    A single correlation's raw point contribution:

        correlation_points = CORRELATION_SEVERITY_WEIGHTS[relationship_type]
                              * combined_confidence

    combined_confidence is always < 1.0 by construction (core/models.py's
    safety rule), so a correlation alone can never reach its full base
    weight — it is always discounted by how confident the underlying
    signals were.
    """
    base = config.CORRELATION_SEVERITY_WEIGHTS.get(correlation.relationship_type.value, 10.0)
    return min(100.0, base * correlation.combined_confidence)


def _correlation_contribution(correlations: list[CrossPostCorrelation]) -> float:
    """
    Aggregate every correlation's points via the same diminishing-returns
    rule as signals (config.DIMINISHING_RETURNS_FACTOR), capped at 100.
    Duplicate correlations (identical signal_ids, e.g. if the same pair
    were somehow supplied twice) are collapsed to one before aggregating,
    so they cannot be double-counted.
    """
    deduped = _dedupe_correlations(correlations)
    points = [_correlation_points(c) for c in deduped]
    return min(100.0, _diminishing_sum(points))


def _dedupe_correlations(
    correlations: list[CrossPostCorrelation],
) -> list[CrossPostCorrelation]:
    seen: set[frozenset[str]] = set()
    result: list[CrossPostCorrelation] = []
    for c in correlations:
        key = frozenset(c.signal_ids)
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
    return result


# ===========================================================================
# E. Inference-chain amplification
# ===========================================================================


def _inference_points(chain: InferenceChain) -> float:
    """
    A single inference chain's raw point contribution:

        inference_points = confidence * chain_length_factor
                            * BASE_INFERENCE_POINTS

    chain_length_factor grows modestly with how many correlations the
    chain links together (config.CHAIN_LENGTH_INCREMENT per extra link,
    capped at config.CHAIN_LENGTH_FACTOR_MAX) — a longer chain of reasoning
    is a stronger (though still explicitly hypothetical — see
    InferenceChain's confidence < 1.0 safety rule) claim than a one-step
    inference.
    """
    extra_links = max(0, len(chain.correlation_ids) - 1)
    chain_length_factor = min(
        config.CHAIN_LENGTH_FACTOR_MAX,
        1.0 + extra_links * config.CHAIN_LENGTH_INCREMENT,
    )
    raw = chain.confidence * chain_length_factor * config.BASE_INFERENCE_POINTS
    return min(100.0, raw)


def _inference_contribution(inference_chains: list[InferenceChain]) -> float:
    """Aggregate every chain's points with the same diminishing-returns rule."""
    deduped = _dedupe_inference_chains(inference_chains)
    points = [_inference_points(c) for c in deduped]
    return min(100.0, _diminishing_sum(points))


def _dedupe_inference_chains(chains: list[InferenceChain]) -> list[InferenceChain]:
    seen: set[str] = set()
    result: list[InferenceChain] = []
    for c in chains:
        if c.id in seen:
            continue
        seen.add(c.id)
        result.append(c)
    return result


# ===========================================================================
# Top risk factors + explanation (both fully deterministic — never Gemini)
# ===========================================================================


def _top_risk_factors(
    signals: list[PrivacySignal],
    correlations: list[CrossPostCorrelation],
    inference_chains: list[InferenceChain],
) -> list[str]:
    """
    Rank every individual signal/correlation/inference-chain by its own
    point contribution and return the top config.TOP_RISK_FACTORS_LIMIT as
    short, cautiously worded strings — "may narrow" / "could contribute
    to" / "potentially reveals", per the project brief, never asserting a
    correlation or inference as a confirmed fact.
    """
    scored: list[tuple[float, str]] = []

    for s in signals:
        category_name = config.CATEGORY_DISPLAY_NAMES.get(s.category.value, s.category.value)
        scored.append(
            (
                _signal_points(s),
                f"'{s.signal}' could contribute to {category_name} exposure",
            )
        )

    for c in correlations:
        text = c.description or (
            f"A {c.relationship_type.value.replace('_', ' ')} correlation across "
            "posts may narrow the plausible context"
        )
        scored.append((_correlation_points(c), text))

    for chain in inference_chains:
        scored.append((_inference_points(chain), chain.resulting_inference))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored[: config.TOP_RISK_FACTORS_LIMIT]]


def _build_explanation(
    signals: list[PrivacySignal],
    correlations: list[CrossPostCorrelation],
    inference_chains: list[InferenceChain],
    category_scores: list[CategoryScore],
    overall_score: float,
    risk_level: RiskLevel,
) -> str:
    """
    Build a deterministic, plain-language explanation from the ALREADY
    CALCULATED values above — this is template text filled in with real
    numbers, never a second AI call. Always ends with the ethical
    clarification required by the project brief: a modeled exposure score,
    not a probability of real-world harm.
    """
    sentences = [
        f"The exposure is primarily driven by {len(signals)} privacy "
        f"signal(s), {len(correlations)} cross-post correlation(s), and "
        f"{len(inference_chains)} inference chain(s)."
    ]

    if category_scores:
        top = category_scores[0]  # already sorted highest-first
        sentences.append(
            f"{top.category} information contributes most strongly to the "
            f"overall exposure (category score: {top.score:.0f}/100)."
        )

    sentences.append(
        f"This yields an overall modeled exposure score of "
        f"{overall_score:.0f}/100, classified as {risk_level.value}."
    )
    sentences.append(
        "This score reflects a MODELED PRIVACY EXPOSURE under AURAlog's "
        "scoring methodology — it is not a probability of being attacked, "
        "doxxed, or identified in the real world."
    )
    return " ".join(sentences)


# ===========================================================================
# F. Risk level classification
# ===========================================================================


def classify_risk_level(score: float) -> RiskLevel:
    """Map a 0-100 score to a RiskLevel using config.RISK_LEVEL_UPPER_BOUNDS."""
    for level_name, upper_bound_exclusive in config.RISK_LEVEL_UPPER_BOUNDS:
        if score < upper_bound_exclusive:
            return RiskLevel(level_name)
    return RiskLevel.CRITICAL  # defensive fallback; unreachable given clamping


# ===========================================================================
# Public entry point
# ===========================================================================


def compute_risk_report(
    signals: list[PrivacySignal],
    correlations: list[CrossPostCorrelation],
    inference_chains: list[InferenceChain] | None = None,
) -> RiskAssessment:
    """
    Compute the full deterministic RiskAssessment for a given set of
    signals, correlations, and (optionally) inference chains.

    Deterministic: given the same inputs, always returns the same output —
    no randomness, no clock, no I/O. Bounded: total_score and every
    category/component score is clamped to [0, 100] (enforced both by this
    function's own min()/max() calls and, redundantly, by RiskAssessment's
    own Pydantic field constraints). Explainable: every intermediate value
    (category_breakdown, signal/correlation/inference_contribution,
    top_risk_factors, explanation) is returned alongside the total, not
    just the final number.

    `inference_chains` are typically Gemini's own semantic narrative chains
    from GeminiService.analyze_posts. They DO contribute to the score (via
    inference_contribution, weighted by config.INFERENCE_COMPONENT_WEIGHT)
    — but note that `correlations` remains the authoritative,
    ONLY-deterministic-engine-produced list used for correlation_contribution;
    Gemini never supplies the correlations this function scores from.

    Handles every edge case from empty input to duplicate signals/
    correlations to an arbitrarily large signal count without exceeding the
    [0, 100] bound or raising.
    """
    inference_chains = inference_chains or []
    deduped_signals = _dedupe_signals(signals)

    category_scores = _category_scores(deduped_signals)
    signal_contribution = round(_signal_contribution(category_scores, deduped_signals), 1)
    correlation_contribution = round(_correlation_contribution(correlations), 1)
    inference_contribution = round(_inference_contribution(inference_chains), 1)

    raw_overall = (
        signal_contribution * config.SIGNAL_COMPONENT_WEIGHT
        + correlation_contribution * config.CORRELATION_COMPONENT_WEIGHT
        + inference_contribution * config.INFERENCE_COMPONENT_WEIGHT
    )
    overall_score = round(max(0.0, min(100.0, raw_overall)), 1)
    risk_level = classify_risk_level(overall_score)

    top_factors = _top_risk_factors(deduped_signals, correlations, inference_chains)
    explanation = _build_explanation(
        deduped_signals,
        correlations,
        inference_chains,
        category_scores,
        overall_score,
        risk_level,
    )

    return RiskAssessment(
        total_score=overall_score,
        risk_level=risk_level,
        category_breakdown=category_scores,
        signals=deduped_signals,
        correlations=correlations,
        inference_chains=inference_chains,
        signal_contribution=signal_contribution,
        correlation_contribution=correlation_contribution,
        inference_contribution=inference_contribution,
        top_risk_factors=top_factors,
        explanation=explanation,
    )


def _dedupe_signals(signals: list[PrivacySignal]) -> list[PrivacySignal]:
    """Keep the first occurrence of each distinct signal id, preserving order."""
    seen: set[str] = set()
    result: list[PrivacySignal] = []
    for s in signals:
        if s.id in seen:
            continue
        seen.add(s.id)
        result.append(s)
    return result


# ===========================================================================
# Before / after sanitization comparison
# ===========================================================================


@dataclass
class RiskComparison:
    """
    Result of comparing a "before" and "after" RiskAssessment (see
    compare_before_after). A plain container, not a core/models.py Pydantic
    model, since it's a derived, presentation-facing summary rather than
    part of the validated pipeline state itself.
    """

    before_score: float
    after_score: float
    absolute_reduction: float
    percentage_reduction: float
    before_risk_level: RiskLevel
    after_risk_level: RiskLevel


def compare_before_after(before: RiskAssessment, after: RiskAssessment) -> RiskComparison:
    """
    Compare two RiskAssessments produced by RUNNING THE SAME
    compute_risk_report PIPELINE TWICE — once on the original signals, once
    on the signals re-extracted from sanitized posts. This function only
    computes the comparison stats; it never re-derives or hardcodes a
    reduction — both scores must already come from real compute_risk_report
    calls.

    Handles the zero-score edge case safely: if before_score is 0,
    percentage_reduction is defined as 0.0 rather than dividing by zero
    (there is nothing to reduce a percentage of).
    """
    absolute_reduction = round(before.total_score - after.total_score, 1)
    if before.total_score <= 0:
        percentage_reduction = 0.0
    else:
        percentage_reduction = round(
            (absolute_reduction / before.total_score) * 100.0, 1
        )

    return RiskComparison(
        before_score=before.total_score,
        after_score=after.total_score,
        absolute_reduction=absolute_reduction,
        percentage_reduction=percentage_reduction,
        before_risk_level=before.risk_level,
        after_risk_level=after.risk_level,
    )

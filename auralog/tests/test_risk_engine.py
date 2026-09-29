"""
Tests for core/risk_engine.py.

Every test constructs real, validated core/models.py objects directly — no
Gemini call, no mocking of the AI layer at all, matching the project's
requirement that the deterministic engine's tests never depend on Gemini.
"""

from __future__ import annotations

import config
from core.models import (
    CorrelationType,
    CrossPostCorrelation,
    InferenceChain,
    PrivacySignal,
    RiskLevel,
    SensitivityLevel,
    SignalCategory,
)
from core.risk_engine import classify_risk_level, compare_before_after, compute_risk_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    category: SignalCategory,
    confidence: float,
    sensitivity: SensitivityLevel = SensitivityLevel.LOW,
    post_id: str = "post-1",
    directly_stated: bool = False,
) -> PrivacySignal:
    return PrivacySignal(
        post_id=post_id,
        category=category,
        signal=f"{category.value} clue",
        confidence=confidence,
        sensitivity=sensitivity,
        directly_stated=directly_stated,
        inferred=not directly_stated,
    )


def _correlation(signal_a: PrivacySignal, signal_b: PrivacySignal, confidence: float = 0.7) -> CrossPostCorrelation:
    return CrossPostCorrelation(
        signal_ids=[signal_a.id, signal_b.id],
        relationship_type=CorrelationType.EDUCATION_LOCATION,
        combined_confidence=confidence,
        description="test correlation",
    )


def _chain(correlation_ids: list[str], confidence: float = 0.6) -> InferenceChain:
    return InferenceChain(
        correlation_ids=correlation_ids,
        resulting_inference="This may narrow the plausible context.",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Test 1 — no signals
# ---------------------------------------------------------------------------


def test_no_signals_yields_zero_score_and_low_risk() -> None:
    report = compute_risk_report(signals=[], correlations=[])
    assert report.total_score == 0.0
    assert report.risk_level == RiskLevel.LOW
    assert report.category_breakdown == []
    assert report.signal_contribution == 0.0
    assert report.correlation_contribution == 0.0
    assert report.inference_contribution == 0.0


# ---------------------------------------------------------------------------
# Test 2 — one low-sensitivity signal stays relatively low
# ---------------------------------------------------------------------------


def test_one_low_sensitivity_signal_stays_low() -> None:
    signal = _signal(
        SignalCategory.LIFESTYLE, confidence=0.4, sensitivity=SensitivityLevel.LOW
    )
    report = compute_risk_report(signals=[signal], correlations=[])
    assert report.total_score < 30.0
    assert report.risk_level in (RiskLevel.LOW, RiskLevel.MODERATE)


# ---------------------------------------------------------------------------
# Test 3 — multiple high-confidence sensitive signals increase exposure
# ---------------------------------------------------------------------------


def test_multiple_high_confidence_sensitive_signals_increase_exposure() -> None:
    low_report = compute_risk_report(
        signals=[_signal(SignalCategory.LIFESTYLE, 0.4, SensitivityLevel.LOW)],
        correlations=[],
    )
    high_signals = [
        _signal(SignalCategory.LOCATION, 0.95, SensitivityLevel.CRITICAL, post_id="p1"),
        _signal(SignalCategory.IDENTITY, 0.9, SensitivityLevel.HIGH, post_id="p2"),
        _signal(SignalCategory.WORKPLACE, 0.9, SensitivityLevel.HIGH, post_id="p3"),
    ]
    high_report = compute_risk_report(signals=high_signals, correlations=[])

    assert high_report.total_score > low_report.total_score
    assert high_report.total_score <= 100.0


# ---------------------------------------------------------------------------
# Test 4 — a cross-post correlation increases exposure over the same signals alone
# ---------------------------------------------------------------------------


def test_correlation_increases_exposure_over_uncorrelated_case() -> None:
    sig_a = _signal(SignalCategory.LOCATION, 0.8, SensitivityLevel.MEDIUM, post_id="p1")
    sig_b = _signal(SignalCategory.EDUCATION, 0.8, SensitivityLevel.MEDIUM, post_id="p2")
    signals = [sig_a, sig_b]

    without_correlation = compute_risk_report(signals=signals, correlations=[])
    with_correlation = compute_risk_report(
        signals=signals, correlations=[_correlation(sig_a, sig_b)]
    )

    assert with_correlation.total_score > without_correlation.total_score
    assert with_correlation.correlation_contribution > 0.0
    assert without_correlation.correlation_contribution == 0.0


# ---------------------------------------------------------------------------
# Test 5 — an inference chain increases exposure appropriately
# ---------------------------------------------------------------------------


def test_inference_chain_increases_exposure() -> None:
    sig_a = _signal(SignalCategory.LOCATION, 0.8, SensitivityLevel.MEDIUM, post_id="p1")
    sig_b = _signal(SignalCategory.EDUCATION, 0.8, SensitivityLevel.MEDIUM, post_id="p2")
    signals = [sig_a, sig_b]
    correlation = _correlation(sig_a, sig_b)

    without_chain = compute_risk_report(signals=signals, correlations=[correlation])
    with_chain = compute_risk_report(
        signals=signals,
        correlations=[correlation],
        inference_chains=[_chain([correlation.id])],
    )

    assert with_chain.total_score > without_chain.total_score
    assert with_chain.inference_contribution > 0.0
    assert without_chain.inference_contribution == 0.0


# ---------------------------------------------------------------------------
# Test 6 — score never exceeds 100
# ---------------------------------------------------------------------------


def test_score_never_exceeds_100_even_with_extreme_input() -> None:
    many_signals = [
        _signal(
            category,
            confidence=0.99,
            sensitivity=SensitivityLevel.CRITICAL,
            post_id=f"post-{i}",
            directly_stated=False,
        )
        for i, category in enumerate(list(SignalCategory) * 10)
    ]
    many_correlations = [
        _correlation(many_signals[i], many_signals[i + 1], confidence=0.99)
        for i in range(0, len(many_signals) - 1, 2)
    ]
    many_chains = [_chain([c.id], confidence=0.95) for c in many_correlations]

    report = compute_risk_report(
        signals=many_signals, correlations=many_correlations, inference_chains=many_chains
    )
    assert report.total_score <= 100.0
    assert report.signal_contribution <= 100.0
    assert report.correlation_contribution <= 100.0
    assert report.inference_contribution <= 100.0
    for cs in report.category_breakdown:
        assert cs.score <= 100.0


# ---------------------------------------------------------------------------
# Test 7 — score never falls below 0
# ---------------------------------------------------------------------------


def test_score_never_falls_below_zero() -> None:
    report = compute_risk_report(signals=[], correlations=[], inference_chains=[])
    assert report.total_score >= 0.0

    minimal_signal = _signal(SignalCategory.LIFESTYLE, confidence=0.0, sensitivity=SensitivityLevel.LOW)
    report2 = compute_risk_report(signals=[minimal_signal], correlations=[])
    assert report2.total_score >= 0.0


# ---------------------------------------------------------------------------
# Test 8 — determinism: same input produces exactly the same score
# ---------------------------------------------------------------------------


def test_same_input_produces_identical_score() -> None:
    sig_a = _signal(SignalCategory.LOCATION, 0.77, SensitivityLevel.HIGH, post_id="p1")
    sig_b = _signal(SignalCategory.ROUTINE, 0.66, SensitivityLevel.MEDIUM, post_id="p2")
    correlation = _correlation(sig_a, sig_b, confidence=0.5)
    chain = _chain([correlation.id], confidence=0.4)

    report1 = compute_risk_report([sig_a, sig_b], [correlation], [chain])
    report2 = compute_risk_report([sig_a, sig_b], [correlation], [chain])

    assert report1.total_score == report2.total_score
    assert report1.risk_level == report2.risk_level
    assert report1.signal_contribution == report2.signal_contribution
    assert report1.correlation_contribution == report2.correlation_contribution
    assert report1.inference_contribution == report2.inference_contribution
    assert [c.score for c in report1.category_breakdown] == [
        c.score for c in report2.category_breakdown
    ]


# ---------------------------------------------------------------------------
# Test 9 — sanitized posts can be rescored via the same pipeline
# ---------------------------------------------------------------------------


def test_sanitized_signals_can_be_rescored_with_same_function() -> None:
    before_signals = [
        _signal(SignalCategory.LOCATION, 0.9, SensitivityLevel.HIGH, post_id="p1"),
        _signal(SignalCategory.EDUCATION, 0.9, SensitivityLevel.HIGH, post_id="p2"),
    ]
    before_correlation = _correlation(before_signals[0], before_signals[1], confidence=0.8)
    before_report = compute_risk_report(before_signals, [before_correlation])

    # Simulates re-running extraction on SANITIZED text: fewer/weaker signals,
    # no more correlation between them (the whole point of sanitizing).
    after_signals = [
        _signal(SignalCategory.LOCATION, 0.3, SensitivityLevel.LOW, post_id="p1"),
    ]
    after_report = compute_risk_report(after_signals, [])

    assert after_report.total_score < before_report.total_score
    comparison = compare_before_after(before_report, after_report)
    assert comparison.absolute_reduction > 0


# ---------------------------------------------------------------------------
# Test 10 — zero-score before/after comparison never divides by zero
# ---------------------------------------------------------------------------


def test_zero_score_before_after_comparison_is_safe() -> None:
    before_report = compute_risk_report(signals=[], correlations=[])
    after_report = compute_risk_report(signals=[], correlations=[])

    comparison = compare_before_after(before_report, after_report)
    assert comparison.percentage_reduction == 0.0
    assert comparison.absolute_reduction == 0.0


# ---------------------------------------------------------------------------
# Risk-level classification
# ---------------------------------------------------------------------------


def test_classify_risk_level_boundaries() -> None:
    assert classify_risk_level(0.0) == RiskLevel.LOW
    assert classify_risk_level(24.9) == RiskLevel.LOW
    assert classify_risk_level(25.0) == RiskLevel.MODERATE
    assert classify_risk_level(49.9) == RiskLevel.MODERATE
    assert classify_risk_level(50.0) == RiskLevel.HIGH
    assert classify_risk_level(74.9) == RiskLevel.HIGH
    assert classify_risk_level(75.0) == RiskLevel.CRITICAL
    assert classify_risk_level(100.0) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------


def test_category_breakdown_only_includes_present_categories() -> None:
    signal = _signal(SignalCategory.TRAVEL, 0.6, SensitivityLevel.MEDIUM)
    report = compute_risk_report(signals=[signal], correlations=[])
    assert len(report.category_breakdown) == 1
    assert report.category_breakdown[0].category == "Travel"


def test_category_scores_property_matches_breakdown() -> None:
    signal = _signal(SignalCategory.WORKPLACE, 0.7, SensitivityLevel.HIGH)
    report = compute_risk_report(signals=[signal], correlations=[])
    assert report.category_scores == {"Workplace": report.category_breakdown[0].score}


# ---------------------------------------------------------------------------
# Edge cases: duplicates
# ---------------------------------------------------------------------------


def test_duplicate_signals_are_not_double_counted() -> None:
    signal = _signal(SignalCategory.LOCATION, 0.9, SensitivityLevel.CRITICAL)
    once = compute_risk_report(signals=[signal], correlations=[])
    twice = compute_risk_report(signals=[signal, signal], correlations=[])
    assert once.total_score == twice.total_score
    assert len(twice.signals) == 1


def test_duplicate_correlations_are_not_double_counted() -> None:
    sig_a = _signal(SignalCategory.LOCATION, 0.8, SensitivityLevel.MEDIUM, post_id="p1")
    sig_b = _signal(SignalCategory.EDUCATION, 0.8, SensitivityLevel.MEDIUM, post_id="p2")
    correlation = _correlation(sig_a, sig_b)

    once = compute_risk_report([sig_a, sig_b], [correlation])
    twice = compute_risk_report([sig_a, sig_b], [correlation, correlation])
    assert once.total_score == twice.total_score


# ---------------------------------------------------------------------------
# Edge cases: values arriving via the Gemini-JSON safe-parsing path
# ---------------------------------------------------------------------------


def test_signal_with_missing_confidence_and_unknown_category_is_bounded() -> None:
    # Exercises PrivacySignal.from_gemini_json's own defaulting, then feeds
    # the result straight into the risk engine to confirm the whole chain
    # stays bounded and doesn't crash.
    signal = PrivacySignal.from_gemini_json(
        {"category": "totally_unknown_category", "signal": "vague"}, post_id="p1"
    )
    report = compute_risk_report(signals=[signal], correlations=[])
    assert 0.0 <= report.total_score <= 100.0
    assert signal.category == SignalCategory.PERSONAL_INFORMATION


def test_confidence_outside_range_is_clamped_before_reaching_engine() -> None:
    signal = PrivacySignal.from_gemini_json(
        {"category": "location", "signal": "x", "confidence": 7.5}, post_id="p1"
    )
    assert 0.0 <= signal.confidence <= 1.0
    report = compute_risk_report(signals=[signal], correlations=[])
    assert 0.0 <= report.total_score <= 100.0


# ---------------------------------------------------------------------------
# top_risk_factors and explanation
# ---------------------------------------------------------------------------


def test_top_risk_factors_are_populated_and_capped() -> None:
    signals = [
        _signal(cat, 0.9, SensitivityLevel.HIGH, post_id=f"p{i}")
        for i, cat in enumerate(list(SignalCategory))
    ]
    report = compute_risk_report(signals=signals, correlations=[])
    assert 0 < len(report.top_risk_factors) <= config.TOP_RISK_FACTORS_LIMIT


def test_explanation_mentions_counts_and_ethical_disclaimer() -> None:
    signal = _signal(SignalCategory.LOCATION, 0.8, SensitivityLevel.HIGH)
    report = compute_risk_report(signals=[signal], correlations=[])
    assert "1 privacy signal" in report.explanation
    assert "modeled" in report.explanation.lower() or "MODELED" in report.explanation
    assert "probability" in report.explanation.lower()


# ---------------------------------------------------------------------------
# Backward-compatible basic contract checks (kept from earlier stages)
# ---------------------------------------------------------------------------


def test_empty_input_returns_zero_score() -> None:
    report = compute_risk_report(signals=[], correlations=[])
    assert report.total_score == 0.0
    assert report.signals == []
    assert report.correlations == []

"""
Tests for core/correlation_engine.py.

Every test constructs real, validated core/models.py objects directly — no
Gemini call, no API key required, matching the project's requirement that
the deterministic engine's tests never depend on Gemini.
"""

from __future__ import annotations

from core.correlation_engine import correlate_signals
from core.models import (
    CorrelationType,
    CrossPostCorrelation,
    PrivacySignal,
    SensitivityLevel,
    SignalCategory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    category: SignalCategory,
    post_id: str,
    signal_text: str = "clue",
    confidence: float = 0.8,
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
    evidence: str = "",
) -> PrivacySignal:
    return PrivacySignal(
        post_id=post_id,
        category=category,
        signal=signal_text,
        confidence=confidence,
        sensitivity=sensitivity,
        evidence=evidence,
        inferred=True,
    )


# ---------------------------------------------------------------------------
# Test 1 — two unrelated posts: no meaningful correlation
# ---------------------------------------------------------------------------


def test_unrelated_signals_produce_no_correlation() -> None:
    # "I like pizza." / "It rained yesterday." — no category pair in the
    # rule table covers LIFESTYLE + PERSONAL_INFORMATION, so nothing fires.
    signals = [
        _signal(SignalCategory.LIFESTYLE, "post-1", "likes pizza"),
        _signal(SignalCategory.PERSONAL_INFORMATION, "post-2", "mentioned weather"),
    ]
    assert correlate_signals(signals) == []


def test_second_unrelated_example_produces_no_correlation() -> None:
    # "I visited Chennai once." (one-off TRAVEL) / "I like football." (LIFESTYLE)
    signals = [
        _signal(SignalCategory.TRAVEL, "post-1", "visited a city once"),
        _signal(SignalCategory.LIFESTYLE, "post-2", "likes football"),
    ]
    assert correlate_signals(signals) == []


# ---------------------------------------------------------------------------
# Test 2 — location + routine across two posts: correlation detected
# ---------------------------------------------------------------------------


def test_location_and_routine_across_posts_is_detected() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.ROUTINE, "post-2", "same walk every day"),
    ]
    correlations = correlate_signals(signals)
    assert len(correlations) == 1
    assert correlations[0].relationship_type == CorrelationType.LOCATION_ROUTINE
    assert set(correlations[0].post_ids) == {"post-1", "post-2"}


# ---------------------------------------------------------------------------
# Test 3 — same-post signals are not a cross-post correlation
# ---------------------------------------------------------------------------


def test_same_post_signals_are_not_cross_post_correlated() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.ROUTINE, "post-1", "every day"),  # SAME post
    ]
    assert correlate_signals(signals) == []


def test_cross_post_correlation_model_rejects_same_post_ids_when_supplied() -> None:
    # Defense in depth: the MODEL itself also refuses a "cross-post"
    # correlation whose post_ids don't actually span 2+ distinct posts.
    import pytest

    with pytest.raises(Exception):
        CrossPostCorrelation(
            signal_ids=["s1", "s2"],
            post_ids=["post-1", "post-1"],
            combined_confidence=0.5,
        )


# ---------------------------------------------------------------------------
# Test 4 — three related posts: multi-signal inference chain
# ---------------------------------------------------------------------------


def test_three_related_posts_produce_multi_signal_correlation() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.EDUCATION, "post-2", "university campus"),
        _signal(SignalCategory.ROUTINE, "post-2", "as usual"),
    ]
    correlations = correlate_signals(signals)
    multi = [c for c in correlations if c.relationship_type == CorrelationType.MULTI_SIGNAL]
    assert len(multi) == 1
    assert len(multi[0].signal_ids) == 3
    assert len(set(multi[0].post_ids)) >= 2
    # Inference-chain construction from these correlations is tested in
    # tests/test_inference_engine.py — this file only tests correlation
    # detection (core/correlation_engine.py's own job).


# ---------------------------------------------------------------------------
# Test 5 — duplicate signal order (s1+s2 vs s2+s1) must not duplicate
# ---------------------------------------------------------------------------


def test_signal_id_order_does_not_create_duplicate_correlations() -> None:
    sig_a = _signal(SignalCategory.LOCATION, "post-1", "near the metro")
    sig_b = _signal(SignalCategory.ROUTINE, "post-2", "every day")

    correlations_ab = correlate_signals([sig_a, sig_b])
    correlations_ba = correlate_signals([sig_b, sig_a])  # reversed input order

    assert len(correlations_ab) == 1
    assert len(correlations_ba) == 1
    assert correlations_ab[0].signal_ids == correlations_ba[0].signal_ids
    # normalized (sorted) regardless of input order
    assert correlations_ab[0].signal_ids == sorted([sig_a.id, sig_b.id])


# ---------------------------------------------------------------------------
# Test 6 — low-confidence unrelated signals: no strong correlation
# ---------------------------------------------------------------------------


def test_low_confidence_unrelated_signals_yield_no_strong_correlation() -> None:
    signals = [
        _signal(
            SignalCategory.LIFESTYLE, "post-1", "vague hobby", confidence=0.1,
            sensitivity=SensitivityLevel.LOW,
        ),
        _signal(
            SignalCategory.PERSONAL_INFORMATION, "post-2", "vague detail",
            confidence=0.1, sensitivity=SensitivityLevel.LOW,
        ),
    ]
    correlations = correlate_signals(signals)
    # No rule covers this category pair at all, so nothing fires regardless
    # of confidence — the strongest guarantee against a "strong correlation".
    assert correlations == []


# ---------------------------------------------------------------------------
# Test 7 — repeated routine across posts: routine correlation, with a
# repetition-language strength boost
# ---------------------------------------------------------------------------


def test_repeated_routine_signal_gets_higher_strength_than_isolated() -> None:
    isolated = [
        _signal(SignalCategory.TIME, "post-1", "morning run", evidence="6 AM"),
        _signal(SignalCategory.ROUTINE, "post-2", "a run", evidence="went for a run"),
    ]
    repeated = [
        _signal(SignalCategory.TIME, "post-1", "morning run", evidence="6 AM"),
        _signal(
            SignalCategory.ROUTINE, "post-2", "another run", evidence="another 6 AM run, every day"
        ),
    ]

    isolated_corr = correlate_signals(isolated)[0]
    repeated_corr = correlate_signals(repeated)[0]

    assert isolated_corr.relationship_type == CorrelationType.TIME_ROUTINE
    assert repeated_corr.strength > isolated_corr.strength


# ---------------------------------------------------------------------------
# Test 8 — location + education + routine: multi-signal correlation
# ---------------------------------------------------------------------------


def test_location_education_routine_yields_multi_signal_correlation() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.EDUCATION, "post-2", "university"),
        _signal(SignalCategory.ROUTINE, "post-2", "as usual"),
    ]
    correlations = correlate_signals(signals)
    combined_signal_ids = {sid for c in correlations for sid in c.signal_ids}
    assert {s.id for s in signals}.issubset(combined_signal_ids)
    for c in correlations:
        assert "may" in c.potential_inference.lower() or "could" in c.potential_inference.lower()


# ---------------------------------------------------------------------------
# Test 9 — no signals: empty correlation list
# ---------------------------------------------------------------------------


def test_no_signals_returns_empty_list() -> None:
    assert correlate_signals([]) == []


# ---------------------------------------------------------------------------
# Test 10 — determinism: same input produces the same correlations
# ---------------------------------------------------------------------------


def test_same_input_produces_identical_correlations() -> None:
    sig_a = _signal(SignalCategory.LOCATION, "post-1", "near the metro")
    sig_b = _signal(SignalCategory.ROUTINE, "post-2", "every day")
    sig_c = _signal(SignalCategory.EDUCATION, "post-3", "university")
    signals = [sig_a, sig_b, sig_c]

    result1 = correlate_signals(signals)
    result2 = correlate_signals(signals)

    assert len(result1) == len(result2)
    for c1, c2 in zip(result1, result2):
        assert c1.signal_ids == c2.signal_ids
        assert c1.relationship_type == c2.relationship_type
        assert c1.strength == c2.strength
        assert c1.combined_confidence == c2.combined_confidence
        assert c1.description == c2.description


# ---------------------------------------------------------------------------
# All six pairwise rule types (section 3, A-F)
# ---------------------------------------------------------------------------


def test_all_pairwise_correlation_types_fire_when_supported() -> None:
    cases = [
        ((SignalCategory.LOCATION, SignalCategory.ROUTINE), CorrelationType.LOCATION_ROUTINE),
        ((SignalCategory.LOCATION, SignalCategory.TIME), CorrelationType.LOCATION_TIME),
        ((SignalCategory.EDUCATION, SignalCategory.LOCATION), CorrelationType.EDUCATION_LOCATION),
        ((SignalCategory.EDUCATION, SignalCategory.IDENTITY), CorrelationType.EDUCATION_IDENTITY),
        ((SignalCategory.WORKPLACE, SignalCategory.ROUTINE), CorrelationType.WORKPLACE_ROUTINE),
        ((SignalCategory.TIME, SignalCategory.ROUTINE), CorrelationType.TIME_ROUTINE),
    ]
    for (cat_a, cat_b), expected_type in cases:
        signals = [
            _signal(cat_a, "post-1", f"{cat_a.value} clue"),
            _signal(cat_b, "post-2", f"{cat_b.value} clue"),
        ]
        correlations = correlate_signals(signals)
        assert len(correlations) == 1, f"expected exactly one correlation for {cat_a}+{cat_b}"
        assert correlations[0].relationship_type == expected_type


# ---------------------------------------------------------------------------
# Strength and confidence bounds
# ---------------------------------------------------------------------------


def test_strength_and_confidence_are_bounded() -> None:
    signals = [
        _signal(
            SignalCategory.LOCATION, "post-1", "near the metro",
            confidence=0.99, sensitivity=SensitivityLevel.CRITICAL,
            evidence="every single morning, always",
        ),
        _signal(
            SignalCategory.ROUTINE, "post-2", "same route",
            confidence=0.99, sensitivity=SensitivityLevel.CRITICAL,
            evidence="every single day, always the same",
        ),
    ]
    correlation = correlate_signals(signals)[0]
    assert 0.0 <= correlation.strength <= 1.0
    assert 0.0 <= correlation.combined_confidence < 1.0  # never fully certain


# ---------------------------------------------------------------------------
# Duplicate signals / correlations are handled gracefully
# ---------------------------------------------------------------------------


def test_duplicate_signal_objects_are_not_double_counted() -> None:
    sig_a = _signal(SignalCategory.LOCATION, "post-1", "near the metro")
    sig_b = _signal(SignalCategory.ROUTINE, "post-2", "every day")

    once = correlate_signals([sig_a, sig_b])
    twice = correlate_signals([sig_a, sig_a, sig_b, sig_b])
    assert len(once) == len(twice) == 1
    assert once[0].signal_ids == twice[0].signal_ids


# ---------------------------------------------------------------------------
# Explanations are grounded in the actual signals (section 14)
# ---------------------------------------------------------------------------


def test_explanation_and_relationship_reference_actual_signal_labels() -> None:
    sig_a = _signal(SignalCategory.LOCATION, "post-1", "cafe near the metro")
    sig_b = _signal(SignalCategory.ROUTINE, "post-2", "recurring commute")
    correlation = correlate_signals([sig_a, sig_b])[0]

    assert "cafe near the metro" in correlation.description
    assert "recurring commute" in correlation.description
    assert correlation.potential_inference  # non-empty
    assert correlation.explanation  # non-empty
    assert "confirmed" not in correlation.potential_inference.lower()


def test_correlations_never_claim_exact_address_or_identity() -> None:
    signals = [
        _signal(SignalCategory.EDUCATION, "post-1", "university"),
        _signal(SignalCategory.IDENTITY, "post-2", "graduation year"),
    ]
    correlation = correlate_signals(signals)[0]
    forbidden_terms = ["exact address", "confirmed identity", "real name", "home address"]
    combined_text = (
        correlation.description + " " + correlation.potential_inference + " " + correlation.explanation
    ).lower()
    for term in forbidden_terms:
        assert term not in combined_text


# ---------------------------------------------------------------------------
# Gemini candidates corroborate but never create a correlation
# ---------------------------------------------------------------------------


def test_gemini_candidate_boosts_strength_when_it_matches() -> None:
    sig_a = _signal(SignalCategory.LOCATION, "post-1", "near the metro")
    sig_b = _signal(SignalCategory.ROUTINE, "post-2", "every day")

    without_candidate = correlate_signals([sig_a, sig_b])[0]

    matching_candidate = CrossPostCorrelation(
        signal_ids=sorted([sig_a.id, sig_b.id]),
        combined_confidence=0.6,
        strength=0.6,
    )
    with_candidate = correlate_signals([sig_a, sig_b], gemini_candidates=[matching_candidate])[0]

    assert with_candidate.strength >= without_candidate.strength


def test_gemini_candidate_alone_cannot_create_an_unsupported_correlation() -> None:
    # Two signals with NO rule covering their category pair — a Gemini
    # candidate referencing them must not conjure a correlation into being.
    sig_a = _signal(SignalCategory.LIFESTYLE, "post-1", "likes pizza")
    sig_b = _signal(SignalCategory.PERSONAL_INFORMATION, "post-2", "mentioned weather")

    phantom_candidate = CrossPostCorrelation(
        signal_ids=sorted([sig_a.id, sig_b.id]),
        combined_confidence=0.9,
        strength=0.9,
    )
    result = correlate_signals([sig_a, sig_b], gemini_candidates=[phantom_candidate])
    assert result == []


# ---------------------------------------------------------------------------
# Performance — should stay fast at realistic scale (section 17)
# ---------------------------------------------------------------------------


def test_performance_with_many_posts_and_signals() -> None:
    import time

    categories = list(SignalCategory)
    signals = [
        _signal(categories[i % len(categories)], f"post-{i}", f"clue {i}")
        for i in range(150)  # ~50 posts worth of signals at 3/post
    ]

    start = time.monotonic()
    correlate_signals(signals)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0  # generous ceiling; should be near-instant in practice

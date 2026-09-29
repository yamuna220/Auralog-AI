"""
Tests for core/models.py.

Covers construction/validation of each of the 8 structured models, the
Gemini JSON deserialization helpers, and — most importantly — the safety
rule that runs through every model: uncertain inference can never be
represented as confirmed personal information.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import (
    CategoryScore,
    CompleteAnalysis,
    CorrelationType,
    CrossPostCorrelation,
    EvidenceLevel,
    InferenceChain,
    PrivacySignal,
    RiskAssessment,
    SanitizedPost,
    SensitivityLevel,
    SignalCategory,
    SocialMediaPost,
)


# ---------------------------------------------------------------------------
# 1. SocialMediaPost
# ---------------------------------------------------------------------------


def test_social_media_post_auto_generates_id() -> None:
    post = SocialMediaPost(text="hello world")
    assert post.id
    assert post.text == "hello world"


def test_social_media_post_strips_whitespace() -> None:
    post = SocialMediaPost(text="  hello world  ")
    assert post.text == "hello world"


def test_social_media_post_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        SocialMediaPost(text="   ")


# ---------------------------------------------------------------------------
# 2 & 3. PrivacySignal / SignalCategory
# ---------------------------------------------------------------------------


def test_privacy_signal_valid_directly_stated() -> None:
    signal = PrivacySignal(
        post_id="post-1",
        category=SignalCategory.LOCATION,
        signal="cafe near the metro station",
        detail="user mentions a specific cafe near a metro station",
        confidence=1.0,
        sensitivity=SensitivityLevel.MEDIUM,
        evidence="the blue cafe near the metro station",
        directly_stated=True,
        inferred=False,
    )
    assert signal.evidence_level == EvidenceLevel.DIRECTLY_STATED


def test_privacy_signal_valid_inferred() -> None:
    signal = PrivacySignal(
        post_id="post-1",
        category=SignalCategory.LOCATION,
        signal="likely lives near university",
        confidence=0.6,
        directly_stated=False,
        inferred=True,
    )
    assert signal.evidence_level == EvidenceLevel.INFERRED


def test_privacy_signal_speculative_when_neither_flag_set() -> None:
    signal = PrivacySignal(
        post_id="post-1",
        category=SignalCategory.LIFESTYLE,
        signal="possibly an early riser",
        confidence=0.2,
    )
    assert signal.evidence_level == EvidenceLevel.SPECULATIVE


def test_privacy_signal_rejects_category_not_in_enum() -> None:
    with pytest.raises(ValidationError):
        PrivacySignal(
            post_id="post-1",
            category="not_a_real_category",  # type: ignore[arg-type]
            signal="x",
            confidence=0.5,
        )


def test_privacy_signal_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        PrivacySignal(
            post_id="post-1",
            category=SignalCategory.LOCATION,
            signal="x",
            confidence=1.5,
        )


def test_privacy_signal_rejects_both_directly_stated_and_inferred() -> None:
    with pytest.raises(ValidationError):
        PrivacySignal(
            post_id="post-1",
            category=SignalCategory.LOCATION,
            signal="x",
            confidence=0.8,
            directly_stated=True,
            inferred=True,
        )


def test_privacy_signal_rejects_full_confidence_when_not_directly_stated() -> None:
    """
    SAFETY RULE: uncertain inference must never be represented as confirmed
    personal information — an inferred (or speculative) signal cannot claim
    confidence 1.0.
    """
    with pytest.raises(ValidationError):
        PrivacySignal(
            post_id="post-1",
            category=SignalCategory.LOCATION,
            signal="x",
            confidence=1.0,
            directly_stated=False,
            inferred=True,
        )


def test_privacy_signal_from_gemini_json_normalizes_bad_category() -> None:
    signal = PrivacySignal.from_gemini_json(
        {"category": "Not A Real Category", "signal": "x", "confidence": 0.5},
        post_id="post-1",
    )
    assert signal.category == SignalCategory.PERSONAL_INFORMATION


def test_privacy_signal_from_gemini_json_clamps_confidence() -> None:
    signal = PrivacySignal.from_gemini_json(
        {"category": "location", "signal": "x", "confidence": 5.0},
        post_id="post-1",
    )
    assert signal.confidence <= 1.0

    signal2 = PrivacySignal.from_gemini_json(
        {"category": "location", "signal": "x", "confidence": -3.0},
        post_id="post-1",
    )
    assert signal2.confidence >= 0.0


def test_privacy_signal_from_gemini_json_resolves_contradictory_flags_safely() -> None:
    """
    If Gemini returns both directly_stated=True and inferred=True (or
    confidence 1.0 without directly_stated), the parser must resolve toward
    the SAFER (less certain) reading, never toward "confirmed".
    """
    signal = PrivacySignal.from_gemini_json(
        {
            "category": "location",
            "signal": "x",
            "confidence": 1.0,
            "directly_stated": True,
            "inferred": True,
        },
        post_id="post-1",
    )
    assert signal.directly_stated is False
    assert signal.inferred is True
    assert signal.confidence < 1.0


def test_privacy_signal_list_from_gemini_json_skips_malformed_items() -> None:
    raw_items = [
        {"category": "location", "signal": "valid one", "confidence": 0.5},
        "not a dict",
        {"category": "education", "signal": "valid two", "confidence": "not a number"},
    ]
    signals = PrivacySignal.list_from_gemini_json(raw_items, post_id="post-1")
    # The malformed string entry is skipped; the two dict entries survive
    # (the bad confidence value falls back to a safe default, not a crash).
    assert len(signals) == 2
    assert all(isinstance(s, PrivacySignal) for s in signals)


# ---------------------------------------------------------------------------
# 4. CrossPostCorrelation
# ---------------------------------------------------------------------------


def _two_signal_ids() -> list[str]:
    return [
        PrivacySignal(
            post_id="post-1", category=SignalCategory.LOCATION, signal="a", confidence=0.7
        ).id,
        PrivacySignal(
            post_id="post-2", category=SignalCategory.TRAVEL, signal="b", confidence=0.6
        ).id,
    ]


def test_cross_post_correlation_valid() -> None:
    correlation = CrossPostCorrelation(
        signal_ids=_two_signal_ids(),
        relationship_type=CorrelationType.EDUCATION_LOCATION,
        combined_confidence=0.7,
    )
    assert correlation.evidence_level == EvidenceLevel.INFERRED


def test_cross_post_correlation_requires_at_least_two_signal_ids() -> None:
    with pytest.raises(ValidationError):
        CrossPostCorrelation(
            signal_ids=["only-one"],
            combined_confidence=0.5,
        )


def test_cross_post_correlation_rejects_duplicate_signal_ids() -> None:
    with pytest.raises(ValidationError):
        CrossPostCorrelation(
            signal_ids=["sig-1", "sig-1"],
            combined_confidence=0.5,
        )


def test_cross_post_correlation_rejects_full_confidence() -> None:
    """
    SAFETY RULE: a correlation is always a derived relationship — it can
    never claim confidence 1.0, since it was never itself "stated" in a post.
    """
    with pytest.raises(ValidationError):
        CrossPostCorrelation(
            signal_ids=_two_signal_ids(),
            combined_confidence=1.0,
        )


def test_cross_post_correlation_evidence_level_downgrades_below_threshold() -> None:
    correlation = CrossPostCorrelation(
        signal_ids=_two_signal_ids(),
        combined_confidence=0.2,
    )
    assert correlation.evidence_level == EvidenceLevel.SPECULATIVE


# ---------------------------------------------------------------------------
# 5. InferenceChain
# ---------------------------------------------------------------------------


def test_inference_chain_valid() -> None:
    chain = InferenceChain(
        correlation_ids=["corr-1"],
        resulting_inference="Posts may narrow the user's commute area.",
        confidence=0.6,
    )
    assert chain.evidence_level == EvidenceLevel.INFERRED


def test_inference_chain_rejects_full_confidence() -> None:
    with pytest.raises(ValidationError):
        InferenceChain(
            correlation_ids=["corr-1"],
            resulting_inference="Confirmed home address.",
            confidence=1.0,
        )


def test_inference_chain_rejects_blank_resulting_inference() -> None:
    with pytest.raises(ValidationError):
        InferenceChain(correlation_ids=["corr-1"], resulting_inference="   ", confidence=0.5)


def test_inference_chain_from_gemini_json_never_reaches_full_confidence() -> None:
    chain = InferenceChain.from_gemini_json(
        {
            "correlation_ids": ["corr-1"],
            "resulting_inference": "attacker could narrow location",
            "confidence": 1.0,
        }
    )
    assert chain.confidence < 1.0


def test_inference_chain_from_gemini_json_defaults_missing_correlation_ids() -> None:
    chain = InferenceChain.from_gemini_json(
        {"resulting_inference": "some inference", "confidence": 0.4}
    )
    assert chain.correlation_ids == ["unknown"]


def test_inference_chain_list_from_gemini_json_skips_malformed_items() -> None:
    raw_items = [
        {"resulting_inference": "valid", "confidence": 0.5, "correlation_ids": ["c1"]},
        42,
    ]
    chains = InferenceChain.list_from_gemini_json(raw_items)
    assert len(chains) == 1


# ---------------------------------------------------------------------------
# 6. RiskAssessment
# ---------------------------------------------------------------------------


def test_risk_assessment_defaults() -> None:
    assessment = RiskAssessment(total_score=0.0)
    assert assessment.category_breakdown == []
    assert assessment.signals == []
    assert assessment.correlations == []


def test_risk_assessment_rejects_score_above_100() -> None:
    with pytest.raises(ValidationError):
        RiskAssessment(total_score=150.0)


def test_risk_assessment_rejects_negative_score() -> None:
    with pytest.raises(ValidationError):
        RiskAssessment(total_score=-1.0)


def test_category_score_bounds() -> None:
    with pytest.raises(ValidationError):
        CategoryScore(category="location_exposure", score=101.0)


# ---------------------------------------------------------------------------
# 7. SanitizedPost
# ---------------------------------------------------------------------------


def test_sanitized_post_from_rewrite() -> None:
    post = SocialMediaPost(text="original text with a location clue")
    sanitized = SanitizedPost.from_rewrite(
        post, "safer rewritten text", removed_signal_ids=["sig-1"]
    )
    assert sanitized.post_id == post.id
    assert sanitized.original_text == post.text
    assert sanitized.sanitized_text == "safer rewritten text"
    assert sanitized.removed_signal_ids == ["sig-1"]


def test_sanitized_post_rejects_blank_sanitized_text() -> None:
    with pytest.raises(ValidationError):
        SanitizedPost(post_id="post-1", original_text="x", sanitized_text="   ")


# ---------------------------------------------------------------------------
# 8. CompleteAnalysis
# ---------------------------------------------------------------------------


def test_complete_analysis_empty_is_valid() -> None:
    analysis = CompleteAnalysis()
    assert analysis.posts == []
    assert analysis.risk_assessment is None


def test_complete_analysis_accepts_signal_referencing_known_post() -> None:
    post = SocialMediaPost(text="hello")
    signal = PrivacySignal(
        post_id=post.id, category=SignalCategory.LOCATION, signal="x", confidence=0.5
    )
    analysis = CompleteAnalysis(posts=[post], signals=[signal])
    assert analysis.signals[0].post_id == post.id


def test_complete_analysis_rejects_signal_with_unknown_post_id() -> None:
    post = SocialMediaPost(text="hello")
    signal = PrivacySignal(
        post_id="does-not-exist", category=SignalCategory.LOCATION, signal="x", confidence=0.5
    )
    with pytest.raises(ValidationError):
        CompleteAnalysis(posts=[post], signals=[signal])


def test_complete_analysis_json_round_trip() -> None:
    post = SocialMediaPost(text="hello world")
    signal = PrivacySignal(
        post_id=post.id,
        category=SignalCategory.LOCATION,
        signal="a clue",
        confidence=0.5,
        inferred=True,
    )
    original = CompleteAnalysis(posts=[post], signals=[signal])

    raw_json = original.to_json()
    restored = CompleteAnalysis.from_json(raw_json)

    assert restored.posts[0].text == "hello world"
    assert restored.signals[0].signal == "a clue"
    assert restored.signals[0].category == SignalCategory.LOCATION

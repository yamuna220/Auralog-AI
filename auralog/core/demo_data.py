"""
Demo data for DEMO MODE (no GEMINI_API_KEY configured).

Holds the three example posts from the project brief AND a pre-computed
analysis result "as if" Gemini had returned it — signals, correlations, and
inference chains, structured EXACTLY like GeminiService.analyze_posts's real
output (same core/models.py types, same validation rules). This lets the
ENTIRE downstream pipeline (correlation engine -> risk engine -> graph ->
UI) run for real against this data; only the Gemini API call itself is
skipped. This is clearly demo/synthetic data, not a claim that Gemini
generated it — see GeminiService._demo_analysis's docstring.

Signal ids (s*), correlation ids (c*) and inference chain ids (i*) below are
deliberately cross-referenced by hand, the same way a real Gemini response
would self-reference them.
"""

from __future__ import annotations

from core.models import (
    CorrelationType,
    CrossPostCorrelation,
    InferenceChain,
    PrivacySignal,
    SensitivityLevel,
    SignalCategory,
    SocialMediaPost,
)

DEMO_POSTS: list[SocialMediaPost] = [
    SocialMediaPost(
        id="demo-1",
        order_index=0,
        text="Loved the cold brew at the blue cafe near the metro station this morning!",
    ),
    SocialMediaPost(
        id="demo-2",
        order_index=1,
        text="Finally made it home! 45 minute walk from my university campus as usual.",
    ),
    SocialMediaPost(
        id="demo-3",
        order_index=2,
        text="Can't wait for our batch of 2020 high school reunion next weekend.",
    ),
]

# ---------------------------------------------------------------------------
# Signals — one list per demo post, keyed by post id.
# ---------------------------------------------------------------------------

DEMO_SIGNALS_BY_POST: dict[str, list[PrivacySignal]] = {
    "demo-1": [
        PrivacySignal(
            id="s1",
            post_id="demo-1",
            category=SignalCategory.LOCATION,
            signal="cafe near a metro station",
            detail="The post names a specific cafe located near a metro station.",
            confidence=0.9,
            sensitivity=SensitivityLevel.MEDIUM,
            evidence="the blue cafe near the metro station",
            directly_stated=True,
            inferred=False,
        ),
        PrivacySignal(
            id="s2",
            post_id="demo-1",
            category=SignalCategory.TIME,
            signal="morning posting time",
            detail="The post is framed as a morning activity ('this morning').",
            confidence=0.7,
            sensitivity=SensitivityLevel.LOW,
            evidence="this morning",
            directly_stated=True,
            inferred=False,
        ),
    ],
    "demo-2": [
        PrivacySignal(
            id="s3",
            post_id="demo-2",
            category=SignalCategory.EDUCATION,
            signal="university campus commute",
            detail="The post states the author walks home from a university campus.",
            confidence=0.9,
            sensitivity=SensitivityLevel.MEDIUM,
            evidence="45 minute walk from my university campus",
            directly_stated=True,
            inferred=False,
        ),
        PrivacySignal(
            id="s4",
            post_id="demo-2",
            category=SignalCategory.ROUTINE,
            signal="recurring commute pattern",
            detail="'as usual' suggests this walk happens repeatedly, not once.",
            confidence=0.6,
            sensitivity=SensitivityLevel.MEDIUM,
            evidence="as usual",
            directly_stated=False,
            inferred=True,
        ),
    ],
    "demo-3": [
        PrivacySignal(
            id="s5",
            post_id="demo-3",
            category=SignalCategory.IDENTITY,
            signal="high school graduation year",
            detail="The post states the author's high school graduating batch/year.",
            confidence=0.85,
            sensitivity=SensitivityLevel.MEDIUM,
            evidence="our batch of 2020 high school reunion",
            directly_stated=True,
            inferred=False,
        ),
    ],
}

# ---------------------------------------------------------------------------
# Correlations — cross-post relationships across the signals above.
# ---------------------------------------------------------------------------

DEMO_CORRELATIONS: list[CrossPostCorrelation] = [
    CrossPostCorrelation(
        id="c1",
        signal_ids=["s1", "s3"],
        post_ids=["demo-1", "demo-2"],
        relationship_type=CorrelationType.EDUCATION_LOCATION,
        combined_confidence=0.65,
        strength=0.62,
        description=(
            "'cafe near a metro station' and 'university campus commute', from "
            "separate posts, together describe an education-related location "
            "alongside another location clue."
        ),
        potential_inference="This combination may narrow the general geographic context.",
        explanation=(
            "These signals occur in separate posts but describe related "
            "Location and Education information."
        ),
    ),
    CrossPostCorrelation(
        id="c2",
        signal_ids=["s2", "s4"],
        post_ids=["demo-1", "demo-2"],
        relationship_type=CorrelationType.TIME_ROUTINE,
        combined_confidence=0.55,
        strength=0.50,
        description=(
            "'morning posting time' and 'recurring commute pattern', from "
            "separate posts, together describe repeated timing alongside "
            "routine information."
        ),
        potential_inference="Repeated timing like this may reveal a predictable routine.",
        explanation=(
            "These signals occur in separate posts but describe related "
            "Time and Routine information."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Inference chains — higher-order narrative built from the correlations above.
# ---------------------------------------------------------------------------

DEMO_INFERENCE_CHAINS: list[InferenceChain] = [
    InferenceChain(
        id="i1",
        correlation_ids=["c1", "c2"],
        signal_ids=["s1", "s2", "s3", "s4"],
        post_ids=["demo-1", "demo-2"],
        steps=[c.description for c in DEMO_CORRELATIONS],
        resulting_inference=(
            "Taken together, these posts could potentially help narrow down "
            "the author's general commute area and a rough daily time "
            "window — this is a hypothetical inference, not a confirmed "
            "location or schedule."
        ),
        confidence=0.5,
        sensitivity=SensitivityLevel.MEDIUM,
        strength=0.45,
        explanation=(
            "Posts demo-1 and demo-2 contribute location, time, education, and "
            "routine signals connected through 2 cross-post correlations. "
            "Together, they may reveal more about the user's routine or "
            "location context than any single post alone."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Sanitized versions — used by GeminiService.sanitize_post in DEMO_MODE.
# ---------------------------------------------------------------------------

DEMO_SANITIZED_TEXT_BY_POST_ID: dict[str, str] = {
    "demo-1": "Loved my coffee out this morning!",
    "demo-2": "Finally home after a long walk from campus!",
    "demo-3": "Can't wait for our high school reunion next weekend.",
}

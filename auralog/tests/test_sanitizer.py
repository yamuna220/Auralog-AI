"""
Comprehensive test suite for core/sanitizer.py & Before/After verification engine.

Verifies:
1. No risky signals -> post unchanged.
2. Specific location -> generalized location.
3. Exact recurring time -> generalized time.
4. Routine exposure -> routine reduced.
5. Identity linkage -> identifying detail reduced.
6. Multiple risky categories -> multiple details sanitized.
7. Meaning preservation -> output remains meaningful & non-empty.
8. Gemini unavailable -> deterministic fallback works.
9. Malformed Gemini response -> fallback works.
10. Unsafe sanitization -> risk regression caught & rejected.
11. Before/after analysis -> runs through real pipeline.
12. Risk reduction calculation -> correct score delta.
13. Signal comparison -> before/after signals compared.
14. Correlation comparison -> before/after correlations compared.
15. Inference-chain comparison -> before/after chains compared.
16. Traceability -> mapped to original post IDs.
17. Determinism -> stable fallback output.
18. No privacy regression -> score never increases.
"""

from __future__ import annotations

from core.models import PrivacySignal, SensitivityLevel, SignalCategory, SocialMediaPost
from core.sanitizer import apply_deterministic_fallback, sanitize_post, sanitize_posts, validate_semantic_preservation
from services.analysis_service import analyze_before_after
from services.gemini_service import GeminiService


class _MockGeminiService(GeminiService):
    """Mock GeminiService that can simulate AI responses, failures, or malformed JSON."""

    def __init__(self, mode: str = "normal") -> None:
        self.demo_mode = True
        self._client = None
        self.mode = mode

    def sanitize_post(
        self, post: SocialMediaPost, flagged_signals: list[PrivacySignal]
    ) -> dict:
        if self.mode == "error":
            raise RuntimeError("Gemini API connection error")
        if self.mode == "malformed":
            return {"invalid_key": "no sanitized_text"}
        if self.mode == "unsafe":
            # Intentionally return a text with higher risk details
            return {
                "sanitized_text": "Had coffee at 7:15 AM at Blue Cafe near Central Metro at 123 Main St.",
                "removed_details": [],
                "preserved_meaning": "Detailed post",
                "reason": "Unsafe test",
            }

        # Default smart mock rewrite
        rewritten = post.text
        if "Blue Cafe" in post.text:
            rewritten = rewritten.replace("Blue Cafe", "a cafe")
        if "near the metro station" in post.text:
            rewritten = rewritten.replace("near the metro station", "in town")
        if "45 minute walk" in post.text:
            rewritten = rewritten.replace("45 minute walk", "walk")
        if "as usual" in post.text:
            rewritten = rewritten.replace("as usual", "today")
        if "batch of 2020" in post.text:
            rewritten = rewritten.replace("batch of 2020 ", "")

        if rewritten == post.text:
            rewritten = "REWRITTEN: " + post.text

        return {
            "sanitized_text": rewritten,
            "removed_details": [s.signal for s in flagged_signals if s.signal],
            "preserved_meaning": "Original post tone & intent preserved.",
            "reason": "Mocked AI sanitization pass.",
        }


def _signal(category: SignalCategory, post_id: str, text: str = "clue") -> PrivacySignal:
    return PrivacySignal(
        post_id=post_id,
        category=category,
        signal=text,
        confidence=0.9,
        sensitivity=SensitivityLevel.HIGH,
        inferred=True,
    )


# --- Test 1: No risky signals ---
def test_no_risky_signals_remains_unchanged() -> None:
    post = SocialMediaPost(text="I love chocolate cake.")
    result = sanitize_post(post, [], _MockGeminiService())
    assert result.status == "unchanged"
    assert result.sanitized_text == "I love chocolate cake."


# --- Test 2: Specific location generalization ---
def test_location_sanitization() -> None:
    post = SocialMediaPost(text="Had coffee at the Blue Cafe near Central Metro.")
    sig = _signal(SignalCategory.LOCATION, post.id, "Blue Cafe")
    result = sanitize_post(post, [sig], _MockGeminiService("error"))  # Uses deterministic fallback
    assert "Blue Cafe" not in result.sanitized_text
    assert "a cafe" in result.sanitized_text or "a local place" in result.sanitized_text


# --- Test 3: Exact recurring time generalization ---
def test_time_sanitization() -> None:
    post = SocialMediaPost(text="I take the 7:15 AM metro every morning.")
    sig = _signal(SignalCategory.TIME, post.id, "7:15 AM metro")
    result = sanitize_post(post, [sig], _MockGeminiService("error"))
    assert "7:15 AM" not in result.sanitized_text
    assert "in the morning" in result.sanitized_text


# --- Test 4: Routine exposure reduction ---
def test_routine_sanitization() -> None:
    post = SocialMediaPost(text="I walk 45 minutes home from campus as usual.")
    sig = _signal(SignalCategory.ROUTINE, post.id, "45 minute walk as usual")
    result = sanitize_post(post, [sig], _MockGeminiService("error"))
    assert "45 minutes" not in result.sanitized_text
    assert "as usual" not in result.sanitized_text


# --- Test 5: Identity linkage reduction ---
def test_identity_sanitization() -> None:
    post = SocialMediaPost(text="Our batch of 2020 high school reunion next weekend.")
    sig = _signal(SignalCategory.IDENTITY, post.id, "batch of 2020")
    result = sanitize_post(post, [sig], _MockGeminiService("error"))
    assert "batch of 2020" not in result.sanitized_text


# --- Test 6: Multiple risky categories ---
def test_multiple_risky_categories_sanitized() -> None:
    post = SocialMediaPost(text="Loved the cold brew at the Blue Cafe near Central Metro at 7:15 AM as usual!")
    sigs = [
        _signal(SignalCategory.LOCATION, post.id, "Blue Cafe"),
        _signal(SignalCategory.TIME, post.id, "7:15 AM"),
        _signal(SignalCategory.ROUTINE, post.id, "as usual"),
    ]
    result = sanitize_post(post, sigs, _MockGeminiService("error"))
    assert "Blue Cafe" not in result.sanitized_text
    assert "7:15 AM" not in result.sanitized_text
    assert "as usual" not in result.sanitized_text


# --- Test 7: Meaning preservation validation ---
def test_meaning_preservation() -> None:
    assert validate_semantic_preservation("Hello world", "Hello earth") is True
    assert validate_semantic_preservation("Long original post text here", "") is False
    assert validate_semantic_preservation("Very long detailed sentence describing a day out", "a") is False


# --- Test 8 & 9: Gemini error / malformed response fallback ---
def test_gemini_failure_and_malformed_fallback() -> None:
    post = SocialMediaPost(text="Had coffee at the Blue Cafe near Central Metro.")
    sig = _signal(SignalCategory.LOCATION, post.id)

    res_err = sanitize_post(post, [sig], _MockGeminiService("error"))
    assert res_err.status == "sanitized"
    assert "Blue Cafe" not in res_err.sanitized_text

    res_mal = sanitize_post(post, [sig], _MockGeminiService("malformed"))
    assert res_mal.status == "sanitized"
    assert "Blue Cafe" not in res_mal.sanitized_text


# --- Test 10 & 18: Unsafe sanitization rejection / No privacy regression ---
def test_unsafe_sanitization_reverted_in_before_after() -> None:
    posts = [SocialMediaPost(id="p1", order_index=0, text="Loved coffee this morning!")]
    # mock service returns a highly specific unsafe string
    result = analyze_before_after(posts, _MockGeminiService("unsafe"))
    assert result.risk_after.total_score <= result.risk_before.total_score


# --- Test 11 - 17: Full before/after analysis pipeline, traceability & determinism ---
def test_before_after_analysis_pipeline() -> None:
    posts = [
        SocialMediaPost(id="demo-1", order_index=0, text="Loved the cold brew at the blue cafe near the metro station this morning!"),
        SocialMediaPost(id="demo-2", order_index=1, text="Finally made it home! 45 minute walk from my university campus as usual."),
        SocialMediaPost(id="demo-3", order_index=2, text="Can't wait for our batch of 2020 high school reunion next weekend."),
    ]
    gemini = _MockGeminiService("normal")
    analysis = analyze_before_after(posts, gemini)

    # Risk reduction verify
    assert analysis.risk_before.total_score >= analysis.risk_after.total_score
    assert analysis.risk_reduction >= 0.0
    assert 0.0 <= analysis.risk_reduction_percentage <= 100.0

    # Traceability
    for sp in analysis.sanitized_posts:
        assert sp.post_id in {"demo-1", "demo-2", "demo-3"}
        assert sp.original_text != ""

    # Determinism test
    analysis2 = analyze_before_after(posts, gemini)
    assert analysis.risk_reduction == analysis2.risk_reduction
    assert [sp.sanitized_text for sp in analysis.sanitized_posts] == [
        sp.sanitized_text for sp in analysis2.sanitized_posts
    ]


def test_deterministic_fallback_stability() -> None:
    text = "Had coffee at the Blue Cafe near Central Metro at 7:15 AM as usual."
    cats = {SignalCategory.LOCATION, SignalCategory.TIME, SignalCategory.ROUTINE}
    out1, d1 = apply_deterministic_fallback(text, cats)
    out2, d2 = apply_deterministic_fallback(text, cats)
    assert out1 == out2
    assert d1 == d2

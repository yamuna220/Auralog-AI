"""
Tests for services/gemini_service.py.

Every test mocks the underlying google-genai client (`_client.models.
generate_content`) directly — no real API key or network call is ever made,
matching the testing strategy from the architecture (Step 1, section 14)
and the explicit Step 4 requirement that tests must not require a real
Gemini API key.
"""

from __future__ import annotations

import json

import pytest

from core.models import SignalCategory, SocialMediaPost
from services.gemini_service import (
    GeminiAuthenticationError,
    GeminiRateLimitError,
    GeminiResponseValidationError,
    GeminiService,
)
from google.genai import errors as genai_errors


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    """Stands in for client.models; call_log records every request's prompt."""

    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self._responses = list(responses)
        self.call_log: list[str] = []

    def generate_content(self, *, model, contents, config):  # noqa: D401
        self.call_log.append(contents)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self.models = _FakeModels(responses)


def _service_with_fake_client(responses: list[_FakeResponse | Exception]) -> GeminiService:
    """Build a GeminiService in 'live mode' backed by a fake client — no real key/network."""
    service = GeminiService(api_key="fake-key-for-tests")
    service._client = _FakeClient(responses)
    return service


def _valid_analysis_json() -> str:
    return json.dumps(
        {
            "signals": [
                {
                    "id": "s1",
                    "post_id": "post-1",
                    "category": "location",
                    "signal": "near a metro station",
                    "detail": "The post mentions a location near a metro station.",
                    "confidence": 0.8,
                    "sensitivity": "medium",
                    "evidence": "near the metro",
                    "directly_stated": True,
                    "inferred": False,
                },
                {
                    "id": "s2",
                    "post_id": "post-2",
                    "category": "routine",
                    "signal": "recurring visit",
                    "detail": "The routine appears to repeat.",
                    "confidence": 0.5,
                    "sensitivity": "low",
                    "evidence": "as usual",
                    "directly_stated": False,
                    "inferred": True,
                },
            ],
            "correlations": [
                {
                    "id": "c1",
                    "signal_ids": ["s1", "s2"],
                    "relationship_type": "location_routine",
                    "combined_confidence": 0.6,
                    "description": "May narrow the general area.",
                }
            ],
            "inference_chains": [
                {
                    "id": "i1",
                    "correlation_ids": ["c1"],
                    "resulting_inference": "Could potentially narrow the commute area.",
                    "confidence": 0.5,
                }
            ],
        }
    )


def _post(text: str = "Coffee near the metro this morning.") -> SocialMediaPost:
    return SocialMediaPost(id="post-1", text=text)


def _second_post(text: str = "Walked the same route home again.") -> SocialMediaPost:
    return SocialMediaPost(id="post-2", text=text)


# ---------------------------------------------------------------------------
# Test 1 — Valid Gemini JSON is successfully parsed
# ---------------------------------------------------------------------------


def test_valid_json_is_parsed_into_validated_models() -> None:
    service = _service_with_fake_client([_FakeResponse(_valid_analysis_json())])
    result = service.analyze_posts([_post(), _second_post()])

    assert len(result.signals) == 2
    assert result.signals[0].category == SignalCategory.LOCATION
    assert result.signals[0].post_id == "post-1"
    assert len(result.correlations) == 1
    assert result.correlations[0].combined_confidence < 1.0
    assert len(result.inference_chains) == 1
    assert result.demo_mode is False


# ---------------------------------------------------------------------------
# Test 2 — Missing field is handled, not a crash
# ---------------------------------------------------------------------------


def test_missing_top_level_field_is_handled_gracefully() -> None:
    # No "correlations" or "inference_chains" key at all.
    raw = json.dumps(
        {
            "signals": [
                {
                    "id": "s1",
                    "post_id": "post-1",
                    "category": "location",
                    "signal": "x",
                    "confidence": 0.7,
                    "directly_stated": True,
                }
            ]
        }
    )
    service = _service_with_fake_client([_FakeResponse(raw)])
    result = service.analyze_posts([_post()])

    assert len(result.signals) == 1
    assert result.correlations == []
    assert result.inference_chains == []


def test_missing_signal_field_falls_back_to_safe_defaults() -> None:
    # Signal missing "confidence" and "category" entirely.
    raw = json.dumps(
        {
            "signals": [{"id": "s1", "post_id": "post-1", "signal": "vague clue"}],
            "correlations": [],
            "inference_chains": [],
        }
    )
    service = _service_with_fake_client([_FakeResponse(raw)])
    result = service.analyze_posts([_post()])

    assert len(result.signals) == 1
    signal = result.signals[0]
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.category == SignalCategory.PERSONAL_INFORMATION  # safe fallback bucket


# ---------------------------------------------------------------------------
# Test 3 — Malformed JSON is a controlled failure (after the corrective retry)
# ---------------------------------------------------------------------------


def test_malformed_json_raises_after_retry_exhausted() -> None:
    # Both the initial call and the corrective retry return unparsable text.
    service = _service_with_fake_client(
        [_FakeResponse("not json at all"), _FakeResponse("still not json")]
    )
    with pytest.raises(GeminiResponseValidationError):
        service.analyze_posts([_post()])
    # Confirms the corrective retry actually happened (2 calls, not 1).
    assert len(service._client.models.call_log) == 2


def test_malformed_json_then_valid_json_on_retry_succeeds() -> None:
    service = _service_with_fake_client(
        [_FakeResponse("not json"), _FakeResponse(_valid_analysis_json())]
    )
    result = service.analyze_posts([_post(), _second_post()])
    assert len(result.signals) == 2


# ---------------------------------------------------------------------------
# Test 4 — Empty response is a controlled failure
# ---------------------------------------------------------------------------


def test_empty_response_raises_controlled_error() -> None:
    service = _service_with_fake_client([_FakeResponse(""), _FakeResponse("")])
    with pytest.raises(GeminiResponseValidationError):
        service.analyze_posts([_post()])


# ---------------------------------------------------------------------------
# Test 5 — No API key means DEMO MODE
# ---------------------------------------------------------------------------


def test_no_api_key_runs_in_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("config.GEMINI_API_KEY", None)
    service = GeminiService(api_key=None)
    assert service.demo_mode is True
    assert service._client is None

    from core.demo_data import DEMO_POSTS

    result = service.analyze_posts(DEMO_POSTS)
    assert result.demo_mode is True
    assert len(result.signals) > 0  # canonical demo posts DO have pre-computed signals


def test_demo_mode_yields_no_signals_for_non_demo_posts() -> None:
    service = GeminiService(api_key=None)
    arbitrary_post = SocialMediaPost(id="not-a-demo-post", text="Some arbitrary text.")
    result = service.analyze_posts([arbitrary_post])
    assert result.demo_mode is True
    assert result.signals == []  # never fabricates analysis it didn't do


# ---------------------------------------------------------------------------
# Test 6 — Ordinary text yields no fabricated sensitive signals
# ---------------------------------------------------------------------------


def test_ordinary_text_with_no_signals_returns_empty_lists() -> None:
    raw = json.dumps({"signals": [], "correlations": [], "inference_chains": []})
    service = _service_with_fake_client([_FakeResponse(raw)])
    result = service.analyze_posts([_post("Just had a nice cup of tea.")])

    assert result.signals == []
    assert result.correlations == []
    assert result.inference_chains == []


# ---------------------------------------------------------------------------
# Test 7 — Location + routine clues extract appropriate signals
# ---------------------------------------------------------------------------


def test_location_and_routine_clues_extract_matching_categories() -> None:
    service = _service_with_fake_client([_FakeResponse(_valid_analysis_json())])
    result = service.analyze_posts([_post(), _second_post()])

    categories = {s.category for s in result.signals}
    assert SignalCategory.LOCATION in categories
    assert SignalCategory.ROUTINE in categories
    # The routine signal was NOT marked directly_stated in the fixture.
    routine_signal = next(s for s in result.signals if s.category == SignalCategory.ROUTINE)
    assert routine_signal.directly_stated is False
    assert routine_signal.inferred is True


# ---------------------------------------------------------------------------
# Test 8 — Sanitized output is produced and preserves meaning (non-empty,
# not identical to a blank/placeholder string)
# ---------------------------------------------------------------------------


def test_sanitize_post_returns_rewritten_text_in_live_mode() -> None:
    from core.models import PrivacySignal

    service = _service_with_fake_client(
        [_FakeResponse("Finally home after a long walk from campus!")]
    )
    post = SocialMediaPost(id="post-1", text="45 minute walk from my university campus.")
    flagged = [
        PrivacySignal(
            post_id="post-1",
            category=SignalCategory.EDUCATION,
            signal="university campus commute",
            confidence=0.8,
            directly_stated=True,
            evidence="university campus",
        )
    ]
    res = service.sanitize_post(post, flagged)
    rewritten = res["sanitized_text"] if isinstance(res, dict) else res
    assert rewritten == "Finally home after a long walk from campus!"
    assert rewritten != post.text


def test_sanitize_post_demo_mode_uses_canonical_rewrite() -> None:
    from core.demo_data import DEMO_POSTS, DEMO_SANITIZED_TEXT_BY_POST_ID, DEMO_SIGNALS_BY_POST

    service = GeminiService(api_key=None)
    demo_post = DEMO_POSTS[1]  # "45 minute walk from my university campus as usual."
    flagged = DEMO_SIGNALS_BY_POST[demo_post.id]

    res = service.sanitize_post(demo_post, flagged)
    rewritten = res["sanitized_text"] if isinstance(res, dict) else res
    assert rewritten == DEMO_SANITIZED_TEXT_BY_POST_ID[demo_post.id]
    assert rewritten != demo_post.text


def test_sanitize_post_with_no_flagged_signals_returns_original_unchanged() -> None:
    service = _service_with_fake_client([])  # no call should be made at all
    post = SocialMediaPost(text="hello world")
    res = service.sanitize_post(post, [])
    rewritten = res["sanitized_text"] if isinstance(res, dict) else res
    assert rewritten == "hello world"


# ---------------------------------------------------------------------------
# Additional error-translation coverage (auth / rate limit)
# ---------------------------------------------------------------------------


def test_authentication_error_is_translated() -> None:
    client_error = genai_errors.ClientError(
        code=401, response_json={"message": "invalid API key"}
    )
    service = _service_with_fake_client([client_error])
    with pytest.raises(GeminiAuthenticationError):
        service.analyze_posts([_post()])


def test_rate_limit_error_is_translated() -> None:
    client_error = genai_errors.ClientError(
        code=429, response_json={"message": "rate limit exceeded"}
    )
    service = _service_with_fake_client([client_error])
    with pytest.raises(GeminiRateLimitError):
        service.analyze_posts([_post()])


def test_api_key_never_appears_in_raised_error_message() -> None:
    secret_key = "sk-super-secret-value-12345"
    client_error = genai_errors.ClientError(
        code=401, response_json={"message": f"key {secret_key} was rejected"}
    )
    service = GeminiService(api_key=secret_key)
    service._client = _FakeClient([client_error])
    try:
        service.analyze_posts([_post()])
    except GeminiAuthenticationError as exc:
        assert secret_key not in str(exc)
    else:
        pytest.fail("expected GeminiAuthenticationError")

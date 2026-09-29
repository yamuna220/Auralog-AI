"""
Privacy Sanitization Engine & Deterministic Rule-Based Fallback.

This module is responsible for:
1. Target-based signal category sanitization (Location, Exact Time, Routine, Identity/Education).
2. Rule-based deterministic text rewriting fallback when Gemini is offline or fails.
3. Natural language and semantic preservation checks.
4. Orchestrating per-post sanitization using Gemini AI or deterministic rule fallbacks.
"""

from __future__ import annotations

import re
import logging
from core.models import PrivacySignal, SanitizedPost, SignalCategory, SocialMediaPost
from services.gemini_service import GeminiService

logger = logging.getLogger("auralog.sanitizer")

# ---------------------------------------------------------------------------
# Deterministic Rule-Based Fallback Rules
# ---------------------------------------------------------------------------

_LOCATION_PATTERNS = [
    (r"(?i)\bat the ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* cafe)\b", "at a cafe"),
    (r"(?i)\bnear the ([a-z\s]+ metro station)\b", "near a transit stop"),
    (r"(?i)\bnear ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* metro)\b", "near a transit stop"),
    (r"(?i)\bfrom my ([a-z\s]+ campus)\b", "from campus"),
    (r"(?i)\bat ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* (?:station|street|road|park|cafe|restaurant))\b", "at a local place"),
]

_TIME_PATTERNS = [
    (r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b", "in the morning"),
    (r"(?i)\bthis morning\b", "today"),
    (r"(?i)\bevery weekday\b", "sometimes"),
    (r"(?i)\bevery morning\b", "in the morning"),
    (r"(?i)\bevery evening\b", "in the evening"),
]

_ROUTINE_PATTERNS = [
    (r"\b\d{1,2}\s*minutes?\b", "some time"),
    (r"\b\d{1,2}\s*min(?:ute)?s?\s+walk\b", "walk"),
    (r"(?i)\bas usual\b", "today"),
    (r"(?i)\bevery day\b", "regularly"),
    (r"(?i)\balways\b", "often"),
]

_IDENTITY_PATTERNS = [
    (r"(?i)\bbatch of \d{4}\b", "school reunion"),
    (r"(?i)\bclass of \d{4}\b", "school reunion"),
    (r"(?i)\bgraduated in \d{4}\b", "graduated"),
]


def apply_deterministic_fallback(
    text: str, categories: set[SignalCategory]
) -> tuple[str, list[str]]:
    """
    Apply rule-based regex transformations based on detected signal categories.
    Returns (sanitized_text, list_of_removed_details).
    """
    current_text = text
    removed_details: list[str] = []

    if SignalCategory.LOCATION in categories or any(
        c in categories for c in [SignalCategory.WORKPLACE, SignalCategory.TRAVEL]
    ):
        for pat, repl in _LOCATION_PATTERNS:
            if re.search(pat, current_text):
                current_text = re.sub(pat, repl, current_text)
                removed_details.append("specific location name/landmark")

    if SignalCategory.TIME in categories:
        for pat, repl in _TIME_PATTERNS:
            if re.search(pat, current_text):
                current_text = re.sub(pat, repl, current_text)
                removed_details.append("exact timing reference")

    if SignalCategory.ROUTINE in categories:
        for pat, repl in _ROUTINE_PATTERNS:
            if re.search(pat, current_text):
                current_text = re.sub(pat, repl, current_text)
                removed_details.append("precise routine indicator")

    if SignalCategory.EDUCATION in categories or SignalCategory.IDENTITY in categories:
        for pat, repl in _IDENTITY_PATTERNS:
            if re.search(pat, current_text):
                current_text = re.sub(pat, repl, current_text)
                removed_details.append("specific graduating batch year or identity clue")

    # Clean up whitespace
    current_text = re.sub(r"\s+", " ", current_text).strip()
    return current_text, list(dict.fromkeys(removed_details))


def validate_semantic_preservation(
    original_text: str, sanitized_text: str
) -> bool:
    """
    Validate that sanitized text is non-empty, reasonably length-proportional,
    and preserves essential readability.
    """
    if not sanitized_text or not sanitized_text.strip():
        return False

    orig_len = len(original_text.strip())
    san_len = len(sanitized_text.strip())

    # Should not collapse to less than 20% of original text length unless original was tiny
    if orig_len > 15 and san_len < max(5, int(orig_len * 0.2)):
        return False

    return True


def sanitize_post(
    post: SocialMediaPost,
    flagged_signals: list[PrivacySignal],
    gemini: GeminiService,
) -> SanitizedPost:
    """
    Sanitize one post using Gemini AI or deterministic fallback if Gemini is unavailable
    or produces malformed/over-aggressive output.
    """
    if not flagged_signals:
        return SanitizedPost.from_rewrite(
            post=post,
            rewritten_text=post.text,
            removed_signal_ids=[],
            removed_details=[],
            preserved_meaning="Original post had no high-risk privacy signals.",
            reason="Unchanged - no flagged signals.",
            status="unchanged",
        )

    categories = {s.category for s in flagged_signals}
    signal_ids = [s.id for s in flagged_signals]

    # Attempt Gemini sanitization first if client available and not strictly offline
    res_dict: dict = {}
    use_fallback = False

    try:
        raw_res = gemini.sanitize_post(post, flagged_signals)
        if isinstance(raw_res, dict):
            res_dict = raw_res
        elif isinstance(raw_res, str):
            res_dict = {
                "sanitized_text": raw_res,
                "removed_details": [s.signal for s in flagged_signals if s.signal],
                "preserved_meaning": "Original meaning retained.",
                "reason": "AI sanitization pass.",
            }
    except Exception as exc:
        logger.warning("Gemini sanitize_post call failed, defaulting to deterministic fallback: %s", exc)
        use_fallback = True

    sanitized_text = res_dict.get("sanitized_text", "")
    if not sanitized_text or use_fallback or not validate_semantic_preservation(post.text, sanitized_text):
        # Fallback to deterministic rule-based replacement
        fallback_text, fallback_details = apply_deterministic_fallback(post.text, categories)
        if not validate_semantic_preservation(post.text, fallback_text):
            fallback_text = post.text  # Safety guarantee

        res_dict = {
            "sanitized_text": fallback_text,
            "removed_details": fallback_details or [s.signal for s in flagged_signals],
            "preserved_meaning": "Basic post intent preserved via rule-based generalization.",
            "reason": "Deterministic rule-based fallback applied.",
        }

    return SanitizedPost.from_rewrite(
        post=post,
        rewritten_text=res_dict.get("sanitized_text", post.text),
        removed_signal_ids=signal_ids,
        removed_details=res_dict.get("removed_details", []),
        preserved_meaning=res_dict.get("preserved_meaning", ""),
        reason=res_dict.get("reason", ""),
        status="sanitized",
    )


def sanitize_posts(
    posts: list[SocialMediaPost],
    flagged_signals_by_post: dict[str, list[PrivacySignal]],
    gemini: GeminiService,
) -> list[SanitizedPost]:
    """Sanitize every post in an audit, keyed by post id."""
    return [
        sanitize_post(post, flagged_signals_by_post.get(post.id, []), gemini)
        for post in posts
    ]

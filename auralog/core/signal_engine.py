"""
Signal extraction — thin legacy-compatible wrapper.

STAGE 4 NOTE: the real extraction call is now GeminiService.analyze_posts,
which sends ALL posts in a single batched request (needed so Gemini can see
cross-post context for correlations/inference chains — see
services/gemini_service.py's module docstring). services/analysis_service.py
calls gemini.analyze_posts(posts) directly for the main pipeline.

This module is kept only as a small, stable per-post/per-audit convenience
wrapper around that same call, for any caller that still wants a
"just give me the signals" interface without touching correlations/
inference chains.
"""

from __future__ import annotations

from core.models import PrivacySignal, SocialMediaPost
from services.gemini_service import GeminiService


def extract_signals_for_post(
    post: SocialMediaPost, gemini: GeminiService
) -> list[PrivacySignal]:
    """Extract privacy-relevant signals from a single post."""
    if not post.text or not post.text.strip():
        return []
    return gemini.analyze_posts([post]).signals


def extract_signals_for_posts(
    posts: list[SocialMediaPost], gemini: GeminiService
) -> dict[str, list[PrivacySignal]]:
    """
    Run extraction across all posts in an audit in ONE batched Gemini call
    (not one call per post — see module docstring), then group the results
    back by post id for callers that want a {post_id: [PrivacySignal, ...]}
    shape. Returns {post_id: []} for any post that yielded no signals.
    """
    result: dict[str, list[PrivacySignal]] = {post.id: [] for post in posts}
    if not posts:
        return result
    analysis = gemini.analyze_posts(posts)
    for signal in analysis.signals:
        result.setdefault(signal.post_id, []).append(signal)
    return result

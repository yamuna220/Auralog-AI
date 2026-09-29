"""
Thin wrapper around the google-genai SDK.

This is the ONLY module that imports the Gemini SDK directly — everything
else in the app talks to `GeminiService`, never to `google.genai` itself.
That makes it trivial to mock in tests, swap models, or fall back to
DEMO_MODE without touching extraction/correlation/scoring logic.

Primary interface:
  - analyze_posts(posts): the main semantic-analysis call. Sends ALL posts
    in one request (not one call per post) so Gemini can see cross-post
    context and propose candidate correlations/inference chains, not just
    per-post signals. Returns a GeminiAnalysisResult of already-validated
    core/models.py objects.
  - sanitize_post(post, flagged_signals): a separate, smaller call used only
    when the user explicitly asks to sanitize specific flagged signals.

ARCHITECTURE NOTE — what Gemini's output is (and is not) used for:
  - PrivacySignal objects from analyze_posts ARE the signals that flow into
    the deterministic core/correlation_engine.py and core/risk_engine.py —
    this is the one place Gemini's output feeds the score.
  - CrossPostCorrelation / InferenceChain objects from analyze_posts are
    Gemini's own SEMANTIC read of how signals might relate — kept and
    exposed for narrative/explanation purposes, but they do NOT replace or
    feed into core/correlation_engine.py's deterministic correlations, which
    remain the only correlations the risk score is computed from. Gemini
    never produces, and this module never accepts, a total/overall risk
    score field.

ERROR HANDLING: every google-genai SDK exception is caught and translated
into one of the typed GeminiServiceError subclasses below, with a
user-safe message (no API keys, no raw stack traces). Malformed JSON gets
one corrective retry (config.GEMINI_MAX_RETRIES) before giving up.
"""

from __future__ import annotations

import logging

import config
from core.models import (
    CrossPostCorrelation,
    InferenceChain,
    PrivacySignal,
    SocialMediaPost,
)
from services.prompt_templates import (
    SYSTEM_INSTRUCTION,
    build_analysis_prompt,
    build_corrective_prompt,
    build_sanitize_prompt,
)
from utils.json_safety import LLMJsonError, parse_llm_json, validate_analysis_shape

logger = logging.getLogger("auralog.gemini")

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - genai is a required dependency,
    genai = None  # but this keeps import-time failures graceful in demo mode.
    genai_errors = None
    genai_types = None


# ---------------------------------------------------------------------------
# Typed errors — safe to show to the user (no secrets, no raw tracebacks)
# ---------------------------------------------------------------------------


class GeminiServiceError(Exception):
    """Base class for all AURAlog Gemini-integration errors."""


class GeminiNotConfiguredError(GeminiServiceError):
    """No API key is configured — the caller should be routed to demo mode."""


class GeminiAuthenticationError(GeminiServiceError):
    """The configured API key was rejected by Gemini (invalid/expired/revoked)."""


class GeminiRateLimitError(GeminiServiceError):
    """Gemini's rate limit was hit."""


class GeminiNetworkError(GeminiServiceError):
    """Could not reach Gemini at all (DNS, connection, timeout)."""


class GeminiAPIError(GeminiServiceError):
    """Gemini reached the server but returned an error response."""


class GeminiResponseValidationError(GeminiServiceError):
    """
    Gemini responded, but the response was empty, not valid JSON (after the
    corrective retry), or otherwise unusable. This is the "never silently
    accept malformed AI output" failure mode.
    """


def _redact(text: str, api_key: str | None) -> str:
    """Strip any occurrence of the API key out of a string before logging it."""
    if not api_key:
        return text
    return text.replace(api_key, "[REDACTED]")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class GeminiAnalysisResult:
    """
    Bundle returned by analyze_posts. A plain container (not a core/models.py
    Pydantic model) since it's a transient service-layer return value, not
    part of the validated pipeline state — every object inside it, however,
    IS a validated core/models.py instance.
    """

    __slots__ = ("signals", "correlations", "inference_chains", "demo_mode")

    def __init__(
        self,
        signals: list[PrivacySignal],
        correlations: list[CrossPostCorrelation],
        inference_chains: list[InferenceChain],
        demo_mode: bool,
    ) -> None:
        self.signals = signals
        self.correlations = correlations
        self.inference_chains = inference_chains
        self.demo_mode = demo_mode


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class GeminiService:
    """Single access point for all Gemini API calls."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else self._resolve_api_key()
        self.demo_mode = self.api_key is None or self.api_key.strip() == ""
        self.model_name = config.GEMINI_MODEL_NAME
        self._client = None
        if not self.demo_mode and genai is not None:
            self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _resolve_api_key() -> str | None:
        """
        Resolve the API key with a fallback chain: config.py (.env) first,
        then Streamlit secrets if a Streamlit context is available. Wrapped
        defensively since st.secrets raises if no secrets.toml exists at
        all, which is a completely normal case for this app (.env is the
        documented/primary path — see .env.example).
        """
        if config.GEMINI_API_KEY:
            return config.GEMINI_API_KEY
        try:
            import streamlit as st

            return st.secrets.get("GEMINI_API_KEY")  # type: ignore[union-attr]
        except Exception:
            return None

    # -- Analysis (signals + correlations + inference chains) ---------------
    def analyze_posts(self, posts: list[SocialMediaPost]) -> GeminiAnalysisResult:
        """
        Run the full semantic analysis pass over ALL given posts in a single
        request. Returns validated PrivacySignal/CrossPostCorrelation/
        InferenceChain objects.

        In DEMO_MODE (no API key configured), returns pre-computed demo data
        for AURAlog's canonical example posts (see core/demo_data.py) rather
        than calling Gemini — structured exactly like a real response so the
        rest of the pipeline (correlation engine, risk engine, graph) runs
        identically either way.

        Raises a GeminiServiceError subclass on failure in live mode. Never
        silently returns fabricated or unvalidated data.
        """
        if not posts:
            return GeminiAnalysisResult([], [], [], demo_mode=self.demo_mode)

        if self.demo_mode or self._client is None:
            return self._demo_analysis(posts)

        prompt = build_analysis_prompt(posts)
        last_error: LLMJsonError | None = None
        data: dict | None = None

        for attempt in range(config.GEMINI_MAX_RETRIES + 1):
            current_prompt = (
                prompt
                if attempt == 0
                else build_corrective_prompt(prompt, str(last_error))
            )
            raw_text = self._call_model(current_prompt)
            try:
                parsed = parse_llm_json(raw_text)
                data = validate_analysis_shape(parsed)
                break
            except LLMJsonError as exc:
                last_error = exc
                logger.warning(
                    "Gemini analysis response failed JSON validation "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    config.GEMINI_MAX_RETRIES + 1,
                    exc,
                )
                continue

        if data is None:
            raise GeminiResponseValidationError(
                "Gemini did not return a usable analysis after "
                f"{config.GEMINI_MAX_RETRIES + 1} attempt(s). Last error: "
                f"{last_error}"
            )

        known_post_ids = {post.id for post in posts}
        signals = self._parse_signals(data["signals"], known_post_ids)
        signal_post_id_lookup = {s.id: s.post_id for s in signals}
        correlations = CrossPostCorrelation.list_from_gemini_json(
            data["correlations"], signal_post_id_lookup
        )
        inference_chains = InferenceChain.list_from_gemini_json(data["inference_chains"])

        return GeminiAnalysisResult(
            signals=signals,
            correlations=correlations,
            inference_chains=inference_chains,
            demo_mode=False,
        )

    @staticmethod
    def _parse_signals(
        raw_signals: list[dict], known_post_ids: set[str]
    ) -> list[PrivacySignal]:
        """
        Parse signal dicts, grouping by their own declared post_id (each
        signal names which post it came from — unlike the old per-post
        extraction design, one batched response covers every post). Any
        signal naming a post_id outside the known set is dropped rather than
        raising, since a hallucinated post_id should never be able to break
        the whole batch.
        """
        signals: list[PrivacySignal] = []
        for item in raw_signals:
            if not isinstance(item, dict):
                continue
            post_id = str(item.get("post_id", "")).strip()
            if post_id not in known_post_ids:
                continue
            parsed = PrivacySignal.list_from_gemini_json([item], post_id=post_id)
            signals.extend(parsed)
        return signals

    def _demo_analysis(self, posts: list[SocialMediaPost]) -> GeminiAnalysisResult:
        """
        DEMO_MODE fallback. Only AURAlog's own canonical example posts (see
        core/demo_data.py) have pre-computed demo signals/correlations/
        inference chains; any other post the user typed while running
        without an API key simply yields no signals — DEMO_MODE never
        pretends to have analyzed text it didn't actually analyze.
        """
        from core.demo_data import (
            DEMO_CORRELATIONS,
            DEMO_INFERENCE_CHAINS,
            DEMO_SIGNALS_BY_POST,
        )

        # Iterate `posts` directly (already ordered) rather than through a
        # `set`, so signal order — and therefore everything downstream that
        # depends on it (correlation_engine's tie-breaking, category
        # ordering, etc.) — stays deterministic across process restarts,
        # not just within a single run.
        signals = [
            s
            for post in posts
            if post.id in DEMO_SIGNALS_BY_POST
            for s in DEMO_SIGNALS_BY_POST[post.id]
        ]
        signal_ids = {s.id for s in signals}
        correlations = [
            c for c in DEMO_CORRELATIONS if set(c.signal_ids).issubset(signal_ids)
        ]
        correlation_ids = {c.id for c in correlations}
        inference_chains = [
            chain
            for chain in DEMO_INFERENCE_CHAINS
            if set(chain.correlation_ids).issubset(correlation_ids)
        ]

        return GeminiAnalysisResult(
            signals=signals,
            correlations=correlations,
            inference_chains=inference_chains,
            demo_mode=True,
        )

    # -- Sanitization ---------------------------------------------------------
    def sanitize_post(
        self, post: SocialMediaPost, flagged_signals: list[PrivacySignal]
    ) -> dict:
        """
        Ask Gemini to rewrite a post, removing/obscuring the flagged
        signals while preserving tone and meaning. Returns a dict containing
        sanitized_text, removed_details, preserved_meaning, and reason.

        In DEMO_MODE, returns a pre-written safer version for AURAlog's
        canonical example posts (see core/demo_data.py).
        """
        if not flagged_signals:
            return {
                "sanitized_text": post.text,
                "removed_details": [],
                "preserved_meaning": "No flagged privacy signals to sanitize.",
                "reason": "Post contains no high-risk privacy signals.",
            }

        if self.demo_mode or self._client is None:
            from core.demo_data import DEMO_SANITIZED_TEXT_BY_POST_ID

            sanitized_text = DEMO_SANITIZED_TEXT_BY_POST_ID.get(post.id, post.text)
            return {
                "sanitized_text": sanitized_text,
                "removed_details": [s.signal for s in flagged_signals if s.signal],
                "preserved_meaning": "Original post theme and intent preserved in demo mode.",
                "reason": "Demo mode pre-configured sanitization.",
            }

        prompt = build_sanitize_prompt(post, flagged_signals)
        try:
            raw_text = self._call_model(prompt, expect_json=False)
            try:
                parsed = parse_llm_json(raw_text)
                if isinstance(parsed, dict) and "sanitized_text" in parsed:
                    sanitized_text = str(parsed.get("sanitized_text", "")).strip()
                    if sanitized_text:
                        removed_details = parsed.get("removed_details", [])
                        if not isinstance(removed_details, list):
                            removed_details = []
                        return {
                            "sanitized_text": sanitized_text,
                            "removed_details": [str(d) for d in removed_details],
                            "preserved_meaning": str(parsed.get("preserved_meaning", "")).strip(),
                            "reason": str(parsed.get("reason", "")).strip(),
                        }
            except LLMJsonError:
                pass

            sanitized_text = raw_text.strip().strip('"')
            if not sanitized_text:
                raise GeminiResponseValidationError(
                    "Gemini returned an empty sanitized post."
                )
            return {
                "sanitized_text": sanitized_text,
                "removed_details": [s.signal for s in flagged_signals if s.signal],
                "preserved_meaning": "General post meaning retained.",
                "reason": "Sanitization pass executed.",
            }
        except (GeminiAuthenticationError, GeminiRateLimitError, GeminiNetworkError, GeminiResponseValidationError):
            raise
        except Exception as exc:
            logger.warning("Gemini sanitization call failed: %s", exc)
            raise GeminiResponseValidationError(f"Sanitization call failed: {exc}") from exc

    # -- Low-level call + error translation ----------------------------------
    def _call_model(self, prompt: str, expect_json: bool = True) -> str:
        """
        Make one call to the Gemini API and return its raw text, translating
        every SDK-level failure into a typed GeminiServiceError with a
        user-safe message. Never logs or raises the raw API key.
        """
        config_kwargs = dict(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
            http_options=genai_types.HttpOptions(
                timeout=config.GEMINI_TIMEOUT_SECONDS * 1000
            ),
        )
        if expect_json:
            config_kwargs["response_mime_type"] = "application/json"

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.ClientError as exc:
            code = getattr(exc, "code", None)
            message = _redact(str(getattr(exc, "message", "") or str(exc)), self.api_key)
            logger.error("Gemini client error (HTTP %s): %s", code, message)
            if code in (401, 403):
                raise GeminiAuthenticationError(
                    "Gemini rejected the configured API key. Check that "
                    "GEMINI_API_KEY is valid and has not been revoked."
                ) from exc
            if code == 429:
                raise GeminiRateLimitError(
                    "Gemini's rate limit was reached. Please wait a moment "
                    "and try again."
                ) from exc
            raise GeminiAPIError(
                f"Gemini rejected the request (HTTP {code})."
            ) from exc
        except genai_errors.ServerError as exc:
            message = _redact(str(getattr(exc, "message", "") or str(exc)), self.api_key)
            logger.error("Gemini server error: %s", message)
            raise GeminiAPIError(
                "Gemini's servers returned an error. Please try again "
                "shortly."
            ) from exc
        except Exception as exc:  # network errors, timeouts, DNS, etc.
            logger.error(
                "Unexpected error calling Gemini: %s", _redact(str(exc), self.api_key)
            )
            raise GeminiNetworkError(
                "Could not reach the Gemini API. Check your network "
                "connection and try again."
            ) from exc

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise GeminiResponseValidationError("Gemini returned an empty response.")
        return text

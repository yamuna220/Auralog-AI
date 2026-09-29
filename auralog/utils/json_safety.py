"""
Strict JSON parsing/validation for LLM output.

Gemini responses are treated as UNTRUSTED input. Every response that's
supposed to be JSON goes through here before it's turned into a Signal,
Correlation, or any other typed object. On failure, callers get a typed
exception instead of a crash or silently-wrong data.
"""

from __future__ import annotations

import json


class LLMJsonError(Exception):
    """Raised when an LLM response cannot be parsed as valid JSON."""


def strip_markdown_fences(raw_text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrapping some models add anyway."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_llm_json(raw_text: str) -> dict | list:
    """
    Parse a raw LLM text response into JSON, tolerating accidental markdown
    fences. Raises LLMJsonError on failure so callers can trigger the
    corrective-retry flow instead of crashing.
    """
    if not raw_text or not raw_text.strip():
        raise LLMJsonError("LLM response was empty")

    cleaned = strip_markdown_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMJsonError(f"Could not parse LLM response as JSON: {exc}") from exc


def validate_analysis_shape(data: object) -> dict:
    """
    Ensure a parsed Gemini analysis response has the expected top-level
    shape (signals/correlations/inference_chains as lists) before it's
    handed to core/models.py's per-item parsers.

    This is deliberately lenient rather than strict: a missing key is
    treated as "the model returned none of that" (empty list) rather than a
    hard failure, since individual item validation already happens one
    level down in PrivacySignal/CrossPostCorrelation/InferenceChain's own
    from_gemini_json parsers. A response that isn't even a JSON object at
    all, though, is not recoverable here and raises.
    """
    if not isinstance(data, dict):
        raise LLMJsonError(
            f"Expected a JSON object at the top level, got {type(data).__name__}"
        )

    result = dict(data)
    for key in ("signals", "correlations", "inference_chains"):
        value = result.get(key, [])
        result[key] = value if isinstance(value, list) else []
    return result

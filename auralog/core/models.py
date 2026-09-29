"""
Structured data models for AURAlog AI.

These are strict Pydantic models (validated at construction time) that flow
through the pipeline:

    SocialMediaPost -> PrivacySignal -> CrossPostCorrelation -> InferenceChain
        -> RiskAssessment -> SanitizedPost -> CompleteAnalysis

Central safety rule enforced throughout this file, per the project brief:
UNCERTAIN INFERENCE MUST NEVER BE REPRESENTED AS CONFIRMED PERSONAL
INFORMATION. Concretely:
  - A PrivacySignal cannot be both directly_stated AND inferred.
  - A PrivacySignal cannot carry confidence 1.0 unless directly_stated.
  - A CrossPostCorrelation is, by definition, a derived relationship — it can
    never claim confidence 1.0 (it is always at least an inference).
  - An InferenceChain (a higher-order claim built from correlations) can
    never claim confidence 1.0 either.

These are enforced as Pydantic validators, not just UI conventions, so no
code path anywhere in the app can construct a "confirmed" claim out of
uncertain evidence.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalCategory(str, Enum):
    """The kinds of privacy-relevant clues the app looks for."""

    LOCATION = "location"
    ROUTINE = "routine"
    TIME = "time"
    EDUCATION = "education"
    WORKPLACE = "workplace"
    IDENTITY = "identity"
    SOCIAL_RELATIONSHIPS = "social_relationships"
    TRAVEL = "travel"
    LIFESTYLE = "lifestyle"
    PERSONAL_INFORMATION = "personal_information"


class SensitivityLevel(str, Enum):
    """How sensitive a single signal is, independent of how certain it is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceLevel(str, Enum):
    """
    How certain a claim is. Shown explicitly everywhere in the UI per the
    project brief's requirement to distinguish directly-stated, inferred,
    and speculative information.
    """

    DIRECTLY_STATED = "directly_stated"
    INFERRED = "inferred"
    SPECULATIVE = "speculative"


class CorrelationType(str, Enum):
    """
    The deterministic correlation rules core/correlation_engine.py applies.
    Each pairwise type names the two SignalCategory values it links;
    MULTI_SIGNAL covers relationships spanning 3+ signals; GENERIC_OVERLAP
    is a defensive fallback for Gemini-sourced candidates only — the
    deterministic engine's own rule table never emits it (see that
    module's docstring on avoiding false correlations).
    """

    LOCATION_ROUTINE = "location_routine"
    LOCATION_TIME = "location_time"
    EDUCATION_LOCATION = "education_location"
    EDUCATION_IDENTITY = "education_identity"
    WORKPLACE_ROUTINE = "workplace_routine"
    TIME_ROUTINE = "time_routine"
    MULTI_SIGNAL = "multi_signal"
    GENERIC_OVERLAP = "generic_overlap"


# ---------------------------------------------------------------------------
# 1. SocialMediaPost
# ---------------------------------------------------------------------------


class SocialMediaPost(BaseModel):
    """A single user-submitted (synthetic/demo) social-media post."""

    id: str = Field(default_factory=_new_id)
    text: str
    order_index: int = 0

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("post text must not be blank")
        return v.strip()


# ---------------------------------------------------------------------------
# 2 & 3. PrivacySignal / SignalCategory
# ---------------------------------------------------------------------------


class PrivacySignal(BaseModel):
    """A privacy-relevant clue extracted from a single post."""

    id: str = Field(default_factory=_new_id)
    post_id: str
    category: SignalCategory
    signal: str  # short label, e.g. "near a metro station"
    detail: str = ""  # fuller extracted description
    confidence: float = Field(ge=0.0, le=1.0)
    sensitivity: SensitivityLevel = SensitivityLevel.LOW
    evidence: str = ""  # verbatim excerpt from the post text supporting this
    directly_stated: bool = False
    inferred: bool = False

    @model_validator(mode="after")
    def _validate_evidence_consistency(self) -> "PrivacySignal":
        if self.directly_stated and self.inferred:
            raise ValueError(
                "a signal cannot be both directly_stated and inferred — "
                "pick the weaker (less certain) of the two claims"
            )
        # SAFETY RULE: an inferred or speculative signal must never be
        # represented as fully certain personal information.
        if not self.directly_stated and self.confidence >= 1.0:
            raise ValueError(
                "a signal that is not directly_stated cannot carry "
                "confidence 1.0 — only directly_stated signals may be "
                "fully certain"
            )
        return self

    @property
    def evidence_level(self) -> EvidenceLevel:
        if self.directly_stated:
            return EvidenceLevel.DIRECTLY_STATED
        if self.inferred:
            return EvidenceLevel.INFERRED
        return EvidenceLevel.SPECULATIVE

    # -- Gemini JSON interop -------------------------------------------------
    @classmethod
    def from_gemini_json(cls, data: dict[str, Any], post_id: str) -> "PrivacySignal":
        """
        Build a PrivacySignal from one entry of a Gemini extraction response.

        Gemini JSON is UNTRUSTED input: every field is normalized/clamped
        rather than trusted verbatim, and any ambiguous or contradictory
        certainty flag always resolves toward the SAFER (less certain)
        reading, never toward "confirmed".
        """
        category_raw = str(data.get("category", "")).strip().lower().replace(" ", "_")
        try:
            category = SignalCategory(category_raw)
        except ValueError:
            category = SignalCategory.PERSONAL_INFORMATION  # safe generic bucket

        sensitivity_raw = str(data.get("sensitivity", "low")).strip().lower()
        try:
            sensitivity = SensitivityLevel(sensitivity_raw)
        except ValueError:
            sensitivity = SensitivityLevel.LOW

        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(confidence, 1.0))

        directly_stated = bool(data.get("directly_stated", False))
        inferred = bool(data.get("inferred", not directly_stated))
        if directly_stated and inferred:
            # Contradictory model output — resolve to the SAFER claim.
            directly_stated = False
            inferred = True
        if not directly_stated and confidence >= 1.0:
            # Non-stated claims can never read as 100% certain.
            confidence = 0.95

        return cls(
            id=str(data.get("id") or _new_id()),
            post_id=post_id,
            category=category,
            signal=str(data.get("signal", "")).strip(),
            detail=str(data.get("detail", "")).strip(),
            confidence=confidence,
            sensitivity=sensitivity,
            evidence=str(data.get("evidence", "")).strip(),
            directly_stated=directly_stated,
            inferred=inferred,
        )

    @classmethod
    def list_from_gemini_json(
        cls, raw_items: list[dict[str, Any]], post_id: str
    ) -> list["PrivacySignal"]:
        """
        Parse a whole Gemini extraction response (a list of signal dicts) for
        one post. Malformed individual entries are skipped rather than
        failing the whole batch — one bad item from the LLM shouldn't drop
        every valid signal alongside it.
        """
        signals: list[PrivacySignal] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                signals.append(cls.from_gemini_json(item, post_id))
            except Exception:
                continue
        return signals


# ---------------------------------------------------------------------------
# 4. CrossPostCorrelation
# ---------------------------------------------------------------------------


class CrossPostCorrelation(BaseModel):
    """
    A deterministic link found between two or more signals across posts.
    Produced entirely by core/correlation_engine.py rules — never by the LLM.

    Two related but distinct 0-1 numbers live on this model, and it's worth
    being precise about which is which:
      - combined_confidence: how confident we are the underlying SIGNALS are
        correct (derived from their own extraction confidence). Like every
        derived value in this file, it can never reach 1.0 — a correlation
        is never a stated fact.
      - strength: how STRONG the modeled relationship itself is (factoring
        in category severity, signal confidence/sensitivity, whether the
        pattern repeats, and how many signals are involved). This one CAN
        reach 1.0 ("a very strong relationship") — 1.0 here describes the
        relationship's intensity, not certainty about the real world, so it
        is not subject to the same "never fully certain" rule.
    """

    id: str = Field(default_factory=_new_id)
    signal_ids: list[str] = Field(min_length=2)
    post_ids: list[str] = Field(default_factory=list)
    relationship_type: CorrelationType = CorrelationType.GENERIC_OVERLAP
    combined_confidence: float = Field(ge=0.0, le=1.0)
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str = ""  # short "relationship" summary
    potential_inference: str = ""  # cautious hedge about what this may suggest
    explanation: str = ""  # fuller, signal-grounded justification

    @field_validator("signal_ids")
    @classmethod
    def _no_duplicate_signal_ids(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("signal_ids must not contain duplicates")
        return v

    @model_validator(mode="after")
    def _correlations_are_never_fully_certain(self) -> "CrossPostCorrelation":
        # SAFETY RULE: a correlation is, by definition, a derived
        # relationship between signals — it was never itself "stated" in any
        # single post, so it must never claim full (1.0) certainty.
        if self.combined_confidence >= 1.0:
            raise ValueError(
                "a correlation cannot carry confidence 1.0 — it is always "
                "a derived inference, never a directly stated fact"
            )
        return self

    @model_validator(mode="after")
    def _must_be_cross_post_when_post_ids_known(self) -> "CrossPostCorrelation":
        # A correlation whose participating posts are known must genuinely
        # span at least two DIFFERENT posts — a same-post relationship is
        # not a cross-post correlation (see core/correlation_engine.py).
        # Left unchecked when post_ids is empty (e.g. an older/external
        # correlation whose originating posts weren't supplied) rather than
        # rejected outright, since the invariant is still enforced by
        # construction wherever this app creates one itself.
        if self.post_ids and len(set(self.post_ids)) < 2:
            raise ValueError(
                "a cross-post correlation must involve at least 2 distinct "
                "post_ids"
            )
        return self

    @property
    def evidence_level(self) -> EvidenceLevel:
        # A correlation is always at least an inference; it downgrades to
        # speculative below a fixed confidence threshold.
        return (
            EvidenceLevel.INFERRED
            if self.combined_confidence >= 0.5
            else EvidenceLevel.SPECULATIVE
        )

    # -- Gemini JSON interop -------------------------------------------------
    @classmethod
    def from_gemini_json(
        cls,
        data: dict[str, Any],
        signal_post_id_lookup: dict[str, str] | None = None,
    ) -> "CrossPostCorrelation":
        """
        Build a CrossPostCorrelation from one entry of a Gemini analysis
        response. Untrusted input, normalized the same way as
        PrivacySignal.from_gemini_json: unknown relationship types fall back
        to a safe generic bucket, confidence is clamped and — per the
        model's own safety rule — NEVER allowed to reach 1.0, since a
        correlation is always a derived relationship, not a stated fact.

        `signal_post_id_lookup` (optional {signal_id: post_id}) lets the
        caller — which already has the real PrivacySignal objects on hand —
        fill in post_ids so the cross-post validator above can actually run;
        without it, post_ids is left empty and that check is skipped.
        """
        relationship_raw = (
            str(data.get("relationship_type", data.get("relationship", "")))
            .strip()
            .lower()
            .replace(" ", "_")
        )
        try:
            relationship_type = CorrelationType(relationship_raw)
        except ValueError:
            relationship_type = CorrelationType.GENERIC_OVERLAP

        raw_signal_ids = data.get("signal_ids", [])
        if not isinstance(raw_signal_ids, list):
            raw_signal_ids = []
        # De-duplicate while preserving order (dict.fromkeys trick).
        signal_ids = list(dict.fromkeys(str(s) for s in raw_signal_ids if s))

        try:
            combined_confidence = float(
                data.get("combined_confidence", data.get("confidence", 0.5))
            )
        except (TypeError, ValueError):
            combined_confidence = 0.5
        combined_confidence = max(0.0, min(combined_confidence, 0.99))

        try:
            strength = float(data.get("strength", combined_confidence))
        except (TypeError, ValueError):
            strength = combined_confidence
        strength = max(0.0, min(strength, 1.0))

        description = str(
            data.get("description", data.get("relationship", ""))
        ).strip()
        potential_inference = str(data.get("potential_inference", "")).strip()
        explanation = str(data.get("explanation", "")).strip()

        post_ids: list[str] = []
        if signal_post_id_lookup:
            post_ids = list(
                dict.fromkeys(
                    signal_post_id_lookup[sid]
                    for sid in signal_ids
                    if sid in signal_post_id_lookup
                )
            )

        return cls(
            id=str(data.get("id") or _new_id()),
            signal_ids=signal_ids,
            post_ids=post_ids,
            relationship_type=relationship_type,
            combined_confidence=combined_confidence,
            strength=strength,
            description=description,
            potential_inference=potential_inference,
            explanation=explanation,
        )

    @classmethod
    def list_from_gemini_json(
        cls,
        raw_items: list[dict[str, Any]],
        signal_post_id_lookup: dict[str, str] | None = None,
    ) -> list["CrossPostCorrelation"]:
        """
        Parse a whole list of correlation dicts. Entries that fail
        validation (e.g. fewer than 2 usable signal_ids after cleanup, or —
        when signal_post_id_lookup is supplied — fewer than 2 distinct
        post_ids) are skipped rather than failing the whole batch.
        """
        correlations: list[CrossPostCorrelation] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                correlations.append(cls.from_gemini_json(item, signal_post_id_lookup))
            except Exception:
                continue
        return correlations


# ---------------------------------------------------------------------------
# 5. InferenceChain
# ---------------------------------------------------------------------------


class InferenceChain(BaseModel):
    """
    A higher-order inference built by chaining one or more correlations
    together into a single narrative claim (e.g. "posts suggest a narrow
    home/commute area"). Built deterministically by
    core/inference_engine.py.build_inference_chains from CrossPostCorrelations
    produced by core/correlation_engine.py — never invented by the LLM,
    though Gemini's analyze_posts can independently propose its own (kept
    separately for narrative display; see services/gemini_service.py).

    Two related but distinct 0-1 numbers live on this model, mirroring
    CrossPostCorrelation's own confidence/strength split:
      - confidence: how strongly the SUPPLIED EVIDENCE supports this
        potential inference — never a claim about real-world probability,
        and (like every derived value in this file) can never reach 1.0.
      - strength: how strong the modeled relationship itself is, factoring
        in the number and strength of contributing correlations, signal
        confidence/sensitivity, and repeated-pattern evidence — with
        diminishing returns so chain LENGTH alone can never make a chain
        of weak, barely-related signals score as strong as a few tightly
        related ones (see core/inference_engine.py for the formula).
    """

    id: str = Field(default_factory=_new_id)
    correlation_ids: list[str] = Field(min_length=1)
    signal_ids: list[str] = Field(default_factory=list)
    post_ids: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)  # short label per contributing correlation
    resulting_inference: str
    confidence: float = Field(ge=0.0, le=1.0)
    sensitivity: SensitivityLevel = SensitivityLevel.LOW
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""

    @field_validator("resulting_inference")
    @classmethod
    def _resulting_inference_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("resulting_inference must not be blank")
        return v.strip()

    @model_validator(mode="after")
    def _chains_are_never_fully_certain(self) -> "InferenceChain":
        # SAFETY RULE: same as CrossPostCorrelation — a chain of inferences
        # cannot become "confirmed" just by adding more inferences together.
        if self.confidence >= 1.0:
            raise ValueError("an inference chain cannot carry confidence 1.0")
        return self

    @property
    def evidence_level(self) -> EvidenceLevel:
        return (
            EvidenceLevel.INFERRED
            if self.confidence >= 0.5
            else EvidenceLevel.SPECULATIVE
        )

    @property
    def potential_outcome(self) -> str:
        """Alias for resulting_inference — same value, alternate name some callers expect."""
        return self.resulting_inference

    # -- Gemini JSON interop -------------------------------------------------
    @classmethod
    def from_gemini_json(cls, data: dict[str, Any]) -> "InferenceChain":
        """
        Build an InferenceChain from one entry of a Gemini explanation
        response. The score/graph are passed INTO the explanation prompt
        (see services/gemini_service.py.explain) — Gemini is only narrating
        already-computed correlations here, not inventing new confidence
        values from scratch, but the output is still treated as untrusted
        and clamped the same way as extraction output.
        """
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(confidence, 0.99))  # never allow 1.0 in

        correlation_ids = data.get("correlation_ids", [])
        if not isinstance(correlation_ids, list) or not correlation_ids:
            correlation_ids = ["unknown"]

        signal_ids = data.get("signal_ids", [])
        if not isinstance(signal_ids, list):
            signal_ids = []

        post_ids = data.get("post_ids", [])
        if not isinstance(post_ids, list):
            post_ids = []

        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        sensitivity_raw = str(data.get("sensitivity", "low")).strip().lower()
        try:
            sensitivity = SensitivityLevel(sensitivity_raw)
        except ValueError:
            sensitivity = SensitivityLevel.LOW

        try:
            strength = float(data.get("strength", confidence))
        except (TypeError, ValueError):
            strength = confidence
        strength = max(0.0, min(strength, 1.0))

        explanation = str(data.get("explanation", "")).strip()

        return cls(
            id=str(data.get("id") or _new_id()),
            correlation_ids=[str(c) for c in correlation_ids],
            signal_ids=[str(s) for s in signal_ids],
            post_ids=[str(p) for p in post_ids],
            steps=[str(s) for s in steps],
            resulting_inference=str(
                data.get("resulting_inference", data.get("potential_outcome", ""))
            ).strip()
            or "Unspecified inference",
            confidence=confidence,
            sensitivity=sensitivity,
            strength=strength,
            explanation=explanation,
        )

    @classmethod
    def list_from_gemini_json(
        cls, raw_items: list[dict[str, Any]]
    ) -> list["InferenceChain"]:
        chains: list[InferenceChain] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                chains.append(cls.from_gemini_json(item))
            except Exception:
                continue
        return chains


# ---------------------------------------------------------------------------
# 6. RiskAssessment
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """
    Score -> risk-level classification (see config.RISK_LEVEL_UPPER_BOUNDS
    for the boundaries and core/risk_engine.classify_risk_level for the
    mapping logic).

    IMPORTANT — what this measures: a risk level (and the numeric score
    behind it) describes AURAlog's MODELED PRIVACY EXPOSURE under this
    application's own scoring methodology. It is not, and must never be
    presented as, a probability of real-world harm (being attacked, doxxed,
    identified, or targeted). See core/risk_engine.py's module docstring.
    """

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CategoryScore(BaseModel):
    """Subscore for one risk category, with the factors that produced it."""

    category: str
    score: float = Field(ge=0.0, le=100.0)
    contributing_signal_ids: list[str] = Field(default_factory=list)
    contributing_correlation_ids: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """
    Full output of the deterministic scoring engine for one audit state
    (either the "before" or "after" sanitization state). Every number here
    is produced entirely by core/risk_engine.py's deterministic formula —
    NEVER by Gemini, which only supplies the underlying signals/
    correlations/inference_chains this is computed from.
    """

    total_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel = RiskLevel.LOW
    category_breakdown: list[CategoryScore] = Field(default_factory=list)
    signals: list[PrivacySignal] = Field(default_factory=list)
    correlations: list[CrossPostCorrelation] = Field(default_factory=list)
    inference_chains: list[InferenceChain] = Field(default_factory=list)

    # The three components total_score is a weighted combination of (see
    # config.SIGNAL_COMPONENT_WEIGHT & friends) — each already normalized to
    # 0-100 on its own, kept here so a judge can see exactly how much of the
    # score came from raw signals vs. cross-post correlation vs. inference
    # chains, without recomputing anything.
    signal_contribution: float = Field(default=0.0, ge=0.0, le=100.0)
    correlation_contribution: float = Field(default=0.0, ge=0.0, le=100.0)
    inference_contribution: float = Field(default=0.0, ge=0.0, le=100.0)

    # Human-readable, deterministically generated (never Gemini-generated —
    # see core/risk_engine.py) summary of what drove the score, plus the
    # top individual contributors.
    top_risk_factors: list[str] = Field(default_factory=list)
    explanation: str = ""

    @property
    def category_scores(self) -> dict[str, float]:
        """Convenience {category_display_name: score} view of category_breakdown."""
        return {c.category: c.score for c in self.category_breakdown}


# ---------------------------------------------------------------------------
# 7. SanitizedPost
# ---------------------------------------------------------------------------


class SanitizedPost(BaseModel):
    """A rewritten, safer version of a post, produced by the sanitizer."""

    post_id: str
    original_text: str
    sanitized_text: str
    removed_signal_ids: list[str] = Field(default_factory=list)
    removed_details: list[str] = Field(default_factory=list)
    preserved_meaning: str = ""
    reason: str = ""
    risk_reduction: float = 0.0
    status: str = "sanitized"  # "sanitized", "unchanged", or "unsafe"
    notes: str = ""

    @field_validator("sanitized_text")
    @classmethod
    def _sanitized_text_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("sanitized_text must not be blank")
        return v.strip()

    @classmethod
    def from_rewrite(
        cls,
        post: SocialMediaPost,
        rewritten_text: str,
        removed_signal_ids: list[str] | None = None,
        removed_details: list[str] | None = None,
        preserved_meaning: str = "",
        reason: str = "",
        risk_reduction: float = 0.0,
        status: str = "sanitized",
        notes: str = "",
    ) -> "SanitizedPost":
        """Build a SanitizedPost from a post plus the sanitizer's output."""
        return cls(
            post_id=post.id,
            original_text=post.text,
            sanitized_text=rewritten_text,
            removed_signal_ids=removed_signal_ids or [],
            removed_details=removed_details or [],
            preserved_meaning=preserved_meaning,
            reason=reason,
            risk_reduction=risk_reduction,
            status=status,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# 8. CompleteAnalysis
# ---------------------------------------------------------------------------


class CompleteAnalysis(BaseModel):
    """
    Top-level container for one full audit: every post, every extracted
    signal, every correlation and inference chain, the risk assessment(s),
    and any sanitized posts. This is also what gets serialized/deserialized
    if an audit needs to be saved, reloaded, or round-tripped through JSON.
    """

    posts: list[SocialMediaPost] = Field(default_factory=list)
    signals: list[PrivacySignal] = Field(default_factory=list)
    correlations: list[CrossPostCorrelation] = Field(default_factory=list)
    inference_chains: list[InferenceChain] = Field(default_factory=list)
    risk_assessment: RiskAssessment | None = None
    sanitized_posts: list[SanitizedPost] = Field(default_factory=list)
    risk_assessment_after: RiskAssessment | None = None

    @model_validator(mode="after")
    def _signals_must_reference_known_posts(self) -> "CompleteAnalysis":
        if not self.posts:
            return self  # nothing to cross-check yet (e.g. a fresh session)
        post_ids = {p.id for p in self.posts}
        for signal in self.signals:
            if signal.post_id not in post_ids:
                raise ValueError(
                    f"signal {signal.id} references unknown post_id "
                    f"{signal.post_id!r}"
                )
        return self

    # -- Serialization --------------------------------------------------------
    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "CompleteAnalysis":
        return cls.model_validate_json(raw)

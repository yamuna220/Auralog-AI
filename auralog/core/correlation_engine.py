"""
Cross-post correlation engine.

Pure, deterministic logic — NO Gemini calls, no Streamlit, no Plotly, no
external web requests, and NO real-person search of any kind. Takes the
full set of PrivacySignals extracted across all posts in an audit and finds
rule-based relationships between them: individually low-risk clues from
DIFFERENT posts that may become more privacy-sensitive when combined.

This module's ONE job is DETECTING relationships between signals — turning
PrivacySignals into CrossPostCorrelations. Combining those correlations
into higher-order narrative chains is a separate concern, owned entirely by
core/inference_engine.py (see that module for why they're split):

    Gemini -> PrivacySignal -> [THIS MODULE] -> CrossPostCorrelation
        -> core/inference_engine.py -> InferenceChain -> core/risk_engine.py

This is one of AURAlog's main technical differentiators, and it is meant to
be legible to a judge: every CrossPostCorrelation produced here is
traceable to one explicit rule in PAIRWISE_RULES / MULTI_SIGNAL_CATEGORY_SETS
below — never a black-box guess. It is also the ONLY source of correlations
that core/risk_engine.py's score is ultimately computed from; Gemini's own
semantic correlation guesses from analyze_posts are accepted here only as
optional, non-authoritative CORROBORATION (see `gemini_candidates` below)
— they can strengthen a correlation this engine already found on its own,
but can never create one by themselves.

--------------------------------------------------------------------------
WHY NO ENTITY-LEVEL SEMANTIC MATCHING

Matching happens at the CATEGORY level (core.models.SignalCategory), which
is itself already a lightweight, conservative form of "entity
normalization": Gemini's extraction step already folds synonyms like
"university"/"campus"/"college" into the single `education` category, so
this engine doesn't need to (and deliberately does not) re-parse raw text
to recognize such synonyms itself. What it explicitly does NOT do is
assume two different named entities are the SAME real-world thing (e.g.
"the metro" and "the Blue Line" are never treated as identical) unless the
signals themselves already say so — that kind of aggressive entity linking
is exactly the false-positive risk the project brief warns against, and a
full semantic-similarity/embedding model would add a real ML dependency
for marginal benefit here, so it's intentionally left out (see
config.REPETITION_KEYWORDS for the one small, deliberately conservative
exception: a plain keyword check for repetition language, not entity
identity).

WHY THIS STAYS FAST

For each of the ~9 rules below, this does at most one pass grouping
signals by category (O(n)) and then, for pairwise rules, compares only the
signals WITHIN the two relevant category groups (not all signals against
all signals) — so the cost is O(rules × avg_category_size²), not O(n²) or
O(n³) over the whole signal set. For multi-signal rules it's cheaper still:
one best-signal lookup per category, O(rules × n). At realistic scale (50
posts × a few signals each, spread across 10 categories) this stays well
under a few thousand comparisons.
"""

from __future__ import annotations

import config
from core.models import (
    CorrelationType,
    CrossPostCorrelation,
    PrivacySignal,
    SensitivityLevel,
    SignalCategory,
)

# ===========================================================================
# Rule tables — the whitelist of MEANINGFUL relationships this engine looks
# for. Deliberately a whitelist, not "correlate anything that co-occurs":
# this is what keeps false correlations (project brief section 9 — e.g.
# "I like pizza" + "it rained yesterday") from ever being generated. If a
# pair of signals doesn't match one of these category combinations, no
# correlation is produced for them, full stop.
# ===========================================================================

# A. Location + Routine, B. Location + Time, C. Education + Location,
# D. Education + Identity, E. Workplace + Routine, F. Time + Routine.
PAIRWISE_RULES: list[dict] = [
    {
        "categories": (SignalCategory.LOCATION, SignalCategory.ROUTINE),
        "type": CorrelationType.LOCATION_ROUTINE,
    },
    {
        "categories": (SignalCategory.LOCATION, SignalCategory.TIME),
        "type": CorrelationType.LOCATION_TIME,
    },
    {
        "categories": (SignalCategory.EDUCATION, SignalCategory.LOCATION),
        "type": CorrelationType.EDUCATION_LOCATION,
    },
    {
        "categories": (SignalCategory.EDUCATION, SignalCategory.IDENTITY),
        "type": CorrelationType.EDUCATION_IDENTITY,
    },
    {
        "categories": (SignalCategory.WORKPLACE, SignalCategory.ROUTINE),
        "type": CorrelationType.WORKPLACE_ROUTINE,
    },
    {
        "categories": (SignalCategory.TIME, SignalCategory.ROUTINE),
        "type": CorrelationType.TIME_ROUTINE,
    },
]

# G. Multi-signal (3+ categories at once) — a curated, still-conservative
# set of category TRIPLES that extend the pairwise rules above rather than
# introducing new relationship semantics. Each of these triples is a
# superset of at least one pairwise rule above (e.g. LOCATION+EDUCATION+
# ROUTINE extends EDUCATION_LOCATION and LOCATION_ROUTINE), so a
# multi-signal correlation only ever fires where the underlying pairwise
# relationships would already be independently meaningful. Kept to a small,
# curated set (not every possible category triple) so a handful of signals
# doesn't produce a combinatorial pile of near-duplicate multi-signal
# correlations that all collapse into one indistinguishable "mega chain"
# once clustered by core/inference_engine.py's chain construction.
MULTI_SIGNAL_CATEGORY_SETS: list[frozenset[SignalCategory]] = [
    frozenset({SignalCategory.LOCATION, SignalCategory.EDUCATION, SignalCategory.ROUTINE}),
    frozenset({SignalCategory.LOCATION, SignalCategory.ROUTINE, SignalCategory.TIME}),
    frozenset({SignalCategory.EDUCATION, SignalCategory.IDENTITY, SignalCategory.ROUTINE}),
]

# Cautious, human-readable relationship / potential_inference text per type
# (project brief section 14: explanations must be grounded in the actual
# signals, section 3/12: always hedged — "may reveal", "could increase",
# never asserted as fact). {a} / {b} are filled in with the actual
# contributing signals' own short labels at generation time.
_PAIRWISE_TEMPLATES: dict[CorrelationType, dict[str, str]] = {
    CorrelationType.LOCATION_ROUTINE: {
        "relationship": "'{a}' and '{b}', from separate posts, together describe a location alongside a recurring routine.",
        "potential_inference": "This combination may reveal a movement pattern.",
    },
    CorrelationType.LOCATION_TIME: {
        "relationship": "'{a}' and '{b}', from separate posts, together describe a location alongside a recurring time reference.",
        "potential_inference": "This combination may reveal a predictable activity pattern.",
    },
    CorrelationType.EDUCATION_LOCATION: {
        "relationship": "'{a}' and '{b}', from separate posts, together describe an education-related location alongside another location clue.",
        "potential_inference": "This combination may narrow the general geographic context.",
    },
    CorrelationType.EDUCATION_IDENTITY: {
        "relationship": "'{a}' and '{b}', from separate posts, together describe education-history information alongside an identity-linked detail.",
        "potential_inference": "This combination could increase identity-linkage exposure.",
    },
    CorrelationType.WORKPLACE_ROUTINE: {
        "relationship": "'{a}' and '{b}', from separate posts, together describe work-related information alongside a recurring routine.",
        "potential_inference": "This combination may reveal routine exposure tied to a work schedule.",
    },
    CorrelationType.TIME_ROUTINE: {
        "relationship": "'{a}' and '{b}', from separate posts, together describe repeated timing alongside routine information.",
        "potential_inference": "Repeated timing like this may reveal a predictable routine.",
    },
}

_MAX_CORRELATION_CONFIDENCE = 0.99  # a correlation is always derived, never certain (see core/models.py)


# ===========================================================================
# Public entry point — correlations
# ===========================================================================


def correlate_signals(
    signals: list[PrivacySignal],
    gemini_candidates: list[CrossPostCorrelation] | None = None,
) -> list[CrossPostCorrelation]:
    """
    Given all signals extracted across an audit's posts, return the list of
    deterministic CrossPostCorrelations found between them.

    `gemini_candidates` (optional): Gemini's OWN semantic correlation
    guesses from GeminiService.analyze_posts. Used ONLY as corroboration —
    if a candidate names the exact same signal pair/group as a correlation
    this engine independently found via its own rules, that correlation's
    `strength` gets a small bonus (config.CORROBORATION_STRENGTH_BONUS).
    Gemini candidates can never introduce a correlation this engine's own
    category rules didn't already support (see module docstring).

    Deterministic: given the same signals (same objects, same ids) in the
    same order, always returns correlations in the same order with the
    same field values — no randomness, no wall-clock dependence.
    """
    if not signals:
        return []

    deduped = _dedupe_by_id(signals)
    by_category = _group_by_category(deduped)
    gemini_candidates = gemini_candidates or []

    correlations: list[CrossPostCorrelation] = []
    seen_signal_id_sets: set[frozenset[str]] = set()

    for rule in PAIRWISE_RULES:
        cat_a, cat_b = rule["categories"]
        correlation = _build_pairwise_correlation(
            by_category.get(cat_a, []),
            by_category.get(cat_b, []),
            rule["type"],
        )
        if correlation is None:
            continue
        key = frozenset(correlation.signal_ids)
        if key in seen_signal_id_sets:
            continue
        seen_signal_id_sets.add(key)
        correlations.append(correlation)

    for category_set in MULTI_SIGNAL_CATEGORY_SETS:
        correlation = _build_multi_signal_correlation(by_category, category_set)
        if correlation is None:
            continue
        key = frozenset(correlation.signal_ids)
        if key in seen_signal_id_sets:
            continue
        seen_signal_id_sets.add(key)
        correlations.append(correlation)

    correlations = [
        _apply_corroboration(c, gemini_candidates) for c in correlations
    ]

    # Deterministic ordering by CONTENT (never by the correlation's own
    # randomly-generated id), so two separate calls over the same input
    # produce lists in the same order, not merely the same set.
    correlations.sort(
        key=lambda c: (-c.strength, c.relationship_type.value, tuple(sorted(c.signal_ids)))
    )
    return correlations


# ===========================================================================
# Pairwise correlation construction
# ===========================================================================


def _build_pairwise_correlation(
    signals_a: list[PrivacySignal],
    signals_b: list[PrivacySignal],
    correlation_type: CorrelationType,
) -> CrossPostCorrelation | None:
    """
    Find the single strongest cross-post pair (one signal from each of the
    two given lists, from two DIFFERENT posts) and build one
    CrossPostCorrelation from it — or return None if no such pair exists
    (both categories present but only within the same post, or one
    category missing entirely). Only the strongest pair is kept per rule,
    which is what prevents a combinatorial explosion of near-duplicate
    correlations when a post has several signals in the same category.
    """
    if not signals_a or not signals_b:
        return None

    candidate_pairs = [
        (a, b) for a in signals_a for b in signals_b if a.post_id != b.post_id
    ]
    if not candidate_pairs:
        return None  # both categories present, but only within the same post

    sig_a, sig_b = max(
        candidate_pairs, key=lambda pair: pair[0].confidence + pair[1].confidence
    )
    return _make_correlation([sig_a, sig_b], correlation_type)


def _build_multi_signal_correlation(
    by_category: dict[SignalCategory, list[PrivacySignal]],
    category_set: frozenset[SignalCategory],
) -> CrossPostCorrelation | None:
    """
    For a category TRIPLE (e.g. Location + Education + Routine), pick the
    single highest-confidence signal in each required category and, if
    together they span at least 2 distinct posts, build one multi-signal
    CrossPostCorrelation from them. Returns None if any required category
    is missing, or if the best signal in each category all happen to come
    from the very same post (not a cross-post relationship).
    """
    best_per_category: list[PrivacySignal] = []
    for category in category_set:
        candidates = by_category.get(category, [])
        if not candidates:
            return None
        best_per_category.append(max(candidates, key=lambda s: s.confidence))

    if len({s.post_id for s in best_per_category}) < 2:
        return None  # all from the same post — not cross-post

    return _make_correlation(best_per_category, CorrelationType.MULTI_SIGNAL)


def _make_correlation(
    signals: list[PrivacySignal], correlation_type: CorrelationType
) -> CrossPostCorrelation:
    signal_ids = sorted({s.id for s in signals})  # normalize order — see section 8
    post_ids = list(dict.fromkeys(s.post_id for s in signals))
    combined_confidence = _combined_confidence(signals)
    strength = _compute_strength(correlation_type, signals)
    relationship, potential_inference, explanation = _describe_correlation(
        correlation_type, signals, post_ids
    )

    return CrossPostCorrelation(
        signal_ids=signal_ids,
        post_ids=post_ids,
        relationship_type=correlation_type,
        combined_confidence=combined_confidence,
        strength=strength,
        description=relationship,
        potential_inference=potential_inference,
        explanation=explanation,
    )


# ===========================================================================
# Strength (0.0-1.0) — see config.py for what each factor means
# ===========================================================================


def _compute_strength(correlation_type: CorrelationType, signals: list[PrivacySignal]) -> float:
    """
    Correlation strength considers: the category relationship itself
    (config.CORRELATION_BASE_STRENGTH), how confident the participating
    signals are, how sensitive they are, whether the underlying text shows
    a REPEATED pattern (not just a one-off), and how many signals are
    involved (diminishing returns — see config.py for each constant).
    Bounded to [0.0, 1.0]; NOT a probability of attack, just how strong the
    modeled relationship between the signals is.
    """
    base = config.CORRELATION_BASE_STRENGTH.get(correlation_type.value, 0.3)
    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    avg_sensitivity = sum(
        config.SENSITIVITY_WEIGHTS.get(s.sensitivity.value, 0.25) for s in signals
    ) / len(signals)

    repetition_bonus = 1.0
    if any(_mentions_repetition(s) for s in signals):
        repetition_bonus += config.REPETITION_STRENGTH_BONUS

    extra_signals = max(0, len(signals) - 2)
    signal_count_factor = min(
        config.MAX_SIGNAL_COUNT_STRENGTH_FACTOR,
        1.0 + extra_signals * config.SIGNAL_COUNT_STRENGTH_INCREMENT,
    )

    raw = base * avg_confidence * avg_sensitivity * repetition_bonus * signal_count_factor
    return max(0.0, min(1.0, raw))


def _mentions_repetition(signal: PrivacySignal) -> bool:
    text = f"{signal.signal} {signal.detail} {signal.evidence}".lower()
    return any(keyword in text for keyword in config.REPETITION_KEYWORDS)


def _apply_corroboration(
    correlation: CrossPostCorrelation, gemini_candidates: list[CrossPostCorrelation]
) -> CrossPostCorrelation:
    """
    If a Gemini-suggested candidate independently names the exact same
    signal set, apply a small strength bonus (corroboration from a second,
    independent method) — capped at 1.0. Returns a NEW CrossPostCorrelation
    (Pydantic models here are treated as immutable-by-convention) rather
    than mutating the one passed in.
    """
    own_signal_set = set(correlation.signal_ids)
    corroborated = any(
        set(candidate.signal_ids) == own_signal_set for candidate in gemini_candidates
    )
    if not corroborated:
        return correlation

    boosted_strength = min(1.0, correlation.strength + config.CORROBORATION_STRENGTH_BONUS)
    return correlation.model_copy(update={"strength": boosted_strength})


# ===========================================================================
# Human-readable descriptions (grounded in the ACTUAL signals — section 14)
# ===========================================================================


def _describe_correlation(
    correlation_type: CorrelationType, signals: list[PrivacySignal], post_ids: list[str]
) -> tuple[str, str, str]:
    """Returns (relationship, potential_inference, explanation) for one correlation."""
    if correlation_type == CorrelationType.MULTI_SIGNAL:
        return _describe_multi_signal(signals, post_ids)

    template = _PAIRWISE_TEMPLATES.get(correlation_type)
    sig_a, sig_b = signals[0], signals[1]
    if template is None:
        relationship = f"'{sig_a.signal}' and '{sig_b.signal}' were found together across posts."
        potential_inference = "This combination could contribute to overall exposure."
    else:
        relationship = template["relationship"].format(a=sig_a.signal, b=sig_b.signal)
        potential_inference = template["potential_inference"]

    cat_a = config.CATEGORY_DISPLAY_NAMES.get(sig_a.category.value, sig_a.category.value)
    cat_b = config.CATEGORY_DISPLAY_NAMES.get(sig_b.category.value, sig_b.category.value)
    explanation = (
        f"These signals occur in separate posts but describe related "
        f"{cat_a} and {cat_b} information."
    )
    return relationship, potential_inference, explanation


def _describe_multi_signal(
    signals: list[PrivacySignal], post_ids: list[str]
) -> tuple[str, str, str]:
    labels = [f"'{s.signal}'" for s in signals]
    relationship = (
        f"{', '.join(labels)}, spanning {len(post_ids)} separate posts, together "
        "describe several related privacy signals."
    )

    categories = {s.category for s in signals}
    if SignalCategory.IDENTITY in categories:
        potential_inference = (
            "This combination could increase identity-linkage exposure beyond "
            "what any single post reveals."
        )
    else:
        potential_inference = (
            "This combination could make an underlying routine or location "
            "context easier to infer than any single post alone."
        )

    category_names = sorted(
        config.CATEGORY_DISPLAY_NAMES.get(c.value, c.value) for c in categories
    )
    explanation = (
        f"These {len(signals)} signals occur across {len(post_ids)} separate "
        f"posts and together describe {', '.join(category_names)} information — "
        "individually each clue is limited, but combined they may narrow the "
        "plausible context more than any one post does alone."
    )
    return relationship, potential_inference, explanation


# ===========================================================================
# Shared helpers
# ===========================================================================


def _dedupe_by_id(signals: list[PrivacySignal]) -> list[PrivacySignal]:
    """Keep the first occurrence of each distinct signal id, preserving order."""
    seen: set[str] = set()
    result: list[PrivacySignal] = []
    for s in signals:
        if s.id in seen:
            continue
        seen.add(s.id)
        result.append(s)
    return result


def _group_by_category(
    signals: list[PrivacySignal],
) -> dict[SignalCategory, list[PrivacySignal]]:
    grouped: dict[SignalCategory, list[PrivacySignal]] = {}
    for s in signals:
        grouped.setdefault(s.category, []).append(s)
    return grouped


def _combined_confidence(signals: list[PrivacySignal]) -> float:
    """
    Derive a single confidence value for a correlation from its constituent
    signals. Deterministic, no LLM involvement. Capped below 1.0 because a
    correlation is always a derived relationship, never a certainty.
    """
    if not signals:
        return 0.0
    raw = sum(s.confidence for s in signals) / len(signals)
    return min(_MAX_CORRELATION_CONFIDENCE, raw)

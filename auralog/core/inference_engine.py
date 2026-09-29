"""
Inference chain engine.

Pure, deterministic logic — NO Gemini calls, no Streamlit, no Plotly, no
external web requests, no real-person search. This is the SECOND stage of
AURAlog's two-stage relationship pipeline:

    Gemini -> PrivacySignal
        -> core/correlation_engine.py  (DETECTS relationships -> CrossPostCorrelation)
        -> core/inference_engine.py    (THIS MODULE: COMBINES correlations -> InferenceChain)
        -> core/risk_engine.py         (CALCULATES exposure score from both)

The split matters: core/correlation_engine.py's only job is "are these two
(or three) signals related at all" — a narrow, local question. This
module's job is the broader one — "do several already-detected
relationships, taken together, tell a bigger story than any one of them
alone" — and, just as importantly, "when should they NOT be combined."
core/risk_engine.py's job is narrower again: given the correlations and
chains this pipeline hands it, compute a number. No stage repeats another
stage's work — see each module's own docstring for what it does NOT do.

--------------------------------------------------------------------------
FACT vs. INFERENCE vs. SPECULATION

Every object this module touches already carries this distinction from
core/models.py:
  - a PrivacySignal marked `directly_stated=True` is close to FACT (the
    post's own words say it) — but even that is capped by how confidently
    the extraction read the text, never treated as ground truth about the
    real world.
  - a CrossPostCorrelation is always INFERENCE — a relationship this app
    noticed between two signals, never something either post stated.
  - an InferenceChain this module builds is, at most, INFERENCE stacked on
    INFERENCE — and below a certain confidence, explicitly labeled
    SPECULATION (see InferenceChain.evidence_level). No matter how many
    correlations combine, a chain can NEVER reach 1.0 confidence (enforced
    both here and, redundantly, by the model's own validator) — this
    module treats that as a hard constraint, not a suggestion.

--------------------------------------------------------------------------
WHY NOT "ONE CONNECTED COMPONENT = ONE CHAIN" (the mega-chain problem)

A naive graph approach — nodes are signals, edges are correlations, each
connected component becomes one chain — sounds appealing but breaks down
fast: a single signal that happens to participate in several UNRELATED
correlations (a "hub") transitively drags every one of those relationships
into one giant, semantically meaningless "mega-chain". The project brief
explicitly calls this out as a defect to fix.

Instead, this module recognizes five NAMED, SEMANTICALLY MEANINGFUL chain
patterns (see CHAIN_PATTERNS below), each defined by a specific TRIPLE of
signal categories with a real-world narrative behind it (e.g. Location +
Time + Routine -> "a recurring movement/time pattern"). A named pattern
only fires when the correlation graph shows at least 2 of the 3 possible
category-pair edges actually present as real CrossPostCorrelations — not
merely "these categories both exist somewhere in the signal list". With
exactly 3 categories, 2 edges is the precise, provable threshold at which
all 3 categories are necessarily connected (see `_pattern_edge_count`);
anything short of that is a "weakly connected" cluster that does NOT get
promoted into a named 3-signal chain — it's left as whatever smaller,
already-meaningful correlation(s) it actually is.

Any correlation not consumed by a named pattern still becomes its own
standalone chain (a "two-signal chain" is a completely valid, meaningful
chain on its own — see CHAIN CONSTRUCTION below) rather than being forced
into, or excluded from, a bigger narrative it doesn't actually support.
This is what keeps multiple independent chains genuinely SEPARATE (see
`build_inference_chains`'s "consumption" bookkeeping) instead of collapsing
into one.
"""

from __future__ import annotations

import hashlib

import config
from core.models import (
    CrossPostCorrelation,
    InferenceChain,
    PrivacySignal,
    SensitivityLevel,
    SignalCategory,
)

# ===========================================================================
# Named chain patterns (project brief section 7, A-E)
#
# Each pattern is a category TRIPLE plus the cautiously-worded
# potential_outcome text used ONLY when that specific pattern's evidence
# threshold (see _pattern_edge_count) is met. A pattern is a strict SUPERSET
# validator, not a new detection rule of its own — it only ever fires using
# CrossPostCorrelations that core/correlation_engine.py already produced.
# ===========================================================================

CHAIN_PATTERNS: list[dict] = [
    {
        "name": "ROUTINE_EXPOSURE",
        "categories": frozenset(
            {SignalCategory.LOCATION, SignalCategory.TIME, SignalCategory.ROUTINE}
        ),
        "potential_outcome": "These clues may reveal a recurring activity or movement pattern.",
    },
    {
        "name": "LOCATION_CONTEXT",
        "categories": frozenset(
            {SignalCategory.LOCATION, SignalCategory.EDUCATION, SignalCategory.ROUTINE}
        ),
        "potential_outcome": (
            "These clues may narrow the general context of a recurring location or commute."
        ),
    },
    {
        "name": "IDENTITY_LINKAGE",
        "categories": frozenset(
            {
                SignalCategory.EDUCATION,
                SignalCategory.IDENTITY,
                SignalCategory.SOCIAL_RELATIONSHIPS,
            }
        ),
        "potential_outcome": (
            "These clues may increase the ability to associate posts with the same "
            "identity context."
        ),
    },
    {
        "name": "WORKPLACE_ROUTINE",
        "categories": frozenset(
            {SignalCategory.WORKPLACE, SignalCategory.LOCATION, SignalCategory.ROUTINE}
        ),
        "potential_outcome": (
            "These clues may reveal a recurring work-related movement pattern."
        ),
    },
    {
        "name": "TRAVEL_PATTERN",
        "categories": frozenset(
            {SignalCategory.TRAVEL, SignalCategory.LOCATION, SignalCategory.TIME}
        ),
        "potential_outcome": "These clues may expose a recurring travel pattern.",
    },
]

# A named pattern requires at least this many of its 3 possible
# category-pair "edges" to be backed by a real correlation. With exactly 3
# categories, 2 edges is mathematically the smallest number that must touch
# all 3 (see module docstring) — the deterministic "strongly related" vs.
# "weakly connected" threshold the project brief asks for.
_MIN_PATTERN_EDGES = 2

_SENSITIVITY_ORDER: dict[SensitivityLevel, int] = {
    SensitivityLevel.LOW: 0,
    SensitivityLevel.MEDIUM: 1,
    SensitivityLevel.HIGH: 2,
    SensitivityLevel.CRITICAL: 3,
}


# ===========================================================================
# Public entry point
# ===========================================================================


def build_inference_chains(
    signals: list[PrivacySignal],
    correlations: list[CrossPostCorrelation],
    gemini_candidates: list[InferenceChain] | None = None,
) -> list[InferenceChain]:
    """
    Build InferenceChains from validated signals and correlations.

    Two kinds of chains are produced, in this order:

    1. NAMED PATTERN chains (project brief section 7, A-E): for each of the
       5 category-triples in CHAIN_PATTERNS, if the correlations show
       enough real evidence (>= _MIN_PATTERN_EDGES category-pair edges —
       see module docstring), one chain is built using every correlation
       and signal that supports that specific pattern.

    2. STANDALONE chains: any correlation NOT already fully "consumed" by a
       named-pattern chain (i.e. whose signal_ids aren't already a subset
       of some chain already built) becomes its own chain, wrapping just
       that one correlation. This is what keeps a plain 2-signal
       correlation with no larger supported narrative as its own valid,
       separate chain, rather than silently dropped or force-merged.

    `gemini_candidates` (optional): Gemini's OWN semantic inference-chain
    guesses from GeminiService.analyze_posts, used ONLY as non-authoritative
    corroboration (project brief section 18) — if a candidate's signal_ids
    substantially overlaps a chain this module already built from validated
    correlations, that chain's strength gets a small bonus. A Gemini
    candidate can never create a chain by itself; a correlation the
    deterministic engine didn't validate is never used, no matter what
    Gemini suggests.

    Deterministic: given the same signals/correlations (same content, any
    input order), always returns the same chains with the same field
    values in the same order — chain IDs are derived from sorted content
    (never a random UUID — see `_stable_chain_id`), and the final ordering
    is (strongest first, then most sensitive, then stable ID) per the
    project brief's own ordering spec.
    """
    if not correlations:
        return []

    signals_by_id = {s.id: s for s in signals}
    gemini_candidates = gemini_candidates or []

    chains: list[InferenceChain] = []
    consumed_signal_id_sets: list[frozenset[str]] = []

    for pattern in CHAIN_PATTERNS:
        chain = _build_pattern_chain(pattern, correlations, signals_by_id)
        if chain is None:
            continue
        chains.append(chain)
        consumed_signal_id_sets.append(frozenset(chain.signal_ids))

    for correlation in correlations:
        signal_id_set = frozenset(correlation.signal_ids)
        if any(signal_id_set <= consumed for consumed in consumed_signal_id_sets):
            continue  # already fully covered by a named-pattern chain
        chain = _build_standalone_chain(correlation, signals_by_id)
        chains.append(chain)
        consumed_signal_id_sets.append(signal_id_set)

    chains = [_apply_corroboration(c, gemini_candidates) for c in chains]

    # Deterministic ordering: strongest first, then most sensitive, then
    # stable id as the final tie-breaker (project brief section 15).
    chains.sort(
        key=lambda c: (-c.strength, -_SENSITIVITY_ORDER[c.sensitivity], c.id)
    )
    return chains


# ===========================================================================
# Named-pattern chain construction
# ===========================================================================


def _build_pattern_chain(
    pattern: dict,
    correlations: list[CrossPostCorrelation],
    signals_by_id: dict[str, PrivacySignal],
) -> InferenceChain | None:
    """
    Attempt to build one chain for a single named pattern. Returns None if
    the evidence threshold (_MIN_PATTERN_EDGES) isn't met.
    """
    categories: frozenset[SignalCategory] = pattern["categories"]

    relevant = [
        c
        for c in correlations
        if _correlation_categories(c, signals_by_id) <= categories
        and len(_correlation_categories(c, signals_by_id)) >= 2
    ]
    if not relevant:
        return None

    edges = _pattern_edge_count(relevant, signals_by_id)
    if edges < _MIN_PATTERN_EDGES:
        return None

    signal_ids = sorted({sid for c in relevant for sid in c.signal_ids})
    correlation_ids = sorted({c.id for c in relevant})
    chain_signals = [signals_by_id[sid] for sid in signal_ids if sid in signals_by_id]
    post_ids = sorted({s.post_id for s in chain_signals})

    ordered_correlations = sorted(
        relevant, key=lambda c: (c.relationship_type.value, tuple(sorted(c.signal_ids)))
    )

    return InferenceChain(
        id=_stable_chain_id(pattern["name"], signal_ids),
        correlation_ids=correlation_ids,
        signal_ids=signal_ids,
        post_ids=post_ids,
        steps=[c.description for c in ordered_correlations if c.description],
        resulting_inference=pattern["potential_outcome"],
        confidence=_chain_confidence(ordered_correlations),
        sensitivity=_max_sensitivity(chain_signals),
        strength=_chain_strength(ordered_correlations, chain_signals),
        explanation=_build_explanation(chain_signals, ordered_correlations, post_ids),
    )


def _correlation_categories(
    correlation: CrossPostCorrelation, signals_by_id: dict[str, PrivacySignal]
) -> frozenset[SignalCategory]:
    return frozenset(
        signals_by_id[sid].category for sid in correlation.signal_ids if sid in signals_by_id
    )


def _pattern_edge_count(
    relevant_correlations: list[CrossPostCorrelation],
    signals_by_id: dict[str, PrivacySignal],
) -> int:
    """
    Count how many DISTINCT category-pair "edges" are backed by at least one
    real correlation. A correlation spanning all 3 categories at once (e.g.
    a MULTI_SIGNAL correlation from core/correlation_engine.py) counts as
    covering all 3 possible pairs simultaneously — it IS a complete,
    already-validated 3-way relationship on its own.
    """
    edges: set[frozenset[SignalCategory]] = set()
    for c in relevant_correlations:
        cats = _correlation_categories(c, signals_by_id)
        if len(cats) == 3:
            a, b, d = sorted(cats, key=lambda x: x.value)
            edges.update({frozenset({a, b}), frozenset({b, d}), frozenset({a, d})})
        elif len(cats) == 2:
            edges.add(cats)
    return len(edges)


# ===========================================================================
# Standalone (single-correlation) chain construction
# ===========================================================================


def _build_standalone_chain(
    correlation: CrossPostCorrelation, signals_by_id: dict[str, PrivacySignal]
) -> InferenceChain:
    """
    Wrap ONE correlation that wasn't consumed by any named pattern as its
    own, still fully meaningful, chain (project brief section 4's "Signal A
    -> Signal B" two-signal chain). Reuses the correlation's own
    already-written, signal-grounded `potential_inference` text rather than
    inventing new wording (project brief section 3: reuse existing
    terminology instead of duplicating it).
    """
    signal_ids = sorted(correlation.signal_ids)
    chain_signals = [signals_by_id[sid] for sid in signal_ids if sid in signals_by_id]
    post_ids = sorted({s.post_id for s in chain_signals} or set(correlation.post_ids))

    return InferenceChain(
        id=_stable_chain_id(correlation.relationship_type.value, signal_ids),
        correlation_ids=[correlation.id],
        signal_ids=signal_ids,
        post_ids=post_ids,
        steps=[correlation.description] if correlation.description else [],
        resulting_inference=correlation.potential_inference
        or "This combination could contribute to overall exposure.",
        confidence=min(0.95, correlation.combined_confidence),
        sensitivity=_max_sensitivity(chain_signals),
        strength=_chain_strength([correlation], chain_signals),
        explanation=_build_explanation(chain_signals, [correlation], post_ids),
    )


# ===========================================================================
# Strength, confidence, sensitivity
# ===========================================================================


def _chain_strength(
    correlations: list[CrossPostCorrelation], signals: list[PrivacySignal]
) -> float:
    """
    Chain strength considers the participating correlations' own strengths
    (diminishing returns — the same config.DIMINISHING_RETURNS_FACTOR used
    throughout this app), the participating signals' confidence and
    sensitivity, whether the underlying text shows a REPEATED pattern, and
    a modest, capped bonus for having more than one supporting
    relationship. Deliberately NOT a function of chain length alone — five
    weak, barely-related signals should not out-score two strong,
    well-evidenced ones (project brief section 8).
    """
    if not correlations or not signals:
        return 0.0

    corr_strengths = sorted((c.strength for c in correlations), reverse=True)
    combined_relationship_strength = min(1.0, _diminishing_sum(corr_strengths))

    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    avg_sensitivity_weight = sum(
        config.SENSITIVITY_WEIGHTS.get(s.sensitivity.value, 0.25) for s in signals
    ) / len(signals)

    repetition_bonus = 1.0
    if any(_mentions_repetition(s) for s in signals):
        repetition_bonus += config.REPETITION_STRENGTH_BONUS

    extra_relationships = max(0, len(correlations) - 1)
    relationship_count_factor = min(
        config.MAX_SIGNAL_COUNT_STRENGTH_FACTOR,
        1.0 + extra_relationships * config.SIGNAL_COUNT_STRENGTH_INCREMENT,
    )

    raw = (
        combined_relationship_strength
        * avg_confidence
        * avg_sensitivity_weight
        * repetition_bonus
        * relationship_count_factor
    )
    return max(0.0, min(1.0, raw))


def _chain_confidence(correlations: list[CrossPostCorrelation]) -> float:
    """
    "How strongly does the supplied evidence support this potential
    inference" — NOT a real-world probability (project brief section 9).
    A diminishing-returns combination of the constituent correlations' own
    combined_confidence, capped at 0.95 (leaving margin below the model's
    own hard <1.0 ceiling).
    """
    if not correlations:
        return 0.0
    values = sorted((c.combined_confidence for c in correlations), reverse=True)
    return max(0.0, min(0.95, _diminishing_sum(values)))


def _max_sensitivity(signals: list[PrivacySignal]) -> SensitivityLevel:
    """
    Chain sensitivity = the MAXIMUM sensitivity among participating
    signals (a conservative choice, per project brief section 10: a chain
    is only as safe as its single most sensitive contributing clue, so it
    should never be UNDER-stated by averaging it away against less
    sensitive signals in the same chain).
    """
    if not signals:
        return SensitivityLevel.LOW
    return max(signals, key=lambda s: _SENSITIVITY_ORDER[s.sensitivity]).sensitivity


def _mentions_repetition(signal: PrivacySignal) -> bool:
    text = f"{signal.signal} {signal.detail} {signal.evidence}".lower()
    return any(keyword in text for keyword in config.REPETITION_KEYWORDS)


def _diminishing_sum(values: list[float]) -> float:
    """Shared diminishing-returns combinator — see config.DIMINISHING_RETURNS_FACTOR."""
    if not values:
        return 0.0
    factor = config.DIMINISHING_RETURNS_FACTOR
    return sum(v * (factor**i) for i, v in enumerate(values))


# ===========================================================================
# Gemini corroboration (non-authoritative — see module docstring)
# ===========================================================================


def _apply_corroboration(
    chain: InferenceChain, gemini_candidates: list[InferenceChain]
) -> InferenceChain:
    """
    If a Gemini-suggested inference-chain candidate substantially overlaps
    this chain's own signal set, apply a small strength bonus — capped at
    1.0. A Gemini candidate can never have created this chain in the first
    place (see build_inference_chains's docstring); this can only
    reinforce a chain already built entirely from validated correlations.
    """
    own_signals = set(chain.signal_ids)
    if not own_signals:
        return chain

    corroborated = any(
        _overlap_ratio(own_signals, set(candidate.signal_ids)) >= 0.5
        for candidate in gemini_candidates
    )
    if not corroborated:
        return chain

    boosted = min(1.0, chain.strength + config.CORROBORATION_STRENGTH_BONUS)
    return chain.model_copy(update={"strength": boosted})


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ===========================================================================
# Explanation (project brief section 12 — always deterministic, never Gemini)
# ===========================================================================


def _build_explanation(
    signals: list[PrivacySignal],
    correlations: list[CrossPostCorrelation],
    post_ids: list[str],
) -> str:
    """
    Deterministically describes: (1) which posts contributed, (2) which
    signals, (3) how they're connected, (4) what potential exposure
    results — grounded entirely in the actual data, never a second AI call.
    """
    posts_label = ", ".join(post_ids) if post_ids else "the supplied posts"
    signal_lines = "; ".join(
        f"'{s.signal}' ({config.CATEGORY_DISPLAY_NAMES.get(s.category.value, s.category.value)}, "
        f"from {s.post_id})"
        for s in sorted(signals, key=lambda s: s.id)
    )
    relationship_count = len(correlations)
    relationship_word = "relationship" if relationship_count == 1 else "relationships"

    return (
        f"Posts {posts_label} contribute the following signals: {signal_lines}. "
        f"These are connected through {relationship_count} cross-post {relationship_word} "
        "identified by the deterministic correlation engine. Together, they may "
        "reveal more about the user's routine, location, or identity context than "
        "any single post alone — this is a potential inference, not a confirmed fact."
    )


# ===========================================================================
# Deterministic chain identity (project brief section 14 — never uuid4)
# ===========================================================================


def _stable_chain_id(pattern_name: str, signal_ids: list[str]) -> str:
    """
    Derive a chain's id purely from its own STABLE content: the pattern/type
    name plus its sorted signal_ids. Same signals + same pattern -> same id,
    every time, in every process — unlike uuid4, which is random per call.

    Deliberately does NOT factor in correlation_ids: a CrossPostCorrelation's
    own `id` is itself a fresh random uuid every time one is constructed
    (see core/models.py), so hashing on it would make the CHAIN's id just as
    volatile as a raw uuid4 would have been — defeating the entire point.
    Two different correlation objects describing the same evidence (the
    same signal_ids) are, by definition, the same relationship, so basing
    identity on signal_ids + pattern name alone is not just a workaround —
    it's the more correct notion of "same chain" in the first place.
    """
    canonical = f"{pattern_name}|{','.join(sorted(signal_ids))}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

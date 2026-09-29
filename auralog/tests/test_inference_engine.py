"""
Tests for core/inference_engine.py.

Every test constructs real, validated core/models.py objects directly (via
core/correlation_engine.py's own correlate_signals where realistic
correlations are needed) — no Gemini call, no API key required, matching
the project's requirement that the deterministic engine's tests never
depend on Gemini.
"""

from __future__ import annotations

import json
import subprocess
import sys

from core.correlation_engine import correlate_signals
from core.inference_engine import build_inference_chains
from core.models import CrossPostCorrelation, PrivacySignal, SensitivityLevel, SignalCategory
from core.risk_engine import compute_risk_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    category: SignalCategory,
    post_id: str,
    signal_text: str = "clue",
    confidence: float = 0.8,
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
    evidence: str = "",
) -> PrivacySignal:
    return PrivacySignal(
        post_id=post_id,
        category=category,
        signal=signal_text,
        confidence=confidence,
        sensitivity=sensitivity,
        evidence=evidence,
        inferred=True,
    )


# ---------------------------------------------------------------------------
# Test 1 — no signals: no inference chains
# ---------------------------------------------------------------------------


def test_no_signals_yields_no_chains() -> None:
    assert build_inference_chains([], []) == []


# ---------------------------------------------------------------------------
# Test 2 — no correlations: no inference chains
# ---------------------------------------------------------------------------


def test_no_correlations_yields_no_chains() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1"),
        _signal(SignalCategory.ROUTINE, "post-2"),
    ]
    assert build_inference_chains(signals, []) == []


# ---------------------------------------------------------------------------
# Test 3 — two related signals: one meaningful chain
# ---------------------------------------------------------------------------


def test_two_related_signals_produce_one_chain() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.ROUTINE, "post-2", "every day"),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)

    assert len(chains) == 1
    assert set(chains[0].signal_ids) == {s.id for s in signals}
    assert chains[0].confidence < 1.0
    assert "confirmed" not in chains[0].resulting_inference.lower()


# ---------------------------------------------------------------------------
# Test 4 — location + time + routine: ROUTINE_EXPOSURE chain
# ---------------------------------------------------------------------------


def test_location_time_routine_yields_routine_exposure_chain() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.TIME, "post-1", "every morning"),
        _signal(SignalCategory.ROUTINE, "post-2", "same walk every day"),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)

    assert len(chains) >= 1
    routine_chain = next(
        c for c in chains if "movement or time pattern" in c.resulting_inference.lower()
        or "movement" in c.resulting_inference.lower()
    )
    assert set(routine_chain.signal_ids) == {s.id for s in signals}


# ---------------------------------------------------------------------------
# Test 5 — location + education + routine: LOCATION_CONTEXT chain
# ---------------------------------------------------------------------------


def test_location_education_routine_yields_location_context_chain() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.EDUCATION, "post-2", "university campus"),
        _signal(SignalCategory.ROUTINE, "post-2", "as usual"),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)

    assert len(chains) >= 1
    location_context_chain = next(
        c for c in chains if "recurring location or commute" in c.resulting_inference.lower()
    )
    assert set(location_context_chain.signal_ids) == {s.id for s in signals}


# ---------------------------------------------------------------------------
# Test 6 — two unrelated signal groups: two SEPARATE chains, never merged
# ---------------------------------------------------------------------------


def test_two_unrelated_groups_produce_two_separate_chains() -> None:
    # Two independently-detected correlations that share NO signal_ids at
    # all — the inference engine must never merge chains that have no
    # actual evidence connecting them, regardless of how many chains exist.
    sig_a = _signal(SignalCategory.LOCATION, "post-1", "near the metro")
    sig_b = _signal(SignalCategory.ROUTINE, "post-2", "every day")
    sig_c = _signal(SignalCategory.EDUCATION, "post-3", "university")
    sig_d = _signal(SignalCategory.IDENTITY, "post-4", "batch of 2020")
    signals = [sig_a, sig_b, sig_c, sig_d]

    correlation_1 = CrossPostCorrelation(
        signal_ids=[sig_a.id, sig_b.id],
        post_ids=["post-1", "post-2"],
        relationship_type="location_routine",
        combined_confidence=0.6,
        strength=0.5,
        description="location + routine",
        potential_inference="This combination may reveal a movement pattern.",
    )
    correlation_2 = CrossPostCorrelation(
        signal_ids=[sig_c.id, sig_d.id],
        post_ids=["post-3", "post-4"],
        relationship_type="education_identity",
        combined_confidence=0.6,
        strength=0.5,
        description="education + identity",
        potential_inference="This combination could increase identity-linkage exposure.",
    )

    chains = build_inference_chains(signals, [correlation_1, correlation_2])

    assert len(chains) == 2
    signal_id_sets = [set(c.signal_ids) for c in chains]
    assert signal_id_sets[0].isdisjoint(signal_id_sets[1])


# ---------------------------------------------------------------------------
# Test 7 — mega-chain prevention: weakly connected subgroups stay separate
# ---------------------------------------------------------------------------


def test_weakly_connected_subgroups_are_not_merged_into_one_mega_chain() -> None:
    # A hub-like ROUTINE signal participates in two DIFFERENT pairwise
    # relationships (with TIME and with WORKPLACE) that do NOT, together,
    # complete any single named 3-category pattern with enough edge
    # support (WORKPLACE+ROUTINE forms one edge; TIME+ROUTINE forms
    # another — but LOCATION is entirely absent, and WORKPLACE+TIME has no
    # rule at all), so this must not collapse into one giant chain.
    signals = [
        _signal(SignalCategory.TIME, "post-1", "every morning"),
        _signal(SignalCategory.ROUTINE, "post-2", "same routine"),  # the "hub"
        _signal(SignalCategory.WORKPLACE, "post-3", "at the office"),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)

    # Two standalone pairwise chains (TIME_ROUTINE, WORKPLACE_ROUTINE),
    # NOT one 3-signal mega-chain, since neither named 3-category pattern
    # (ROUTINE_EXPOSURE needs LOCATION; WORKPLACE_ROUTINE needs LOCATION
    # too) has enough distinct category-pair evidence to fire.
    assert len(chains) == 2
    for chain in chains:
        assert len(chain.signal_ids) == 2


# ---------------------------------------------------------------------------
# Test 8 — duplicate correlations: no duplicate inference chains
# ---------------------------------------------------------------------------


def test_duplicate_correlations_do_not_produce_duplicate_chains() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.ROUTINE, "post-2", "every day"),
    ]
    correlation = correlate_signals(signals)[0]
    chains = build_inference_chains(signals, [correlation, correlation])
    assert len(chains) == 1


# ---------------------------------------------------------------------------
# Test 9 — input-order determinism (shuffled signals/correlations)
# ---------------------------------------------------------------------------


def test_shuffled_input_order_produces_same_chains() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.EDUCATION, "post-2", "university"),
        _signal(SignalCategory.ROUTINE, "post-2", "as usual"),
    ]
    correlations = correlate_signals(signals)

    chains_normal = build_inference_chains(signals, correlations)
    chains_shuffled = build_inference_chains(
        list(reversed(signals)), list(reversed(correlations))
    )

    assert [c.id for c in chains_normal] == [c.id for c in chains_shuffled]
    assert [c.signal_ids for c in chains_normal] == [c.signal_ids for c in chains_shuffled]


# ---------------------------------------------------------------------------
# Test 10 — process-level determinism
# ---------------------------------------------------------------------------


def test_process_level_determinism() -> None:
    script = """
import sys
sys.path.insert(0, %r)
from core.correlation_engine import correlate_signals
from core.inference_engine import build_inference_chains
from core.models import PrivacySignal, SensitivityLevel, SignalCategory
import json

signals = [
    PrivacySignal(id="fixed-s1", post_id="post-1", category=SignalCategory.LOCATION,
                  signal="near the metro", confidence=0.8, sensitivity=SensitivityLevel.MEDIUM,
                  inferred=True),
    PrivacySignal(id="fixed-s2", post_id="post-2", category=SignalCategory.ROUTINE,
                  signal="every day", confidence=0.8, sensitivity=SensitivityLevel.MEDIUM,
                  inferred=True),
]
correlations = correlate_signals(signals)
chains = build_inference_chains(signals, correlations)
print(json.dumps([
    {"id": c.id, "signal_ids": c.signal_ids, "strength": c.strength, "confidence": c.confidence}
    for c in chains
]))
"""
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script % project_root],
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(result.stdout))

    assert outputs[0] == outputs[1]


# ---------------------------------------------------------------------------
# Test 11 — chain strength bounds
# ---------------------------------------------------------------------------


def test_chain_strength_is_bounded() -> None:
    signals = [
        _signal(
            SignalCategory.LOCATION, "post-1", "near the metro",
            confidence=0.99, sensitivity=SensitivityLevel.CRITICAL,
            evidence="every single morning, always",
        ),
        _signal(
            SignalCategory.TIME, "post-1", "morning",
            confidence=0.99, sensitivity=SensitivityLevel.CRITICAL,
            evidence="every single morning, always",
        ),
        _signal(
            SignalCategory.ROUTINE, "post-2", "same route",
            confidence=0.99, sensitivity=SensitivityLevel.CRITICAL,
            evidence="every single day, always the same",
        ),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)
    for chain in chains:
        assert 0.0 <= chain.strength <= 1.0


# ---------------------------------------------------------------------------
# Test 12 — confidence bounds
# ---------------------------------------------------------------------------


def test_chain_confidence_is_bounded() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro", confidence=0.99),
        _signal(SignalCategory.ROUTINE, "post-2", "every day", confidence=0.99),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)
    for chain in chains:
        assert 0.0 <= chain.confidence < 1.0


# ---------------------------------------------------------------------------
# Test 13 — evidence traceability
# ---------------------------------------------------------------------------


def test_every_chain_is_traceable_to_real_signals_and_correlations() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.EDUCATION, "post-2", "university"),
        _signal(SignalCategory.ROUTINE, "post-2", "as usual"),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)

    known_signal_ids = {s.id for s in signals}
    known_correlation_ids = {c.id for c in correlations}
    known_post_ids = {s.post_id for s in signals}

    for chain in chains:
        assert set(chain.signal_ids) <= known_signal_ids
        assert set(chain.correlation_ids) <= known_correlation_ids
        assert set(chain.post_ids) <= known_post_ids
        assert len(chain.signal_ids) > 0
        assert len(chain.correlation_ids) > 0


# ---------------------------------------------------------------------------
# Test 14 — no false inference from unrelated signals
# ---------------------------------------------------------------------------


def test_unrelated_signals_yield_no_chain() -> None:
    # "I like pizza." / "It rained yesterday." / "I watched a movie."
    signals = [
        _signal(SignalCategory.LIFESTYLE, "post-1", "likes pizza"),
        _signal(SignalCategory.PERSONAL_INFORMATION, "post-2", "mentioned weather"),
        _signal(SignalCategory.LIFESTYLE, "post-3", "watched a movie"),
    ]
    correlations = correlate_signals(signals)
    assert correlations == []  # no meaningful correlation at all
    chains = build_inference_chains(signals, correlations)
    assert chains == []  # therefore no inference chain either


# ---------------------------------------------------------------------------
# Test 15 — Risk Engine integration
# ---------------------------------------------------------------------------


def test_inference_chains_reach_and_affect_risk_assessment() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.ROUTINE, "post-2", "every day"),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)
    assert len(chains) > 0

    with_chains = compute_risk_report(signals, correlations, inference_chains=chains)
    without_chains = compute_risk_report(signals, correlations, inference_chains=[])

    assert with_chains.inference_contribution > 0.0
    assert without_chains.inference_contribution == 0.0
    assert with_chains.total_score > without_chains.total_score
    assert with_chains.inference_chains == chains


# ---------------------------------------------------------------------------
# Test 16 — sanitized input reduces or removes inference chains
# ---------------------------------------------------------------------------


def test_sanitized_signals_reduce_inference_chains() -> None:
    before_signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro", confidence=0.9),
        _signal(SignalCategory.TIME, "post-1", "every morning", confidence=0.9),
        _signal(SignalCategory.ROUTINE, "post-2", "same walk every day", confidence=0.9),
    ]
    before_correlations = correlate_signals(before_signals)
    before_chains = build_inference_chains(before_signals, before_correlations)

    # Simulates re-running extraction on SANITIZED text: the location and
    # routine signals are gone, leaving only a vague lifestyle mention.
    after_signals = [
        _signal(SignalCategory.LIFESTYLE, "post-1", "had a nice morning", confidence=0.3),
    ]
    after_correlations = correlate_signals(after_signals)
    after_chains = build_inference_chains(after_signals, after_correlations)

    assert len(before_chains) > len(after_chains)
    assert after_chains == []


# ---------------------------------------------------------------------------
# Ethical safety: never claim confirmed real-world facts
# ---------------------------------------------------------------------------


def test_chains_never_claim_confirmed_real_world_facts() -> None:
    signals = [
        _signal(SignalCategory.EDUCATION, "post-1", "university"),
        _signal(SignalCategory.IDENTITY, "post-2", "graduation year"),
        _signal(SignalCategory.SOCIAL_RELATIONSHIPS, "post-3", "old classmates"),
    ]
    correlations = correlate_signals(signals)
    chains = build_inference_chains(signals, correlations)

    forbidden_terms = [
        "home address", "exact address", "confirmed identity", "real name",
        "we identified", "definitely the user's",
    ]
    for chain in chains:
        combined_text = (chain.resulting_inference + " " + chain.explanation).lower()
        for term in forbidden_terms:
            assert term not in combined_text


# ---------------------------------------------------------------------------
# Gemini candidates corroborate but never create a chain
# ---------------------------------------------------------------------------


def test_gemini_candidate_cannot_create_a_chain_alone() -> None:
    signals = [
        _signal(SignalCategory.LIFESTYLE, "post-1", "likes pizza"),
        _signal(SignalCategory.PERSONAL_INFORMATION, "post-2", "mentioned weather"),
    ]
    correlations = correlate_signals(signals)  # empty — no rule covers this pair
    assert correlations == []

    from core.models import InferenceChain

    phantom_candidate = InferenceChain(
        correlation_ids=["phantom"],
        signal_ids=[s.id for s in signals],
        resulting_inference="Phantom inference that should never be created.",
        confidence=0.5,
    )
    chains = build_inference_chains(signals, correlations, gemini_candidates=[phantom_candidate])
    assert chains == []


def test_gemini_candidate_boosts_strength_when_it_overlaps() -> None:
    signals = [
        _signal(SignalCategory.LOCATION, "post-1", "near the metro"),
        _signal(SignalCategory.ROUTINE, "post-2", "every day"),
    ]
    correlations = correlate_signals(signals)

    without_candidate = build_inference_chains(signals, correlations)[0]

    from core.models import InferenceChain

    matching_candidate = InferenceChain(
        correlation_ids=["c-gemini"],
        signal_ids=[s.id for s in signals],
        resulting_inference="Gemini's own guess at the same relationship.",
        confidence=0.5,
    )
    with_candidate = build_inference_chains(
        signals, correlations, gemini_candidates=[matching_candidate]
    )[0]

    assert with_candidate.strength >= without_candidate.strength


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_performance_with_many_signals() -> None:
    import time

    categories = list(SignalCategory)
    signals = [
        _signal(categories[i % len(categories)], f"post-{i}", f"clue {i}")
        for i in range(150)
    ]
    correlations = correlate_signals(signals)

    start = time.monotonic()
    build_inference_chains(signals, correlations)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0

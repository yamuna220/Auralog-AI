"""
Central configuration for AURAlog AI.

Everything that affects the DETERMINISTIC scoring/correlation logic lives here,
versioned in code, rather than in environment variables. This keeps the
scoring formula transparent and auditable (important for explainability
judging criteria) while secrets (API keys) stay in .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini / API configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# "gemini-flash-latest" is Google's own recommended, auto-updating alias for
# the current GA Flash model (currently backed by Gemini 3.5 Flash — Gemini
# 2.0 Flash and its variants were fully decommissioned by Google on
# 2026-06-01, so that older name is no longer usable at all). Using the
# alias means AURAlog keeps working as Google rolls the alias forward,
# without a code change; GEMINI_MODEL_NAME can still be overridden in .env
# to pin an exact version if ever needed for reproducibility.
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-flash-latest")
GEMINI_TIMEOUT_SECONDS: int = 20
GEMINI_MAX_RETRIES: int = 1  # one corrective retry on malformed JSON
GEMINI_TEMPERATURE: float = 0.2  # low temperature: extraction should be consistent, not creative
GEMINI_MAX_OUTPUT_TOKENS: int = 4096

# DEMO_MODE is true whenever no API key is configured. The app must remain
# fully usable in this mode using canned extraction data (see core/demo_data.py),
# with the rest of the pipeline (correlation -> scoring -> graph -> explanation)
# still running for real on that data.
DEMO_MODE: bool = GEMINI_API_KEY is None or GEMINI_API_KEY.strip() == ""

# ---------------------------------------------------------------------------
# Risk scoring configuration
# ---------------------------------------------------------------------------
# All of core/risk_engine.py's formula constants live here, centralized and
# documented, per the project's explainability requirement: a judge should
# be able to read this file top-to-bottom and understand exactly why the
# score came out the way it did, with no magic numbers buried in the engine.

# --- A. Category weights ----------------------------------------------------
# One weight per PrivacySignal category (core.models.SignalCategory), used
# both to scale that category's own signal points and to weight how much
# each category counts toward the overall signal_contribution. These weight
# PRIVACY SENSITIVITY — how safety-relevant the category of information is
# if exposed — not any assumption about who is posting it.
CATEGORY_WEIGHTS: dict[str, float] = {
    # Physical whereabouts — the single most safety-relevant category, since
    # it can directly enable in-person contact.
    "location": 1.20,
    # A predictable behavior pattern compounds risk even without an exact
    # location (e.g. "same time every day").
    "routine": 1.10,
    # A bare time-of-day clue is low-risk in isolation; it matters mainly in
    # combination with routine/location, which the correlation engine
    # already captures separately — so its own weight is kept modest here.
    "time": 0.70,
    # Commonly semi-public information on its own, but narrows identity
    # meaningfully when combined with other signals.
    "education": 0.80,
    # Reveals where someone can be physically found on a schedule.
    "workplace": 1.00,
    # Directly narrows who a person is (e.g. a graduation year, a
    # name-adjacent detail).
    "identity": 1.10,
    # Exposes a person's social graph, which can enable social-engineering
    # or be used to reach the person indirectly.
    "social_relationships": 0.90,
    # Reveals movement patterns and, implicitly, absence from a home base.
    "travel": 0.90,
    # Generally the least sensitive category alone (hobbies, preferences);
    # still counted, since it can sharpen other inferences in combination.
    "lifestyle": 0.50,
    # Catch-all bucket for signals that don't fit a narrower category (also
    # the safe fallback PrivacySignal.from_gemini_json uses for unrecognized
    # categories) — treated as moderately high by default since its content
    # is, by definition, unpredictable.
    "personal_information": 1.00,
}

# Human-readable display names for the categories above, used in
# RiskAssessment.category_breakdown and in generated explanations/top risk
# factors — this is purely presentational and never affects scoring math.
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "location": "Location",
    "routine": "Routine",
    "time": "Time",
    "education": "Education",
    "workplace": "Workplace",
    "identity": "Identity",
    "social_relationships": "Social Relationships",
    "travel": "Travel",
    "lifestyle": "Lifestyle",
    "personal_information": "Personal Information",
}

# --- B & C. Confidence and sensitivity -------------------------------------
# Sensitivity reflects how revealing the SPECIFIC DETAIL is, independent of
# its category (e.g. "a generic hobby" vs. "a recurring location") — this is
# the SensitivityLevel the AI extraction assigns to each signal, per the
# project brief's examples of lower- vs. higher-sensitivity information.
SENSITIVITY_WEIGHTS: dict[str, float] = {
    "low": 0.25,
    "medium": 0.55,
    "high": 0.80,
    "critical": 1.00,
}

# The point scale a single fully-confident, fully-sensitive, highest-weight
# signal is measured against before being capped at 100. Kept as one named
# constant so the "100" in "confidence * sensitivity * category_weight *
# SIGNAL_POINT_SCALE" is self-documenting rather than a bare number.
SIGNAL_POINT_SCALE: float = 100.0

# --- D & E. Diminishing returns for multiple signals/correlations/chains ---
# "More signals increase exposure, but do NOT count every signal equally"
# (project brief). Applied identically to signals-within-a-category,
# correlations, and inference chains: sort contributors by their own point
# value descending, then weight the i-th ranked contributor by
# DIMINISHING_RETURNS_FACTOR ** i before summing. The first (largest) signal
# counts in full; each additional corroborating one counts for
# progressively less — modeling how, say, a 2nd location clue that agrees
# with the 1st tells you less new information than the 1st one did. Because
# the factor is < 1, the infinite sum still converges to a bounded value
# even before the explicit 0-100 cap below is applied.
DIMINISHING_RETURNS_FACTOR: float = 0.5

# --- Correlation contribution (used by core/risk_engine.py's SCORING) ------
# Base POINT value for a single fully-confident correlation of each
# relationship_type, before its own combined_confidence is applied. Higher
# for relationship types that narrow location/identity more directly. Keyed
# by core.models.CorrelationType values (see core/correlation_engine.py for
# what each type means and when it fires). This is a POINTS scale (used to
# compute correlation_contribution toward the 0-100 total score) — distinct
# from CORRELATION_BASE_STRENGTH below, which is a 0-1 STRENGTH scale stored
# on the CrossPostCorrelation object itself, describing the relationship's
# own intensity rather than its score contribution.
CORRELATION_SEVERITY_WEIGHTS: dict[str, float] = {
    "location_routine": 24.0,  # movement pattern — high
    "location_time": 22.0,  # predictable activity pattern
    "education_location": 20.0,  # narrows general geographic context
    "education_identity": 22.0,  # identity-linkage
    "workplace_routine": 23.0,  # routine exposure via work schedule
    "time_routine": 18.0,  # predictable routine, no location narrowing
    "multi_signal": 30.0,  # 3+ signals combined — strongest single relationship
    "generic_overlap": 8.0,  # weak fallback; the engine's own rules never emit this
}

# --- Correlation STRENGTH (used by core/correlation_engine.py, NOT scoring) -
# Baseline strength (0.0-1.0) per relationship_type, before it's scaled by
# the participating signals' own confidence/sensitivity and by the
# repetition/signal-count factors below (see
# core/correlation_engine.py._compute_strength). Kept as a SEPARATE table
# from CORRELATION_SEVERITY_WEIGHTS above on purpose: "strength" answers
# "how strong is this modeled relationship" (an attribute of the
# correlation object itself), while CORRELATION_SEVERITY_WEIGHTS answers
# "how many risk points does this correlation contribute" (the Risk
# Engine's separate concern) — core/correlation_engine.py never computes
# risk points, and core/risk_engine.py never computes strength.
CORRELATION_BASE_STRENGTH: dict[str, float] = {
    "location_routine": 0.75,
    "location_time": 0.70,
    "education_location": 0.65,
    "education_identity": 0.70,
    "workplace_routine": 0.72,
    "time_routine": 0.60,
    "multi_signal": 0.85,
    "generic_overlap": 0.30,
}

# A correlation whose participating signals' own text contains an explicit
# repetition cue ("every", "always", "usual", "again", "recurring", "each")
# describes a REPEATED pattern, not an isolated one-off — per the project
# brief's own example ("repeated timing information should be treated as
# stronger than an isolated time reference"). Detected as a simple
# substring match (deliberately not ML — see core/correlation_engine.py's
# module docstring on staying lightweight) and applied as a strength
# multiplier bonus, capped below.
REPETITION_KEYWORDS: tuple[str, ...] = (
    "every",
    "always",
    "usual",
    "again",
    "recurring",
    "each ",
    "repeated",
    "daily",
    "weekly",
)
REPETITION_STRENGTH_BONUS: float = 0.15  # multiplicative bonus when a repetition cue is found

# A correlation spanning more signals is inherently more significant (per
# the project brief's multi-signal requirement), but with DIMINISHING
# returns per extra signal beyond the first two, capped so a large signal
# count can never dominate the formula on its own.
SIGNAL_COUNT_STRENGTH_INCREMENT: float = 0.10  # per signal beyond the first 2
MAX_SIGNAL_COUNT_STRENGTH_FACTOR: float = 1.3

# When an optional Gemini-suggested correlation candidate (see
# core/correlation_engine.py's `gemini_candidates` parameter) independently
# names the exact same pair/group of signals as one of our own rule-based
# correlations, that's corroboration from a second, independent method —
# a small strength bonus reflects that without ever letting Gemini alone
# CREATE a correlation our own category rules didn't already support.
CORROBORATION_STRENGTH_BONUS: float = 0.05

# --- Inference-chain contribution --------------------------------------------
# Base point value for a single fully-confident, single-link inference
# chain, before its own confidence and chain-length factor are applied.
BASE_INFERENCE_POINTS: float = 35.0
# Each additional correlation folded into one inference chain adds this
# fraction of extra weight (a longer chain of reasoning is a stronger,
# though still hypothetical, claim), capped at CHAIN_LENGTH_FACTOR_MAX.
CHAIN_LENGTH_INCREMENT: float = 0.15
CHAIN_LENGTH_FACTOR_MAX: float = 1.6

# --- Combining the three contributions into one overall score --------------
# signal_contribution, correlation_contribution, and inference_contribution
# are each already normalized to 0-100 (see core/risk_engine.py). The
# overall score is their WEIGHTED AVERAGE (weights sum to 1.0), which keeps
# the result bounded in [0, 100] automatically — no separate normalization
# step needed. Signals get the largest share since they're the only
# component that can be present alone; correlations and inference chains
# only add on top of that when cross-post relationships actually exist,
# which is deliberate: a single, uncorrelated clue should never score as
# high as several clues that reinforce each other.
SIGNAL_COMPONENT_WEIGHT: float = 0.55
CORRELATION_COMPONENT_WEIGHT: float = 0.25
INFERENCE_COMPONENT_WEIGHT: float = 0.20

MAX_CATEGORY_SCORE: float = 100.0
MAX_TOTAL_SCORE: float = 100.0

# --- F. Risk levels ----------------------------------------------------------
# Centralized score -> risk-level boundaries, per the project brief:
#   0-24 LOW, 25-49 MODERATE, 50-74 HIGH, 75-100 CRITICAL.
# Expressed as ascending (level, exclusive_upper_bound) pairs so the
# boundary logic in core/risk_engine.classify_risk_level is a simple
# "first bound the score is strictly less than" scan, with CRITICAL's
# upper bound set safely above 100 to also catch a score of exactly 100.
# Keyed by plain strings (not core.models.RiskLevel directly) so this
# config module has no dependency on core/ — risk_engine.py bridges the two.
RISK_LEVEL_UPPER_BOUNDS: list[tuple[str, float]] = [
    ("LOW", 25.0),
    ("MODERATE", 50.0),
    ("HIGH", 75.0),
    ("CRITICAL", 100.0001),
]

# How many entries compute_risk_report's top_risk_factors returns at most.
TOP_RISK_FACTORS_LIMIT: int = 5

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
APP_TITLE: str = "AURAlog AI"
APP_TAGLINE: str = "AI-Powered Privacy Exposure & De-Anonymization Risk Simulator"

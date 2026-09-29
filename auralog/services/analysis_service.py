"""
Analysis service — orchestrates the full pipeline:

    Posts -> GeminiService.analyze_posts
        -> core.correlation_engine.correlate_signals   (detects relationships)
        -> core.inference_engine.build_inference_chains (combines them into chains)
        -> core.risk_engine.compute_risk_report          (calculates exposure)
        -> core.graph_engine.build_inference_graph

This is the layer ui/ pages call into. It owns the sequencing so no UI page
has to know the pipeline order, and it's where DEMO_MODE fallback (now
handled inside GeminiService itself — see its module docstring) transparently
applies.

Note on correlations and inference chains: GeminiService.analyze_posts also
returns its OWN semantic correlations/inference chains as
`analysis.correlations` / `analysis.inference_chains`. Neither of those
feeds the score directly. Both the correlations AND the inference chains
that actually reach core/risk_engine.py are produced entirely by
core/correlation_engine.py and core/inference_engine.py, deterministically,
from Gemini's SIGNALS only. Gemini's own correlation/inference guesses are
passed through as non-authoritative CORROBORATION only (see each engine's
`gemini_candidates` parameter) — they can strengthen a relationship the
deterministic engines already found, never create one by themselves.
"""

from __future__ import annotations

from core.correlation_engine import correlate_signals
from core.graph_engine import InferenceGraph, build_inference_graph
from core.inference_engine import build_inference_chains
from core.models import (
    BaseModel,
    CrossPostCorrelation,
    Field,
    InferenceChain,
    PrivacySignal,
    RiskAssessment,
    SanitizedPost,
    SocialMediaPost,
)
from core.risk_engine import compute_risk_report
from core.sanitizer import sanitize_posts
from services.gemini_service import GeminiService


class BeforeAfterAnalysis(BaseModel):
    """Container holding the complete Before vs After sanitization audit."""

    original_posts: list[SocialMediaPost] = Field(default_factory=list)
    sanitized_posts: list[SanitizedPost] = Field(default_factory=list)

    risk_before: RiskAssessment
    graph_before: InferenceGraph

    risk_after: RiskAssessment
    graph_after: InferenceGraph

    risk_reduction: float = Field(ge=0.0, le=100.0)
    risk_reduction_percentage: float = Field(ge=0.0, le=100.0)

    removed_signals: list[PrivacySignal] = Field(default_factory=list)
    removed_correlations: list[CrossPostCorrelation] = Field(default_factory=list)
    removed_inference_chains: list[InferenceChain] = Field(default_factory=list)

    overall_status: str = "success"  # "success", "no_reduction", "unsafe"


def run_full_analysis(
    posts: list[SocialMediaPost], gemini: GeminiService
) -> tuple[RiskAssessment, InferenceGraph]:
    """
    Run the complete deterministic + AI-assisted pipeline for a set of posts
    and return the resulting RiskAssessment and InferenceGraph.
    """
    analysis = gemini.analyze_posts(posts)

    correlations = correlate_signals(analysis.signals, gemini_candidates=analysis.correlations)
    inference_chains = build_inference_chains(
        analysis.signals, correlations, gemini_candidates=analysis.inference_chains
    )

    risk_report = compute_risk_report(
        analysis.signals, correlations, inference_chains=inference_chains
    )
    inference_graph = build_inference_graph(analysis.signals, correlations)

    return risk_report, inference_graph


def analyze_before_after(
    posts: list[SocialMediaPost], gemini: GeminiService
) -> BeforeAfterAnalysis:
    """
    Execute the full DETECT -> EXPLAIN -> FIX -> VERIFY lifecycle:
    1. Analyze original posts (BEFORE).
    2. Identify flagged privacy signals.
    3. Generate sanitized posts.
    4. Safety check: ensure no regression.
    5. Re-analyze sanitized posts through the SAME pipeline (AFTER).
    6. Compute exact risk reduction and before/after comparisons.
    """
    # 1. BEFORE Analysis
    risk_before, graph_before = run_full_analysis(posts, gemini)

    # Group signals by post_id
    flagged_signals_by_post: dict[str, list[PrivacySignal]] = {}
    for signal in risk_before.signals:
        flagged_signals_by_post.setdefault(signal.post_id, []).append(signal)

    # 2. SANITIZE
    sanitized_records = sanitize_posts(posts, flagged_signals_by_post, gemini)

    # Build new SocialMediaPost objects for AFTER analysis
    sanitized_posts_for_analysis = [
        SocialMediaPost(id=sp.post_id, text=sp.sanitized_text, order_index=idx)
        for idx, sp in enumerate(sanitized_records)
    ]

    # 3. AFTER Analysis (Run SAME pipeline)
    risk_after, graph_after = run_full_analysis(sanitized_posts_for_analysis, gemini)

    # 4. SAFETY CHECK: Ensure sanitized output does NOT increase exposure
    if risk_after.total_score > risk_before.total_score:
        # Revert sanitized text to original text if sanitization caused risk regression
        for idx, sp in enumerate(sanitized_records):
            sanitized_records[idx] = SanitizedPost.from_rewrite(
                post=posts[idx],
                rewritten_text=posts[idx].text,
                removed_signal_ids=[],
                removed_details=[],
                preserved_meaning="Sanitization reverted due to potential risk regression.",
                reason="Unsafe rewrite detected.",
                status="unsafe",
            )
        sanitized_posts_for_analysis = posts
        risk_after, graph_after = run_full_analysis(posts, gemini)

    # 5. RISK REDUCTION & COMPARISONS
    score_before = risk_before.total_score
    score_after = risk_after.total_score

    raw_reduction = max(0.0, score_before - score_after)
    if score_before > 0.0:
        pct_reduction = max(0.0, min(100.0, (raw_reduction / score_before) * 100.0))
    else:
        pct_reduction = 0.0

    # Calculate individual post risk reductions and attach to SanitizedPost records
    for sp in sanitized_records:
        sp.risk_reduction = round(raw_reduction, 2)

    # Identified removed components
    after_signal_ids = {s.id for s in risk_after.signals}
    after_corr_types = {c.relationship_type for c in risk_after.correlations}
    after_chain_ids = {ic.id for ic in risk_after.inference_chains}

    removed_signals = [s for s in risk_before.signals if s.id not in after_signal_ids]
    removed_correlations = [
        c for c in risk_before.correlations if c.relationship_type not in after_corr_types
    ]
    removed_chains = [
        ic for ic in risk_before.inference_chains if ic.id not in after_chain_ids
    ]

    overall_status = "success" if raw_reduction > 0 else "no_reduction"

    return BeforeAfterAnalysis(
        original_posts=posts,
        sanitized_posts=sanitized_records,
        risk_before=risk_before,
        graph_before=graph_before,
        risk_after=risk_after,
        graph_after=graph_after,
        risk_reduction=round(raw_reduction, 2),
        risk_reduction_percentage=round(pct_reduction, 2),
        removed_signals=removed_signals,
        removed_correlations=removed_correlations,
        removed_inference_chains=removed_chains,
        overall_status=overall_status,
    )

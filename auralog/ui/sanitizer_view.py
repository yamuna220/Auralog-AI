"""
Sanitizer page — lets the user request safer rewrites of flagged posts, then
shows the before/after exposure score comparison.
"""

from __future__ import annotations

import streamlit as st

from state.session import get_audit_session


def render() -> None:
    st.header("Privacy Sanitizer")

    session = get_audit_session()

    if session.risk_assessment is None:
        st.info("Run an analysis from the Add Posts page first.")
        return

    if st.button("Sanitize flagged posts", type="primary"):
        # TODO (later stage): call core.sanitizer.sanitize_posts, then re-run
        # services.analysis_service.run_full_analysis on the rewritten posts
        # to get a real risk_assessment_after (never trust the rewrite as "safe"
        # on the LLM's word alone).
        st.info("Sanitization is not implemented yet.")

    if session.risk_assessment_after is not None:
        before = session.risk_assessment.total_score
        after = session.risk_assessment_after.total_score
        col_before, col_after = st.columns(2)
        col_before.metric("Before", f"{before:.0f} / 100")
        col_after.metric("After", f"{after:.0f} / 100", delta=f"{after - before:.0f}")

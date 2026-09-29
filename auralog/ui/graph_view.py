"""
Graph view page — renders the correlation/inference graph and the
plain-language explanation of what an attacker could potentially infer.
"""

from __future__ import annotations

import streamlit as st

from state.session import get_audit_session


def render() -> None:
    st.header("Correlation / Inference Graph")

    session = get_audit_session()

    if session.risk_assessment is None:
        st.info("Run an analysis from the Add Posts page first.")
        return

    report = session.risk_assessment

    if not report.correlations:
        st.write(
            "No cross-post correlations were found for this audit yet."
        )
        return

    # TODO (later stage): render the InferenceGraph with Plotly/NetworkX,
    # and show the AI-generated explanation text with directly-stated /
    # inferred / speculative labels per signal and correlation node.
    st.write(f"{len(report.correlations)} correlation(s) found.")

"""Dashboard page — exposure score overview and category breakdown."""

from __future__ import annotations

import streamlit as st

from state.session import get_audit_session


def render() -> None:
    st.header("Exposure Dashboard")

    session = get_audit_session()

    if session.risk_assessment is None:
        st.info("Run an analysis from the Add Posts page first.")
        return

    report = session.risk_assessment
    st.metric("Total Privacy Exposure Score", f"{report.total_score:.0f} / 100")

    st.subheader("Category Breakdown")
    for category in report.category_breakdown:
        st.progress(
            min(1.0, category.score / 100),
            text=f"{category.category.replace('_', ' ').title()}: {category.score:.0f}",
        )

    # TODO (later stage): render signal chips, contributing-factor lists, and
    # link out to Signal Analysis / Correlation Graph / Inference Explanation
    # pages once those are implemented.

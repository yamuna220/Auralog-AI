"""Home page — landing screen, entry point into a new privacy audit."""

from __future__ import annotations

import streamlit as st

import config


def render() -> None:
    st.title(config.APP_TITLE)
    st.caption(config.APP_TAGLINE)

    if config.DEMO_MODE:
        st.warning(
            "Running in **Demo Mode** — no GEMINI_API_KEY detected. "
            "The full scoring/correlation/graph pipeline still runs live "
            "against sample data.",
            icon="⚠️",
        )

    st.markdown(
        "AURAlog AI shows how several individually harmless public posts "
        "can be combined to reveal more than any single post gives away — "
        "an approximate location, a routine, or an identity clue. "
        "Everything here runs on synthetic or user-entered demo data only."
    )

    if st.button("Start a new privacy audit", type="primary"):
        st.session_state["_nav_target"] = "add_posts"
        st.rerun()

"""Input page — add, edit, and remove posts for the current audit."""

from __future__ import annotations

import streamlit as st

from core.models import SocialMediaPost
from state.session import get_audit_session


def render() -> None:
    st.header("Add Posts")
    st.caption(
        "Enter several public social-media posts (synthetic or demo text only) "
        "to analyze together."
    )

    session = get_audit_session()

    with st.form("add_post_form", clear_on_submit=True):
        new_post_text = st.text_area("Post text", height=100)
        submitted = st.form_submit_button("Add post")
        if submitted and new_post_text.strip():
            session.posts.append(
                SocialMediaPost(
                    text=new_post_text.strip(), order_index=len(session.posts)
                )
            )

    if session.posts:
        st.subheader(f"Posts in this audit ({len(session.posts)})")
        for post in session.posts:
            col_text, col_remove = st.columns([5, 1])
            col_text.write(f"**{post.order_index + 1}.** {post.text}")
            if col_remove.button("Remove", key=f"remove_{post.id}"):
                session.posts = [p for p in session.posts if p.id != post.id]
                st.rerun()

        if st.button("Analyze posts", type="primary"):
            # TODO (later stage): call services.analysis_service.run_full_analysis
            # and store the resulting RiskAssessment/InferenceGraph on the session
            # before navigating to the dashboard.
            st.session_state["_nav_target"] = "dashboard"
            st.rerun()
    else:
        st.info("Add at least one post to continue.")

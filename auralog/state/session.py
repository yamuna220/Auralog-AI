"""
Session-state management.

Single owner of all `st.session_state` reads/writes. No ui/ page should
touch `st.session_state` directly — they call these helpers instead. That
way the session-state shape can change in one place without hunting through
every page.

The session container is core.models.CompleteAnalysis itself — the same
top-level model used to represent (and, if needed later, serialize) one
full audit — rather than a separate parallel "AuditSession" type.
"""

from __future__ import annotations

import streamlit as st

from core.models import CompleteAnalysis

SESSION_KEY = "audit_session"


def init_session_state() -> None:
    """Ensure a CompleteAnalysis exists in session state. Safe to call every run."""
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = CompleteAnalysis()


def get_audit_session() -> CompleteAnalysis:
    """Return the current audit's CompleteAnalysis, initializing it if needed."""
    init_session_state()
    return st.session_state[SESSION_KEY]


def reset_audit_session() -> None:
    """Start a brand-new audit, discarding all current posts/results."""
    st.session_state[SESSION_KEY] = CompleteAnalysis()

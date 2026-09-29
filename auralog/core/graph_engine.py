"""
Inference/attack graph builder.

Turns a RiskAssessment's PrivacySignals + CrossPostCorrelations into an
InferenceGraph (nodes + edges) that ui/graph_view.py renders with
Plotly/NetworkX.

Added as its own module (not folded into risk_engine.py) because graph
construction is a distinct concern from scoring, and ui/graph_view.py needs
a stable, deterministic data contract to render against — this keeps that
contract out of the UI layer.

InferenceGraphNode/Edge/InferenceGraph are simple rendering-only structures
(not validated pipeline data, unlike core/models.py's Pydantic models), so
they're kept local to this module rather than in core/models.py.

STAGE 2 NOTE: stubbed with a clear TODO; return contract is final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from core.models import CrossPostCorrelation, EvidenceLevel, PrivacySignal


@dataclass
class InferenceGraphNode:
    id: str
    label: str
    node_type: Literal["signal", "correlation"]
    evidence_level: EvidenceLevel


@dataclass
class InferenceGraphEdge:
    source_id: str
    target_id: str
    weight: float = 0.0


@dataclass
class InferenceGraph:
    nodes: list[InferenceGraphNode] = field(default_factory=list)
    edges: list[InferenceGraphEdge] = field(default_factory=list)


def build_inference_graph(
    signals: list[PrivacySignal], correlations: list[CrossPostCorrelation]
) -> InferenceGraph:
    """
    Build the node/edge graph representing signals, the correlations between
    them, and the resulting higher-order inferences.

    TODO (later stage): implement real node/edge construction —
    one node per Signal (evidence_level=DIRECTLY_STATED or as extracted),
    one node per Correlation (evidence_level=INFERRED or SPECULATIVE
    depending on combined_confidence), and edges from each correlation's
    constituent signal_ids to that correlation node.
    """
    if not signals:
        return InferenceGraph()

    nodes: list[InferenceGraphNode] = []
    edges: list[InferenceGraphEdge] = []
    return InferenceGraph(nodes=nodes, edges=edges)

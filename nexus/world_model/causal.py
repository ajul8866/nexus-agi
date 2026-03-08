"""
NEXUS-AGI Causal Reasoner
Cause-effect reasoning untuk world model
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CausalNode:
    id: str
    name: str
    description: str
    node_type: str = "event"  # event, state, action, outcome
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    relationship: str  # causes, prevents, enables, inhibits
    strength: float = 1.0  # 0.0 to 1.0
    confidence: float = 1.0
    conditions: List[str] = field(default_factory=list)


@dataclass
class CausalChain:
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    probability: float
    description: str


class CausalReasoner:
    """Graph-based causal reasoning engine."""

    def __init__(self):
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: List[CausalEdge] = []
        self._adjacency: Dict[str, List[str]] = {}  # source -> [targets]
        self._reverse_adj: Dict[str, List[str]] = {}  # target -> [sources]
        self._inference_cache: Dict[str, Any] = {}
        self._build_default_graph()

    def add_node(self, node_id: str, name: str, description: str,
                 node_type: str = "event",
                 properties: Optional[Dict] = None) -> CausalNode:
        node = CausalNode(id=node_id, name=name, description=description,
                          node_type=node_type, properties=properties or {})
        self._nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, relationship: str,
                 strength: float = 1.0, confidence: float = 1.0,
                 conditions: Optional[List[str]] = None) -> CausalEdge:
        edge = CausalEdge(source_id=source_id, target_id=target_id,
                          relationship=relationship, strength=strength,
                          confidence=confidence, conditions=conditions or [])
        self._edges.append(edge)
        self._adjacency.setdefault(source_id, []).append(target_id)
        self._reverse_adj.setdefault(target_id, []).append(source_id)
        return edge

    def infer_causes(self, effect_id: str, depth: int = 3) -> List[Dict[str, Any]]:
        """Find causes of an effect by traversing causal graph backwards."""
        causes = []
        visited: Set[str] = set()
        self._traverse_causes(effect_id, depth, 1.0, visited, causes)
        return sorted(causes, key=lambda x: x["probability"], reverse=True)

    def _traverse_causes(self, node_id: str, remaining_depth: int,
                         prob: float, visited: Set[str],
                         results: List[Dict]) -> None:
        if remaining_depth == 0 or node_id in visited:
            return
        visited.add(node_id)
        cause_ids = self._reverse_adj.get(node_id, [])
        for cause_id in cause_ids:
            edge = self._get_edge(cause_id, node_id)
            if edge:
                cause_prob = prob * edge.strength * edge.confidence
                cause_node = self._nodes.get(cause_id)
                if cause_node:
                    results.append({
                        "node_id": cause_id,
                        "name": cause_node.name,
                        "relationship": edge.relationship,
                        "probability": round(cause_prob, 3),
                        "depth": 4 - remaining_depth
                    })
                self._traverse_causes(cause_id, remaining_depth - 1,
                                      cause_prob, visited, results)

    def predict_effects(self, cause_id: str, depth: int = 3) -> List[Dict[str, Any]]:
        """Predict effects of a cause."""
        effects = []
        visited: Set[str] = set()
        self._traverse_effects(cause_id, depth, 1.0, visited, effects)
        return sorted(effects, key=lambda x: x["probability"], reverse=True)

    def _traverse_effects(self, node_id: str, remaining_depth: int,
                           prob: float, visited: Set[str],
                           results: List[Dict]) -> None:
        if remaining_depth == 0 or node_id in visited:
            return
        visited.add(node_id)
        for target_id in self._adjacency.get(node_id, []):
            edge = self._get_edge(node_id, target_id)
            if edge:
                effect_prob = prob * edge.strength * edge.confidence
                effect_node = self._nodes.get(target_id)
                if effect_node:
                    results.append({
                        "node_id": target_id,
                        "name": effect_node.name,
                        "relationship": edge.relationship,
                        "probability": round(effect_prob, 3),
                        "depth": 4 - remaining_depth
                    })
                self._traverse_effects(target_id, remaining_depth - 1,
                                       effect_prob, visited, results)

    def _get_edge(self, source_id: str, target_id: str) -> Optional[CausalEdge]:
        for edge in self._edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None

    def _build_default_graph(self):
        """Build a small default causal knowledge graph."""
        nodes = [
            ("poor_planning", "Poor Planning", "Insufficient task planning", "cause"),
            ("task_failure", "Task Failure", "Task execution fails", "outcome"),
            ("low_quality", "Low Quality Output", "Output quality is poor", "outcome"),
            ("high_latency", "High Latency", "Response takes too long", "state"),
            ("resource_limit", "Resource Limit", "Memory or compute exhausted", "state"),
        ]
        for nid, name, desc, ntype in nodes:
            self.add_node(nid, name, desc, ntype)

        edges = [
            ("poor_planning", "task_failure", "causes", 0.8, 0.9),
            ("poor_planning", "low_quality", "causes", 0.7, 0.85),
            ("resource_limit", "high_latency", "causes", 0.9, 0.95),
            ("high_latency", "task_failure", "enables", 0.4, 0.7),
        ]
        for src, tgt, rel, strength, conf in edges:
            self.add_edge(src, tgt, rel, strength, conf)

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "cache_entries": len(self._inference_cache)
        }

"""CausalReasoner - causal graph inference and do-calculus reasoning."""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("nexus.world_model.causal")


@dataclass
class CausalNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    domain: str = "general"
    observed: bool = True
    value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cause: str = ""       # node name
    effect: str = ""      # node name
    strength: float = 1.0  # 0=none, 1=deterministic
    mechanism: str = ""    # description of how cause -> effect
    is_confounded: bool = False
    lag: int = 0           # time lag (0 = instantaneous)


class CausalReasoner:
    """
    Causal reasoning engine using a Directed Acyclic Graph (DAG).
    Supports:
      - Building causal graphs
      - Do-calculus interventions (do(X=x))
      - Counterfactual queries
      - Finding causal paths
      - Identifying confounders and mediators
    """

    def __init__(self):
        self._nodes: Dict[str, CausalNode] = {}  # name -> node
        self._edges: List[CausalEdge] = []
        self._adjacency: Dict[str, List[str]] = {}   # cause -> [effects]
        self._reverse_adj: Dict[str, List[str]] = {} # effect -> [causes]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, name: str, domain: str = "general", observed: bool = True, value: Optional[Any] = None) -> CausalNode:
        if name not in self._nodes:
            node = CausalNode(name=name, domain=domain, observed=observed, value=value)
            self._nodes[name] = node
        return self._nodes[name]

    def add_edge(self, cause: str, effect: str, strength: float = 1.0, mechanism: str = "", lag: int = 0) -> CausalEdge:
        self.add_node(cause)
        self.add_node(effect)
        edge = CausalEdge(cause=cause, effect=effect, strength=strength, mechanism=mechanism, lag=lag)
        self._edges.append(edge)
        self._adjacency.setdefault(cause, []).append(effect)
        self._reverse_adj.setdefault(effect, []).append(cause)
        logger.debug("Causal edge: %s -> %s (strength=%.2f)", cause, effect, strength)
        return edge

    def remove_edge(self, cause: str, effect: str) -> bool:
        before = len(self._edges)
        self._edges = [e for e in self._edges if not (e.cause == cause and e.effect == effect)]
        if cause in self._adjacency:
            self._adjacency[cause] = [e for e in self._adjacency[cause] if e != effect]
        if effect in self._reverse_adj:
            self._reverse_adj[effect] = [c for c in self._reverse_adj[effect] if c != cause]
        return len(self._edges) < before

    # ------------------------------------------------------------------
    # Causal queries
    # ------------------------------------------------------------------

    def causes_of(self, effect: str, direct_only: bool = False) -> List[str]:
        direct = self._reverse_adj.get(effect, [])
        if direct_only:
            return list(direct)
        all_causes: Set[str] = set()
        frontier = list(direct)
        while frontier:
            cause = frontier.pop()
            if cause not in all_causes:
                all_causes.add(cause)
                frontier.extend(self._reverse_adj.get(cause, []))
        return list(all_causes)

    def effects_of(self, cause: str, direct_only: bool = False) -> List[str]:
        direct = self._adjacency.get(cause, [])
        if direct_only:
            return list(direct)
        all_effects: Set[str] = set()
        frontier = list(direct)
        while frontier:
            effect = frontier.pop()
            if effect not in all_effects:
                all_effects.add(effect)
                frontier.extend(self._adjacency.get(effect, []))
        return list(all_effects)

    def find_paths(self, source: str, target: str, max_length: int = 8) -> List[List[str]]:
        paths: List[List[str]] = []
        stack: List[Tuple[str, List[str]]] = [(source, [source])]
        while stack:
            node, path = stack.pop()
            if node == target:
                paths.append(path)
                continue
            if len(path) >= max_length:
                continue
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return paths

    def find_common_causes(self, node_a: str, node_b: str) -> List[str]:
        causes_a = set(self.causes_of(node_a))
        causes_b = set(self.causes_of(node_b))
        return list(causes_a & causes_b)

    def find_mediators(self, cause: str, effect: str) -> List[str]:
        paths = self.find_paths(cause, effect)
        mediators: Set[str] = set()
        for path in paths:
            for node in path[1:-1]:
                mediators.add(node)
        return list(mediators)

    def is_confounded(self, node_a: str, node_b: str) -> bool:
        return len(self.find_common_causes(node_a, node_b)) > 0

    # ------------------------------------------------------------------
    # Intervention (do-calculus)
    # ------------------------------------------------------------------

    def intervene(self, interventions: Dict[str, Any]) -> "CausalReasoner":
        """
        Apply do(X=x) interventions: remove all incoming edges to X
        and set X's value. Returns a mutilated graph copy.
        """
        mutilated = CausalReasoner()
        # Copy nodes
        for name, node in self._nodes.items():
            mutilated.add_node(name, node.domain, node.observed, node.value)
        # Copy edges, skipping incoming edges to intervened nodes
        for edge in self._edges:
            if edge.effect not in interventions:
                mutilated.add_edge(edge.cause, edge.effect, edge.strength, edge.mechanism, edge.lag)
        # Set intervention values
        for node_name, value in interventions.items():
            if node_name in mutilated._nodes:
                mutilated._nodes[node_name].value = value
        logger.info("Intervention applied on: %s", list(interventions.keys()))
        return mutilated

    def counterfactual(self, factual_obs: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Simple counterfactual: what would Y be if we had done X=x instead?"""
        mutilated = self.intervene(intervention)
        # Propagate values through the mutilated graph (simple linear pass)
        result: Dict[str, Any] = {}
        for name, node in mutilated._nodes.items():
            if name in intervention:
                result[name] = intervention[name]
            elif name in factual_obs:
                result[name] = factual_obs[name]
            else:
                causes = mutilated._reverse_adj.get(name, [])
                if causes and all(c in result for c in causes):
                    edges = [e for e in mutilated._edges if e.effect == name]
                    weighted = sum(result[e.cause] * e.strength for e in edges if isinstance(result.get(e.cause), (int, float)))
                    result[name] = weighted if weighted != 0 else None
        return result

    # ------------------------------------------------------------------
    # Graph analysis
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for edge in self._edges:
            in_degree[edge.effect] = in_degree.get(edge.effect, 0) + 1
        queue = [n for n, d in in_degree.items() if d == 0]
        sorted_nodes: List[str] = []
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for neighbor in self._adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return sorted_nodes

    def has_cycle(self) -> bool:
        return len(self.topological_sort()) < len(self._nodes)

    def stats(self) -> Dict[str, Any]:
        return {"nodes": len(self._nodes), "edges": len(self._edges), "has_cycle": self.has_cycle(), "root_nodes": [n for n in self._nodes if not self._reverse_adj.get(n)], "leaf_nodes": [n for n in self._nodes if not self._adjacency.get(n)]}

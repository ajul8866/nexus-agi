"""
NEXUS-AGI Causal Reasoner
Causal graph untuk forward/backward reasoning
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CausalLink:
    cause: str
    effect: str
    strength: float  # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    observations: int = 1


@dataclass
class CausalPath:
    nodes: List[str]
    total_strength: float
    links: List[CausalLink]


class CausalReasoner:
    """Causal reasoning engine dengan graph-based inference."""

    def __init__(self):
        # adjacency: cause -> {effect -> CausalLink}
        self._forward: Dict[str, Dict[str, CausalLink]] = {}
        # reverse: effect -> {cause -> CausalLink}
        self._backward: Dict[str, Dict[str, CausalLink]] = {}

    def add_causal_link(self, cause: str, effect: str, strength: float = 0.8,
                        evidence: Optional[List[str]] = None) -> None:
        evidence = evidence or []
        if cause not in self._forward:
            self._forward[cause] = {}
        if effect not in self._backward:
            self._backward[effect] = {}

        if effect in self._forward.get(cause, {}):
            # Update existing link
            link = self._forward[cause][effect]
            # Bayesian update
            link.strength = (link.strength * link.observations + strength) / (link.observations + 1)
            link.observations += 1
            link.evidence.extend(evidence)
        else:
            link = CausalLink(cause=cause, effect=effect, strength=strength, evidence=evidence)
            self._forward[cause][effect] = link
            self._backward[effect][cause] = link

    def predict_effects(self, cause: str, depth: int = 3) -> List[Dict[str, Any]]:
        """Forward chaining: given cause, predict downstream effects."""
        results = []
        visited: Set[str] = set()

        def _recurse(node: str, path: List[str], strength: float, d: int):
            if d == 0 or node in visited:
                return
            visited.add(node)
            for effect, link in self._forward.get(node, {}).items():
                combined = strength * link.strength
                results.append({
                    "effect": effect,
                    "path": path + [effect],
                    "strength": round(combined, 3),
                    "direct": len(path) == 1
                })
                _recurse(effect, path + [effect], combined, d - 1)

        _recurse(cause, [cause], 1.0, depth)
        return sorted(results, key=lambda x: x["strength"], reverse=True)

    def infer_causes(self, effect: str, depth: int = 3) -> List[Dict[str, Any]]:
        """Backward chaining: given effect, infer possible causes."""
        results = []
        visited: Set[str] = set()

        def _recurse(node: str, path: List[str], strength: float, d: int):
            if d == 0 or node in visited:
                return
            visited.add(node)
            for cause, link in self._backward.get(node, {}).items():
                combined = strength * link.strength
                results.append({
                    "cause": cause,
                    "path": list(reversed(path + [cause])),
                    "strength": round(combined, 3),
                    "direct": len(path) == 1
                })
                _recurse(cause, path + [cause], combined, d - 1)

        _recurse(effect, [effect], 1.0, depth)
        return sorted(results, key=lambda x: x["strength"], reverse=True)

    def find_intervention(self, goal: str, constraints: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Find interventions that lead to goal while respecting constraints."""
        constraints = constraints or []
        causes = self.infer_causes(goal)
        interventions = []
        for c in causes:
            cause_node = c["cause"]
            if cause_node not in constraints:
                interventions.append({
                    "intervention": cause_node,
                    "expected_effect": goal,
                    "confidence": c["strength"],
                    "path": c["path"]
                })
        return interventions[:5]

    def counterfactual(self, cause: str, effect: str) -> Dict[str, Any]:
        """What if 'cause' had not happened? Would 'effect' still occur?"""
        direct_effects = self.predict_effects(cause, depth=1)
        effect_names = [e["effect"] for e in direct_effects]
        alternative_causes = self.infer_causes(effect)
        alt_causes = [c["cause"] for c in alternative_causes if c["cause"] != cause]

        return {
            "question": f"What if '{cause}' had not happened?",
            "would_effect_occur": effect not in effect_names or len(alt_causes) > 0,
            "alternative_causes": alt_causes,
            "confidence": 1.0 - (self._forward.get(cause, {}).get(effect, CausalLink(cause, effect, 0)).strength)
        }

    def get_graph_summary(self) -> Dict[str, Any]:
        all_nodes = set(self._forward.keys()) | set(self._backward.keys())
        total_links = sum(len(v) for v in self._forward.values())
        return {
            "nodes": len(all_nodes),
            "links": total_links,
            "root_causes": [n for n in self._forward if n not in self._backward],
            "terminal_effects": [n for n in self._backward if n not in self._forward]
        }

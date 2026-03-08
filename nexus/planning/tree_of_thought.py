"""TreeOfThought - branching reasoning paths with BFS/DFS/beam search and pruning."""

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.planning.tree_of_thought")


@dataclass
class ThoughtNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thought: str = ""
    evaluation_score: float = 0.0    # 0.0 – 1.0
    depth: int = 0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    is_terminal: bool = False
    is_pruned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "thought": self.thought[:100],
            "score": round(self.evaluation_score, 3),
            "depth": self.depth,
            "children": len(self.children),
            "is_terminal": self.is_terminal,
            "is_pruned": self.is_pruned,
        }


@dataclass
class ToTResult:
    best_path: List[ThoughtNode]
    best_score: float
    search_method: str
    nodes_explored: int
    nodes_pruned: int
    final_thought: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_method": self.search_method,
            "best_score": round(self.best_score, 3),
            "path_length": len(self.best_path),
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "final_thought": self.final_thought,
            "path": [n.to_dict() for n in self.best_path],
        }


class TreeOfThought:
    """
    Tree-of-Thought reasoning:
    - Expand multiple thought branches from each node
    - Evaluate and score each branch
    - Prune low-scoring branches
    - Select best path using BFS, DFS, or beam search
    """

    def __init__(
        self,
        branching_factor: int = 3,
        max_depth: int = 4,
        beam_width: int = 2,
        prune_threshold: float = 0.3,
    ):
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
        self._nodes: Dict[str, ThoughtNode] = {}

    # ── Node management ────────────────────────────────────────────────────────────
    def _add_node(self, node: ThoughtNode) -> None:
        self._nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self._nodes:
            self._nodes[node.parent_id].children.append(node.node_id)

    def _get_node(self, node_id: str) -> Optional[ThoughtNode]:
        return self._nodes.get(node_id)

    # ── Expansion ──────────────────────────────────────────────────────────────────
    def expand(
        self,
        parent: ThoughtNode,
        thought_generator: Callable[[ThoughtNode, int], List[str]],
        evaluator: Callable[[str, ThoughtNode], float],
    ) -> List[ThoughtNode]:
        """Expand a node into branching_factor children."""
        if parent.depth >= self.max_depth:
            parent.is_terminal = True
            return []

        thoughts = thought_generator(parent, self.branching_factor)
        children: List[ThoughtNode] = []

        for thought in thoughts[:self.branching_factor]:
            score = evaluator(thought, parent)
            child = ThoughtNode(
                thought=thought,
                evaluation_score=score,
                depth=parent.depth + 1,
                parent_id=parent.node_id,
                is_terminal=(parent.depth + 1 >= self.max_depth),
            )
            if score < self.prune_threshold:
                child.is_pruned = True
            self._add_node(child)
            children.append(child)

        logger.debug("Expanded node depth=%d -> %d children", parent.depth, len(children))
        return children

    # ── Search methods ─────────────────────────────────────────────────────────────
    def bfs(
        self,
        root: ThoughtNode,
        thought_generator: Callable,
        evaluator: Callable,
    ) -> ToTResult:
        """Breadth-first search across thought tree."""
        queue: deque = deque([root])
        explored = 0
        pruned = 0
        best_node = root

        while queue:
            node = queue.popleft()
            explored += 1
            if node.is_pruned:
                pruned += 1
                continue
            if node.evaluation_score > best_node.evaluation_score:
                best_node = node
            if not node.is_terminal:
                children = self.expand(node, thought_generator, evaluator)
                for child in children:
                    if not child.is_pruned:
                        queue.append(child)
                    else:
                        pruned += 1

        path = self._trace_path(best_node)
        return ToTResult(
            best_path=path,
            best_score=best_node.evaluation_score,
            search_method="BFS",
            nodes_explored=explored,
            nodes_pruned=pruned,
            final_thought=best_node.thought,
        )

    def dfs(
        self,
        root: ThoughtNode,
        thought_generator: Callable,
        evaluator: Callable,
    ) -> ToTResult:
        """Depth-first search across thought tree."""
        stack = [root]
        explored = 0
        pruned = 0
        best_node = root

        while stack:
            node = stack.pop()
            explored += 1
            if node.is_pruned:
                pruned += 1
                continue
            if node.evaluation_score > best_node.evaluation_score:
                best_node = node
            if not node.is_terminal:
                children = self.expand(node, thought_generator, evaluator)
                for child in reversed(children):
                    if not child.is_pruned:
                        stack.append(child)
                    else:
                        pruned += 1

        path = self._trace_path(best_node)
        return ToTResult(
            best_path=path,
            best_score=best_node.evaluation_score,
            search_method="DFS",
            nodes_explored=explored,
            nodes_pruned=pruned,
            final_thought=best_node.thought,
        )

    def beam_search(
        self,
        root: ThoughtNode,
        thought_generator: Callable,
        evaluator: Callable,
    ) -> ToTResult:
        """Beam search: keep only top beam_width nodes at each depth level."""
        beam: List[ThoughtNode] = [root]
        explored = 0
        pruned = 0
        best_node = root

        for depth in range(self.max_depth):
            if not beam:
                break
            next_beam: List[ThoughtNode] = []
            for node in beam:
                explored += 1
                if node.evaluation_score > best_node.evaluation_score:
                    best_node = node
                if not node.is_terminal:
                    children = self.expand(node, thought_generator, evaluator)
                    for child in children:
                        if child.is_pruned:
                            pruned += 1
                        else:
                            next_beam.append(child)
            next_beam.sort(key=lambda n: n.evaluation_score, reverse=True)
            pruned += max(0, len(next_beam) - self.beam_width)
            beam = next_beam[:self.beam_width]

        path = self._trace_path(best_node)
        return ToTResult(
            best_path=path,
            best_score=best_node.evaluation_score,
            search_method="BeamSearch",
            nodes_explored=explored,
            nodes_pruned=pruned,
            final_thought=best_node.thought,
        )

    # ── Path tracing ──────────────────────────────────────────────────────────────
    def _trace_path(self, node: ThoughtNode) -> List[ThoughtNode]:
        path: List[ThoughtNode] = []
        current: Optional[ThoughtNode] = node
        while current:
            path.append(current)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    # ── High-level interface ────────────────────────────────────────────────────────
    def solve(
        self,
        problem: str,
        method: str = "beam",
        thought_generator: Optional[Callable] = None,
        evaluator: Optional[Callable] = None,
    ) -> ToTResult:
        """Solve a problem using ToT with default or custom generators."""
        if thought_generator is None:
            thought_generator = self._default_thought_generator(problem)
        if evaluator is None:
            evaluator = self._default_evaluator

        root = ThoughtNode(thought=f"Initial: {problem[:80]}", evaluation_score=0.5)
        self._add_node(root)

        if method == "bfs":
            return self.bfs(root, thought_generator, evaluator)
        elif method == "dfs":
            return self.dfs(root, thought_generator, evaluator)
        else:
            return self.beam_search(root, thought_generator, evaluator)

    def _default_thought_generator(self, problem: str) -> Callable:
        approaches = [
            f"Approach A: decompose '{problem[:30]}' into sub-problems",
            f"Approach B: use analogy to solve '{problem[:30]}'",
            f"Approach C: apply systematic elimination for '{problem[:30]}'",
            f"Approach D: use first-principles reasoning for '{problem[:30]}'",
        ]
        import itertools
        cycle = itertools.cycle(approaches)

        def generator(node: ThoughtNode, n: int) -> List[str]:
            return [f"{next(cycle)} (depth {node.depth+1})" for _ in range(n)]

        return generator

    @staticmethod
    def _default_evaluator(thought: str, parent: ThoughtNode) -> float:
        """Simple length-based heuristic (demo only)."""
        base = min(len(thought) / 100.0, 1.0)
        depth_bonus = max(0, 0.1 - parent.depth * 0.02)
        return min(1.0, base + depth_bonus)

    # ── Stats ──────────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        nodes = list(self._nodes.values())
        pruned = [n for n in nodes if n.is_pruned]
        return {
            "total_nodes": len(nodes),
            "pruned": len(pruned),
            "avg_score": round(sum(n.evaluation_score for n in nodes) / max(len(nodes), 1), 3),
            "max_depth": max((n.depth for n in nodes), default=0),
        }

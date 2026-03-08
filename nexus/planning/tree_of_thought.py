"""TreeOfThought - deliberate reasoning via tree exploration and pruning."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("nexus.planning.tree_of_thought")


@dataclass
class ThoughtNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thought: str = ""
    parent_id: Optional[str] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    pruned: bool = False
    is_solution: bool = False
    votes: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "ThoughtNode") -> None:
        child.parent_id = self.node_id
        child.depth = self.depth + 1
        self.children.append(child)

    def active_children(self) -> List["ThoughtNode"]:
        return [c for c in self.children if not c.pruned]

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "thought": self.thought[:120], "score": round(self.score, 3), "depth": self.depth, "pruned": self.pruned, "is_solution": self.is_solution, "votes": self.votes, "children": len(self.children)}


class TreeOfThought:
    """
    Tree of Thoughts (ToT) reasoning.
    Generates multiple thought branches, evaluates them via BFS or DFS,
    prunes low-scoring branches, and returns the best solution path.

    Requires:
      - generate_thoughts(state, thought) -> List[str]   (branching)
      - evaluate_thought(state, thought)  -> float       (0-1 score)
      - is_solved(state, thought)         -> bool
    """

    def __init__(
        self,
        generate_thoughts: Callable,
        evaluate_thought: Callable,
        is_solved: Callable,
        breadth: int = 3,
        max_depth: int = 5,
        prune_threshold: float = 0.3,
        strategy: str = "bfs",
    ):
        self.generate_thoughts = generate_thoughts
        self.evaluate_thought = evaluate_thought
        self.is_solved = is_solved
        self.breadth = breadth
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold
        self.strategy = strategy  # "bfs" or "dfs"
        self._roots: Dict[str, ThoughtNode] = {}
        self._all_nodes: Dict[str, ThoughtNode] = {}

    def solve(self, problem: str, initial_state: Any = None) -> Optional[List[ThoughtNode]]:
        root = ThoughtNode(thought=f"Problem: {problem}")
        self._roots[root.node_id] = root
        self._all_nodes[root.node_id] = root

        if self.strategy == "dfs":
            solution = self._dfs(root, initial_state)
        else:
            solution = self._bfs(root, initial_state)

        if solution:
            logger.info("ToT solved problem in %d steps", len(solution))
        else:
            logger.info("ToT exhausted search without finding solution")
        return solution

    def _bfs(self, root: ThoughtNode, state: Any) -> Optional[List[ThoughtNode]]:
        frontier: List[ThoughtNode] = [root]
        for depth in range(self.max_depth):
            if not frontier:
                break
            next_frontier: List[ThoughtNode] = []
            for node in frontier:
                if self.is_solved(state, node.thought):
                    node.is_solution = True
                    return self._path_to_root(node)
                candidates = self._expand(node, state)
                next_frontier.extend(candidates)
            # Score and prune
            for candidate in next_frontier:
                candidate.score = self.evaluate_thought(state, candidate.thought)
            next_frontier = [c for c in next_frontier if not c.pruned]
            next_frontier.sort(key=lambda n: n.score, reverse=True)
            frontier = next_frontier[:self.breadth * (depth + 1)]
        # Return best path if no solution found
        if not frontier:
            return None
        best = max(self._all_nodes.values(), key=lambda n: n.score if not n.pruned else -1)
        return self._path_to_root(best)

    def _dfs(self, node: ThoughtNode, state: Any, path: Optional[List[ThoughtNode]] = None) -> Optional[List[ThoughtNode]]:
        if path is None:
            path = []
        path = path + [node]
        if self.is_solved(state, node.thought):
            node.is_solution = True
            return path
        if node.depth >= self.max_depth:
            return None
        candidates = self._expand(node, state)
        candidates.sort(key=lambda n: n.score, reverse=True)
        for candidate in candidates:
            if candidate.pruned:
                continue
            result = self._dfs(candidate, state, path)
            if result:
                return result
        return None

    def _expand(self, node: ThoughtNode, state: Any) -> List[ThoughtNode]:
        raw_thoughts = self.generate_thoughts(state, node.thought)
        children: List[ThoughtNode] = []
        for thought_text in raw_thoughts[:self.breadth]:
            child = ThoughtNode(thought=thought_text)
            score = self.evaluate_thought(state, thought_text)
            child.score = score
            if score < self.prune_threshold:
                child.pruned = True
            node.add_child(child)
            self._all_nodes[child.node_id] = child
            children.append(child)
        return children

    def _path_to_root(self, node: ThoughtNode) -> List[ThoughtNode]:
        path: List[ThoughtNode] = []
        current: Optional[ThoughtNode] = node
        visited = set()
        while current is not None and current.node_id not in visited:
            path.append(current)
            visited.add(current.node_id)
            if current.parent_id:
                current = self._all_nodes.get(current.parent_id)
            else:
                break
        return list(reversed(path))

    def vote_on_thoughts(self, node_ids: List[str], evaluator: Callable) -> str:
        best_id = ""
        best_score = -1.0
        for nid in node_ids:
            node = self._all_nodes.get(nid)
            if node:
                score = evaluator(node.thought)
                node.votes += 1
                if score > best_score:
                    best_score = score
                    best_id = nid
        return best_id

    def get_best_leaf(self) -> Optional[ThoughtNode]:
        leaves = [n for n in self._all_nodes.values() if not n.children and not n.pruned]
        if not leaves:
            return None
        return max(leaves, key=lambda n: n.score)

    def stats(self) -> Dict[str, Any]:
        all_nodes = list(self._all_nodes.values())
        pruned = [n for n in all_nodes if n.pruned]
        solutions = [n for n in all_nodes if n.is_solution]
        return {"total_nodes": len(all_nodes), "pruned_nodes": len(pruned), "solution_nodes": len(solutions), "max_depth_reached": max((n.depth for n in all_nodes), default=0), "avg_score": round(sum(n.score for n in all_nodes) / max(len(all_nodes), 1), 3)}

"""MonteCarloTreeSearch - MCTS planning for decision making under uncertainty."""

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.planning.mcts")


@dataclass
class MCTSNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: Any = None
    action: Optional[str] = None
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[str] = field(default_factory=list)
    depth: int = 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return len(self.children) == 0 and self.is_fully_expanded()

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits
        return (self.value / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, exploration: float = 1.414) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def most_visited_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.visits)

    def add_child(self, action: str, state: Any) -> "MCTSNode":
        child = MCTSNode(state=state, action=action, parent=self, depth=self.depth + 1)
        self.children.append(child)
        return child


class MonteCarloTreeSearch:
    """
    Generic MCTS implementation.
    Requires caller-supplied callbacks:
      - get_actions(state) -> List[str]
      - apply_action(state, action) -> new_state
      - is_terminal(state) -> bool
      - evaluate(state) -> float  (0.0 = worst, 1.0 = best)
    """

    def __init__(
        self,
        get_actions: Callable,
        apply_action: Callable,
        is_terminal: Callable,
        evaluate: Callable,
        iterations: int = 1000,
        max_depth: int = 20,
        exploration: float = 1.414,
        rollout_depth: int = 10,
    ):
        self.get_actions = get_actions
        self.apply_action = apply_action
        self.is_terminal = is_terminal
        self.evaluate = evaluate
        self.iterations = iterations
        self.max_depth = max_depth
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self._root: Optional[MCTSNode] = None
        self._stats: Dict[str, int] = {"selections": 0, "expansions": 0, "rollouts": 0, "backprops": 0}

    def search(self, initial_state: Any) -> Tuple[Optional[str], MCTSNode]:
        actions = self.get_actions(initial_state)
        self._root = MCTSNode(state=initial_state, untried_actions=list(actions))

        for _ in range(self.iterations):
            node = self._select(self._root)
            if not self.is_terminal(node.state) and not node.is_fully_expanded():
                node = self._expand(node)
            reward = self._rollout(node)
            self._backpropagate(node, reward)

        if not self._root.children:
            return None, self._root

        best = self._root.most_visited_child()
        logger.info("MCTS search done: iterations=%d best_action=%s visits=%d value=%.3f",
                    self.iterations, best.action, best.visits, best.value / max(best.visits, 1))
        return best.action, self._root

    def _select(self, node: MCTSNode) -> MCTSNode:
        self._stats["selections"] += 1
        current = node
        while not self.is_terminal(current.state) and current.is_fully_expanded() and current.children:
            current = current.best_child(self.exploration)
            if current.depth >= self.max_depth:
                break
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        self._stats["expansions"] += 1
        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        new_state = self.apply_action(node.state, action)
        child = node.add_child(action, new_state)
        new_actions = self.get_actions(new_state)
        child.untried_actions = list(new_actions)
        return child

    def _rollout(self, node: MCTSNode) -> float:
        self._stats["rollouts"] += 1
        state = node.state
        depth = 0
        while not self.is_terminal(state) and depth < self.rollout_depth:
            actions = self.get_actions(state)
            if not actions:
                break
            action = random.choice(actions)
            state = self.apply_action(state, action)
            depth += 1
        return self.evaluate(state)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        self._stats["backprops"] += 1
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def get_action_values(self) -> List[Dict[str, Any]]:
        if not self._root:
            return []
        return sorted(
            [{"action": c.action, "visits": c.visits, "avg_value": round(c.value / max(c.visits, 1), 4), "ucb1": round(c.ucb1(self.exploration), 4)} for c in self._root.children],
            key=lambda x: x["visits"], reverse=True
        )

    def tree_stats(self) -> Dict[str, Any]:
        if not self._root:
            return {}
        return {"iterations": self.iterations, "root_visits": self._root.visits, "children": len(self._root.children), "search_stats": self._stats, "action_values": self.get_action_values()[:5]}

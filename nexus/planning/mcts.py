"""MonteCarloTreeSearch - UCT-based planning with expansion, simulation, backprop."""

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.planning.mcts")

UCT_CONSTANT = math.sqrt(2)


@dataclass
class MCTSNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: Any = None
    action: Optional[str] = None      # action that led to this state
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    visit_count: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False
    untried_actions: List[str] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.visit_count, 1)

    def uct_score(self, parent_visits: int, c: float = UCT_CONSTANT) -> float:
        """Upper Confidence Bound for Trees (UCT)."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.avg_reward
        exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / self.visit_count)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "action": self.action,
            "visits": self.visit_count,
            "avg_reward": round(self.avg_reward, 4),
            "children": len(self.children),
            "is_terminal": self.is_terminal,
        }


@dataclass
class MCTSResult:
    best_action: Optional[str]
    best_node_id: str
    iterations: int
    nodes_created: int
    root_visits: int
    best_avg_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_action": self.best_action,
            "iterations": self.iterations,
            "nodes_created": self.nodes_created,
            "root_visits": self.root_visits,
            "best_avg_reward": round(self.best_avg_reward, 4),
        }


class MonteCarloTreeSearch:
    """
    MCTS with UCT selection strategy.
    Requires:
    - get_actions(state) -> List[str]
    - apply_action(state, action) -> new_state
    - is_terminal(state) -> bool
    - simulate(state) -> float (reward 0.0 – 1.0)
    """

    def __init__(
        self,
        iterations: int = 100,
        uct_constant: float = UCT_CONSTANT,
        max_simulation_depth: int = 20,
        seed: Optional[int] = None,
    ):
        self.iterations = iterations
        self.uct_constant = uct_constant
        self.max_simulation_depth = max_simulation_depth
        self._rng = random.Random(seed)
        self._nodes: Dict[str, MCTSNode] = {}

    # ── Core MCTS phases ──────────────────────────────────────────────────────────
    def _selection(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCT until a non-fully-expanded node is reached."""
        while node.is_fully_expanded() and node.children and not node.is_terminal:
            child_nodes = [self._nodes[cid] for cid in node.children if cid in self._nodes]
            if not child_nodes:
                break
            node = max(child_nodes, key=lambda c: c.uct_score(node.visit_count, self.uct_constant))
        return node

    def _expansion(
        self,
        node: MCTSNode,
        apply_action: Callable,
        get_actions: Callable,
        is_terminal: Callable,
    ) -> MCTSNode:
        """Expand one untried action from the node."""
        if not node.untried_actions or node.is_terminal:
            return node

        action = node.untried_actions.pop(self._rng.randint(0, len(node.untried_actions) - 1))
        new_state = apply_action(node.state, action)
        terminal = is_terminal(new_state)

        child = MCTSNode(
            state=new_state,
            action=action,
            parent_id=node.node_id,
            is_terminal=terminal,
            untried_actions=get_actions(new_state) if not terminal else [],
        )
        self._nodes[child.node_id] = child
        node.children.append(child.node_id)
        return child

    def _simulation(
        self,
        state: Any,
        apply_action: Callable,
        get_actions: Callable,
        is_terminal: Callable,
        simulate: Callable,
    ) -> float:
        """Run a random rollout from state and return reward."""
        current_state = state
        for _ in range(self.max_simulation_depth):
            if is_terminal(current_state):
                break
            actions = get_actions(current_state)
            if not actions:
                break
            action = self._rng.choice(actions)
            current_state = apply_action(current_state, action)
        return simulate(current_state)

    def _backpropagation(self, node: MCTSNode, reward: float) -> None:
        """Propagate reward up the tree."""
        current: Optional[MCTSNode] = node
        while current:
            current.visit_count += 1
            current.total_reward += reward
            current = self._nodes.get(current.parent_id) if current.parent_id else None

    # ── Main search ──────────────────────────────────────────────────────────────────
    def search(
        self,
        initial_state: Any,
        get_actions: Callable[[Any], List[str]],
        apply_action: Callable[[Any, str], Any],
        is_terminal: Callable[[Any], bool],
        simulate: Callable[[Any], float],
        iterations: Optional[int] = None,
    ) -> MCTSResult:
        """Run MCTS and return the best action from the root."""
        iters = iterations or self.iterations
        self._nodes.clear()

        root = MCTSNode(
            state=initial_state,
            action=None,
            is_terminal=is_terminal(initial_state),
            untried_actions=get_actions(initial_state),
        )
        self._nodes[root.node_id] = root

        for i in range(iters):
            node = self._selection(root)
            if not node.is_terminal and not node.is_fully_expanded():
                node = self._expansion(node, apply_action, get_actions, is_terminal)
            reward = self._simulation(
                node.state, apply_action, get_actions, is_terminal, simulate
            )
            self._backpropagation(node, reward)

        best_child: Optional[MCTSNode] = None
        if root.children:
            child_nodes = [self._nodes[cid] for cid in root.children if cid in self._nodes]
            best_child = max(child_nodes, key=lambda c: c.avg_reward)

        return MCTSResult(
            best_action=best_child.action if best_child else None,
            best_node_id=best_child.node_id if best_child else root.node_id,
            iterations=iters,
            nodes_created=len(self._nodes),
            root_visits=root.visit_count,
            best_avg_reward=best_child.avg_reward if best_child else 0.0,
        )

    # ── Action selection policies ───────────────────────────────────────────────────────
    def select_action_robust(self) -> Optional[str]:
        """Robust child: most visited (less variance than max reward)."""
        root = next((n for n in self._nodes.values() if n.parent_id is None), None)
        if not root or not root.children:
            return None
        child_nodes = [self._nodes[cid] for cid in root.children if cid in self._nodes]
        best = max(child_nodes, key=lambda c: c.visit_count)
        return best.action

    # ── Stats ──────────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        nodes = list(self._nodes.values())
        return {
            "total_nodes": len(nodes),
            "iterations": self.iterations,
            "uct_constant": self.uct_constant,
            "avg_visits": round(sum(n.visit_count for n in nodes) / max(len(nodes), 1), 2),
            "avg_reward": round(sum(n.avg_reward for n in nodes) / max(len(nodes), 1), 4),
        }

    # ── Demo helper ────────────────────────────────────────────────────────────────
    @staticmethod
    def demo_problem() -> MCTSResult:
        """Run MCTS on a simple numerical maximisation problem."""
        import random as _r

        def get_actions(state: int) -> List[str]:
            return ["+1", "+2", "+3", "-1"] if state < 10 else []

        def apply_action(state: int, action: str) -> int:
            return state + int(action)

        def is_terminal(state: int) -> bool:
            return state >= 10

        def simulate(state: int) -> float:
            return state / 10.0

        mcts = MonteCarloTreeSearch(iterations=50, seed=42)
        return mcts.search(0, get_actions, apply_action, is_terminal, simulate)

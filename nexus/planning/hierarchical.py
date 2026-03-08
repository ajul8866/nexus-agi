"""HierarchicalPlanner - goal decomposition into multi-level plan trees."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.planning.hierarchical")


class GoalStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    BLOCKED = auto()


@dataclass
class Goal:
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: float = 0.5
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_criteria: str = ""

    def is_terminal(self) -> bool:
        return self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED)


@dataclass
class PlanNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: Goal = field(default_factory=Goal)
    children: List["PlanNode"] = field(default_factory=list)
    depth: int = 0
    action: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    estimated_cost: float = 1.0
    actual_cost: float = 0.0

    def add_child(self, child: "PlanNode") -> None:
        child.depth = self.depth + 1
        self.children.append(child)

    def is_leaf(self) -> bool:
        return not self.children

    def all_leaves(self) -> List["PlanNode"]:
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves

    def total_cost(self) -> float:
        if self.is_leaf():
            return self.estimated_cost
        return sum(c.total_cost() for c in self.children)

    def to_dict(self, depth: int = 0) -> Dict[str, Any]:
        return {"node_id": self.node_id, "goal": self.goal.description, "status": self.goal.status.name, "depth": depth, "action": self.action, "estimated_cost": self.estimated_cost, "children": [c.to_dict(depth + 1) for c in self.children]}


class HierarchicalPlanner:
    """
    Hierarchical Task Network (HTN) style planner.
    Decomposes high-level goals into executable leaf tasks.
    Supports priority scheduling, precondition checking, and plan repair.
    """

    MAX_DEPTH = 6

    def __init__(self):
        self._goals: Dict[str, Goal] = {}
        self._plan_trees: Dict[str, PlanNode] = {}  # goal_id -> root PlanNode
        self._completed_goals: List[str] = []
        self._world_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def add_goal(self, goal: Goal) -> str:
        self._goals[goal.goal_id] = goal
        logger.info("Goal added: %s  priority=%.2f", goal.description, goal.priority)
        return goal.goal_id

    def create_goal(self, description: str, priority: float = 0.5, parent_id: Optional[str] = None, success_criteria: str = "") -> Goal:
        g = Goal(description=description, priority=priority, parent_id=parent_id, success_criteria=success_criteria)
        self.add_goal(g)
        return g

    def update_goal_status(self, goal_id: str, status: GoalStatus) -> bool:
        if goal_id not in self._goals:
            return False
        self._goals[goal_id].status = status
        if status == GoalStatus.COMPLETED:
            self._completed_goals.append(goal_id)
        return True

    def get_active_goals(self) -> List[Goal]:
        return sorted([g for g in self._goals.values() if g.status in (GoalStatus.ACTIVE, GoalStatus.PENDING)], key=lambda g: g.priority, reverse=True)

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose(self, goal: Goal, depth: int = 0) -> PlanNode:
        root = PlanNode(goal=goal, depth=depth)
        if depth >= self.MAX_DEPTH or self._is_primitive(goal):
            root.action = self._derive_action(goal)
            return root
        subtasks = self._decompose_goal(goal)
        for subtask_desc in subtasks:
            child_goal = Goal(description=subtask_desc, priority=goal.priority * 0.9, parent_id=goal.goal_id)
            self.add_goal(child_goal)
            child_node = self.decompose(child_goal, depth + 1)
            root.add_child(child_node)
        return root

    def _is_primitive(self, goal: Goal) -> bool:
        primitives = ["execute", "call", "send", "fetch", "store", "compute", "display", "log", "write", "read"]
        first_word = goal.description.lower().split()[0] if goal.description.split() else ""
        return first_word in primitives

    def _derive_action(self, goal: Goal) -> str:
        words = goal.description.lower().split()
        if not words:
            return "no_op"
        verb_map = {"research": "search_and_summarize", "analyze": "run_analysis", "implement": "write_code", "test": "run_tests", "deploy": "deploy_service", "summarize": "generate_summary", "validate": "run_validation"}
        return verb_map.get(words[0], f"execute_{words[0]}")

    def _decompose_goal(self, goal: Goal) -> List[str]:
        desc = goal.description.lower()
        if "research" in desc or "analyze" in desc:
            return [f"Search information about: {goal.description}", f"Analyze findings for: {goal.description}", f"Summarize analysis of: {goal.description}"]
        if "implement" in desc or "build" in desc or "create" in desc:
            return [f"Design architecture for: {goal.description}", f"Implement core logic for: {goal.description}", f"Test implementation of: {goal.description}", f"Document: {goal.description}"]
        if "test" in desc or "verify" in desc:
            return [f"Write test cases for: {goal.description}", f"Execute tests for: {goal.description}", f"Report test results for: {goal.description}"]
        return [f"Understand requirements: {goal.description}", f"Execute primary action: {goal.description}", f"Validate outcome: {goal.description}"]

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, goal: Goal) -> PlanNode:
        root = self.decompose(goal)
        self._plan_trees[goal.goal_id] = root
        logger.info("Plan created for goal=%s  leaves=%d  cost=%.2f", goal.goal_id, len(root.all_leaves()), root.total_cost())
        return root

    def get_next_actions(self, goal_id: str) -> List[str]:
        tree = self._plan_trees.get(goal_id)
        if not tree:
            return []
        return [n.action for n in tree.all_leaves() if n.action and n.goal.status == GoalStatus.PENDING]

    def repair_plan(self, goal_id: str, failed_node_id: str) -> Optional[PlanNode]:
        goal = self._goals.get(goal_id)
        if not goal:
            return None
        logger.info("Repairing plan for goal=%s failed_node=%s", goal_id, failed_node_id)
        return self.plan(goal)

    # ------------------------------------------------------------------
    # World state
    # ------------------------------------------------------------------

    def update_world_state(self, key: str, value: Any) -> None:
        self._world_state[key] = value

    def check_preconditions(self, node: PlanNode) -> bool:
        return all(self._world_state.get(pre, False) for pre in node.preconditions)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {"total_goals": len(self._goals), "active_goals": len(self.get_active_goals()), "completed_goals": len(self._completed_goals), "plan_trees": len(self._plan_trees)}

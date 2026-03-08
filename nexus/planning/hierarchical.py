"""HierarchicalPlanner - goal decomposition into sub-goals, tasks, and actions."""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("nexus.planning.hierarchical")


class NodeType(Enum):
    GOAL = auto()
    SUB_GOAL = auto()
    TASK = auto()
    ACTION = auto()


class NodeStatus(Enum):
    PENDING = auto()
    READY = auto()        # dependencies met
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class Goal:
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: int = 5          # 1 (lowest) – 10 (highest)
    success_criteria: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.TASK
    description: str = ""
    priority: int = 5
    status: NodeStatus = NodeStatus.PENDING
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)        # child node_ids
    dependencies: List[str] = field(default_factory=list)    # sibling node_ids that must be DONE first
    estimated_cost: float = 1.0
    actual_cost: float = 0.0
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self, depth: int = 0) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "type": self.node_type.name,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.name,
            "children": self.children,
            "dependencies": self.dependencies,
            "estimated_cost": self.estimated_cost,
            "result": str(self.result)[:100] if self.result else None,
            "depth": depth,
        }


class HierarchicalPlanner:
    """
    Decomposes high-level goals into a tree of sub-goals → tasks → actions.
    Supports:
    - Multi-level decomposition
    - Dependency resolution (topological ordering)
    - Priority assignment
    - Plan validation
    - Cost estimation
    """

    def __init__(self):
        self._nodes: Dict[str, PlanNode] = {}
        self._roots: List[str] = []   # top-level goal node ids

    # ── Node management ────────────────────────────────────────────────────────────
    def add_node(self, node: PlanNode) -> str:
        self._nodes[node.node_id] = node
        if node.parent_id is None:
            self._roots.append(node.node_id)
        else:
            parent = self._nodes.get(node.parent_id)
            if parent and node.node_id not in parent.children:
                parent.children.append(node.node_id)
        return node.node_id

    def get_node(self, node_id: str) -> Optional[PlanNode]:
        return self._nodes.get(node_id)

    def update_status(self, node_id: str, status: NodeStatus, result: Any = None) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.status = status
            if result is not None:
                node.result = result

    # ── Decomposition ─────────────────────────────────────────────────────────────
    def decompose_goal(self, goal: Goal, depth: int = 0, max_depth: int = 3) -> PlanNode:
        """Recursively decompose a goal into a plan tree."""
        root = PlanNode(
            node_type=NodeType.GOAL,
            description=goal.description,
            priority=goal.priority,
        )
        self._nodes[root.node_id] = root
        self._roots.append(root.node_id)

        if depth < max_depth:
            sub_goals = self._generate_sub_goals(goal.description, root.node_id, goal.priority)
            for sg in sub_goals:
                self._nodes[sg.node_id] = sg
                root.children.append(sg.node_id)
                tasks = self._generate_tasks(sg.description, sg.node_id, sg.priority)
                for t in tasks:
                    self._nodes[t.node_id] = t
                    sg.children.append(t.node_id)
                    actions = self._generate_actions(t.description, t.node_id, t.priority)
                    for a in actions:
                        self._nodes[a.node_id] = a
                        t.children.append(a.node_id)

        logger.info("Decomposed goal '%s' into %d nodes", goal.description[:40], len(self._nodes))
        return root

    def _generate_sub_goals(self, description: str, parent_id: str, base_priority: int) -> List[PlanNode]:
        desc_lower = description.lower()
        sub_goals = []
        templates = [
            ("Gather requirements and context", base_priority),
            ("Design solution approach", base_priority - 1),
            ("Implement and execute", base_priority - 1),
            ("Validate and review", base_priority - 2),
        ]
        if "research" in desc_lower:
            templates = [("Literature survey", base_priority), ("Data collection", base_priority - 1),
                         ("Analysis", base_priority - 1), ("Synthesis", base_priority - 2)]
        elif "code" in desc_lower or "implement" in desc_lower:
            templates = [("Architecture design", base_priority), ("Core implementation", base_priority),
                         ("Testing", base_priority - 1), ("Documentation", base_priority - 2)]

        prev_id: Optional[str] = None
        for tmpl_desc, prio in templates:
            node = PlanNode(
                node_type=NodeType.SUB_GOAL,
                description=f"{tmpl_desc} for: {description[:30]}",
                priority=max(1, prio),
                parent_id=parent_id,
                dependencies=[prev_id] if prev_id else [],
            )
            sub_goals.append(node)
            prev_id = node.node_id
        return sub_goals

    def _generate_tasks(self, sub_goal_desc: str, parent_id: str, priority: int) -> List[PlanNode]:
        tasks = []
        templates = [
            f"Plan: {sub_goal_desc[:40]}",
            f"Execute: {sub_goal_desc[:40]}",
            f"Review: {sub_goal_desc[:40]}",
        ]
        prev_id: Optional[str] = None
        for desc in templates:
            node = PlanNode(
                node_type=NodeType.TASK,
                description=desc,
                priority=max(1, priority - 1),
                parent_id=parent_id,
                estimated_cost=1.0,
                dependencies=[prev_id] if prev_id else [],
            )
            tasks.append(node)
            prev_id = node.node_id
        return tasks

    def _generate_actions(self, task_desc: str, parent_id: str, priority: int) -> List[PlanNode]:
        return [
            PlanNode(
                node_type=NodeType.ACTION,
                description=f"Action: {task_desc[:50]}",
                priority=max(1, priority - 1),
                parent_id=parent_id,
                estimated_cost=0.5,
            )
        ]

    # ── Dependency resolution ────────────────────────────────────────────────────────
    def topological_order(self, root_id: str) -> List[str]:
        """Return execution order respecting dependencies (Kahn's algorithm)."""
        subtree = self._collect_subtree(root_id)
        in_degree: Dict[str, int] = {nid: 0 for nid in subtree}
        for nid in subtree:
            node = self._nodes[nid]
            for dep in node.dependencies:
                if dep in subtree:
                    in_degree[nid] = in_degree.get(nid, 0) + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            queue.sort(key=lambda nid: -self._nodes[nid].priority)
            nid = queue.pop(0)
            order.append(nid)
            for candidate in subtree:
                if nid in self._nodes[candidate].dependencies:
                    in_degree[candidate] -= 1
                    if in_degree[candidate] == 0:
                        queue.append(candidate)
        return order

    def _collect_subtree(self, node_id: str) -> Set[str]:
        visited: Set[str] = set()
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            node = self._nodes.get(nid)
            if node:
                stack.extend(node.children)
        return visited

    # ── Validation ─────────────────────────────────────────────────────────────────
    def validate(self, root_id: str) -> Dict[str, Any]:
        issues: List[str] = []
        subtree = self._collect_subtree(root_id)

        for nid in subtree:
            node = self._nodes[nid]
            for dep in node.dependencies:
                if dep not in self._nodes:
                    issues.append(f"Node {nid} has missing dependency {dep}")
            if node.priority < 1 or node.priority > 10:
                issues.append(f"Node {nid} has invalid priority {node.priority}")

        try:
            order = self.topological_order(root_id)
            if len(order) != len(subtree):
                issues.append("Cycle detected in plan dependencies")
        except Exception as e:
            issues.append(f"Validation error: {e}")

        total_cost = sum(self._nodes[nid].estimated_cost for nid in subtree)
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "node_count": len(subtree),
            "estimated_total_cost": total_cost,
        }

    # ── Reporting ─────────────────────────────────────────────────────────────────
    def plan_summary(self) -> Dict[str, Any]:
        by_status = {}
        by_type = {}
        for node in self._nodes.values():
            s = node.status.name
            t = node.node_type.name
            by_status[s] = by_status.get(s, 0) + 1
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "roots": len(self._roots),
            "by_status": by_status,
            "by_type": by_type,
        }

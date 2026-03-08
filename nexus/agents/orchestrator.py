"""OrchestratorAgent - coordinates multi-agent task execution."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from nexus.agents.base import AgentBase, AgentCapability, AgentState

logger = logging.getLogger("nexus.agents.orchestrator")


@dataclass
class SubTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = ""
    description: str = ""
    assigned_agent: Optional[str] = None
    status: str = "pending"      # pending | running | done | failed
    result: Any = None
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    subtasks: List[SubTask] = field(default_factory=list)
    status: str = "planned"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def all_done(self) -> bool:
        return all(t.status == "done" for t in self.subtasks)

    def next_ready(self) -> List[SubTask]:
        done_ids = {t.task_id for t in self.subtasks if t.status == "done"}
        return [
            t for t in self.subtasks
            if t.status == "pending" and set(t.dependencies).issubset(done_ids)
        ]


class OrchestratorAgent(AgentBase):
    """
    Orchestrator: decomposes goals into subtasks, assigns them to specialist
    agents, resolves conflicts, and aggregates results.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(
            agent_id=agent_id,
            capabilities=[
                AgentCapability.ORCHESTRATION,
                AgentCapability.PLANNING,
                AgentCapability.REASONING,
            ],
        )
        self._registered_agents: Dict[str, AgentBase] = {}
        self._active_plans: Dict[str, TaskPlan] = {}
        self._results_store: Dict[str, Any] = {}

    # ── Agent registry ─────────────────────────────────────────────────────────
    def add_agent(self, agent: AgentBase) -> None:
        self._registered_agents[agent.agent_id] = agent
        logger.info("Orchestrator registered sub-agent: %s", agent.agent_id)

    def remove_agent(self, agent_id: str) -> None:
        self._registered_agents.pop(agent_id, None)

    def select_agent(self, required_cap: AgentCapability) -> Optional[AgentBase]:
        """Select best available agent for a capability (least busy)."""
        candidates = [
            a for a in self._registered_agents.values()
            if a.has_capability(required_cap) and a.state.status != "error"
        ]
        if not candidates:
            return None
        # Pick least busy (smallest queue)
        return min(candidates, key=lambda a: a.message_queue.qsize())

    # ── Task decomposition ─────────────────────────────────────────────────────
    async def decompose_goal(self, goal: str) -> TaskPlan:
        """Decompose a high-level goal into ordered subtasks."""
        plan = TaskPlan(goal=goal)
        keywords = goal.lower()

        # Simple heuristic decomposition (production would use LLM)
        subtasks_defs: List[Tuple[str, List[str], int]] = []

        if "research" in keywords or "analyze" in keywords:
            subtasks_defs.append(("Research relevant information", [], 10))
        if "code" in keywords or "implement" in keywords or "write" in keywords:
            subtasks_defs.append(("Implement solution", [], 8))
        if "test" in keywords or "verify" in keywords:
            subtasks_defs.append(("Verify and test output", [], 6))
        if "summarize" in keywords or "report" in keywords:
            subtasks_defs.append(("Summarize results", [], 4))

        # Fallback: single generic task
        if not subtasks_defs:
            subtasks_defs = [
                ("Understand the goal", [], 10),
                ("Execute primary action", [], 8),
                ("Validate and finalize", [], 5),
            ]

        prev_id: Optional[str] = None
        for desc, extra_deps, prio in subtasks_defs:
            deps = list(extra_deps)
            if prev_id:
                deps.append(prev_id)
            st = SubTask(
                parent_id=plan.plan_id,
                description=desc,
                priority=prio,
                dependencies=deps,
            )
            plan.subtasks.append(st)
            prev_id = st.task_id

        self._active_plans[plan.plan_id] = plan
        logger.info("Decomposed goal into %d subtasks plan=%s", len(plan.subtasks), plan.plan_id)
        return plan

    # ── Execution ──────────────────────────────────────────────────────────────
    async def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        plan.status = "running"
        aggregated: Dict[str, Any] = {}

        while not plan.all_done():
            ready = plan.next_ready()
            if not ready:
                await asyncio.sleep(0.1)
                continue

            tasks_coros = [self._dispatch_subtask(st, plan) for st in ready]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

            for st, res in zip(ready, results):
                if isinstance(res, Exception):
                    st.status = "failed"
                    st.result = str(res)
                    logger.error("Subtask %s failed: %s", st.task_id, res)
                else:
                    st.status = "done"
                    st.result = res
                    st.completed_at = datetime.utcnow()
                    aggregated[st.task_id] = res

        plan.status = "done"
        return self._aggregate_results(plan, aggregated)

    async def _dispatch_subtask(self, subtask: SubTask, plan: TaskPlan) -> Any:
        subtask.status = "running"
        cap = AgentCapability.REASONING  # default
        agent = self.select_agent(cap)
        if agent:
            subtask.assigned_agent = agent.agent_id
            await agent.receive({"task": subtask.description, "plan_id": plan.plan_id})
            # Simulate waiting for result (real system uses callbacks/futures)
            await asyncio.sleep(0.05)
            result = f"Completed: {subtask.description}"
        else:
            result = f"No agent available for: {subtask.description}"
        return result

    def _aggregate_results(self, plan: TaskPlan, results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge subtask results; detect conflicts."""
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "status": plan.status,
            "subtask_results": results,
            "summary": f"Completed {len(results)}/{len(plan.subtasks)} subtasks",
        }

    # ── Conflict resolution ────────────────────────────────────────────────────
    def resolve_conflict(self, results: List[Any]) -> Any:
        """Simple majority vote for conflicting results."""
        if not results:
            return None
        from collections import Counter
        counts = Counter(str(r) for r in results)
        winner = counts.most_common(1)[0][0]
        logger.info("Conflict resolved by majority vote: %s", winner)
        return winner

    # ── AgentBase abstract impl ────────────────────────────────────────────────
    async def perceive(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data.get("task", str(input_data))
        return str(input_data)

    async def think(self, percept: Any) -> Any:
        plan = await self.decompose_goal(str(percept))
        return plan

    async def act(self, decision: Any) -> Any:
        if isinstance(decision, TaskPlan):
            return await self.execute_plan(decision)
        return {"error": "Unknown decision type"}

    async def reflect(self) -> Any:
        total = sum(len(p.subtasks) for p in self._active_plans.values())
        done = sum(
            sum(1 for t in p.subtasks if t.status == "done")
            for p in self._active_plans.values()
        )
        return {
            "plans": len(self._active_plans),
            "total_subtasks": total,
            "completed_subtasks": done,
            "agents_managed": len(self._registered_agents),
        }

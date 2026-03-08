"""
NEXUS-AGI Orchestrator Agent
Coordinates multi-agent task execution
"""
from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
from .base import AgentBase, AgentResult, AgentStatus


class OrchestratorAgent(AgentBase):
    """
    Master orchestrator that breaks tasks into subtasks
    and delegates to specialist agents.
    """

    def __init__(self, name: str = "Orchestrator"):
        super().__init__(name=name, role="orchestrator",
                         capabilities=["planning", "delegation", "synthesis", "monitoring"])
        self._agents: Dict[str, AgentBase] = {}
        self._active_tasks: Dict[str, Dict] = {}

    def register_agent(self, agent: AgentBase) -> None:
        self._agents[agent.agent_id] = agent

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    async def plan(self, goal: str) -> List[str]:
        steps = [
            f"Analyze goal: {goal[:50]}",
            "Identify required capabilities",
            "Select appropriate specialist agents",
            "Decompose into parallel subtasks",
            "Execute subtasks and monitor progress",
            "Synthesize results",
            "Validate output quality",
            "Return final result",
        ]
        return steps

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        start = time.time()
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        self.status = AgentStatus.RUNNING
        self._active_tasks[task_id] = {"task": task, "start": start, "status": "running"}

        try:
            plan = await self.plan(task)
            results = []

            for agent in list(self._agents.values())[:3]:
                sub_result = await agent.execute(task, context)
                results.append(sub_result)

            success_results = [r for r in results if r.success]
            output = {
                "task_id": task_id,
                "plan": plan,
                "subtask_results": len(results),
                "successful": len(success_results),
                "synthesis": f"Completed task with {len(success_results)}/{len(results)} agents succeeding"
            }

            self._active_tasks[task_id]["status"] = "completed"
            result = AgentResult(
                agent_id=self.agent_id, task_id=task_id, success=True,
                output=output, latency=time.time() - start
            )
        except Exception as e:
            self._active_tasks[task_id]["status"] = "failed"
            result = AgentResult(
                agent_id=self.agent_id, task_id=task_id, success=False,
                output=None, error=str(e), latency=time.time() - start
            )
        finally:
            self.status = AgentStatus.IDLE

        self.record_result(result)
        return result

    def get_agent_roster(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._agents.values()]

    def get_active_tasks(self) -> Dict[str, Any]:
        return dict(self._active_tasks)

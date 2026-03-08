"""SpecialistAgent - domain-specific execution with skill registry."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from nexus.agents.base import AgentBase, AgentCapability

logger = logging.getLogger("nexus.agents.specialist")


@dataclass
class Skill:
    name: str
    description: str
    handler: Callable
    domain: str = "general"
    call_count: int = 0
    total_latency_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.call_count, 1)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(total, 1)


@dataclass
class SkillResult:
    skill_name: str
    success: bool
    output: Any
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class SpecialistAgent(AgentBase):
    """
    Specialist agent focused on a specific domain.
    Maintains a skill registry and tracks per-skill performance.
    """

    def __init__(
        self,
        domain: str = "general",
        agent_id: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            capabilities=capabilities or [AgentCapability.REASONING, AgentCapability.TOOL_USE],
        )
        self.domain = domain
        self._skills: Dict[str, Skill] = {}
        self._result_history: List[SkillResult] = []
        self._register_default_skills()

    # ── Skill management ───────────────────────────────────────────────────────
    def _register_default_skills(self) -> None:
        self.register_skill(
            name="summarize",
            description="Summarize text input",
            handler=self._skill_summarize,
            domain="nlp",
        )
        self.register_skill(
            name="analyze",
            description="Analyze and extract key insights",
            handler=self._skill_analyze,
            domain="analysis",
        )
        self.register_skill(
            name="transform",
            description="Transform data from one format to another",
            handler=self._skill_transform,
            domain="data",
        )

    def register_skill(
        self,
        name: str,
        description: str,
        handler: Callable,
        domain: str = "general",
    ) -> None:
        self._skills[name] = Skill(
            name=name,
            description=description,
            handler=handler,
            domain=domain,
        )
        logger.info("Specialist %s registered skill: %s", self.agent_id, name)

    def unregister_skill(self, name: str) -> None:
        self._skills.pop(name, None)

    def list_skills(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.name,
                "domain": s.domain,
                "description": s.description,
                "avg_latency_ms": round(s.avg_latency_ms, 2),
                "success_rate": round(s.success_rate, 3),
                "calls": s.call_count,
            }
            for s in self._skills.values()
        ]

    # ── Skill execution ────────────────────────────────────────────────────────
    async def execute_skill(self, skill_name: str, **kwargs) -> SkillResult:
        if skill_name not in self._skills:
            raise ValueError(f"Skill not found: {skill_name}")

        skill = self._skills[skill_name]
        start = time.perf_counter()
        skill.call_count += 1

        try:
            if asyncio.iscoroutinefunction(skill.handler):
                output = await skill.handler(**kwargs)
            else:
                output = skill.handler(**kwargs)
            latency = (time.perf_counter() - start) * 1000
            skill.success_count += 1
            skill.total_latency_ms += latency
            result = SkillResult(
                skill_name=skill_name,
                success=True,
                output=output,
                latency_ms=round(latency, 2),
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            skill.failure_count += 1
            skill.total_latency_ms += latency
            result = SkillResult(
                skill_name=skill_name,
                success=False,
                output=None,
                latency_ms=round(latency, 2),
                error=str(exc),
            )
            logger.exception("Skill %s failed: %s", skill_name, exc)

        self._result_history.append(result)
        return result

    # ── Default skill implementations ──────────────────────────────────────────
    async def _skill_summarize(self, text: str, max_words: int = 50) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    async def _skill_analyze(self, data: Any) -> Dict[str, Any]:
        text = str(data)
        return {
            "length": len(text),
            "words": len(text.split()),
            "sentences": text.count(".") + text.count("!") + text.count("?"),
            "has_numbers": any(c.isdigit() for c in text),
            "preview": text[:100],
        }

    async def _skill_transform(self, data: Any, target_format: str = "str") -> Any:
        if target_format == "str":
            return str(data)
        if target_format == "list":
            return list(data) if hasattr(data, "__iter__") else [data]
        if target_format == "dict":
            return {"value": data}
        return data

    # ── Performance ────────────────────────────────────────────────────────────
    def performance_report(self) -> Dict[str, Any]:
        total_calls = sum(s.call_count for s in self._skills.values())
        total_success = sum(s.success_count for s in self._skills.values())
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "total_skill_calls": total_calls,
            "overall_success_rate": total_success / max(total_calls, 1),
            "skills": self.list_skills(),
            "recent_results": len(self._result_history),
        }

    # ── AgentBase abstract impl ────────────────────────────────────────────────
    async def perceive(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {"task": str(input_data), "skill": "analyze"}

    async def think(self, percept: Any) -> Any:
        task = percept.get("task", "") if isinstance(percept, dict) else str(percept)
        # Determine best skill based on task keywords
        task_lower = task.lower()
        if "summarize" in task_lower or "summary" in task_lower:
            return {"skill": "summarize", "kwargs": {"text": task}}
        if "analyze" in task_lower or "analysis" in task_lower:
            return {"skill": "analyze", "kwargs": {"data": task}}
        return {"skill": "analyze", "kwargs": {"data": task}}

    async def act(self, decision: Any) -> Any:
        if isinstance(decision, dict) and "skill" in decision:
            result = await self.execute_skill(
                decision["skill"], **decision.get("kwargs", {})
            )
            return result.output if result.success else f"Error: {result.error}"
        return str(decision)

    async def reflect(self) -> Any:
        return self.performance_report()

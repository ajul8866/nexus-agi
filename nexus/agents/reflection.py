"""
NEXUS-AGI Reflection Agent
Meta-cognition and self-evaluation agent
"""
from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
from .base import AgentBase, AgentResult, AgentStatus


class ReflectionAgent(AgentBase):
    """
    Reflection agent that evaluates outputs, identifies errors,
    and provides improvement suggestions.
    """

    def __init__(self, name: str = "Reflector"):
        super().__init__(name=name, role="reflection",
                         capabilities=["evaluation", "critique", "improvement", "meta-cognition"])
        self._evaluations: List[Dict[str, Any]] = []

    async def plan(self, goal: str) -> List[str]:
        return [
            "Review previous outputs",
            "Identify logical errors or inconsistencies",
            "Assess quality against criteria",
            "Generate improvement suggestions",
            "Provide confidence scores",
        ]

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        start = time.time()
        task_id = f"reflect_{uuid.uuid4().hex[:8]}"
        self.status = AgentStatus.RUNNING

        try:
            prior_results = (context or {}).get("prior_results", [])
            evaluation = self.evaluate_results(prior_results, task)
            self._evaluations.append(evaluation)
            result = AgentResult(
                agent_id=self.agent_id, task_id=task_id, success=True,
                output=evaluation, latency=time.time() - start
            )
        except Exception as e:
            result = AgentResult(
                agent_id=self.agent_id, task_id=task_id, success=False,
                output=None, error=str(e), latency=time.time() - start
            )
        finally:
            self.status = AgentStatus.IDLE

        self.record_result(result)
        return result

    def evaluate_results(self, results: List[Any], original_task: str) -> Dict[str, Any]:
        if not results:
            return {
                "task": original_task[:80], "quality_score": 0.0,
                "issues": ["No results to evaluate"],
                "suggestions": ["Ensure agents produce output before reflection"],
                "should_retry": False, "confidence": 0.0
            }

        quality_score = 0.8
        issues = []
        suggestions = []

        if len(results) < 2:
            quality_score -= 0.1
            suggestions.append("Consider using multiple agents for cross-validation")

        for r in results:
            if isinstance(r, dict) and not r.get("success", True):
                quality_score -= 0.2
                issues.append(f"Agent failure: {r.get('error', 'unknown error')}")

        if quality_score < 0.6:
            suggestions.append("Retry with different approach or parameters")
        else:
            suggestions.append("Results appear satisfactory")

        return {
            "task": original_task[:80],
            "quality_score": round(max(0.0, quality_score), 3),
            "issues": issues,
            "suggestions": suggestions,
            "should_retry": quality_score < 0.5,
            "confidence": round(min(1.0, quality_score + 0.1), 3),
            "results_evaluated": len(results)
        }

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        return list(self._evaluations)

    def get_avg_quality_score(self) -> float:
        if not self._evaluations:
            return 0.0
        return sum(e.get("quality_score", 0) for e in self._evaluations) / len(self._evaluations)

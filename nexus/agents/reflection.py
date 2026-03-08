"""ReflectionAgent - meta-cognition, self-critique, and strategy improvement."""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from nexus.agents.base import AgentBase, AgentCapability

logger = logging.getLogger("nexus.agents.reflection")


@dataclass
class Anomaly:
    agent_id: str
    metric: str
    observed: float
    expected: float
    severity: str   # low | medium | high | critical
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


@dataclass
class StrategyImprovement:
    target_agent: str
    current_strategy: str
    proposed_strategy: str
    rationale: str
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReflectionReport:
    report_id: str
    timestamp: datetime
    agents_analysed: List[str]
    anomalies: List[Anomaly]
    improvements: List[StrategyImprovement]
    overall_health: float          # 0.0 – 1.0
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "agents_analysed": self.agents_analysed,
            "anomaly_count": len(self.anomalies),
            "improvement_count": len(self.improvements),
            "overall_health": self.overall_health,
            "summary": self.summary,
            "anomalies": [
                {
                    "agent": a.agent_id,
                    "metric": a.metric,
                    "observed": a.observed,
                    "expected": a.expected,
                    "severity": a.severity,
                    "description": a.description,
                }
                for a in self.anomalies
            ],
            "improvements": [
                {
                    "target": s.target_agent,
                    "current": s.current_strategy,
                    "proposed": s.proposed_strategy,
                    "rationale": s.rationale,
                    "confidence": s.confidence,
                }
                for s in self.improvements
            ],
        }


class ReflectionAgent(AgentBase):
    """
    Meta-cognitive agent that monitors other agents, performs self-critique,
    detects anomalies, and suggests strategy improvements.
    """

    # Thresholds
    ERROR_RATE_WARN = 0.1
    ERROR_RATE_CRIT = 0.3
    LATENCY_WARN_MS = 2000
    LATENCY_CRIT_MS = 10000

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(
            agent_id=agent_id,
            capabilities=[AgentCapability.REFLECTION, AgentCapability.REASONING],
        )
        self._agent_snapshots: Dict[str, List[Dict[str, Any]]] = {}
        self._report_history: List[ReflectionReport] = []
        self._metric_baselines: Dict[str, Dict[str, float]] = {}
        import uuid
        self._uuid = uuid

    # ── Data ingestion ─────────────────────────────────────────────────────────
    def ingest_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        if agent_id not in self._agent_snapshots:
            self._agent_snapshots[agent_id] = []
        state["_ingested_at"] = datetime.utcnow().isoformat()
        self._agent_snapshots[agent_id].append(state)
        # Keep last 200 snapshots per agent
        if len(self._agent_snapshots[agent_id]) > 200:
            self._agent_snapshots[agent_id] = self._agent_snapshots[agent_id][-200:]

    # ── Analysis ───────────────────────────────────────────────────────────────
    def _detect_anomalies(self) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        for agent_id, snapshots in self._agent_snapshots.items():
            if len(snapshots) < 2:
                continue
            recent = snapshots[-1]
            metrics = recent.get("metrics", {})

            error_count = metrics.get("error_count", 0)
            task_count = len(recent.get("task_history", [])) or 1
            error_rate = error_count / task_count

            if error_rate >= self.ERROR_RATE_CRIT:
                anomalies.append(Anomaly(
                    agent_id=agent_id,
                    metric="error_rate",
                    observed=error_rate,
                    expected=self.ERROR_RATE_WARN,
                    severity="critical",
                    description=f"Error rate {error_rate:.1%} exceeds critical threshold",
                ))
            elif error_rate >= self.ERROR_RATE_WARN:
                anomalies.append(Anomaly(
                    agent_id=agent_id,
                    metric="error_rate",
                    observed=error_rate,
                    expected=0.02,
                    severity="medium",
                    description=f"Error rate {error_rate:.1%} above warning threshold",
                ))

            # Status anomaly
            if recent.get("status") == "error":
                anomalies.append(Anomaly(
                    agent_id=agent_id,
                    metric="status",
                    observed=1.0,
                    expected=0.0,
                    severity="high",
                    description="Agent is in ERROR state",
                ))

        return anomalies

    def _suggest_improvements(self, anomalies: List[Anomaly]) -> List[StrategyImprovement]:
        suggestions: List[StrategyImprovement] = []
        for anomaly in anomalies:
            if anomaly.metric == "error_rate" and anomaly.severity in ("medium", "high", "critical"):
                suggestions.append(StrategyImprovement(
                    target_agent=anomaly.agent_id,
                    current_strategy="continue_as_is",
                    proposed_strategy="add_retry_with_backoff",
                    rationale=f"High error rate ({anomaly.observed:.1%}) suggests transient failures; retry logic may help.",
                    confidence=0.75,
                ))
            if anomaly.metric == "status" and anomaly.severity == "high":
                suggestions.append(StrategyImprovement(
                    target_agent=anomaly.agent_id,
                    current_strategy="auto_recover",
                    proposed_strategy="restart_with_state_preservation",
                    rationale="Agent stuck in ERROR state; controlled restart recommended.",
                    confidence=0.9,
                ))
        return suggestions

    def _compute_health(self, anomalies: List[Anomaly]) -> float:
        if not anomalies:
            return 1.0
        severity_weights = {"low": 0.05, "medium": 0.15, "high": 0.3, "critical": 0.5}
        total_penalty = sum(severity_weights.get(a.severity, 0.1) for a in anomalies)
        return max(0.0, 1.0 - total_penalty)

    # ── Full reflection cycle ──────────────────────────────────────────────────
    async def run_reflection(self) -> ReflectionReport:
        anomalies = self._detect_anomalies()
        improvements = self._suggest_improvements(anomalies)
        health = self._compute_health(anomalies)
        agents_analysed = list(self._agent_snapshots.keys())

        if health > 0.8:
            summary = "System health is GOOD. No critical issues detected."
        elif health > 0.5:
            summary = f"System health is DEGRADED. {len(anomalies)} anomaly(ies) found."
        else:
            summary = f"System health is CRITICAL. Immediate attention required."

        report = ReflectionReport(
            report_id=str(self._uuid.uuid4()),
            timestamp=datetime.utcnow(),
            agents_analysed=agents_analysed,
            anomalies=anomalies,
            improvements=improvements,
            overall_health=round(health, 3),
            summary=summary,
        )
        self._report_history.append(report)
        logger.info(
            "Reflection complete health=%.2f anomalies=%d improvements=%d",
            health, len(anomalies), len(improvements),
        )
        return report

    def get_latest_report(self) -> Optional[ReflectionReport]:
        return self._report_history[-1] if self._report_history else None

    # ── Self-critique ──────────────────────────────────────────────────────────
    def self_critique(self) -> Dict[str, Any]:
        my_snapshots = self._agent_snapshots.get(self.agent_id, [])
        critique: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "reports_generated": len(self._report_history),
            "agents_monitored": len(self._agent_snapshots),
        }
        if len(self._report_history) >= 2:
            prev_health = self._report_history[-2].overall_health
            curr_health = self._report_history[-1].overall_health
            delta = curr_health - prev_health
            critique["health_trend"] = "improving" if delta > 0 else "declining" if delta < 0 else "stable"
            critique["health_delta"] = round(delta, 3)
        return critique

    # ── AgentBase abstract impl ────────────────────────────────────────────────
    async def perceive(self, input_data: Any) -> Any:
        if isinstance(input_data, dict) and "agent_id" in input_data:
            self.ingest_agent_state(input_data["agent_id"], input_data)
        return input_data

    async def think(self, percept: Any) -> Any:
        return await self.run_reflection()

    async def act(self, decision: Any) -> Any:
        if isinstance(decision, ReflectionReport):
            return decision.to_dict()
        return str(decision)

    async def reflect(self) -> Any:
        return self.self_critique()

"""
NEXUS-AGI Self-Improver
Main RSI orchestrator - koordinasi semua komponen improvement
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .monitor import PerformanceMonitor
from .optimizer import PromptOptimizer
from .generator import ToolGenerator


@dataclass
class ImprovementProposal:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = "prompt"  # prompt, tool, architecture
    description: str = ""
    expected_gain: float = 0.0
    confidence: float = 0.0
    dry_run_result: Optional[Dict] = None
    applied: bool = False
    rolled_back: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class ImprovementReport:
    cycle_id: str
    proposals: List[ImprovementProposal]
    applied_count: int
    failed_count: int
    net_improvement: float
    timestamp: float = field(default_factory=time.time)


class SelfImprover:
    """Recursive Self-Improvement orchestrator."""

    CONFIDENCE_THRESHOLD = 0.7
    MIN_SAMPLES_FOR_IMPROVEMENT = 10

    def __init__(self, confidence_threshold: float = 0.7):
        self.monitor = PerformanceMonitor()
        self.optimizer = PromptOptimizer()
        self.generator = ToolGenerator()
        self._proposals: Dict[str, ImprovementProposal] = {}
        self._improvement_history: List[ImprovementReport] = []
        self._audit_log: List[Dict[str, Any]] = []
        self.confidence_threshold = confidence_threshold

    def run_improvement_cycle(self) -> ImprovementReport:
        """Run one full RSI cycle."""
        cycle_id = f"cycle_{int(time.time())}"
        self._log(cycle_id, "start", "Starting improvement cycle")

        proposals = self.propose_improvements()
        applied = 0
        failed = 0

        for proposal in proposals:
            if proposal.confidence >= self.confidence_threshold:
                success = self.apply_improvement(proposal)
                if success:
                    applied += 1
                else:
                    failed += 1
                    self.rollback_improvement(proposal.id)

        before = self.monitor.compute_metrics()
        net_improvement = sum(p.expected_gain for p in proposals if p.applied)

        report = ImprovementReport(
            cycle_id=cycle_id,
            proposals=proposals,
            applied_count=applied,
            failed_count=failed,
            net_improvement=round(net_improvement, 3)
        )
        self._improvement_history.append(report)
        self._log(cycle_id, "complete", f"Applied {applied} improvements")
        return report

    def evaluate_current_capabilities(self) -> Dict[str, Any]:
        """Evaluate current system capability matrix."""
        metrics = self.monitor.compute_metrics()
        bottlenecks = self.monitor.detect_bottlenecks()
        return {
            "metrics": metrics,
            "bottlenecks": len(bottlenecks),
            "critical_bottlenecks": sum(1 for b in bottlenecks if b.severity == "critical"),
            "optimizer_stats": self.optimizer.stats(),
            "generator_stats": self.generator.stats(),
            "overall_health": self._compute_health_score(metrics)
        }

    def propose_improvements(self) -> List[ImprovementProposal]:
        """Generate prioritized improvement proposals."""
        proposals = []
        opportunities = self.monitor.get_improvement_opportunities()

        for opp in opportunities:
            if opp["metric"] in ["task_success_rate", "planning_accuracy"]:
                proposals.append(ImprovementProposal(
                    type="prompt",
                    description=f"Optimize prompts for metric: {opp['metric']}",
                    expected_gain=opp["expected_gain"],
                    confidence=0.75
                ))
            elif opp["metric"] == "avg_latency":
                proposals.append(ImprovementProposal(
                    type="architecture",
                    description="Enable response caching to reduce latency",
                    expected_gain=opp["expected_gain"],
                    confidence=0.8
                ))
            elif opp["metric"] == "token_efficiency":
                proposals.append(ImprovementProposal(
                    type="prompt",
                    description="Apply prompt compression strategy",
                    expected_gain=opp["expected_gain"],
                    confidence=0.7
                ))

        for p in proposals:
            self._proposals[p.id] = p

        return sorted(proposals, key=lambda x: x.expected_gain, reverse=True)

    def apply_improvement(self, proposal: ImprovementProposal,
                          dry_run: bool = False) -> bool:
        """Apply an improvement proposal."""
        if proposal.confidence < self.confidence_threshold:
            self._log(proposal.id, "skip", f"Confidence {proposal.confidence} below threshold")
            return False

        if dry_run:
            proposal.dry_run_result = {
                "would_apply": True,
                "type": proposal.type,
                "description": proposal.description,
                "estimated_gain": proposal.expected_gain
            }
            return True

        # Simulate applying improvement
        proposal.applied = True
        self._log(proposal.id, "apply", f"Applied: {proposal.description}")
        return True

    def rollback_improvement(self, improvement_id: str) -> bool:
        """Rollback a previously applied improvement."""
        proposal = self._proposals.get(improvement_id)
        if not proposal or not proposal.applied:
            return False
        proposal.applied = False
        proposal.rolled_back = True
        self._log(improvement_id, "rollback", f"Rolled back: {proposal.description}")
        return True

    def _compute_health_score(self, metrics: Dict[str, float]) -> float:
        weights = {
            "task_success_rate": 0.4,
            "token_efficiency": 0.2,
            "planning_accuracy": 0.3,
            "memory_utilization": 0.1,
        }
        score = 0.0
        for metric, weight in weights.items():
            val = metrics.get(metric, 0)
            if metric == "memory_utilization":
                score += weight * (1 - val)
            else:
                score += weight * val
        return round(score, 3)

    def _log(self, ref_id: str, event: str, message: str) -> None:
        self._audit_log.append({
            "ref_id": ref_id, "event": event,
            "message": message, "timestamp": time.time()
        })

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._audit_log[-limit:]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_cycles": len(self._improvement_history),
            "total_proposals": len(self._proposals),
            "applied_proposals": sum(1 for p in self._proposals.values() if p.applied),
            "audit_entries": len(self._audit_log),
            "monitor": self.monitor.generate_performance_report()
        }

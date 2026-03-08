"""
NEXUS-AGI Performance Monitor
Track metrics dan detect bottlenecks untuk RSI
"""
from __future__ import annotations
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque


@dataclass
class TaskRecord:
    task_id: str
    success: bool
    latency: float
    tokens_used: int
    quality_score: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckReport:
    metric: str
    severity: str  # low, medium, high, critical
    current_value: float
    threshold: float
    suggestion: str
    priority: int  # 1=highest


class PerformanceMonitor:
    """Monitor performa sistem dan deteksi bottleneck."""

    THRESHOLDS = {
        "task_success_rate": 0.85,
        "avg_latency": 10.0,       # seconds
        "token_efficiency": 0.6,   # quality/token ratio
        "memory_utilization": 0.9,
        "planning_accuracy": 0.75,
    }

    def __init__(self, window_size: int = 100):
        self._records: deque = deque(maxlen=window_size)
        self._alerts: List[Dict[str, Any]] = []
        self._window_size = window_size

    def record_task(self, task_id: str, success: bool, latency: float,
                    tokens_used: int, quality_score: float,
                    metadata: Optional[Dict] = None) -> None:
        record = TaskRecord(
            task_id=task_id,
            success=success,
            latency=latency,
            tokens_used=tokens_used,
            quality_score=quality_score,
            metadata=metadata or {}
        )
        self._records.append(record)
        self._check_alerts(record)

    def _check_alerts(self, record: TaskRecord) -> None:
        metrics = self.compute_metrics()
        for metric, threshold in self.THRESHOLDS.items():
            val = metrics.get(metric, 0)
            if metric == "avg_latency" and val > threshold:
                self._alerts.append({
                    "metric": metric, "value": val,
                    "threshold": threshold, "timestamp": time.time()
                })
            elif metric != "avg_latency" and val < threshold:
                self._alerts.append({
                    "metric": metric, "value": val,
                    "threshold": threshold, "timestamp": time.time()
                })

    def compute_metrics(self) -> Dict[str, float]:
        if not self._records:
            return {k: 0.0 for k in self.THRESHOLDS}
        records = list(self._records)
        success_rate = sum(1 for r in records if r.success) / len(records)
        avg_latency = statistics.mean(r.latency for r in records)
        avg_tokens = statistics.mean(r.tokens_used for r in records) or 1
        avg_quality = statistics.mean(r.quality_score for r in records)
        token_efficiency = avg_quality / (avg_tokens / 1000) if avg_tokens > 0 else 0
        return {
            "task_success_rate": round(success_rate, 3),
            "avg_latency": round(avg_latency, 3),
            "token_efficiency": round(min(token_efficiency, 1.0), 3),
            "memory_utilization": 0.5,  # placeholder
            "planning_accuracy": round(avg_quality, 3),
        }

    def detect_bottlenecks(self) -> List[BottleneckReport]:
        metrics = self.compute_metrics()
        bottlenecks = []
        suggestions = {
            "task_success_rate": "Improve task decomposition and error handling",
            "avg_latency": "Optimize prompt length and use caching",
            "token_efficiency": "Use more concise prompts with examples",
            "memory_utilization": "Implement memory compression and pruning",
            "planning_accuracy": "Improve hierarchical planning depth",
        }
        priority = 1
        for metric, threshold in self.THRESHOLDS.items():
            val = metrics.get(metric, 0)
            is_bad = (metric == "avg_latency" and val > threshold) or \
                     (metric != "avg_latency" and val < threshold)
            if is_bad:
                gap = abs(val - threshold) / threshold
                severity = "critical" if gap > 0.3 else "high" if gap > 0.2 else "medium" if gap > 0.1 else "low"
                bottlenecks.append(BottleneckReport(
                    metric=metric,
                    severity=severity,
                    current_value=val,
                    threshold=threshold,
                    suggestion=suggestions.get(metric, "Investigate further"),
                    priority=priority
                ))
                priority += 1
        return sorted(bottlenecks, key=lambda x: x.priority)

    def get_improvement_opportunities(self) -> List[Dict[str, Any]]:
        bottlenecks = self.detect_bottlenecks()
        return [
            {
                "metric": b.metric,
                "severity": b.severity,
                "suggestion": b.suggestion,
                "expected_gain": round(abs(b.current_value - b.threshold) / b.threshold, 2)
            }
            for b in bottlenecks
        ]

    def generate_performance_report(self) -> Dict[str, Any]:
        metrics = self.compute_metrics()
        records = list(self._records)
        latencies = [r.latency for r in records] if records else [0]
        return {
            "summary": metrics,
            "bottlenecks": [
                {"metric": b.metric, "severity": b.severity, "suggestion": b.suggestion}
                for b in self.detect_bottlenecks()
            ],
            "trends": {
                "latency_p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 3),
                "latency_p50": round(sorted(latencies)[int(len(latencies) * 0.50)] if latencies else 0, 3),
                "recent_success_rate": round(
                    sum(1 for r in records[-10:] if r.success) / max(len(records[-10:]), 1), 3
                )
            },
            "total_tasks": len(self._records),
            "alerts_count": len(self._alerts),
            "timestamp": time.time()
        }

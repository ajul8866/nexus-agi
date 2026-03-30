"""
NEXUS-AGI Budget Manager
Cost control dan token tracking untuk production safety
"""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum, auto
from datetime import datetime, timedelta

logger = logging.getLogger("nexus.budget")


class AlertSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    HARD_LIMIT = auto()


@dataclass
class TokenUsage:
    """Track token usage per task/agent."""
    task_id: str
    agent_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    model: str = "unknown"


@dataclass
class BudgetAlert:
    """Alert when budget thresholds exceeded."""
    severity: AlertSeverity
    message: str
    current_spend: float
    limit: float
    percentage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action_taken: str = ""  # e.g., "blocked_task", "notified_user"


@dataclass
class BudgetConfig:
    """Budget configuration."""
    daily_limit_usd: float = 10.0
    task_limit_usd: float = 2.0
    warning_threshold: float = 0.8  # 80% of limit
    critical_threshold: float = 0.95  # 95% of limit
    hard_limit: float = 1.0  # 100% - block all
    enabled: bool = True


class BudgetManager:
    """
    Manage token budgets dan cost control untuk NEXUS-AGI.
    
    Features:
    - Daily budget limits dengan automatic reset
    - Per-task budget tracking
    - Real-time cost estimation
    - Alert system dengan escalating severity
    - Hard limits untuk prevent runaway costs
    """
    
    # Model pricing (per 1K tokens) - update sesuai provider
    MODEL_PRICING = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
        "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
        "qwen-3.5": {"prompt": 0.0005, "completion": 0.001},  # Estimate
        "default": {"prompt": 0.001, "completion": 0.002},
    }
    
    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self._daily_spend: float = 0.0
        self._task_spend: Dict[str, float] = {}
        self._usage_history: List[TokenUsage] = []
        self._alerts: List[BudgetAlert] = []
        self._last_reset: datetime = datetime.utcnow()
        self._enabled = self.config.enabled
        logger.info("BudgetManager initialized with daily_limit=$%.2f", 
                    self.config.daily_limit_usd)
    
    # ── Cost Calculation ───────────────────────────────────────────────────────
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, 
                      model: str = "default") -> float:
        """Estimate cost for given token counts."""
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["default"])
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        return round(prompt_cost + completion_cost, 6)
    
    def record_usage(self, task_id: str, agent_id: str, 
                     prompt_tokens: int, completion_tokens: int,
                     model: str = "default") -> TokenUsage:
        """Record token usage and update budget tracking."""
        total_tokens = prompt_tokens + completion_tokens
        cost = self.estimate_cost(prompt_tokens, completion_tokens, model)
        
        usage = TokenUsage(
            task_id=task_id,
            agent_id=agent_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            model=model
        )
        
        self._usage_history.append(usage)
        self._daily_spend += cost
        self._task_spend[task_id] = self._task_spend.get(task_id, 0) + cost
        
        logger.debug("Recorded usage: task=%s cost=$%.6f total_daily=$%.2f",
                     task_id, cost, self._daily_spend)
        
        # Check thresholds
        self._check_thresholds(task_id)
        
        return usage
    
    # ── Threshold Checking ─────────────────────────────────────────────────────
    def _check_thresholds(self, task_id: str) -> None:
        """Check budget thresholds and trigger alerts."""
        daily_pct = self._daily_spend / self.config.daily_limit_usd
        task_spend = self._task_spend.get(task_id, 0)
        task_pct = task_spend / self.config.task_limit_usd
        
        # Check daily limit
        if daily_pct >= self.config.hard_limit:
            self._create_alert(
                AlertSeverity.HARD_LIMIT,
                f"Daily budget HARD LIMIT reached (${self.config.daily_limit_usd})",
                self._daily_spend,
                self.config.daily_limit_usd,
                daily_pct * 100,
                action_taken="blocked_further_tasks"
            )
        elif daily_pct >= self.config.critical_threshold:
            self._create_alert(
                AlertSeverity.CRITICAL,
                f"Daily budget critical: {daily_pct*100:.1f}% used",
                self._daily_spend,
                self.config.daily_limit_usd,
                daily_pct * 100
            )
        elif daily_pct >= self.config.warning_threshold:
            self._create_alert(
                AlertSeverity.WARNING,
                f"Daily budget warning: {daily_pct*100:.1f}% used",
                self._daily_spend,
                self.config.daily_limit_usd,
                daily_pct * 100
            )
        
        # Check task limit
        if task_pct >= self.config.hard_limit:
            self._create_alert(
                AlertSeverity.HARD_LIMIT,
                f"Task {task_id} budget HARD LIMIT (${self.config.task_limit_usd})",
                task_spend,
                self.config.task_limit_usd,
                task_pct * 100,
                action_taken="blocked_task_continuation"
            )
    
    def _create_alert(self, severity: AlertSeverity, message: str,
                      current: float, limit: float, percentage: float,
                      action_taken: str = "") -> BudgetAlert:
        """Create and store budget alert."""
        alert = BudgetAlert(
            severity=severity,
            message=message,
            current_spend=current,
            limit=limit,
            percentage=percentage,
            action_taken=action_taken
        )
        self._alerts.append(alert)
        logger.warning("BUDGET ALERT [%s]: %s", severity.name, message)
    
    # ── Budget Control ─────────────────────────────────────────────────────────
    def can_proceed(self, task_id: str, estimated_cost: float = 0) -> bool:
        """Check if task can proceed within budget."""
        if not self._enabled:
            return True
        
        # Check daily limit
        if self._daily_spend >= self.config.daily_limit_usd * self.config.hard_limit:
            logger.error("Task %s BLOCKED: daily budget exceeded", task_id)
            return False
        
        # Check task limit
        task_spend = self._task_spend.get(task_id, 0)
        if task_spend >= self.config.task_limit_usd * self.config.hard_limit:
            logger.error("Task %s BLOCKED: task budget exceeded", task_id)
            return False
        
        # Check if estimated cost would exceed limits
        if self._daily_spend + estimated_cost > self.config.daily_limit_usd:
            logger.warning("Task %s BLOCKED: estimated cost exceeds daily budget", task_id)
            return False
        
        return True
    
    def block_task(self, task_id: str) -> None:
        """Manually block a task (called by orchestrator on hard limit)."""
        self._create_alert(
            AlertSeverity.HARD_LIMIT,
            f"Task {task_id} manually blocked",
            self._task_spend.get(task_id, 0),
            self.config.task_limit_usd,
            100,
            action_taken="orchestrator_blocked"
        )
    
    # ── Reset & Maintenance ────────────────────────────────────────────────────
    def reset_daily(self) -> None:
        """Reset daily budget (call at midnight UTC or configured timezone)."""
        self._daily_spend = 0.0
        self._task_spend.clear()
        self._last_reset = datetime.utcnow()
        logger.info("Daily budget reset. Last reset: %s", self._last_reset)
    
    def should_reset(self) -> bool:
        """Check if daily reset is needed (24h since last reset)."""
        return datetime.utcnow() - self._last_reset > timedelta(days=1)
    
    # ── Reporting ──────────────────────────────────────────────────────────────
    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        daily_pct = (self._daily_spend / self.config.daily_limit_usd * 100) \
                    if self.config.daily_limit_usd > 0 else 0
        
        return {
            "enabled": self._enabled,
            "daily_spend_usd": round(self._daily_spend, 4),
            "daily_limit_usd": self.config.daily_limit_usd,
            "daily_percentage": round(daily_pct, 2),
            "task_count": len(self._task_spend),
            "total_tasks_tracked": len(self._usage_history),
            "alerts_count": len(self._alerts),
            "last_reset": self._last_reset.isoformat(),
            "status": "blocked" if daily_pct >= 100 else 
                      "critical" if daily_pct >= 95 else
                      "warning" if daily_pct >= 80 else "ok"
        }
    
    def get_task_spend(self, task_id: str) -> Dict[str, Any]:
        """Get spending details for specific task."""
        spend = self._task_spend.get(task_id, 0)
        usage_records = [u for u in self._usage_history if u.task_id == task_id]
        
        return {
            "task_id": task_id,
            "total_cost_usd": round(spend, 4),
            "limit_usd": self.config.task_limit_usd,
            "percentage": round(spend / self.config.task_limit_usd * 100, 2) 
                          if self.config.task_limit_usd > 0 else 0,
            "usage_count": len(usage_records),
            "total_tokens": sum(u.total_tokens for u in usage_records),
            "status": "blocked" if spend >= self.config.task_limit_usd else "ok"
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent budget alerts."""
        return [
            {
                "severity": a.severity.name,
                "message": a.message,
                "percentage": round(a.percentage, 2),
                "action": a.action_taken,
                "timestamp": a.timestamp.isoformat()
            }
            for a in sorted(self._alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
        ]
    
    def export_usage_log(self) -> List[Dict[str, Any]]:
        """Export full usage history for analytics."""
        return [
            {
                "task_id": u.task_id,
                "agent_id": u.agent_id,
                "model": u.model,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "cost_usd": u.cost_usd,
                "timestamp": datetime.utcfromtimestamp(u.timestamp).isoformat()
            }
            for u in self._usage_history
        ]
    
    # ── Enable/Disable ─────────────────────────────────────────────────────────
    def enable(self) -> None:
        """Enable budget tracking."""
        self._enabled = True
        logger.info("Budget tracking ENABLED")
    
    def disable(self) -> None:
        """Disable budget tracking (use with caution)."""
        self._enabled = False
        logger.warning("Budget tracking DISABLED - no cost limits enforced")


# ── Integration Helper ─────────────────────────────────────────────────────────
def create_budget_manager(daily_limit: float = 10.0, 
                          task_limit: float = 2.0,
                          warning_pct: float = 0.8) -> BudgetManager:
    """Factory function to create configured BudgetManager."""
    config = BudgetConfig(
        daily_limit_usd=daily_limit,
        task_limit_usd=task_limit,
        warning_threshold=warning_pct,
        critical_threshold=0.95,
        hard_limit=1.0,
        enabled=True
    )
    return BudgetManager(config)

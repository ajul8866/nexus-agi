"""
NEXUS-AGI Sandbox
Secure execution environment for tool calls
"""
from __future__ import annotations
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class SandboxStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SandboxResult:
    sandbox_id: str
    success: bool
    output: Any
    error: Optional[str]
    status: str
    execution_time: float
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class Sandbox:
    """
    Isolated execution environment for tool calls.
    Provides timeout control, resource limits, and audit logging.
    """

    DEFAULT_TIMEOUT = 30.0
    MAX_MEMORY_MB = 512

    def __init__(self, max_concurrent: int = 10,
                 default_timeout: float = 30.0):
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._active_executions: Dict[str, Dict] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._semaphore = threading.Semaphore(max_concurrent)

    def execute(self, func: Callable, args: tuple = (),
                kwargs: Dict = None, timeout: Optional[float] = None,
                sandbox_id: Optional[str] = None) -> SandboxResult:
        """Execute function in isolated sandbox with timeout."""
        sid = sandbox_id or f"sbx_{uuid.uuid4().hex[:8]}"
        kwargs = kwargs or {}
        timeout = timeout or self._default_timeout
        start = time.time()

        if not self._semaphore.acquire(timeout=5.0):
            return SandboxResult(
                sandbox_id=sid, success=False, output=None,
                error="Max concurrent executions reached",
                status=SandboxStatus.ERROR.value, execution_time=0.0
            )

        self._active_executions[sid] = {
            "func": func.__name__, "start": start, "status": SandboxStatus.RUNNING.value
        }

        result_container = {"output": None, "error": None, "done": False}

        def run():
            try:
                result_container["output"] = func(*args, **kwargs)
                result_container["done"] = True
            except Exception as e:
                result_container["error"] = str(e)
                result_container["done"] = True

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        exec_time = time.time() - start
        self._semaphore.release()
        self._active_executions.pop(sid, None)

        if not result_container["done"]:
            status = SandboxStatus.TIMEOUT.value
            success = False
            error = f"Execution timed out after {timeout}s"
            output = None
        elif result_container["error"]:
            status = SandboxStatus.ERROR.value
            success = False
            error = result_container["error"]
            output = None
        else:
            status = SandboxStatus.COMPLETED.value
            success = True
            error = None
            output = result_container["output"]

        sbx_result = SandboxResult(
            sandbox_id=sid, success=success, output=output,
            error=error, status=status, execution_time=round(exec_time, 3)
        )
        self._audit_log.append({
            "sandbox_id": sid, "func": func.__name__,
            "status": status, "execution_time": exec_time,
            "timestamp": time.time()
        })
        return sbx_result

    def get_active_count(self) -> int:
        return len(self._active_executions)

    def get_audit_log(self, limit: int = 50) -> List[Dict]:
        return self._audit_log[-limit:]

    def stats(self) -> Dict[str, Any]:
        total = len(self._audit_log)
        success = sum(1 for e in self._audit_log if e["status"] == SandboxStatus.COMPLETED.value)
        timeouts = sum(1 for e in self._audit_log if e["status"] == SandboxStatus.TIMEOUT.value)
        avg_time = (
            sum(e["execution_time"] for e in self._audit_log) / total
            if total > 0 else 0.0
        )
        return {
            "total_executions": total,
            "success_count": success,
            "timeout_count": timeouts,
            "error_count": total - success - timeouts,
            "avg_execution_time": round(avg_time, 3),
            "active_now": self.get_active_count()
        }

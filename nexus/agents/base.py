"""
NEXUS-AGI Agent Base Class
"""
from __future__ import annotations
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    recipient_id: str = ""
    content: Any = None
    msg_type: str = "task"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent_id: str
    task_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    latency: float = 0.0
    tokens_used: int = 0
    timestamp: float = field(default_factory=time.time)


class AgentBase(ABC):
    """
    Abstract base class for all NEXUS-AGI agents.
    Provides common infrastructure: messaging, state, logging.
    """

    def __init__(self, name: str, role: str, capabilities: Optional[List[str]] = None):
        self.agent_id = f"{role}_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.role = role
        self.capabilities = capabilities or []
        self.status = AgentStatus.IDLE
        self._message_queue: List[AgentMessage] = []
        self._history: List[AgentResult] = []
        self._created_at = time.time()
        self._metadata: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute a task and return the result."""
        pass

    @abstractmethod
    async def plan(self, goal: str) -> List[str]:
        """Generate a plan (list of steps) to achieve a goal."""
        pass

    async def send_message(self, recipient_id: str, content: Any, msg_type: str = "task") -> AgentMessage:
        msg = AgentMessage(sender_id=self.agent_id, recipient_id=recipient_id,
                           content=content, msg_type=msg_type)
        return msg

    def receive_message(self, message: AgentMessage) -> None:
        self._message_queue.append(message)

    def get_pending_messages(self) -> List[AgentMessage]:
        msgs = list(self._message_queue)
        self._message_queue.clear()
        return msgs

    def record_result(self, result: AgentResult) -> None:
        self._history.append(result)

    def get_success_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.success) / len(self._history)

    def get_avg_latency(self) -> float:
        if not self._history:
            return 0.0
        return sum(r.latency for r in self._history) / len(self._history)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id, "name": self.name, "role": self.role,
            "capabilities": self.capabilities, "status": self.status.value,
            "success_rate": round(self.get_success_rate(), 3),
            "avg_latency": round(self.get_avg_latency(), 3),
            "tasks_completed": len(self._history),
            "created_at": self._created_at
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name} status={self.status.value}>"

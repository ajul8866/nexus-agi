"""Base Agent class for NEXUS-AGI."""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.agents.base")


class AgentCapability(Enum):
    REASONING = auto()
    PLANNING = auto()
    CODING = auto()
    RESEARCH = auto()
    REFLECTION = auto()
    ORCHESTRATION = auto()
    TOOL_USE = auto()
    MEMORY_MANAGEMENT = auto()
    SAFETY = auto()


@dataclass
class AgentState:
    agent_id: str
    status: str = "idle"
    current_task: Optional[str] = None
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "current_task": self.current_task,
            "task_count": len(self.task_history),
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }


class AgentBase(ABC):
    """
    Abstract base class for all NEXUS-AGI agents.
    Provides async execution loop, memory, tools, and messaging.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        max_queue_size: int = 100,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.capabilities = capabilities or []
        self.state = AgentState(agent_id=self.agent_id)
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.memory_store: Dict[str, Any] = {}
        self.tool_registry: Dict[str, Any] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        logger.info("Agent created id=%s caps=%s", self.agent_id, [c.name for c in self.capabilities])

    # ── Abstract interface ─────────────────────────────────────────────────────
    @abstractmethod
    async def perceive(self, input_data: Any) -> Any:
        """Process incoming percepts/messages."""

    @abstractmethod
    async def think(self, percept: Any) -> Any:
        """Reason about a percept and produce a plan or decision."""

    @abstractmethod
    async def act(self, decision: Any) -> Any:
        """Execute the decision and return a result."""

    @abstractmethod
    async def reflect(self) -> Any:
        """Meta-cognitive reflection on recent actions."""

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._running = True
        self.state.status = "running"
        logger.info("Agent %s started", self.agent_id)
        await self._execution_loop()

    async def stop(self) -> None:
        self._running = False
        self.state.status = "stopped"
        logger.info("Agent %s stopped", self.agent_id)

    async def pause(self) -> None:
        self._running = False
        self.state.status = "paused"

    async def resume(self) -> None:
        self._running = True
        self.state.status = "running"
        await self._execution_loop()

    # ── Execution loop ─────────────────────────────────────────────────────────
    async def _execution_loop(self) -> None:
        while self._running:
            try:
                try:
                    raw_input = await asyncio.wait_for(
                        self.message_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue

                self.state.last_active = datetime.utcnow()
                self.state.current_task = str(raw_input)[:80]
                self.state.status = "thinking"

                percept = await self.perceive(raw_input)
                decision = await self.think(percept)
                result = await self.act(decision)

                self.state.task_history.append(
                    {
                        "input": str(raw_input)[:200],
                        "result": str(result)[:200],
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                self.state.status = "idle"
                self.state.current_task = None
                self.message_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Agent %s error: %s", self.agent_id, exc)
                self.state.status = "error"
                self.state.metrics["error_count"] = (
                    self.state.metrics.get("error_count", 0) + 1
                )
                await asyncio.sleep(0.5)

    # ── Utilities ──────────────────────────────────────────────────────────────
    async def receive(self, message: Any) -> None:
        """Put a message onto this agent's queue."""
        await self.message_queue.put(message)

    def register_tool(self, name: str, func: Any, description: str = "") -> None:
        self.tool_registry[name] = {"func": func, "description": description}
        logger.debug("Agent %s registered tool: %s", self.agent_id, name)

    async def use_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tool_registry:
            raise ValueError(f"Tool not found: {name}")
        tool = self.tool_registry[name]
        if asyncio.iscoroutinefunction(tool["func"]):
            return await tool["func"](**kwargs)
        return tool["func"](**kwargs)

    def store_memory(self, key: str, value: Any) -> None:
        self.memory_store[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self.memory_store.get(key, default)

    def has_capability(self, cap: AgentCapability) -> bool:
        return cap in self.capabilities

    def get_state(self) -> Dict[str, Any]:
        return self.state.to_dict()

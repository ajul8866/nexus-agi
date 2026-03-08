"""NEXUS-AGI Kernel - Core event loop, agent registry, and message bus."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("nexus.kernel")


class AgentLifecycle(Enum):
    INIT = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""          # "" = broadcast
    topic: str = ""
    payload: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0            # higher = more urgent


class MessageBus:
    """Async pub/sub message bus with topic routing."""

    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._global_subscribers: List[asyncio.Queue] = []
        self._history: List[Message] = []
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(topic, []).append(q)
        logger.debug("Subscribed to topic: %s", topic)
        return q

    async def subscribe_all(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._global_subscribers.append(q)
        return q

    async def publish(self, message: Message) -> None:
        self._history.append(message)
        async with self._lock:
            # Topic-specific
            for q in self._subscribers.get(message.topic, []):
                await q.put(message)
            # Global listeners
            for q in self._global_subscribers:
                await q.put(message)
        logger.debug("Published message topic=%s sender=%s", message.topic, message.sender)

    async def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [
                    q for q in self._subscribers[topic] if q is not queue
                ]

    def get_history(self, limit: int = 100) -> List[Message]:
        return self._history[-limit:]


@dataclass
class AgentRecord:
    agent_id: str
    agent_type: str
    instance: Any
    lifecycle: AgentLifecycle = AgentLifecycle.INIT
    registered_at: datetime = field(default_factory=datetime.utcnow)
    task_count: int = 0
    error_count: int = 0


class NexusKernel:
    """
    Core AGI kernel - manages lifecycle, agent registry, and messaging.
    Uses asyncio event loop as backbone.
    """

    def __init__(self, max_agents: int = 10):
        self.kernel_id = str(uuid.uuid4())
        self.max_agents = max_agents
        self.message_bus = MessageBus()
        self._agents: Dict[str, AgentRecord] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._hooks: Dict[str, List[Callable]] = {
            "on_agent_registered": [],
            "on_agent_stopped": [],
            "on_message": [],
        }
        logger.info("NexusKernel initialised  id=%s", self.kernel_id)

    async def start(self) -> None:
        if self._running:
            logger.warning("Kernel already running")
            return
        self._running = True
        logger.info("NexusKernel STARTING")
        for rec in self._agents.values():
            await self._start_agent(rec)
        logger.info("NexusKernel RUNNING with %d agents", len(self._agents))

    async def stop(self) -> None:
        logger.info("NexusKernel STOPPING")
        self._running = False
        for rec in self._agents.values():
            rec.lifecycle = AgentLifecycle.STOPPED
            if hasattr(rec.instance, "stop"):
                await rec.instance.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("NexusKernel STOPPED")

    async def _start_agent(self, rec: AgentRecord) -> None:
        rec.lifecycle = AgentLifecycle.RUNNING
        if hasattr(rec.instance, "start"):
            task = asyncio.create_task(rec.instance.start(), name=rec.agent_id)
            self._tasks.append(task)

    def register_agent(self, agent_instance: Any, agent_type: str = "generic") -> str:
        if len(self._agents) >= self.max_agents:
            raise RuntimeError(f"Max agent limit reached: {self.max_agents}")
        agent_id = getattr(agent_instance, "agent_id", str(uuid.uuid4()))
        rec = AgentRecord(
            agent_id=agent_id,
            agent_type=agent_type,
            instance=agent_instance,
        )
        self._agents[agent_id] = rec
        for hook in self._hooks["on_agent_registered"]:
            hook(rec)
        logger.info("Registered agent id=%s type=%s", agent_id, agent_type)
        return agent_id

    def unregister_agent(self, agent_id: str) -> None:
        if agent_id in self._agents:
            rec = self._agents.pop(agent_id)
            rec.lifecycle = AgentLifecycle.STOPPED
            for hook in self._hooks["on_agent_stopped"]:
                hook(rec)
            logger.info("Unregistered agent id=%s", agent_id)

    def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentRecord]:
        return list(self._agents.values())

    async def send_message(self, sender: str, recipient: str, topic: str, payload: Any, priority: int = 0) -> Message:
        msg = Message(sender=sender, recipient=recipient, topic=topic, payload=payload, priority=priority)
        await self.message_bus.publish(msg)
        return msg

    async def broadcast(self, sender: str, topic: str, payload: Any) -> Message:
        return await self.send_message(sender, "", topic, payload)

    def add_hook(self, event: str, callback: Callable) -> None:
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}")
        self._hooks[event].append(callback)

    def status(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "running": self._running,
            "agent_count": len(self._agents),
            "max_agents": self.max_agents,
            "message_history": len(self.message_bus._history),
            "agents": [
                {"id": r.agent_id, "type": r.agent_type, "lifecycle": r.lifecycle.name, "tasks": r.task_count}
                for r in self._agents.values()
            ],
        }

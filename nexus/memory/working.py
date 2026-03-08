"""
NEXUS-AGI Working Memory
Short-term context storage for active reasoning
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import OrderedDict


@dataclass
class WorkingMemorySlot:
    key: str
    value: Any
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # seconds until expiry

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class WorkingMemory:
    """
    Short-term working memory with LRU eviction and TTL support.
    Holds active context for ongoing reasoning tasks.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._slots: OrderedDict[str, WorkingMemorySlot] = OrderedDict()
        self._context_stack: List[Dict[str, Any]] = []

    def set(self, key: str, value: Any, priority: float = 0.5,
            ttl: Optional[float] = None) -> WorkingMemorySlot:
        # Evict expired first
        self._evict_expired()

        if key in self._slots:
            slot = self._slots[key]
            slot.value = value
            slot.priority = priority
            slot.last_accessed = time.time()
            self._slots.move_to_end(key)
            return slot

        if len(self._slots) >= self.capacity:
            self._evict_lru()

        slot = WorkingMemorySlot(key=key, value=value, priority=priority, ttl=ttl)
        self._slots[key] = slot
        return slot

    def get(self, key: str, default: Any = None) -> Any:
        slot = self._slots.get(key)
        if slot is None:
            return default
        if slot.is_expired():
            del self._slots[key]
            return default
        slot.last_accessed = time.time()
        slot.access_count += 1
        self._slots.move_to_end(key)
        return slot.value

    def delete(self, key: str) -> bool:
        if key in self._slots:
            del self._slots[key]
            return True
        return False

    def clear(self) -> int:
        count = len(self._slots)
        self._slots.clear()
        return count

    def push_context(self, context: Dict[str, Any]) -> None:
        self._context_stack.append(context)
        for k, v in context.items():
            self.set(k, v, priority=0.8, ttl=300)

    def pop_context(self) -> Optional[Dict[str, Any]]:
        if self._context_stack:
            return self._context_stack.pop()
        return None

    def get_current_context(self) -> Dict[str, Any]:
        return self._context_stack[-1] if self._context_stack else {}

    def get_all(self, include_expired: bool = False) -> Dict[str, Any]:
        result = {}
        for key, slot in list(self._slots.items()):
            if not include_expired and slot.is_expired():
                continue
            result[key] = slot.value
        return result

    def _evict_expired(self) -> int:
        expired = [k for k, s in self._slots.items() if s.is_expired()]
        for k in expired:
            del self._slots[k]
        return len(expired)

    def _evict_lru(self) -> None:
        # Evict lowest priority + least recently used
        if not self._slots:
            return
        lru_key = min(self._slots.keys(),
                      key=lambda k: (self._slots[k].priority, self._slots[k].last_accessed))
        del self._slots[lru_key]

    def stats(self) -> Dict[str, Any]:
        self._evict_expired()
        return {
            "capacity": self.capacity,
            "used": len(self._slots),
            "utilization": round(len(self._slots) / self.capacity, 3),
            "context_depth": len(self._context_stack),
            "avg_priority": round(sum(s.priority for s in self._slots.values()) / len(self._slots), 3)
                            if self._slots else 0,
        }

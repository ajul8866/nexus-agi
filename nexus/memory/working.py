"""WorkingMemory - short-term buffer with Miller's Law and attention mechanism."""

import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.memory.working")

# Miller's Law: cognitive chunk limit = 7 ± 2
MILLER_CAPACITY = 7
MILLER_MIN = 5
MILLER_MAX = 9


@dataclass
class MemoryItem:
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    label: str = ""
    relevance: float = 1.0       # 0.0 – 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None   # None = no expiry

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds

    def attention_weight(self, recency_bias: float = 0.6, relevance_bias: float = 0.4) -> float:
        """Attention = weighted combo of recency and relevance."""
        recency = max(0.0, 1.0 - self.age_seconds / 3600.0)  # decays over 1 hour
        return recency_bias * recency + relevance_bias * self.relevance

    def access(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "content": str(self.content)[:200],
            "relevance": self.relevance,
            "age_seconds": round(self.age_seconds, 1),
            "access_count": self.access_count,
            "attention_weight": round(self.attention_weight(), 3),
            "expired": self.is_expired,
        }


class WorkingMemory:
    """
    Short-term working memory buffer implementing Miller's Law (7±2 items).
    Uses attention mechanism to decide what to keep when over capacity.
    """

    def __init__(self, capacity: int = MILLER_CAPACITY):
        self.capacity = max(MILLER_MIN, min(MILLER_MAX, capacity))
        self._buffer: Dict[str, MemoryItem] = {}
        self._context: Dict[str, Any] = {}   # named context slots

    # ── Basic operations ───────────────────────────────────────────────────────────
    def push(
        self,
        content: Any,
        label: str = "",
        relevance: float = 1.0,
        ttl_seconds: Optional[float] = None,
    ) -> MemoryItem:
        self._evict_expired()
        item = MemoryItem(
            content=content,
            label=label,
            relevance=relevance,
            ttl_seconds=ttl_seconds,
        )
        if len(self._buffer) >= self.capacity:
            self._evict_lowest_attention()
        self._buffer[item.item_id] = item
        logger.debug("WorkingMemory push label=%s items=%d/%d", label, len(self._buffer), self.capacity)
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        item = self._buffer.get(item_id)
        if item:
            if item.is_expired:
                del self._buffer[item_id]
                return None
            item.access()
        return item

    def get_by_label(self, label: str) -> List[MemoryItem]:
        return [i for i in self._buffer.values() if i.label == label and not i.is_expired]

    def remove(self, item_id: str) -> bool:
        return self._buffer.pop(item_id, None) is not None

    def clear(self) -> None:
        self._buffer.clear()

    # ── Attention ──────────────────────────────────────────────────────────────────
    def get_attended_items(self, top_k: Optional[int] = None) -> List[MemoryItem]:
        """Return items sorted by attention weight (highest first)."""
        valid = [i for i in self._buffer.values() if not i.is_expired]
        valid.sort(key=lambda i: i.attention_weight(), reverse=True)
        return valid[:top_k] if top_k else valid

    def focus(self, top_k: int = 3) -> List[MemoryItem]:
        """Return the top-k most attended items (cognitive focus)."""
        return self.get_attended_items(top_k=top_k)

    # ── Eviction ──────────────────────────────────────────────────────────────────
    def _evict_expired(self) -> int:
        expired = [iid for iid, i in self._buffer.items() if i.is_expired]
        for iid in expired:
            del self._buffer[iid]
        return len(expired)

    def _evict_lowest_attention(self) -> Optional[MemoryItem]:
        if not self._buffer:
            return None
        lowest_id = min(self._buffer, key=lambda iid: self._buffer[iid].attention_weight())
        evicted = self._buffer.pop(lowest_id)
        logger.debug("Evicted lowest-attention item: %s", evicted.label)
        return evicted

    # ── Context slots ─────────────────────────────────────────────────────────────
    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def update_context(self, data: Dict[str, Any]) -> None:
        self._context.update(data)

    def clear_context(self) -> None:
        self._context.clear()

    # ── Snapshot ──────────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        self._evict_expired()
        return {
            "capacity": self.capacity,
            "used": len(self._buffer),
            "utilization": round(len(self._buffer) / self.capacity, 2),
            "context_keys": list(self._context.keys()),
            "items": [i.to_dict() for i in self.get_attended_items()],
        }

    def __len__(self) -> int:
        self._evict_expired()
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemory(used={len(self._buffer)}/{self.capacity})"

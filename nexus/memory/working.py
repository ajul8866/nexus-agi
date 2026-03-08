"""WorkingMemory - fast, limited-capacity scratchpad with priority eviction."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("nexus.memory.working")


@dataclass
class MemoryItem:
    key: str
    value: Any
    priority: float = 0.5          # 0.0 = low, 1.0 = critical
    ttl_seconds: Optional[float] = None
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    pinned: bool = False            # pinned items are never evicted

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    def eviction_score(self) -> float:
        """Lower score = evict first."""
        if self.pinned:
            return float("inf")
        age = time.monotonic() - self.last_accessed
        recency = 1.0 / (1.0 + age)
        return self.priority * 0.5 + recency * 0.3 + min(self.access_count, 100) / 100.0 * 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "priority": self.priority, "ttl_seconds": self.ttl_seconds, "access_count": self.access_count, "tags": self.tags, "pinned": self.pinned, "expired": self.is_expired(), "eviction_score": round(self.eviction_score(), 4)}


class WorkingMemory:
    """
    Fixed-capacity working memory (cognitive scratchpad).
    Evicts lowest-priority / least-recently-used items when full.
    Supports TTL expiry, pinning, tagging, and snapshots.
    """

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._store: Dict[str, MemoryItem] = {}
        self._tag_index: Dict[str, List[str]] = {}  # tag -> [keys]
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, priority: float = 0.5, ttl: Optional[float] = None, tags: Optional[List[str]] = None, pin: bool = False) -> MemoryItem:
        self._purge_expired()
        if key not in self._store and len(self._store) >= self.capacity:
            self._evict_one()
        item = MemoryItem(key=key, value=value, priority=max(0.0, min(1.0, priority)), ttl_seconds=ttl, tags=tags or [], pinned=pin)
        self._store[key] = item
        for tag in item.tags:
            self._tag_index.setdefault(tag, []).append(key)
        return item

    def get(self, key: str, default: Any = None) -> Any:
        self._purge_expired()
        item = self._store.get(key)
        if item is None or item.is_expired():
            self._miss_count += 1
            if item:
                self._remove(key)
            return default
        item.last_accessed = time.monotonic()
        item.access_count += 1
        self._hit_count += 1
        return item.value

    def get_item(self, key: str) -> Optional[MemoryItem]:
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        if key in self._store:
            self._remove(key)
            return True
        return False

    def exists(self, key: str) -> bool:
        item = self._store.get(key)
        if item and item.is_expired():
            self._remove(key)
            return False
        return item is not None

    def update_priority(self, key: str, priority: float) -> bool:
        if key in self._store:
            self._store[key].priority = max(0.0, min(1.0, priority))
            return True
        return False

    def pin(self, key: str) -> bool:
        if key in self._store:
            self._store[key].pinned = True
            return True
        return False

    def unpin(self, key: str) -> bool:
        if key in self._store:
            self._store[key].pinned = False
            return True
        return False

    def clear(self, include_pinned: bool = False) -> int:
        if include_pinned:
            count = len(self._store)
            self._store.clear()
            self._tag_index.clear()
            return count
        unpinned = [k for k, v in self._store.items() if not v.pinned]
        for k in unpinned:
            self._remove(k)
        return len(unpinned)

    # ------------------------------------------------------------------
    # Tag-based retrieval
    # ------------------------------------------------------------------

    def get_by_tag(self, tag: str) -> List[MemoryItem]:
        keys = self._tag_index.get(tag, [])
        result = []
        for k in list(keys):
            item = self._store.get(k)
            if item and not item.is_expired():
                result.append(item)
        return result

    def keys_by_tag(self, tag: str) -> List[str]:
        return [item.key for item in self.get_by_tag(tag)]

    # ------------------------------------------------------------------
    # Iteration & snapshots
    # ------------------------------------------------------------------

    def items(self) -> Iterator[Tuple[str, Any]]:
        self._purge_expired()
        for key, item in self._store.items():
            yield key, item.value

    def snapshot(self) -> Dict[str, Any]:
        self._purge_expired()
        return {k: v.value for k, v in self._store.items()}

    def restore(self, snapshot: Dict[str, Any], priority: float = 0.5) -> None:
        for k, v in snapshot.items():
            self.set(k, v, priority=priority)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        evictable = {k: v for k, v in self._store.items() if not v.pinned}
        if not evictable:
            return
        victim = min(evictable, key=lambda k: evictable[k].eviction_score())
        self._remove(victim)
        self._eviction_count += 1
        logger.debug("Evicted key=%s", victim)

    def _purge_expired(self) -> int:
        expired = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired:
            self._remove(k)
        return len(expired)

    def _remove(self, key: str) -> None:
        item = self._store.pop(key, None)
        if item:
            for tag in item.tags:
                if tag in self._tag_index:
                    self._tag_index[tag] = [k for k in self._tag_index[tag] if k != key]
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        self._purge_expired()
        total = self._hit_count + self._miss_count
        return {"capacity": self.capacity, "used": len(self._store), "utilisation": round(len(self._store) / max(self.capacity, 1), 3), "hit_rate": round(self._hit_count / max(total, 1), 3), "hits": self._hit_count, "misses": self._miss_count, "evictions": self._eviction_count, "pinned": sum(1 for v in self._store.values() if v.pinned), "tags": list(self._tag_index.keys())}

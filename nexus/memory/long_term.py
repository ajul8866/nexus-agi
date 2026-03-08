"""
NEXUS-AGI Long-Term Memory
Persistent storage for knowledge and experiences
"""
from __future__ import annotations
import time
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: Any = None
    memory_type: str = "general"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source_agent: str = ""
    consolidated: bool = False


class LongTermMemory:
    """
    Persistent long-term memory with consolidation and retrieval.
    In production, backed by a database (SQLite/PostgreSQL).
    """

    CONSOLIDATION_THRESHOLD = 3  # access_count before consolidation

    def __init__(self, storage_path: Optional[str] = None):
        self._records: Dict[str, MemoryRecord] = {}
        self._type_index: Dict[str, List[str]] = {}
        self._tag_index: Dict[str, List[str]] = {}
        self.storage_path = storage_path

    def store(self, content: Any, memory_type: str = "general",
              importance: float = 0.5, tags: Optional[List[str]] = None,
              source_agent: str = "") -> MemoryRecord:
        record = MemoryRecord(
            content=content, memory_type=memory_type, importance=importance,
            tags=tags or [], source_agent=source_agent
        )
        self._records[record.id] = record
        self._type_index.setdefault(memory_type, []).append(record.id)
        for tag in (tags or []):
            self._tag_index.setdefault(tag.lower(), []).append(record.id)
        return record

    def retrieve(self, query: str, limit: int = 20,
                 memory_type: Optional[str] = None,
                 min_importance: float = 0.0) -> List[MemoryRecord]:
        candidates = list(self._records.values())
        if memory_type:
            ids = self._type_index.get(memory_type, [])
            candidates = [self._records[rid] for rid in ids if rid in self._records]
        candidates = [r for r in candidates if r.importance >= min_importance]
        query_lower = query.lower()
        scored = []
        for record in candidates:
            score = record.importance
            content_str = str(record.content).lower()
            if query_lower in content_str:
                score += 0.4
            tag_match = sum(1 for t in record.tags if query_lower in t.lower())
            score += tag_match * 0.1
            score += min(record.access_count * 0.02, 0.2)
            scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:limit]]
        for r in results:
            r.last_accessed = time.time()
            r.access_count += 1
        return results

    def get_by_id(self, record_id: str) -> Optional[MemoryRecord]:
        return self._records.get(record_id)

    def consolidate(self) -> int:
        """Mark frequently accessed memories as consolidated."""
        consolidated = 0
        for record in self._records.values():
            if record.access_count >= self.CONSOLIDATION_THRESHOLD and not record.consolidated:
                record.consolidated = True
                record.importance = min(1.0, record.importance + 0.1)
                consolidated += 1
        return consolidated

    def forget(self, record_id: str) -> bool:
        record = self._records.pop(record_id, None)
        if record:
            ids = self._type_index.get(record.memory_type, [])
            if record_id in ids:
                ids.remove(record_id)
            for tag in record.tags:
                tag_ids = self._tag_index.get(tag.lower(), [])
                if record_id in tag_ids:
                    tag_ids.remove(record_id)
            return True
        return False

    def prune_low_importance(self, threshold: float = 0.2) -> int:
        to_delete = [rid for rid, r in self._records.items() if r.importance < threshold and not r.consolidated]
        for rid in to_delete:
            self.forget(rid)
        return len(to_delete)

    def stats(self) -> Dict[str, Any]:
        records = list(self._records.values())
        return {
            "total_records": len(records),
            "memory_types": list(self._type_index.keys()),
            "consolidated": sum(1 for r in records if r.consolidated),
            "avg_importance": round(sum(r.importance for r in records) / len(records), 3) if records else 0,
            "total_tags": len(self._tag_index),
        }

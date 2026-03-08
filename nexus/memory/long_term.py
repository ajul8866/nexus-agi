"""LongTermMemory - persistent JSON-based storage with vector similarity search."""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nexus.memory.long_term")

EMBEDDING_DIM = 128


@dataclass
class LongTermRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str = ""
    value: Any = None
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None   # stored as list for JSON serialisation
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = 0
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "tags": self.tags,
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "importance": self.importance,
        }


def _make_embedding(text: str) -> List[float]:
    """Deterministic pseudo-embedding (demo). Replace with real model in production."""
    rng = np.random.default_rng(seed=abs(hash(str(text))) % (2**31))
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    vec = vec / norm if norm > 0 else vec
    return vec.tolist()


def _cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class LongTermMemory:
    """
    Persistent long-term memory backed by a JSON file.
    Supports:
    - Key/value storage with categories and tags
    - Vector similarity search (numpy)
    - Memory compression (deduplicate + merge)
    - Import/export
    """

    def __init__(self, storage_path: str = "data/long_term_memory.json", auto_save: bool = True):
        self.storage_path = storage_path
        self.auto_save = auto_save
        self._records: Dict[str, LongTermRecord] = {}     # record_id -> record
        self._key_index: Dict[str, str] = {}              # key -> record_id
        self._dirty = False

        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            logger.info("No existing LTM file at %s; starting fresh.", self.storage_path)
            return
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            for raw in data.get("records", []):
                rec = LongTermRecord(**raw)
                self._records[rec.record_id] = rec
                self._key_index[rec.key] = rec.record_id
            logger.info("Loaded %d LTM records from %s", len(self._records), self.storage_path)
        except Exception as exc:
            logger.error("Failed to load LTM: %s", exc)

    def save(self) -> None:
        try:
            data = {"records": [asdict(r) for r in self._records.values()]}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._dirty = False
            logger.debug("LTM saved: %d records → %s", len(self._records), self.storage_path)
        except Exception as exc:
            logger.error("Failed to save LTM: %s", exc)

    def _maybe_save(self) -> None:
        if self.auto_save and self._dirty:
            self.save()

    # ── CRUD ───────────────────────────────────────────────────────────────────
    def store(
        self,
        key: str,
        value: Any,
        category: str = "general",
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        embed: bool = True,
    ) -> LongTermRecord:
        now = datetime.utcnow().isoformat()

        if key in self._key_index:
            rec = self._records[self._key_index[key]]
            rec.value = value
            rec.category = category
            rec.tags = tags or rec.tags
            rec.importance = importance
            rec.updated_at = now
            if embed:
                rec.embedding = _make_embedding(f"{key} {str(value)}")
            logger.debug("Updated LTM key=%s", key)
        else:
            emb = _make_embedding(f"{key} {str(value)}") if embed else None
            rec = LongTermRecord(
                key=key,
                value=value,
                category=category,
                tags=tags or [],
                embedding=emb,
                importance=importance,
            )
            self._records[rec.record_id] = rec
            self._key_index[key] = rec.record_id
            logger.debug("Stored new LTM key=%s id=%s", key, rec.record_id)

        self._dirty = True
        self._maybe_save()
        return rec

    def retrieve(self, key: str) -> Optional[LongTermRecord]:
        rid = self._key_index.get(key)
        if rid and rid in self._records:
            rec = self._records[rid]
            rec.access_count += 1
            self._dirty = True
            self._maybe_save()
            return rec
        return None

    def delete(self, key: str) -> bool:
        rid = self._key_index.pop(key, None)
        if rid and rid in self._records:
            del self._records[rid]
            self._dirty = True
            self._maybe_save()
            return True
        return False

    def get_by_category(self, category: str) -> List[LongTermRecord]:
        return [r for r in self._records.values() if r.category == category]

    def get_by_tags(self, tags: List[str]) -> List[LongTermRecord]:
        tag_set = set(tags)
        return [r for r in self._records.values() if tag_set & set(r.tags)]

    # ── Semantic search ────────────────────────────────────────────────────────
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[LongTermRecord, float]]:
        q_emb = _make_embedding(query)
        scored = []
        for rec in self._records.values():
            if rec.embedding:
                sim = _cosine(q_emb, rec.embedding)
                scored.append((rec, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── Compression ─────────────────────────────────────────────────────────────
    def compress(self, similarity_threshold: float = 0.95) -> int:
        """Merge near-duplicate records (cosine similarity > threshold)."""
        records = list(self._records.values())
        merged_count = 0
        merged_ids: set = set()

        for i, rec_a in enumerate(records):
            if rec_a.record_id in merged_ids or rec_a.embedding is None:
                continue
            for rec_b in records[i+1:]:
                if rec_b.record_id in merged_ids or rec_b.embedding is None:
                    continue
                sim = _cosine(rec_a.embedding, rec_b.embedding)
                if sim >= similarity_threshold:
                    # Keep record with higher importance + access count
                    score_a = rec_a.importance + rec_a.access_count * 0.01
                    score_b = rec_b.importance + rec_b.access_count * 0.01
                    keep, discard = (rec_a, rec_b) if score_a >= score_b else (rec_b, rec_a)
                    # Merge tags
                    keep.tags = list(set(keep.tags + discard.tags))
                    merged_ids.add(discard.record_id)
                    merged_count += 1

        for rid in merged_ids:
            rec = self._records.pop(rid, None)
            if rec:
                self._key_index.pop(rec.key, None)

        if merged_count:
            self._dirty = True
            self._maybe_save()
        logger.info("Compression merged %d duplicate records", merged_count)
        return merged_count

    # ── Import / Export ────────────────────────────────────────────────────────
    def export_json(self, path: str) -> None:
        data = {"records": [asdict(r) for r in self._records.values()]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Exported %d records to %s", len(self._records), path)

    def import_json(self, path: str) -> int:
        with open(path, "r") as f:
            data = json.load(f)
        added = 0
        for raw in data.get("records", []):
            key = raw.get("key", "")
            if key and key not in self._key_index:
                rec = LongTermRecord(**raw)
                self._records[rec.record_id] = rec
                self._key_index[key] = rec.record_id
                added += 1
        self._dirty = True
        self._maybe_save()
        logger.info("Imported %d new records from %s", added, path)
        return added

    # ── Stats ──────────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        from collections import Counter
        cats = Counter(r.category for r in self._records.values())
        return {
            "total_records": len(self._records),
            "storage_path": self.storage_path,
            "categories": dict(cats),
            "avg_importance": round(
                sum(r.importance for r in self._records.values()) / max(len(self._records), 1), 3
            ),
            "dirty": self._dirty,
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"LongTermMemory(records={len(self._records)}, path={self.storage_path!r})"

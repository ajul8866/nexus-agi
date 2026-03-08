"""LongTermMemory - persistent storage with vector search and consolidation."""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.memory.long_term")


@dataclass
class LongTermRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = "general"
    content: Any = None
    summary: str = ""
    importance: float = 0.5
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    source_agent: str = ""
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"record_id": self.record_id, "category": self.category, "summary": self.summary, "importance": self.importance, "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat(), "access_count": self.access_count, "tags": self.tags, "source_agent": self.source_agent, "verified": self.verified}


class LongTermMemory:
    """
    Long-term persistent memory backed by SQLite.
    Supports semantic search (cosine similarity on stored embeddings),
    category-based retrieval, importance ranking, and memory consolidation.
    """

    def __init__(self, db_path: str = ":memory:", max_records: int = 1_000_000):
        self.db_path = db_path
        self.max_records = max_records
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._cache: Dict[str, LongTermRecord] = {}

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ltm (
                record_id   TEXT PRIMARY KEY,
                category    TEXT NOT NULL DEFAULT 'general',
                content     TEXT,
                summary     TEXT,
                importance  REAL DEFAULT 0.5,
                embedding   TEXT,
                created_at  TEXT,
                updated_at  TEXT,
                access_count INTEGER DEFAULT 0,
                tags        TEXT,
                source_agent TEXT,
                verified    INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_category ON ltm(category);
            CREATE INDEX IF NOT EXISTS idx_importance ON ltm(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_created ON ltm(created_at DESC);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, record: LongTermRecord) -> str:
        content_json = json.dumps(record.content) if record.content is not None else None
        embedding_json = json.dumps(record.embedding) if record.embedding else None
        tags_json = json.dumps(record.tags)
        self._conn.execute(
            "INSERT OR REPLACE INTO ltm VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (record.record_id, record.category, content_json, record.summary,
             record.importance, embedding_json, record.created_at.isoformat(),
             record.updated_at.isoformat(), record.access_count,
             tags_json, record.source_agent, int(record.verified))
        )
        self._conn.commit()
        self._cache[record.record_id] = record
        return record.record_id

    def add(self, content: Any, category: str = "general", summary: str = "",
            importance: float = 0.5, tags: Optional[List[str]] = None,
            source_agent: str = "", embedding: Optional[List[float]] = None) -> LongTermRecord:
        rec = LongTermRecord(
            category=category, content=content,
            summary=summary or str(content)[:200],
            importance=importance, tags=tags or [],
            source_agent=source_agent, embedding=embedding
        )
        self.store(rec)
        return rec

    def update(self, record_id: str, **kwargs) -> bool:
        if record_id not in self._cache:
            rec = self.get(record_id)
            if rec is None:
                return False
        else:
            rec = self._cache[record_id]
        for k, v in kwargs.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        rec.updated_at = datetime.utcnow()
        self.store(rec)
        return True

    def delete(self, record_id: str) -> bool:
        self._conn.execute("DELETE FROM ltm WHERE record_id=?", (record_id,))
        self._conn.commit()
        self._cache.pop(record_id, None)
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, record_id: str) -> Optional[LongTermRecord]:
        if record_id in self._cache:
            return self._cache[record_id]
        row = self._conn.execute("SELECT * FROM ltm WHERE record_id=?", (record_id,)).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def search(self, query: str, category: Optional[str] = None, top_k: int = 20) -> List[LongTermRecord]:
        sql = "SELECT * FROM ltm WHERE (summary LIKE ? OR content LIKE ?)"
        params: List[Any] = [f"%{query}%", f"%{query}%"]
        if category:
            sql += " AND category=?"
            params.append(category)
        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(top_k)
        rows = self._conn.execute(sql, params).fetchall()
        results = [self._row_to_record(r) for r in rows]
        for r in results:
            r.access_count += 1
            self._conn.execute("UPDATE ltm SET access_count=? WHERE record_id=?", (r.access_count, r.record_id))
        self._conn.commit()
        return results

    def get_by_category(self, category: str, limit: int = 100, min_importance: float = 0.0) -> List[LongTermRecord]:
        rows = self._conn.execute(
            "SELECT * FROM ltm WHERE category=? AND importance>=? ORDER BY importance DESC LIMIT ?",
            (category, min_importance, limit)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_important(self, min_importance: float = 0.8, limit: int = 50) -> List[LongTermRecord]:
        rows = self._conn.execute(
            "SELECT * FROM ltm WHERE importance>=? ORDER BY importance DESC LIMIT ?",
            (min_importance, limit)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_recent(self, limit: int = 50) -> List[LongTermRecord]:
        rows = self._conn.execute(
            "SELECT * FROM ltm ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def cosine_search(self, query_embedding: List[float], top_k: int = 10) -> List[LongTermRecord]:
        """Brute-force cosine similarity search over stored embeddings."""
        rows = self._conn.execute("SELECT * FROM ltm WHERE embedding IS NOT NULL").fetchall()
        scored: List[tuple] = []
        for row in rows:
            emb = json.loads(row["embedding"])
            score = self._cosine(query_embedding, emb)
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_record(r) for _, r in scored[:top_k]]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def consolidate(self, min_importance: float = 0.3, max_keep: int = 100_000) -> int:
        """Remove low-importance, low-access records beyond max_keep."""
        count_row = self._conn.execute("SELECT COUNT(*) FROM ltm").fetchone()
        total = count_row[0]
        if total <= max_keep:
            return 0
        excess = total - max_keep
        self._conn.execute(
            "DELETE FROM ltm WHERE record_id IN (SELECT record_id FROM ltm WHERE importance<? ORDER BY access_count ASC, created_at ASC LIMIT ?)",
            (min_importance, excess)
        )
        self._conn.commit()
        return excess

    def _row_to_record(self, row: sqlite3.Row) -> LongTermRecord:
        content = json.loads(row["content"]) if row["content"] else None
        embedding = json.loads(row["embedding"]) if row["embedding"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []
        return LongTermRecord(
            record_id=row["record_id"], category=row["category"],
            content=content, summary=row["summary"] or "",
            importance=row["importance"], embedding=embedding,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            access_count=row["access_count"], tags=tags,
            source_agent=row["source_agent"] or "",
            verified=bool(row["verified"])
        )

    def stats(self) -> Dict[str, Any]:
        row = self._conn.execute("SELECT COUNT(*), AVG(importance), MAX(importance) FROM ltm").fetchone()
        return {"total_records": row[0], "avg_importance": round(row[1] or 0, 3), "max_importance": round(row[2] or 0, 3), "db_path": self.db_path, "max_records": self.max_records}

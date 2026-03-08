"""EpisodicMemory - stores and retrieves experience episodes with forgetting curve."""

import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.memory.episodic")


@dataclass
class Episode:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recall_count: int = 0
    last_recalled: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    def retention(self, stability: float = 24.0) -> float:
        hours_elapsed = (datetime.utcnow() - self.timestamp).total_seconds() / 3600
        raw = math.exp(-hours_elapsed / max(stability, 0.1))
        boost = min(self.recall_count * 0.05, 0.5)
        return min(1.0, raw + boost)

    def relevance_score(self, query_tags: List[str], stability: float = 24.0) -> float:
        if not query_tags:
            return self.importance * self.retention(stability)
        overlap = len(set(self.tags) & set(query_tags)) / max(len(query_tags), 1)
        return 0.4 * overlap + 0.3 * self.importance + 0.3 * self.retention(stability)

    def to_dict(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "event": self.event, "context": self.context, "outcome": self.outcome, "importance": self.importance, "timestamp": self.timestamp.isoformat(), "recall_count": self.recall_count, "tags": self.tags, "retention": round(self.retention(), 3)}


class EpisodicMemory:
    """Episodic memory with in-memory storage, optional SQLite persistence, Ebbinghaus forgetting curve, and memory consolidation."""

    def __init__(self, capacity: int = 10000, db_path: Optional[str] = None, forgetting_stability: float = 24.0):
        self.capacity = capacity
        self.forgetting_stability = forgetting_stability
        self._episodes: Dict[str, Episode] = {}
        self._db: Optional[sqlite3.Connection] = None
        if db_path:
            self._init_db(db_path)

    def _init_db(self, db_path: str) -> None:
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("""CREATE TABLE IF NOT EXISTS episodes (episode_id TEXT PRIMARY KEY, event TEXT, context TEXT, outcome TEXT, importance REAL, timestamp TEXT, recall_count INTEGER, tags TEXT)""")
        self._db.commit()
        self._load_from_db()

    def _load_from_db(self) -> None:
        if not self._db:
            return
        cur = self._db.execute("SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (self.capacity,))
        for row in cur.fetchall():
            ep = Episode(episode_id=row[0], event=row[1], context=json.loads(row[2] or "{}"), outcome=row[3], importance=row[4], timestamp=datetime.fromisoformat(row[5]), recall_count=row[6], tags=json.loads(row[7] or "[]"))
            self._episodes[ep.episode_id] = ep

    def _persist_episode(self, ep: Episode) -> None:
        if not self._db:
            return
        self._db.execute("INSERT OR REPLACE INTO episodes (episode_id, event, context, outcome, importance, timestamp, recall_count, tags) VALUES (?,?,?,?,?,?,?,?)", (ep.episode_id, ep.event, json.dumps(ep.context), ep.outcome, ep.importance, ep.timestamp.isoformat(), ep.recall_count, json.dumps(ep.tags)))
        self._db.commit()

    def store(self, episode: Episode) -> str:
        if len(self._episodes) >= self.capacity:
            self._evict_oldest()
        self._episodes[episode.episode_id] = episode
        self._persist_episode(episode)
        return episode.episode_id

    def add(self, event: str, context: Optional[Dict[str, Any]] = None, outcome: Optional[str] = None, importance: float = 0.5, tags: Optional[List[str]] = None) -> Episode:
        ep = Episode(event=event, context=context or {}, outcome=outcome, importance=importance, tags=tags or [])
        self.store(ep)
        return ep

    def get(self, episode_id: str) -> Optional[Episode]:
        ep = self._episodes.get(episode_id)
        if ep:
            ep.recall_count += 1
            ep.last_recalled = datetime.utcnow()
        return ep

    def delete(self, episode_id: str) -> bool:
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            if self._db:
                self._db.execute("DELETE FROM episodes WHERE episode_id=?", (episode_id,))
                self._db.commit()
            return True
        return False

    def retrieve_by_tags(self, tags: List[str], top_k: int = 10) -> List[Episode]:
        scored = [(ep, ep.relevance_score(tags, self.forgetting_stability)) for ep in self._episodes.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [ep for ep, _ in scored[:top_k]]
        for ep in results:
            ep.recall_count += 1
        return results

    def retrieve_recent(self, hours: float = 24.0, limit: int = 50) -> List[Episode]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [ep for ep in self._episodes.values() if ep.timestamp >= cutoff]
        recent.sort(key=lambda e: e.timestamp, reverse=True)
        return recent[:limit]

    def retrieve_by_importance(self, min_importance: float = 0.7, limit: int = 20) -> List[Episode]:
        important = [ep for ep in self._episodes.values() if ep.importance >= min_importance]
        important.sort(key=lambda e: e.importance, reverse=True)
        return important[:limit]

    def search(self, query: str, top_k: int = 10) -> List[Episode]:
        q = query.lower()
        matched = [ep for ep in self._episodes.values() if q in ep.event.lower()]
        matched.sort(key=lambda e: e.importance * e.retention(self.forgetting_stability), reverse=True)
        return matched[:top_k]

    def consolidate(self, retention_threshold: float = 0.05) -> int:
        to_delete = [eid for eid, ep in self._episodes.items() if ep.retention(self.forgetting_stability) < retention_threshold and ep.importance < 0.8]
        for eid in to_delete:
            self.delete(eid)
        return len(to_delete)

    def _evict_oldest(self) -> None:
        if not self._episodes:
            return
        oldest_id = min(self._episodes, key=lambda k: self._episodes[k].timestamp)
        self.delete(oldest_id)

    def stats(self) -> Dict[str, Any]:
        episodes = list(self._episodes.values())
        avg_retention = sum(e.retention(self.forgetting_stability) for e in episodes) / max(len(episodes), 1)
        return {"total_episodes": len(episodes), "capacity": self.capacity, "avg_retention": round(avg_retention, 3), "avg_importance": round(sum(e.importance for e in episodes) / max(len(episodes), 1), 3), "has_db": self._db is not None}

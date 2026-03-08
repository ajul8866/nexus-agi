"""
NEXUS-AGI Episodic Memory
Event-based memory storage with temporal indexing
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque


@dataclass
class Episode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    event_type: str = "task"
    content: Any = None
    agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    Stores sequences of events/experiences with temporal ordering.
    Supports importance-based retention and keyword search.
    """

    def __init__(self, max_episodes: int = 10000, importance_threshold: float = 0.3):
        self._episodes: deque = deque(maxlen=max_episodes)
        self._index: Dict[str, Episode] = {}
        self.importance_threshold = importance_threshold
        self.max_episodes = max_episodes

    def store(self, event_type: str, content: Any, agent_id: str = "",
              importance: float = 0.5, tags: Optional[List[str]] = None,
              metadata: Optional[Dict] = None) -> Episode:
        episode = Episode(
            event_type=event_type, content=content, agent_id=agent_id,
            importance=importance, tags=tags or [], metadata=metadata or {}
        )
        if importance >= self.importance_threshold:
            self._episodes.append(episode)
            self._index[episode.id] = episode
        return episode

    def retrieve(self, query: str, limit: int = 10,
                 event_type: Optional[str] = None,
                 since: Optional[float] = None) -> List[Episode]:
        results = list(self._episodes)
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if since:
            results = [e for e in results if e.timestamp >= since]
        query_lower = query.lower()
        scored = []
        for ep in results:
            score = ep.importance
            content_str = str(ep.content).lower()
            if query_lower in content_str:
                score += 0.3
            if any(query_lower in tag.lower() for tag in ep.tags):
                score += 0.2
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def get_by_id(self, episode_id: str) -> Optional[Episode]:
        return self._index.get(episode_id)

    def get_recent(self, n: int = 20) -> List[Episode]:
        episodes = list(self._episodes)
        return sorted(episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def get_by_agent(self, agent_id: str, limit: int = 50) -> List[Episode]:
        return [e for e in self._episodes if e.agent_id == agent_id][-limit:]

    def forget_old(self, keep_important: float = 0.8) -> int:
        before = len(self._episodes)
        important = [e for e in self._episodes if e.importance >= keep_important]
        recent_cutoff = time.time() - 3600
        recent = [e for e in self._episodes if e.timestamp >= recent_cutoff]
        kept = list(set(important + recent))
        self._episodes = deque(kept, maxlen=self.max_episodes)
        self._index = {e.id: e for e in self._episodes}
        return before - len(self._episodes)

    def stats(self) -> Dict[str, Any]:
        episodes = list(self._episodes)
        return {
            "total_episodes": len(episodes),
            "max_capacity": self.max_episodes,
            "utilization": round(len(episodes) / self.max_episodes, 3) if self.max_episodes else 0,
            "avg_importance": round(sum(e.importance for e in episodes) / len(episodes), 3) if episodes else 0,
            "event_types": list(set(e.event_type for e in episodes)),
        }

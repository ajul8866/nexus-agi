"""
NEXUS-AGI World Model
Internal state representation dengan entity-relation graph
"""
from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


@dataclass
class Entity:
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update(self, properties: Dict[str, Any]):
        self.properties.update(properties)
        self.updated_at = time.time()


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class StateSnapshot:
    timestamp: float
    entities: Dict[str, Any]
    relations: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorldModel:
    """Internal world state representation for NEXUS-AGI."""

    def __init__(self, max_history: int = 1000):
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._history: deque = deque(maxlen=max_history)
        self._lock = asyncio.Lock()
        self._version = 0

    async def update_state(self, observation: Dict[str, Any]) -> None:
        async with self._lock:
            self._snapshot()
            obs_type = observation.get("type", "generic")
            if obs_type == "entity":
                eid = observation.get("id", f"ent_{self._version}")
                if eid in self._entities:
                    self._entities[eid].update(observation.get("properties", {}))
                else:
                    self._entities[eid] = Entity(
                        id=eid,
                        type=observation.get("entity_type", "unknown"),
                        properties=observation.get("properties", {})
                    )
            elif obs_type == "relation":
                rel = Relation(
                    source_id=observation["source"],
                    target_id=observation["target"],
                    relation_type=observation.get("relation_type", "related_to"),
                    strength=observation.get("strength", 1.0),
                    properties=observation.get("properties", {})
                )
                self._relations.append(rel)
            elif obs_type == "fact":
                key = observation.get("key", f"fact_{self._version}")
                self._entities[key] = Entity(
                    id=key,
                    type="fact",
                    properties={"value": observation.get("value"), "source": observation.get("source", "unknown")}
                )
            self._version += 1

    def query_state(self, query: Dict[str, Any]) -> List[Any]:
        results = []
        entity_type = query.get("entity_type")
        properties = query.get("properties", {})
        relation_type = query.get("relation_type")

        if relation_type:
            for rel in self._relations:
                if rel.relation_type == relation_type:
                    results.append(asdict(rel))
        else:
            for entity in self._entities.values():
                if entity_type and entity.type != entity_type:
                    continue
                match = all(
                    entity.properties.get(k) == v
                    for k, v in properties.items()
                )
                if match:
                    results.append(asdict(entity))
        return results

    def get_context(self, task: str) -> Dict[str, Any]:
        keywords = set(task.lower().split())
        relevant_entities = []
        for entity in self._entities.values():
            entity_text = f"{entity.type} {' '.join(str(v) for v in entity.properties.values())}".lower()
            if any(kw in entity_text for kw in keywords):
                relevant_entities.append(asdict(entity))
        return {
            "task": task,
            "relevant_entities": relevant_entities[:20],
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "world_version": self._version
        }

    def _snapshot(self):
        self._history.append(StateSnapshot(
            timestamp=time.time(),
            entities={k: asdict(v) for k, v in self._entities.items()},
            relations=[asdict(r) for r in self._relations],
            metadata={"version": self._version}
        ))

    def serialize(self) -> str:
        return json.dumps({
            "entities": {k: asdict(v) for k, v in self._entities.items()},
            "relations": [asdict(r) for r in self._relations],
            "version": self._version
        })

    @classmethod
    def deserialize(cls, data: str) -> "WorldModel":
        obj = json.loads(data)
        wm = cls()
        wm._version = obj.get("version", 0)
        for k, v in obj.get("entities", {}).items():
            wm._entities[k] = Entity(**v)
        for r in obj.get("relations", []):
            wm._relations.append(Relation(**r))
        return wm

    def stats(self) -> Dict[str, Any]:
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "history_size": len(self._history),
            "version": self._version
        }

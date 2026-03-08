"""SemanticMemory - concept graph, embeddings, and knowledge relationships."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.memory.semantic")


@dataclass
class Concept:
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    domain: str = "general"
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    source: str = "agent"
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"concept_id": self.concept_id, "name": self.name, "description": self.description, "domain": self.domain, "attributes": self.attributes, "confidence": self.confidence, "source": self.source, "access_count": self.access_count, "created_at": self.created_at.isoformat()}


@dataclass
class Relation:
    relation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: str = "related_to"
    weight: float = 1.0
    bidirectional: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class SemanticMemory:
    """Semantic memory as a concept graph. Supports text similarity search, relation traversal, and concept clustering."""

    RELATION_TYPES = ["is_a", "has_a", "part_of", "related_to", "causes", "enables", "contradicts", "synonym_of"]

    def __init__(self, max_concepts: int = 50000):
        self.max_concepts = max_concepts
        self._concepts: Dict[str, Concept] = {}
        self._name_index: Dict[str, str] = {}  # name.lower() -> concept_id
        self._domain_index: Dict[str, List[str]] = {}
        self._relations: Dict[str, Relation] = {}
        self._adjacency: Dict[str, List[str]] = {}  # concept_id -> [related concept_ids]

    def add_concept(self, concept: Concept) -> str:
        if len(self._concepts) >= self.max_concepts:
            self._evict_least_used()
        self._concepts[concept.concept_id] = concept
        self._name_index[concept.name.lower()] = concept.concept_id
        self._domain_index.setdefault(concept.domain, []).append(concept.concept_id)
        return concept.concept_id

    def create_concept(self, name: str, description: str = "", domain: str = "general", attributes: Optional[Dict[str, Any]] = None, confidence: float = 1.0) -> Concept:
        existing_id = self._name_index.get(name.lower())
        if existing_id:
            return self._concepts[existing_id]
        c = Concept(name=name, description=description, domain=domain, attributes=attributes or {}, confidence=confidence)
        self.add_concept(c)
        return c

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        c = self._concepts.get(concept_id)
        if c:
            c.access_count += 1
        return c

    def find_by_name(self, name: str) -> Optional[Concept]:
        cid = self._name_index.get(name.lower())
        return self.get_concept(cid) if cid else None

    def search(self, query: str, top_k: int = 10, domain: Optional[str] = None) -> List[Concept]:
        q = query.lower()
        candidates = self._concepts.values()
        if domain:
            domain_ids = set(self._domain_index.get(domain, []))
            candidates = [c for c in candidates if c.concept_id in domain_ids]
        else:
            candidates = list(candidates)
        scored: List[Tuple[Concept, float]] = []
        for c in candidates:
            score = 0.0
            if q == c.name.lower():
                score = 1.0
            elif q in c.name.lower():
                score = 0.8
            elif q in c.description.lower():
                score = 0.5
            elif any(q in str(v).lower() for v in c.attributes.values()):
                score = 0.3
            if score > 0:
                scored.append((c, score * c.confidence))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    def add_relation(self, source_id: str, target_id: str, relation_type: str = "related_to", weight: float = 1.0, bidirectional: bool = False) -> Relation:
        if source_id not in self._concepts or target_id not in self._concepts:
            raise ValueError("Both concepts must exist before adding a relation")
        rel = Relation(source_id=source_id, target_id=target_id, relation_type=relation_type, weight=weight, bidirectional=bidirectional)
        self._relations[rel.relation_id] = rel
        self._adjacency.setdefault(source_id, []).append(target_id)
        if bidirectional:
            self._adjacency.setdefault(target_id, []).append(source_id)
        return rel

    def get_relations(self, concept_id: str, relation_type: Optional[str] = None) -> List[Relation]:
        rels = [r for r in self._relations.values() if r.source_id == concept_id or (r.bidirectional and r.target_id == concept_id)]
        if relation_type:
            rels = [r for r in rels if r.relation_type == relation_type]
        return rels

    def traverse(self, start_id: str, max_hops: int = 3) -> Dict[str, List[str]]:
        visited: Dict[str, List[str]] = {start_id: []}
        frontier = [start_id]
        for _ in range(max_hops):
            next_frontier = []
            for cid in frontier:
                for neighbor in self._adjacency.get(cid, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[cid] + [cid]
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def update_concept(self, concept_id: str, **kwargs) -> bool:
        if concept_id not in self._concepts:
            return False
        c = self._concepts[concept_id]
        for k, v in kwargs.items():
            if hasattr(c, k):
                setattr(c, k, v)
        c.updated_at = datetime.utcnow()
        return True

    def delete_concept(self, concept_id: str) -> bool:
        if concept_id not in self._concepts:
            return False
        c = self._concepts.pop(concept_id)
        self._name_index.pop(c.name.lower(), None)
        if c.domain in self._domain_index:
            self._domain_index[c.domain] = [i for i in self._domain_index[c.domain] if i != concept_id]
        self._relations = {rid: r for rid, r in self._relations.items() if r.source_id != concept_id and r.target_id != concept_id}
        self._adjacency.pop(concept_id, None)
        return True

    def _evict_least_used(self) -> None:
        if not self._concepts:
            return
        least = min(self._concepts, key=lambda k: self._concepts[k].access_count)
        self.delete_concept(least)

    def get_by_domain(self, domain: str) -> List[Concept]:
        ids = self._domain_index.get(domain, [])
        return [self._concepts[i] for i in ids if i in self._concepts]

    def stats(self) -> Dict[str, Any]:
        return {"total_concepts": len(self._concepts), "total_relations": len(self._relations), "domains": list(self._domain_index.keys()), "max_concepts": self.max_concepts}

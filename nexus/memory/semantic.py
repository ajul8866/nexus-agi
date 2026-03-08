"""SemanticMemory - knowledge graph with cosine similarity search."""

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nexus.memory.semantic")


@dataclass
class Concept:
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class Edge:
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation: str = "related_to"
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class SemanticMemory:
    """
    Knowledge graph with:
    - Concept nodes with optional embeddings
    - Typed, weighted edges
    - Cosine similarity search (numpy)
    - Knowledge merging and update
    """

    EMBEDDING_DIM = 128

    def __init__(self):
        self._concepts: Dict[str, Concept] = {}      # id -> Concept
        self._name_index: Dict[str, str] = {}         # name -> id
        self._edges: Dict[str, Edge] = {}             # edge_id -> Edge
        self._adjacency: Dict[str, List[str]] = {}    # concept_id -> [edge_ids]

    # ── Random embedding placeholder ─────────────────────────────────────────────
    @classmethod
    def _make_embedding(cls, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding from text hash (demo only)."""
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        vec = rng.standard_normal(cls.EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ── Concepts ───────────────────────────────────────────────────────────────────
    def add_concept(
        self,
        name: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Concept:
        if name in self._name_index:
            existing = self._concepts[self._name_index[name]]
            existing.frequency += 1
            if description:
                existing.description = description
            if properties:
                existing.properties.update(properties)
            logger.debug("Updated existing concept: %s", name)
            return existing

        emb = embedding if embedding is not None else self._make_embedding(name + " " + description)
        concept = Concept(
            name=name,
            description=description,
            embedding=emb,
            properties=properties or {},
        )
        self._concepts[concept.concept_id] = concept
        self._name_index[name] = concept.concept_id
        self._adjacency[concept.concept_id] = []
        logger.debug("Added concept: %s id=%s", name, concept.concept_id)
        return concept

    def get_concept(self, name_or_id: str) -> Optional[Concept]:
        if name_or_id in self._concepts:
            return self._concepts[name_or_id]
        cid = self._name_index.get(name_or_id)
        return self._concepts.get(cid) if cid else None

    def update_concept(self, concept_id: str, **kwargs) -> bool:
        if concept_id not in self._concepts:
            return False
        c = self._concepts[concept_id]
        for k, v in kwargs.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return True

    def remove_concept(self, concept_id: str) -> bool:
        if concept_id not in self._concepts:
            return False
        name = self._concepts[concept_id].name
        del self._concepts[concept_id]
        self._name_index.pop(name, None)
        self._adjacency.pop(concept_id, None)
        # Remove related edges
        to_remove = [eid for eid, e in self._edges.items()
                     if e.source_id == concept_id or e.target_id == concept_id]
        for eid in to_remove:
            del self._edges[eid]
        return True

    # ── Edges ──────────────────────────────────────────────────────────────────
    def add_relation(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[Edge]:
        src = self.get_concept(source)
        tgt = self.get_concept(target)
        if not src or not tgt:
            logger.warning("Cannot add relation: concept not found (%s -> %s)", source, target)
            return None
        edge = Edge(
            source_id=src.concept_id,
            target_id=tgt.concept_id,
            relation=relation,
            weight=weight,
            properties=properties or {},
        )
        self._edges[edge.edge_id] = edge
        self._adjacency.setdefault(src.concept_id, []).append(edge.edge_id)
        return edge

    def get_neighbours(self, name_or_id: str, relation: Optional[str] = None) -> List[Concept]:
        concept = self.get_concept(name_or_id)
        if not concept:
            return []
        edge_ids = self._adjacency.get(concept.concept_id, [])
        neighbours = []
        for eid in edge_ids:
            edge = self._edges.get(eid)
            if edge and (relation is None or edge.relation == relation):
                neighbour = self._concepts.get(edge.target_id)
                if neighbour:
                    neighbours.append(neighbour)
        return neighbours

    # ── Similarity search ────────────────────────────────────────────────────────
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[Concept, float]]:
        query_emb = self._make_embedding(query)
        scored = []
        for concept in self._concepts.values():
            if concept.embedding is not None:
                sim = self._cosine(query_emb, concept.embedding)
                scored.append((concept, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def find_by_name(self, pattern: str) -> List[Concept]:
        p = pattern.lower()
        return [c for c in self._concepts.values() if p in c.name.lower()]

    # ── Merge ──────────────────────────────────────────────────────────────────
    def merge(self, other: "SemanticMemory") -> int:
        """Merge another SemanticMemory into this one. Returns # concepts added."""
        added = 0
        for concept in other._concepts.values():
            if concept.name not in self._name_index:
                self.add_concept(
                    name=concept.name,
                    description=concept.description,
                    properties=concept.properties,
                    embedding=concept.embedding,
                )
                added += 1
        for edge in other._edges.values():
            src = other._concepts.get(edge.source_id)
            tgt = other._concepts.get(edge.target_id)
            if src and tgt:
                self.add_relation(src.name, tgt.name, edge.relation, edge.weight)
        logger.info("Merged %d new concepts from external memory", added)
        return added

    # ── Stats ──────────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        return {
            "concepts": len(self._concepts),
            "edges": len(self._edges),
            "most_frequent": sorted(
                [(c.name, c.frequency) for c in self._concepts.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }

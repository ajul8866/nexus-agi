"""
NEXUS-AGI Semantic Memory
Concept and knowledge storage with vector similarity search
"""
from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Concept:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    category: str = "general"
    relations: List[str] = field(default_factory=list)  # IDs of related concepts
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    source: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0


class SemanticMemory:
    """
    Stores concepts, facts, and knowledge with semantic search.
    Uses simple TF-IDF style scoring when embeddings unavailable.
    """

    def __init__(self, max_concepts: int = 50000):
        self._concepts: Dict[str, Concept] = {}
        self._name_index: Dict[str, str] = {}  # name -> id
        self._category_index: Dict[str, List[str]] = {}  # category -> [ids]
        self.max_concepts = max_concepts

    def store_concept(self, name: str, description: str, category: str = "general",
                      relations: Optional[List[str]] = None, confidence: float = 1.0,
                      source: str = "", embedding: Optional[List[float]] = None) -> Concept:
        # Update if exists
        if name.lower() in self._name_index:
            existing_id = self._name_index[name.lower()]
            concept = self._concepts[existing_id]
            concept.description = description
            concept.confidence = confidence
            concept.updated_at = time.time()
            if embedding:
                concept.embedding = embedding
            return concept

        concept = Concept(
            name=name, description=description, category=category,
            relations=relations or [], confidence=confidence,
            source=source, embedding=embedding
        )
        self._concepts[concept.id] = concept
        self._name_index[name.lower()] = concept.id
        self._category_index.setdefault(category, []).append(concept.id)
        return concept

    def retrieve(self, query: str, limit: int = 10,
                 category: Optional[str] = None) -> List[Tuple[float, Concept]]:
        query_words = set(query.lower().split())
        results = []

        candidates = self._concepts.values()
        if category:
            cat_ids = self._category_index.get(category, [])
            candidates = [self._concepts[cid] for cid in cat_ids if cid in self._concepts]

        for concept in candidates:
            score = self._score_concept(concept, query_words, query)
            if score > 0:
                results.append((score, concept))

        results.sort(key=lambda x: x[0], reverse=True)
        # Update access counts
        for _, concept in results[:limit]:
            concept.access_count += 1
        return results[:limit]

    def _score_concept(self, concept: Concept, query_words: set, query: str) -> float:
        score = 0.0
        name_words = set(concept.name.lower().split())
        desc_words = set(concept.description.lower().split())

        name_overlap = len(query_words & name_words) / max(len(query_words), 1)
        desc_overlap = len(query_words & desc_words) / max(len(query_words), 1)

        score += name_overlap * 0.6 + desc_overlap * 0.3

        if query.lower() in concept.name.lower():
            score += 0.4
        if query.lower() in concept.description.lower():
            score += 0.2

        score *= concept.confidence
        return round(score, 4)

    def get_by_name(self, name: str) -> Optional[Concept]:
        cid = self._name_index.get(name.lower())
        if cid:
            concept = self._concepts.get(cid)
            if concept:
                concept.access_count += 1
            return concept
        return None

    def get_related(self, concept_id: str, depth: int = 1) -> List[Concept]:
        concept = self._concepts.get(concept_id)
        if not concept:
            return []
        related = []
        for rid in concept.relations:
            rc = self._concepts.get(rid)
            if rc:
                related.append(rc)
        return related

    def get_by_category(self, category: str, limit: int = 50) -> List[Concept]:
        ids = self._category_index.get(category, [])
        return [self._concepts[cid] for cid in ids[:limit] if cid in self._concepts]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_concepts": len(self._concepts),
            "categories": list(self._category_index.keys()),
            "category_counts": {k: len(v) for k, v in self._category_index.items()},
            "with_embeddings": sum(1 for c in self._concepts.values() if c.embedding),
            "max_capacity": self.max_concepts,
        }

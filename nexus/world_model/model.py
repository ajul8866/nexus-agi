"""WorldModel - internal model of environment state and dynamics."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.world_model.model")


@dataclass
class StateVariable:
    name: str
    value: Any
    uncertainty: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    source: str = "observation"
    history: List[Tuple[datetime, Any]] = field(default_factory=list)

    def update(self, new_value: Any, source: str = "observation", uncertainty: float = 0.0) -> None:
        self.history.append((self.last_updated, self.value))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        self.value = new_value
        self.uncertainty = uncertainty
        self.source = source
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "uncertainty": self.uncertainty, "last_updated": self.last_updated.isoformat(), "source": self.source}


@dataclass
class WorldSnapshot:
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    state: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    confidence: float = 1.0


class WorldModel:
    """
    Internal world model: maintains state variables, tracks history,
    supports counterfactual queries, and provides state snapshots.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or str(uuid.uuid4())
        self._state: Dict[str, StateVariable] = {}
        self._snapshots: List[WorldSnapshot] = []
        self._transition_rules: Dict[str, Any] = {}
        self._created_at = datetime.utcnow()
        self._update_count = 0
        logger.info("WorldModel created: %s", self.model_id)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def observe(self, key: str, value: Any, uncertainty: float = 0.0, source: str = "observation") -> StateVariable:
        if key in self._state:
            self._state[key].update(value, source, uncertainty)
        else:
            self._state[key] = StateVariable(name=key, value=value, uncertainty=uncertainty, source=source)
        self._update_count += 1
        return self._state[key]

    def observe_batch(self, observations: Dict[str, Any], source: str = "batch") -> None:
        for k, v in observations.items():
            self.observe(k, v, source=source)

    def get(self, key: str, default: Any = None) -> Any:
        var = self._state.get(key)
        return var.value if var is not None else default

    def get_variable(self, key: str) -> Optional[StateVariable]:
        return self._state.get(key)

    def get_with_uncertainty(self, key: str) -> Tuple[Any, float]:
        var = self._state.get(key)
        if var is None:
            return None, 1.0
        return var.value, var.uncertainty

    def delete(self, key: str) -> bool:
        if key in self._state:
            del self._state[key]
            return True
        return False

    def keys(self) -> List[str]:
        return list(self._state.keys())

    def items(self) -> Dict[str, Any]:
        return {k: v.value for k, v in self._state.items()}

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(self, description: str = "") -> WorldSnapshot:
        snap = WorldSnapshot(
            state={k: v.to_dict() for k, v in self._state.items()},
            description=description,
            confidence=self._compute_confidence()
        )
        self._snapshots.append(snap)
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]
        logger.debug("Snapshot taken: %s", snap.snapshot_id)
        return snap

    def restore(self, snapshot: WorldSnapshot) -> None:
        self._state.clear()
        for k, v_dict in snapshot.state.items():
            self._state[k] = StateVariable(
                name=k, value=v_dict["value"],
                uncertainty=v_dict.get("uncertainty", 0.0),
                source=v_dict.get("source", "restored"),
                last_updated=datetime.fromisoformat(v_dict["last_updated"])
            )
        logger.info("World state restored from snapshot %s", snapshot.snapshot_id)

    def get_latest_snapshot(self) -> Optional[WorldSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    # ------------------------------------------------------------------
    # Transition rules
    # ------------------------------------------------------------------

    def add_transition_rule(self, name: str, condition: Any, effect: Any) -> None:
        self._transition_rules[name] = {"condition": condition, "effect": effect}

    def apply_transitions(self) -> List[str]:
        applied: List[str] = []
        for name, rule in self._transition_rules.items():
            try:
                condition_fn = rule["condition"]
                if callable(condition_fn) and condition_fn(self):
                    effect_fn = rule["effect"]
                    if callable(effect_fn):
                        effect_fn(self)
                    applied.append(name)
            except Exception as exc:
                logger.warning("Transition rule %s failed: %s", name, exc)
        return applied

    # ------------------------------------------------------------------
    # Counterfactual / simulation
    # ------------------------------------------------------------------

    def simulate(self, hypothetical_observations: Dict[str, Any]) -> "WorldModel":
        """Create a hypothetical copy of the world with different observations."""
        sim = WorldModel(model_id=f"{self.model_id}_sim")
        for k, var in self._state.items():
            sim.observe(k, var.value, var.uncertainty, "copy")
        for k, v in hypothetical_observations.items():
            sim.observe(k, v, source="hypothetical")
        return sim

    def diff(self, other: "WorldModel") -> Dict[str, Any]:
        """Return differences between this world model and another."""
        diff: Dict[str, Any] = {}
        all_keys = set(self._state) | set(other._state)
        for key in all_keys:
            v1 = self.get(key)
            v2 = other.get(key)
            if v1 != v2:
                diff[key] = {"self": v1, "other": v2}
        return diff

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_confidence(self) -> float:
        if not self._state:
            return 1.0
        avg_uncertainty = sum(v.uncertainty for v in self._state.values()) / len(self._state)
        return round(max(0.0, 1.0 - avg_uncertainty), 3)

    def stats(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "state_variables": len(self._state), "snapshots": len(self._snapshots), "transition_rules": len(self._transition_rules), "update_count": self._update_count, "confidence": self._compute_confidence(), "created_at": self._created_at.isoformat()}

"""
NEXUS-AGI Prediction Engine
Prediksi state masa depan berdasarkan action sequences
"""
from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class Prediction:
    predicted_state: Dict[str, Any]
    confidence: float
    horizon: int
    action_sequence: List[str]
    timestamp: float = field(default_factory=time.time)
    prediction_id: str = field(default_factory=lambda: f"pred_{int(time.time()*1000)}")


@dataclass
class PredictionEvaluation:
    prediction_id: str
    accuracy: float
    key_matches: int
    key_total: int
    timestamp: float = field(default_factory=time.time)


class PredictionEngine:
    """State prediction dengan ensemble strategies."""

    def __init__(self):
        self._predictions: Dict[str, Prediction] = {}
        self._evaluations: List[PredictionEvaluation] = []
        self._action_effects: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._accuracy_history: List[float] = []

    def record_action_effect(self, action: str, before_state: Dict[str, Any],
                              after_state: Dict[str, Any]) -> None:
        """Learn from observed action -> state transitions."""
        diff = {}
        for k in set(list(before_state.keys()) + list(after_state.keys())):
            if before_state.get(k) != after_state.get(k):
                diff[k] = {"from": before_state.get(k), "to": after_state.get(k)}
        self._action_effects[action].append(diff)

    def predict(self, current_state: Dict[str, Any],
                action_sequence: List[str], horizon: int = 5) -> Prediction:
        """Predict future state after executing action_sequence."""
        simulated = dict(current_state)
        for action in action_sequence[:horizon]:
            simulated = self._apply_action(simulated, action)

        confidence = self._estimate_confidence(action_sequence)
        pred = Prediction(
            predicted_state=simulated,
            confidence=confidence,
            horizon=min(horizon, len(action_sequence)),
            action_sequence=action_sequence
        )
        self._predictions[pred.prediction_id] = pred
        return pred

    def _apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply learned action effects to state."""
        new_state = dict(state)
        effects = self._action_effects.get(action, [])
        if effects:
            # Use most recent effect as prediction
            latest_effect = effects[-1]
            for k, change in latest_effect.items():
                new_state[k] = change.get("to")
        else:
            # No learned effect -- mark as uncertain
            new_state[f"_uncertain_{action}"] = True
        return new_state

    def evaluate_prediction(self, prediction_id: str,
                             actual_state: Dict[str, Any]) -> PredictionEvaluation:
        """Compare prediction vs actual state, update accuracy."""
        pred = self._predictions.get(prediction_id)
        if not pred:
            return PredictionEvaluation(prediction_id=prediction_id, accuracy=0.0,
                                        key_matches=0, key_total=0)

        predicted = pred.predicted_state
        keys = [k for k in actual_state if not k.startswith("_")]
        matches = sum(1 for k in keys if predicted.get(k) == actual_state.get(k))
        accuracy = matches / len(keys) if keys else 0.0

        self._accuracy_history.append(accuracy)
        if len(self._accuracy_history) > 200:
            self._accuracy_history = self._accuracy_history[-200:]

        ev = PredictionEvaluation(
            prediction_id=prediction_id,
            accuracy=accuracy,
            key_matches=matches,
            key_total=len(keys)
        )
        self._evaluations.append(ev)
        return ev

    def _estimate_confidence(self, action_sequence: List[str]) -> float:
        """Estimate confidence based on how well we know each action."""
        if not action_sequence:
            return 0.5
        known = sum(1 for a in action_sequence if a in self._action_effects)
        base_confidence = known / len(action_sequence)
        avg_accuracy = self.get_average_accuracy()
        return round((base_confidence * 0.5 + avg_accuracy * 0.5), 3)

    def get_confidence(self, prediction: Prediction) -> Dict[str, Any]:
        avg = self.get_average_accuracy()
        std = self._std_accuracy()
        lower = max(0.0, prediction.confidence - std)
        upper = min(1.0, prediction.confidence + std)
        return {
            "point_estimate": prediction.confidence,
            "interval": [round(lower, 3), round(upper, 3)],
            "historical_accuracy": avg,
            "sample_size": len(self._accuracy_history)
        }

    def get_average_accuracy(self) -> float:
        if not self._accuracy_history:
            return 0.5
        return round(sum(self._accuracy_history) / len(self._accuracy_history), 3)

    def _std_accuracy(self) -> float:
        if len(self._accuracy_history) < 2:
            return 0.2
        avg = self.get_average_accuracy()
        variance = sum((x - avg) ** 2 for x in self._accuracy_history) / len(self._accuracy_history)
        return round(math.sqrt(variance), 3)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_predictions": len(self._predictions),
            "total_evaluations": len(self._evaluations),
            "average_accuracy": self.get_average_accuracy(),
            "known_actions": len(self._action_effects),
            "std_accuracy": self._std_accuracy()
        }

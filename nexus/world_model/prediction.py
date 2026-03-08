"""PredictionEngine - forecasting future states using models and uncertainty."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.world_model.prediction")


@dataclass
class Prediction:
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    variable: str = ""
    predicted_value: Any = None
    confidence: float = 0.5
    horizon_seconds: float = 60.0
    made_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime = field(default_factory=datetime.utcnow)
    model_name: str = "default"
    actual_value: Optional[Any] = None
    verified: bool = False
    error: Optional[float] = None

    def __post_init__(self):
        self.valid_until = self.made_at + timedelta(seconds=self.horizon_seconds)

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.valid_until

    def verify(self, actual: Any) -> float:
        self.actual_value = actual
        self.verified = True
        if isinstance(self.predicted_value, (int, float)) and isinstance(actual, (int, float)):
            self.error = abs(float(self.predicted_value) - float(actual))
        else:
            self.error = 0.0 if self.predicted_value == actual else 1.0
        return self.error

    def to_dict(self) -> Dict[str, Any]:
        return {"prediction_id": self.prediction_id, "variable": self.variable, "predicted_value": self.predicted_value, "confidence": self.confidence, "horizon_seconds": self.horizon_seconds, "model_name": self.model_name, "made_at": self.made_at.isoformat(), "valid_until": self.valid_until.isoformat(), "verified": self.verified, "error": self.error}


@dataclass
class PredictionModel:
    name: str
    predict_fn: Callable
    description: str = ""
    accuracy_history: List[float] = field(default_factory=list)
    call_count: int = 0

    @property
    def avg_accuracy(self) -> float:
        return sum(self.accuracy_history) / max(len(self.accuracy_history), 1)

    def record_accuracy(self, error: float, scale: float = 1.0) -> None:
        accuracy = max(0.0, 1.0 - error / max(scale, 1e-9))
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > 1000:
            self.accuracy_history = self.accuracy_history[-1000:]


class PredictionEngine:
    """
    Multi-model prediction engine with ensemble support,
    confidence calibration, and accuracy tracking.
    """

    def __init__(self):
        self._models: Dict[str, PredictionModel] = {}
        self._predictions: Dict[str, Prediction] = {}
        self._history: List[Prediction] = []
        self._register_default_models()

    def _register_default_models(self) -> None:
        self.register_model("last_value", self._model_last_value, "Predict using most recent observed value")
        self.register_model("trend", self._model_trend, "Linear trend extrapolation")
        self.register_model("mean", self._model_mean, "Predict using historical mean")

    def register_model(self, name: str, predict_fn: Callable, description: str = "") -> None:
        self._models[name] = PredictionModel(name=name, predict_fn=predict_fn, description=description)
        logger.debug("Registered prediction model: %s", name)

    def predict(self, variable: str, history: List[Any], horizon_seconds: float = 60.0, model_name: str = "last_value", context: Optional[Dict[str, Any]] = None) -> Prediction:
        model = self._models.get(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")
        model.call_count += 1
        try:
            predicted, confidence = model.predict_fn(history, context or {})
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_name, exc)
            predicted, confidence = None, 0.0
        pred = Prediction(variable=variable, predicted_value=predicted, confidence=confidence, horizon_seconds=horizon_seconds, model_name=model_name)
        self._predictions[pred.prediction_id] = pred
        self._history.append(pred)
        return pred

    def predict_ensemble(self, variable: str, history: List[Any], horizon_seconds: float = 60.0, model_names: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> Prediction:
        names = model_names or list(self._models.keys())
        individual: List[Prediction] = []
        for name in names:
            try:
                p = self.predict(variable, history, horizon_seconds, name, context)
                individual.append(p)
            except Exception:
                pass
        if not individual:
            return Prediction(variable=variable, predicted_value=None, confidence=0.0, horizon_seconds=horizon_seconds, model_name="ensemble")
        # Confidence-weighted average for numeric predictions
        numeric = [p for p in individual if isinstance(p.predicted_value, (int, float))]
        if numeric:
            total_conf = sum(p.confidence for p in numeric) or 1.0
            weighted_val = sum(p.predicted_value * p.confidence for p in numeric) / total_conf
            avg_conf = sum(p.confidence for p in numeric) / len(numeric)
            ensemble_pred = Prediction(variable=variable, predicted_value=round(weighted_val, 6), confidence=round(avg_conf, 4), horizon_seconds=horizon_seconds, model_name="ensemble")
        else:
            best = max(individual, key=lambda p: p.confidence)
            ensemble_pred = Prediction(variable=variable, predicted_value=best.predicted_value, confidence=best.confidence, horizon_seconds=horizon_seconds, model_name="ensemble")
        self._predictions[ensemble_pred.prediction_id] = ensemble_pred
        return ensemble_pred

    def verify(self, prediction_id: str, actual_value: Any) -> Optional[float]:
        pred = self._predictions.get(prediction_id)
        if not pred:
            return None
        error = pred.verify(actual_value)
        model = self._models.get(pred.model_name)
        if model:
            model.record_accuracy(error)
        return error

    def get_active_predictions(self) -> List[Prediction]:
        return [p for p in self._predictions.values() if not p.is_expired() and not p.verified]

    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        return self._predictions.get(prediction_id)

    # ------------------------------------------------------------------
    # Built-in models
    # ------------------------------------------------------------------

    @staticmethod
    def _model_last_value(history: List[Any], context: Dict) -> Tuple[Any, float]:
        if not history:
            return None, 0.0
        return history[-1], 0.7

    @staticmethod
    def _model_trend(history: List[Any], context: Dict) -> Tuple[Any, float]:
        numeric = [float(v) for v in history if isinstance(v, (int, float))]
        if len(numeric) < 2:
            return numeric[-1] if numeric else None, 0.4
        slope = (numeric[-1] - numeric[0]) / len(numeric)
        predicted = numeric[-1] + slope
        confidence = min(0.85, 0.5 + 0.05 * len(numeric))
        return round(predicted, 6), confidence

    @staticmethod
    def _model_mean(history: List[Any], context: Dict) -> Tuple[Any, float]:
        numeric = [float(v) for v in history if isinstance(v, (int, float))]
        if not numeric:
            return None, 0.0
        mean = sum(numeric) / len(numeric)
        confidence = min(0.75, 0.3 + 0.05 * len(numeric))
        return round(mean, 6), confidence

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def model_performance(self) -> Dict[str, Any]:
        return {name: {"calls": m.call_count, "avg_accuracy": round(m.avg_accuracy, 4), "description": m.description} for name, m in self._models.items()}

    def stats(self) -> Dict[str, Any]:
        verified = [p for p in self._history if p.verified]
        return {"total_predictions": len(self._predictions), "verified": len(verified), "active": len(self.get_active_predictions()), "models": len(self._models), "avg_error": round(sum(p.error for p in verified if p.error is not None) / max(len(verified), 1), 4), "model_performance": self.model_performance()}

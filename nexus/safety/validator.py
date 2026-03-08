"""
Action Validator - validates AGI actions before execution
"""
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ActionRisk(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ActionSpec:
    name: str
    action_type: str
    parameters: Dict[str, Any]
    requesting_agent: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    is_valid: bool
    risk_level: str
    reason: str
    requires_approval: bool
    modifications: Dict[str, Any] = field(default_factory=dict)

class ActionValidator:
    """
    Validates AGI actions before execution.
    Implements a risk-based approval system.
    """

    DANGEROUS_ACTION_TYPES = {
        "delete_file", "format_disk", "execute_system_command",
        "modify_system_config", "network_broadcast", "send_mass_email"
    }

    HIGH_RISK_ACTION_TYPES = {
        "write_file", "modify_database", "send_email", "api_call_external",
        "create_process", "modify_memory"
    }

    APPROVAL_REQUIRED_RISK = {ActionRisk.HIGH, ActionRisk.CRITICAL}

    def __init__(self, custom_dangerous: Optional[Set[str]] = None):
        self.dangerous_actions = set(self.DANGEROUS_ACTION_TYPES)
        self.high_risk_actions = set(self.HIGH_RISK_ACTION_TYPES)
        if custom_dangerous:
            self.dangerous_actions.update(custom_dangerous)
        self._validators: List[Callable[[ActionSpec], Optional[ValidationResult]]] = []

    def validate(self, action: ActionSpec) -> ValidationResult:
        for validator in self._validators:
            result = validator(action)
            if result is not None:
                return result

        risk = self._assess_risk(action)

        if risk == ActionRisk.CRITICAL:
            return ValidationResult(
                is_valid=False,
                risk_level=risk.value,
                reason=f"Action type '{action.action_type}' is critically dangerous",
                requires_approval=True,
            )

        requires_approval = risk in self.APPROVAL_REQUIRED_RISK

        return ValidationResult(
            is_valid=True,
            risk_level=risk.value,
            reason=f"Action assessed as {risk.value} risk",
            requires_approval=requires_approval,
        )

    def _assess_risk(self, action: ActionSpec) -> ActionRisk:
        if action.action_type in self.dangerous_actions:
            return ActionRisk.CRITICAL
        if action.action_type in self.high_risk_actions:
            return ActionRisk.HIGH
        return ActionRisk.SAFE

    def add_validator(self, validator: Callable[[ActionSpec], Optional[ValidationResult]]):
        self._validators.append(validator)

    def get_risk_summary(self) -> Dict[str, Any]:
        return {
            "dangerous_action_types": list(self.dangerous_actions),
            "high_risk_action_types": list(self.high_risk_actions),
            "custom_validators": len(self._validators),
        }

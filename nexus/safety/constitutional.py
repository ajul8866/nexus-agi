"""
Constitutional AI - rule-based safety constraints for AGI outputs
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RuleCategory(Enum):
    HARM_PREVENTION = "harm_prevention"
    PRIVACY = "privacy"
    HONESTY = "honesty"
    FAIRNESS = "fairness"
    SAFETY = "safety"
    LEGAL = "legal"

@dataclass
class ConstitutionalRule:
    id: str
    name: str
    description: str
    category: RuleCategory
    severity: Severity
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class ViolationReport:
    rule_id: str
    rule_name: str
    category: str
    severity: str
    description: str
    matched_content: str
    suggestion: str

class ConstitutionalAI:
    """
    Implements Constitutional AI principles for safe AGI operation.
    Enforces a set of rules that the AGI must follow at all times.
    """

    DEFAULT_RULES = [
        ConstitutionalRule(
            id="harm_001",
            name="No Physical Harm Instructions",
            description="Prevent instructions for causing physical harm",
            category=RuleCategory.HARM_PREVENTION,
            severity=Severity.CRITICAL,
            keywords=["how to kill", "how to hurt", "bomb", "weapon", "poison someone"],
        ),
        ConstitutionalRule(
            id="privacy_001",
            name="No PII Exposure",
            description="Prevent exposure of personally identifiable information",
            category=RuleCategory.PRIVACY,
            severity=Severity.HIGH,
            patterns=[r"\b\d{3}-\d{2}-\d{4}\b", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        ),
        ConstitutionalRule(
            id="honesty_001",
            name="No Deception",
            description="Prevent deceptive statements",
            category=RuleCategory.HONESTY,
            severity=Severity.HIGH,
            keywords=["i am human", "i am not an ai", "i am a person"],
        ),
        ConstitutionalRule(
            id="safety_001",
            name="No Dangerous Code",
            description="Prevent generation of dangerous system commands",
            category=RuleCategory.SAFETY,
            severity=Severity.CRITICAL,
            patterns=[r"rm\s+-rf\s+/", r"format\s+c:", r"del\s+/f\s+/s\s+/q"],
        ),
        ConstitutionalRule(
            id="legal_001",
            name="No Illegal Activity",
            description="Prevent assistance with clearly illegal activities",
            category=RuleCategory.LEGAL,
            severity=Severity.HIGH,
            keywords=["how to steal", "how to hack illegally", "how to commit fraud"],
        ),
    ]

    def __init__(self, custom_rules: Optional[List[ConstitutionalRule]] = None):
        self.rules = list(self.DEFAULT_RULES)
        if custom_rules:
            self.rules.extend(custom_rules)
        self._compiled_patterns: Dict[str, List] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        for rule in self.rules:
            compiled = []
            for pattern in rule.patterns:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid pattern in rule {rule.id}: {e}")
            self._compiled_patterns[rule.id] = compiled

    def check(self, content: str) -> Tuple[bool, List[ViolationReport]]:
        """
        Check content against all constitutional rules.
        Returns (is_safe, violations)
        """
        violations = []
        content_lower = content.lower()

        for rule in self.rules:
            if not rule.enabled:
                continue

            matched = False
            matched_content = ""

            for keyword in rule.keywords:
                if keyword.lower() in content_lower:
                    matched = True
                    matched_content = keyword
                    break

            if not matched:
                for pattern in self._compiled_patterns.get(rule.id, []):
                    match = pattern.search(content)
                    if match:
                        matched = True
                        matched_content = match.group()
                        break

            if matched:
                violations.append(ViolationReport(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    category=rule.category.value,
                    severity=rule.severity.value,
                    description=rule.description,
                    matched_content=matched_content,
                    suggestion=f"Remove or rephrase content related to: {rule.name}",
                ))

        is_safe = len(violations) == 0
        return is_safe, violations

    def add_rule(self, rule: ConstitutionalRule):
        self.rules.append(rule)
        compiled = []
        for pattern in rule.patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass
        self._compiled_patterns[rule.id] = compiled

    def get_rules_summary(self) -> Dict[str, Any]:
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules if r.enabled),
            "by_category": {cat.value: sum(1 for r in self.rules if r.category == cat) for cat in RuleCategory},
            "by_severity": {sev.value: sum(1 for r in self.rules if r.severity == sev) for sev in Severity},
        }

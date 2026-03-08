"""
Output Filter - post-processing safety layer
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FilterResult:
    original: str
    filtered: str
    was_modified: bool
    modifications: List[str]

class OutputFilter:
    """
    Filters and sanitizes AGI outputs before delivery.
    Removes or redacts sensitive information.
    """

    PII_PATTERNS = {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    REDACTION_MAP = {
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
        "email": "[EMAIL REDACTED]",
        "phone_us": "[PHONE REDACTED]",
        "ip_address": "[IP REDACTED]",
    }

    def __init__(self, redact_pii: bool = True, max_length: Optional[int] = None):
        self.redact_pii = redact_pii
        self.max_length = max_length

    def filter(self, content: str) -> FilterResult:
        filtered = content
        modifications = []

        if self.redact_pii:
            for pii_type, pattern in self.PII_PATTERNS.items():
                matches = pattern.findall(filtered)
                if matches:
                    filtered = pattern.sub(self.REDACTION_MAP[pii_type], filtered)
                    modifications.append(f"Redacted {len(matches)} {pii_type} instance(s)")

        if self.max_length and len(filtered) > self.max_length:
            filtered = filtered[:self.max_length] + "... [TRUNCATED]"
            modifications.append(f"Truncated to {self.max_length} characters")

        return FilterResult(
            original=content,
            filtered=filtered,
            was_modified=filtered != content,
            modifications=modifications,
        )

    def batch_filter(self, contents: List[str]) -> List[FilterResult]:
        return [self.filter(c) for c in contents]

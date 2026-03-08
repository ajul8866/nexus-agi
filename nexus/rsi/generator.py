"""
NEXUS-AGI Tool Generator
Dynamic tool generation untuk capability gaps
"""
from __future__ import annotations
import time
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CapabilityGap:
    description: str
    failed_tasks: List[str]
    missing_capability: str
    frequency: int
    severity: float


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    code_template: str
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    validated: bool = False


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    safety_score: float


TOOL_TEMPLATES = {
    "data_fetcher": '''
def {name}(url: str, timeout: int = 30) -> dict:
    """Fetch data from URL and return parsed result."""
    import urllib.request
    import json
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return {{"data": json.loads(resp.read()), "status": resp.status}}
    except Exception as e:
        return {{"error": str(e), "data": None}}
''',
    "text_processor": '''
def {name}(text: str, operation: str = "clean") -> str:
    """Process text with specified operation."""
    import re
    if operation == "clean":
        text = re.sub(r"\\s+", " ", text).strip()
    elif operation == "extract_numbers":
        numbers = re.findall(r"\\d+\\.?\\d*", text)
        return " ".join(numbers)
    elif operation == "summarize":
        sentences = text.split(".")
        return ". ".join(sentences[:3])
    return text
''',
    "calculator": '''
def {name}(expression: str) -> dict:
    """Safely evaluate mathematical expression."""
    import ast
    import operator
    ops = {{ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv}}
    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        raise ValueError(f"Unsupported: {{type(node)}}")
    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree.body)
        return {{"result": result, "expression": expression}}
    except Exception as e:
        return {{"error": str(e), "result": None}}
''',
}

DANGEROUS_PATTERNS = [
    "exec(", "eval(", "__import__", "subprocess", "os.system",
    "open(", "shutil", "pickle", "marshal", "importlib"
]


class ToolGenerator:
    """Generate new tools dynamically untuk mengisi capability gaps."""

    def __init__(self):
        self._generated_tools: Dict[str, ToolSpec] = {}
        self._capability_gaps: List[CapabilityGap] = []

    def identify_capability_gap(self, failed_tasks: List[str]) -> CapabilityGap:
        keywords: Dict[str, int] = {}
        for task in failed_tasks:
            for word in task.lower().split():
                if len(word) > 4:
                    keywords[word] = keywords.get(word, 0) + 1
        top_keyword = max(keywords, key=keywords.get) if keywords else "unknown"
        capability_map = {
            "fetch": "data_fetcher", "download": "data_fetcher", "http": "data_fetcher",
            "calculate": "calculator", "compute": "calculator", "math": "calculator",
            "process": "text_processor", "parse": "text_processor", "extract": "text_processor",
        }
        missing = capability_map.get(top_keyword, "generic_processor")
        gap = CapabilityGap(
            description=f"Unable to handle tasks requiring '{top_keyword}' capability",
            failed_tasks=failed_tasks, missing_capability=missing,
            frequency=len(failed_tasks), severity=min(len(failed_tasks) / 10.0, 1.0)
        )
        self._capability_gaps.append(gap)
        return gap

    def generate_tool_spec(self, gap: CapabilityGap) -> ToolSpec:
        template_key = gap.missing_capability if gap.missing_capability in TOOL_TEMPLATES else "text_processor"
        tool_name = f"auto_{gap.missing_capability}_{int(time.time())}"
        code = TOOL_TEMPLATES[template_key].format(name=tool_name)
        return ToolSpec(
            name=tool_name,
            description=f"Auto-generated tool for: {gap.description}",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "object"},
            code_template=code,
            tags=["auto-generated", gap.missing_capability]
        )

    def validate_tool(self, tool_spec: ToolSpec) -> ValidationResult:
        errors = []
        warnings = []
        safety_score = 1.0
        try:
            ast.parse(tool_spec.code_template)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        for pattern in DANGEROUS_PATTERNS:
            if pattern in tool_spec.code_template:
                errors.append(f"Dangerous pattern detected: {pattern}")
                safety_score -= 0.3
        if len(tool_spec.code_template) > 5000:
            warnings.append("Tool code is very long (>5000 chars)")
        safety_score = max(0.0, safety_score)
        valid = len(errors) == 0 and safety_score > 0.5
        if valid:
            tool_spec.validated = True
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, safety_score=round(safety_score, 2))

    def register_tool(self, tool_spec: ToolSpec) -> bool:
        if not tool_spec.validated:
            result = self.validate_tool(tool_spec)
            if not result.valid:
                return False
        self._generated_tools[tool_spec.name] = tool_spec
        return True

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "tags": t.tags, "validated": t.validated}
                for t in self._generated_tools.values()]

    def stats(self) -> Dict[str, Any]:
        return {
            "generated_tools": len(self._generated_tools),
            "capability_gaps": len(self._capability_gaps),
            "validated_tools": sum(1 for t in self._generated_tools.values() if t.validated)
        }

"""
NEXUS-AGI Sandbox
Safe Python code execution environment
"""
from __future__ import annotations
import io
import sys
import time
import ast
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExecutionResult:
    success: bool
    output: str
    return_value: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""


BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open", "input",
    "memoryview", "breakpoint"
}

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "http",
    "shutil", "pickle", "marshal", "importlib", "ctypes", "multiprocessing"
}

SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "vars", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError"
}


class Sandbox:
    """Sandboxed Python execution with resource limits."""

    def __init__(self, max_output_size: int = 10_000):
        self._max_output_size = max_output_size
        self._execution_count = 0

    def execute_code(self, code: str, timeout: float = 10.0,
                     extra_globals: Optional[Dict] = None) -> ExecutionResult:
        start = time.time()
        self._execution_count += 1

        # AST safety check
        safety_check = self._check_safety(code)
        if not safety_check["safe"]:
            return ExecutionResult(
                success=False, output="", return_value=None,
                error=f"Safety violation: {safety_check['reason']}"
            )

        # Prepare safe globals
        safe_globals = {
            "__builtins__": {k: __builtins__[k] if isinstance(__builtins__, dict)
                             else getattr(__builtins__, k, None)
                             for k in SAFE_BUILTINS
                             if (isinstance(__builtins__, dict) and k in __builtins__)
                             or hasattr(__builtins__, k)},
            "__name__": "__sandbox__",
        }
        if extra_globals:
            safe_globals.update(extra_globals)

        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        return_value = None

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr
            exec(compile(code, "<sandbox>", "exec"), safe_globals)  # noqa: S102
            # Try to get last expression value
            try:
                tree = ast.parse(code, mode="exec")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    return_value = eval(  # noqa: S307
                        compile(ast.Expression(body=tree.body[-1].value), "<sandbox>", "eval"),
                        safe_globals
                    )
            except Exception:
                pass
        except Exception as e:
            err = traceback.format_exc()
            return ExecutionResult(
                success=False,
                output=captured_stdout.getvalue()[:self._max_output_size],
                return_value=None,
                error=err[:2000],
                execution_time=time.time() - start,
                stdout=captured_stdout.getvalue()[:self._max_output_size],
                stderr=captured_stderr.getvalue()[:1000]
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_val = captured_stdout.getvalue()[:self._max_output_size]
        return ExecutionResult(
            success=True,
            output=stdout_val,
            return_value=self._serialize(return_value),
            execution_time=round(time.time() - start, 4),
            stdout=stdout_val,
            stderr=captured_stderr.getvalue()[:1000]
        )

    def _check_safety(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"safe": False, "reason": f"Syntax error: {e}"}

        for node in ast.walk(tree):
            # Block import statements for dangerous modules
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [n.name for n in getattr(node, "names", [])]
                module = getattr(node, "module", "") or ""
                all_names = names + [module]
                for name in all_names:
                    root = name.split(".")[0] if name else ""
                    if root in BLOCKED_MODULES:
                        return {"safe": False, "reason": f"Blocked module: {root}"}

            # Block dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        return {"safe": False, "reason": f"Blocked builtin: {node.func.id}"}

        return {"safe": True, "reason": None}

    def _serialize(self, value: Any) -> Any:
        if value is None:
            return None
        try:
            import json
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    def stats(self) -> Dict[str, Any]:
        return {"total_executions": self._execution_count}

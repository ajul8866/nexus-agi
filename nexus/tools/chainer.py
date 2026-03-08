"""
NEXUS-AGI Tool Chainer
Build dan execute tool pipelines
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import asyncio

from .executor import ToolExecutor, ToolResult


@dataclass
class ChainResult:
    steps: List[ToolResult]
    final_output: Any
    success: bool
    error: Optional[str] = None


class ToolChainer:
    """Pipeline composition untuk tool chains."""

    def __init__(self, executor: ToolExecutor):
        self._executor = executor

    def chain(self, tool_sequence: List[Dict[str, Any]],
              initial_input: Optional[Dict] = None) -> ChainResult:
        """Execute tools sequentially, piping output to next input."""
        steps: List[ToolResult] = []
        current_input = initial_input or {}

        for step in tool_sequence:
            tool_name = step["tool"]
            # Merge current pipeline data with step-specific params
            params = {**current_input, **step.get("params", {})}
            result = self._executor.execute(tool_name, params)
            steps.append(result)

            if not result.success:
                return ChainResult(steps=steps, final_output=None,
                                   success=False, error=result.error)

            # Output becomes next input (flatten if dict)
            if isinstance(result.output, dict):
                current_input = result.output
            else:
                current_input = {"input": result.output, "previous": result.output}

        return ChainResult(
            steps=steps,
            final_output=current_input,
            success=True
        )

    async def parallel(self, tool_list: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        return await self._executor.batch_execute(tool_list)

    def conditional(self, condition_tool: Dict, true_tool: Dict,
                    false_tool: Dict, input_data: Dict) -> ToolResult:
        """Execute true_tool or false_tool based on condition_tool result."""
        condition_result = self._executor.execute(
            condition_tool["tool"],
            {**input_data, **condition_tool.get("params", {})}
        )
        branch = true_tool if (condition_result.success and condition_result.output) else false_tool
        return self._executor.execute(
            branch["tool"],
            {**input_data, **branch.get("params", {})}
        )

    async def map_reduce(self, map_tool: Dict, reduce_tool: Dict,
                         inputs: List[Any]) -> ToolResult:
        """Map tool over inputs, then reduce results."""
        map_calls = [{"tool": map_tool["tool"], "params": {"input": inp}} for inp in inputs]
        map_results = await self.parallel(map_calls)
        mapped_outputs = [r.output for r in map_results if r.success]

        return self._executor.execute(
            reduce_tool["tool"],
            {"inputs": mapped_outputs, **reduce_tool.get("params", {})}
        )

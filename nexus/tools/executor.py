"""
NEXUS-AGI Tool Executor
Execute tools dengan timeout, retry, caching
"""
from __future__ import annotations
import asyncio
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    latency: float = 0.0
    cached: bool = False
    timestamp: float = field(default_factory=time.time)
    retries: int = 0


class ToolExecutor:
    """Execute tools dengan timeout, retry, caching, dan rate limiting."""

    def __init__(self, registry: ToolRegistry, cache_ttl: int = 300):
        self._registry = registry
        self._cache: Dict[str, tuple] = {}  # key -> (result, expires_at)
        self._cache_ttl = cache_ttl
        self._rate_limits: Dict[str, List[float]] = {}
        self._history: List[ToolResult] = []
        self._stats: Dict[str, Dict] = {}

    def execute(self, tool_name: str, params: Dict[str, Any],
                timeout: float = 30.0, max_retries: int = 3,
                use_cache: bool = True) -> ToolResult:
        start = time.time()

        # Cache check
        cache_key = self._cache_key(tool_name, params)
        if use_cache:
            cached = self._get_cache(cache_key)
            if cached:
                return ToolResult(tool_name=tool_name, success=True,
                                  output=cached, cached=True,
                                  latency=time.time() - start)

        # Rate limit check
        if not self._check_rate_limit(tool_name):
            return ToolResult(tool_name=tool_name, success=False,
                              error="Rate limit exceeded", latency=0.0)

        tool = self._registry.get_tool(tool_name)
        if not tool or not tool.func:
            return ToolResult(tool_name=tool_name, success=False,
                              error=f"Tool '{tool_name}' not found")

        last_error = None
        for attempt in range(max_retries):
            try:
                result = tool.func(**params)
                latency = time.time() - start
                tr = ToolResult(tool_name=tool_name, success=True,
                                output=result, latency=latency, retries=attempt)
                self._set_cache(cache_key, result)
                self._record_stat(tool_name, latency, True)
                self._history.append(tr)
                return tr
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))

        latency = time.time() - start
        tr = ToolResult(tool_name=tool_name, success=False,
                        error=last_error, latency=latency, retries=max_retries)
        self._record_stat(tool_name, latency, False)
        self._history.append(tr)
        return tr

    async def execute_async(self, tool_name: str, params: Dict[str, Any],
                            timeout: float = 30.0) -> ToolResult:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: self.execute(tool_name, params, timeout)),
            timeout=timeout
        )

    async def batch_execute(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        tasks = [self.execute_async(tc["tool"], tc.get("params", {})) for tc in tool_calls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _cache_key(self, tool_name: str, params: Dict) -> str:
        raw = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cache(self, key: str) -> Optional[Any]:
        if key in self._cache:
            val, expires = self._cache[key]
            if time.time() < expires:
                return val
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time() + self._cache_ttl)

    def _check_rate_limit(self, tool_name: str, max_per_minute: int = 60) -> bool:
        now = time.time()
        calls = self._rate_limits.setdefault(tool_name, [])
        # Remove calls older than 60s
        self._rate_limits[tool_name] = [t for t in calls if now - t < 60]
        if len(self._rate_limits[tool_name]) >= max_per_minute:
            return False
        self._rate_limits[tool_name].append(now)
        return True

    def _record_stat(self, tool_name: str, latency: float, success: bool):
        s = self._stats.setdefault(tool_name, {"calls": 0, "success": 0, "total_latency": 0.0})
        s["calls"] += 1
        s["success"] += int(success)
        s["total_latency"] += latency

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for name, s in self._stats.items():
            result[name] = {
                "calls": s["calls"],
                "success_rate": round(s["success"] / s["calls"], 3) if s["calls"] else 0,
                "avg_latency": round(s["total_latency"] / s["calls"], 3) if s["calls"] else 0
            }
        return result

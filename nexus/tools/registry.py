"""
NEXUS-AGI Tool Registry
Register, discover, dan manage tools
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolInfo:
    name: str
    description: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    tags: List[str]
    func: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    call_count: int = 0
    avg_latency: float = 0.0
    created_at: float = field(default_factory=time.time)


class ToolRegistry:
    """Centralized tool registry dengan discovery dan versioning."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, ToolInfo]] = {}  # name -> {version -> ToolInfo}
        self._default_versions: Dict[str, str] = {}
        self._tags_index: Dict[str, List[str]] = {}
        self._register_builtins()

    def register(self, func: Callable, name: str, description: str,
                 input_schema: Dict, output_schema: Dict,
                 tags: List[str] = None, version: str = "1.0.0",
                 dependencies: List[str] = None) -> ToolInfo:
        tags = tags or []
        dependencies = dependencies or []
        info = ToolInfo(
            name=name, description=description, version=version,
            input_schema=input_schema, output_schema=output_schema,
            tags=tags, func=func, dependencies=dependencies
        )
        self._tools.setdefault(name, {})[version] = info
        self._default_versions[name] = version

        for tag in tags:
            self._tags_index.setdefault(tag, [])
            if name not in self._tags_index[tag]:
                self._tags_index[tag].append(name)

        return info

    def get_tool(self, name: str, version: Optional[str] = None) -> Optional[ToolInfo]:
        if name not in self._tools:
            return None
        ver = version or self._default_versions.get(name)
        return self._tools[name].get(ver)

    def discover(self, capability_description: str) -> List[ToolInfo]:
        """Semantic-lite discovery: match keywords against tool descriptions."""
        keywords = set(capability_description.lower().split())
        scored: List[tuple] = []
        for name, versions in self._tools.items():
            tool = list(versions.values())[-1]
            tool_text = f"{tool.name} {tool.description} {' '.join(tool.tags)}".lower()
            score = sum(1 for kw in keywords if kw in tool_text)
            if score > 0:
                scored.append((score, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:5]]

    def list_tools(self, tags: Optional[List[str]] = None) -> List[ToolInfo]:
        if tags:
            matching_names: set = set()
            for tag in tags:
                matching_names.update(self._tags_index.get(tag, []))
            return [self.get_tool(name) for name in matching_names if self.get_tool(name)]
        return [list(versions.values())[-1] for versions in self._tools.values()]

    def _register_builtins(self):
        """Register built-in tools."""
        import urllib.request
        import json as _json

        def web_search(query: str) -> Dict:
            return {"query": query, "results": [], "note": "Configure search API key"}

        def file_read(path: str) -> Dict:
            try:
                with open(path, "r") as f:
                    return {"content": f.read(), "path": path}
            except Exception as e:
                return {"error": str(e)}

        def file_write(path: str, content: str) -> Dict:
            try:
                with open(path, "w") as f:
                    f.write(content)
                return {"success": True, "path": path, "bytes": len(content)}
            except Exception as e:
                return {"error": str(e)}

        def http_request(url: str, method: str = "GET") -> Dict:
            try:
                req = urllib.request.Request(url, method=method)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return {"status": resp.status, "body": resp.read().decode()[:2000]}
            except Exception as e:
                return {"error": str(e)}

        builtins = [
            (web_search, "web_search", "Search the web for information",
             {"query": "string"}, {"results": "array"}, ["search", "web"]),
            (file_read, "file_read", "Read content from a file",
             {"path": "string"}, {"content": "string"}, ["file", "io"]),
            (file_write, "file_write", "Write content to a file",
             {"path": "string", "content": "string"}, {"success": "boolean"}, ["file", "io"]),
            (http_request, "http_request", "Make HTTP request to a URL",
             {"url": "string"}, {"status": "integer", "body": "string"}, ["http", "web"]),
        ]
        for func, name, desc, inp, out, tags in builtins:
            self.register(func, name, desc, inp, out, tags)

    def stats(self) -> Dict[str, Any]:
        total = sum(len(v) for v in self._tools.values())
        return {
            "tool_names": len(self._tools),
            "total_versions": total,
            "tags": list(self._tags_index.keys())
        }

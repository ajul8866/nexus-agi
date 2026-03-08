"""
NEXUS-AGI FastAPI Server
REST API + WebSocket interface
"""
from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --- Pydantic Models ---
class TaskRequest(BaseModel):
    description: str
    priority: int = 5
    timeout: int = 300
    metadata: Dict[str, Any] = {}


class TaskResponse(BaseModel):
    task_id: str
    status: str
    description: str
    created_at: float
    result: Optional[Any] = None
    error: Optional[str] = None


class MemoryQueryRequest(BaseModel):
    query: str
    limit: int = 10
    memory_type: str = "all"


class AgentCreateRequest(BaseModel):
    name: str
    role: str = "specialist"
    capabilities: List[str] = []


# --- In-memory task store ---
_tasks: Dict[str, Dict[str, Any]] = {}
_agents: Dict[str, Dict[str, Any]] = {}
_connections: List[WebSocket] = []


def create_app() -> FastAPI:
    app = FastAPI(
        title="NEXUS-AGI API",
        description="Advanced General Intelligence Framework REST API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}

    @app.get("/status")
    async def status():
        return {
            "status": "operational",
            "tasks": {"total": len(_tasks), "pending": sum(1 for t in _tasks.values() if t["status"] == "pending")},
            "agents": {"total": len(_agents)},
            "uptime": time.time(),
        }

    @app.post("/tasks", response_model=TaskResponse)
    async def create_task(req: TaskRequest):
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id, "status": "pending",
            "description": req.description, "priority": req.priority,
            "created_at": time.time(), "result": None, "error": None,
            "metadata": req.metadata
        }
        _tasks[task_id] = task
        await _broadcast({"event": "task_created", "task_id": task_id, "description": req.description})
        return TaskResponse(**{k: task[k] for k in TaskResponse.model_fields})

    @app.get("/tasks", response_model=List[TaskResponse])
    async def list_tasks(status: Optional[str] = None, limit: int = 50):
        tasks = list(_tasks.values())
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        tasks = sorted(tasks, key=lambda x: x["created_at"], reverse=True)[:limit]
        return [TaskResponse(**{k: t[k] for k in TaskResponse.model_fields}) for t in tasks]

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: str):
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        t = _tasks[task_id]
        return TaskResponse(**{k: t[k] for k in TaskResponse.model_fields})

    @app.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str):
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        _tasks[task_id]["status"] = "cancelled"
        return {"message": f"Task {task_id} cancelled"}

    @app.post("/memory/query")
    async def query_memory(req: MemoryQueryRequest):
        return {
            "query": req.query, "results": [],
            "total": 0, "memory_type": req.memory_type
        }

    @app.get("/memory/stats")
    async def memory_stats():
        return {
            "episodic": {"count": 0, "size_mb": 0},
            "semantic": {"count": 0, "size_mb": 0},
            "working": {"count": 0, "size_mb": 0},
            "total_mb": 0
        }

    @app.get("/agents")
    async def list_agents():
        return {"agents": list(_agents.values()), "total": len(_agents)}

    @app.post("/agents")
    async def create_agent(req: AgentCreateRequest):
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        agent = {
            "agent_id": agent_id, "name": req.name,
            "role": req.role, "capabilities": req.capabilities,
            "status": "idle", "created_at": time.time()
        }
        _agents[agent_id] = agent
        return agent

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        _connections.append(ws)
        try:
            while True:
                data = await ws.receive_json()
                await ws.send_json({"echo": data, "timestamp": time.time()})
        except WebSocketDisconnect:
            _connections.remove(ws)

    return app


async def _broadcast(message: Dict[str, Any]):
    dead = []
    for ws in _connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections.remove(ws)


app = create_app()

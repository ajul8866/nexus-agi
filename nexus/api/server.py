"""
NEXUS-AGI REST API + WebSocket Server (FastAPI)
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import asyncio
import json
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# === Pydantic Models ===

class TaskRequest(BaseModel):
    task: str = Field(..., description="Task description for the agent")
    agent_type: str = Field("orchestrator", description="Type of agent to handle task")
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(1, ge=1, le=10)
    max_iterations: int = Field(10, ge=1, le=100)

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class MemoryQuery(BaseModel):
    query: str
    memory_type: str = Field("episodic", description="episodic, semantic, or working")
    top_k: int = Field(5, ge=1, le=50)

class AgentConfig(BaseModel):
    name: str
    agent_type: str
    capabilities: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)

# === In-memory task store (replace with Redis/DB in production) ===
_tasks: Dict[str, Dict] = {}
_ws_connections: List[WebSocket] = []

def create_app(nexus_kernel=None) -> FastAPI:
    app = FastAPI(
        title="NEXUS-AGI API",
        description="REST API and WebSocket server for the NEXUS AGI framework",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === Health Endpoints ===

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}

    @app.get("/status")
    async def system_status():
        return {
            "kernel": "running" if nexus_kernel else "not_initialized",
            "active_tasks": len([t for t in _tasks.values() if t["status"] == "running"]),
            "total_tasks": len(_tasks),
            "ws_connections": len(_ws_connections),
        }

    # === Task Endpoints ===

    @app.post("/tasks", response_model=TaskResponse)
    async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "status": "queued",
            "task": request.task,
            "agent_type": request.agent_type,
            "context": request.context,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
        _tasks[task_id] = task_data

        if nexus_kernel:
            background_tasks.add_task(_run_task, task_id, request, nexus_kernel)
        else:
            task_data["status"] = "error"
            task_data["error"] = "Kernel not initialized"

        return TaskResponse(**task_data)

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: str):
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskResponse(**_tasks[task_id])

    @app.get("/tasks")
    async def list_tasks(status: Optional[str] = None, limit: int = 20):
        tasks = list(_tasks.values())
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        return {"tasks": tasks[-limit:], "total": len(tasks)}

    @app.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str):
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        _tasks[task_id]["status"] = "cancelled"
        return {"message": "Task cancelled", "task_id": task_id}

    # === Memory Endpoints ===

    @app.post("/memory/query")
    async def query_memory(query: MemoryQuery):
        if not nexus_kernel:
            raise HTTPException(status_code=503, detail="Kernel not initialized")

        return {
            "query": query.query,
            "memory_type": query.memory_type,
            "results": [],
            "message": "Memory system integration pending kernel initialization",
        }

    @app.get("/memory/stats")
    async def memory_stats():
        return {
            "episodic": {"count": 0, "size_mb": 0},
            "semantic": {"count": 0, "size_mb": 0},
            "working": {"slots": 7, "used": 0},
        }

    # === Agent Endpoints ===

    @app.get("/agents")
    async def list_agents():
        if not nexus_kernel:
            return {"agents": [], "message": "Kernel not initialized"}
        return {"agents": [], "total": 0}

    @app.post("/agents")
    async def create_agent(config: AgentConfig):
        agent_id = str(uuid.uuid4())
        return {
            "agent_id": agent_id,
            "name": config.name,
            "type": config.agent_type,
            "status": "created",
        }

    # === WebSocket ===

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        _ws_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(_ws_connections)}")

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                response = await _handle_ws_message(message, nexus_kernel)
                await websocket.send_text(json.dumps(response))

        except WebSocketDisconnect:
            _ws_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total: {len(_ws_connections)}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in _ws_connections:
                _ws_connections.remove(websocket)

    return app


async def _run_task(task_id: str, request: TaskRequest, kernel):
    _tasks[task_id]["status"] = "running"
    try:
        result = await kernel.process(request.task, request.context)
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["result"] = result
        _tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
    except Exception as e:
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()


async def _handle_ws_message(message: Dict, kernel) -> Dict:
    msg_type = message.get("type", "unknown")

    if msg_type == "ping":
        return {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
    elif msg_type == "task":
        task_id = str(uuid.uuid4())
        return {"type": "task_accepted", "task_id": task_id}
    elif msg_type == "status":
        return {"type": "status", "kernel": "running" if kernel else "not_initialized"}
    else:
        return {"type": "error", "message": f"Unknown message type: {msg_type}"}

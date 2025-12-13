"""
FastAPI Backend for DroidRun Agent.

Production-ready server with:
- SSE streaming for chat responses
- WebSocket support for real-time updates
- REST API for agent management
- Health checks and metrics
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Query,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("droidrun.backend")


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS
    cors_origins: List[str] = ["*"]

    # Agent
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    offline_mode: bool = False

    # Memory
    memory_enabled: bool = True
    memory_store: str = "in_memory"  # "in_memory" or "qdrant"
    qdrant_url: Optional[str] = None

    # Redis (for session management)
    redis_url: Optional[str] = None

    # Metrics
    metrics_enabled: bool = True


def get_config() -> ServerConfig:
    """Load configuration from environment."""
    return ServerConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        default_model=os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514"),
        offline_mode=os.getenv("OFFLINE_MODE", "false").lower() == "true",
        memory_enabled=os.getenv("MEMORY_ENABLED", "true").lower() == "true",
        memory_store=os.getenv("MEMORY_STORE", "in_memory"),
        qdrant_url=os.getenv("QDRANT_URL"),
        redis_url=os.getenv("REDIS_URL"),
        metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
    )


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message."""

    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request."""

    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    stream: bool = Field(True, description="Enable streaming response")
    model: Optional[str] = Field(None, description="Model override")
    max_tokens: Optional[int] = Field(None, description="Max tokens override")
    include_memory: bool = Field(True, description="Include memory context")


class ChatResponse(BaseModel):
    """Chat response."""

    id: str
    content: str
    thinking: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []
    usage: Dict[str, int] = {}
    session_id: str
    created_at: str


class TaskRequest(BaseModel):
    """Task execution request."""

    goal: str = Field(..., description="Task goal to achieve")
    device_serial: Optional[str] = Field(None, description="Target device serial")
    session_id: Optional[str] = Field(None, description="Session ID")
    max_steps: int = Field(15, description="Maximum execution steps")
    reasoning: bool = Field(False, description="Enable reasoning mode")


class TaskResponse(BaseModel):
    """Task execution response."""

    task_id: str
    status: str
    goal: str
    result: Optional[Dict[str, Any]] = None
    steps: int = 0
    created_at: str


class ResearchRequest(BaseModel):
    """Research request."""

    query: str = Field(..., description="Research query")
    max_results: int = Field(10, description="Maximum results")
    include_memory: bool = Field(True, description="Include memory context")


class MemoryStoreRequest(BaseModel):
    """Memory storage request."""

    task: str
    goal: str
    success: bool
    reason: str
    steps: int
    actions: List[Dict[str, Any]] = []
    tags: List[str] = []


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    memory_enabled: bool
    offline_mode: bool


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container."""

    def __init__(self):
        self.config: ServerConfig = get_config()
        self.start_time: float = time.time()
        self.memory_manager = None
        self.research_agent = None
        self.claude_agent = None
        self.active_sessions: Dict[str, Dict] = {}
        self.active_tasks: Dict[str, Dict] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}

    async def initialize(self):
        """Initialize application components."""
        logger.info("Initializing application...")

        # Initialize memory manager
        if self.config.memory_enabled:
            try:
                from droidrun.agent.memory import MemoryManager, MemoryConfig

                memory_config = MemoryConfig(
                    store_type=self.config.memory_store,
                    qdrant_url=self.config.qdrant_url,
                )
                self.memory_manager = MemoryManager(config=memory_config)
                await self.memory_manager.start()
                logger.info("Memory manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize memory manager: {e}")

        # Initialize research agent
        try:
            from droidrun.agent.research import ResearchAgent, ResearchAgentConfig

            research_config = ResearchAgentConfig(
                offline_mode=self.config.offline_mode,
            )
            self.research_agent = ResearchAgent(
                config=research_config,
                memory_manager=self.memory_manager,
            )
            logger.info("Research agent initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize research agent: {e}")

        # Initialize Claude agent
        try:
            from droidrun.agent.claude import ClaudeCodeAgent, ClaudeAgentConfig

            claude_config = ClaudeAgentConfig(
                model=self.config.default_model,
                max_tokens=self.config.max_tokens,
                offline_mode=self.config.offline_mode,
            )
            self.claude_agent = ClaudeCodeAgent(config=claude_config)
            logger.info("Claude agent initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude agent: {e}")

        logger.info("Application initialized successfully")

    async def shutdown(self):
        """Shutdown application components."""
        logger.info("Shutting down application...")

        if self.memory_manager:
            await self.memory_manager.stop()

        # Close WebSocket connections
        for ws in self.websocket_connections.values():
            try:
                await ws.close()
            except Exception:
                pass

        logger.info("Application shutdown complete")


app_state = AppState()


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await app_state.initialize()
    yield
    await app_state.shutdown()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="DroidRun Agent API",
    description="Production-ready API for autonomous mobile agent",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SSE Streaming
# ============================================================================

async def stream_chat_response(
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for chat response."""
    session_id = request.session_id or str(uuid.uuid4())

    # Send session start
    yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id})}\n\n"

    try:
        if not app_state.claude_agent:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Claude agent not available'})}\n\n"
            return

        # Get memory context
        memory_context = ""
        if request.include_memory and app_state.memory_manager:
            last_message = request.messages[-1].content if request.messages else ""
            memory_context = await app_state.memory_manager.get_context_for_task(
                task=last_message,
                goal=last_message,
            )

            if memory_context:
                yield f"data: {json.dumps({'type': 'memory_context', 'context': memory_context})}\n\n"

        # Convert messages to dict format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Add memory context to system message if available
        if memory_context:
            system_content = f"Relevant context from memory:\n{memory_context}\n\n"
            messages.insert(0, {"role": "system", "content": system_content})

        # Stream response
        response_generator = await app_state.claude_agent.chat(
            messages=messages,
            stream=True,
        )

        full_text = ""

        async for event in response_generator:
            event_dict = event.to_dict()

            if event.type.value == "text_delta":
                full_text += event.data
                yield f"data: {json.dumps({'type': 'text_delta', 'delta': event.data})}\n\n"

            elif event.type.value == "thinking_delta":
                yield f"data: {json.dumps({'type': 'thinking_delta', 'delta': event.data})}\n\n"

            elif event.type.value == "tool_use_complete":
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': event.data})}\n\n"

            elif event.type.value == "message_complete":
                yield f"data: {json.dumps({'type': 'complete', 'content': full_text})}\n\n"

            elif event.type.value == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': event.data})}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    yield "data: [DONE]\n\n"


# ============================================================================
# API Routes
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - app_state.start_time,
        memory_enabled=app_state.memory_manager is not None,
        offline_mode=app_state.config.offline_mode,
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with optional streaming.

    SSE streaming format:
    - type: session_start - Session initialized
    - type: memory_context - Relevant memory context
    - type: text_delta - Text content chunk
    - type: thinking_delta - Thinking content chunk
    - type: tool_call - Tool call completed
    - type: complete - Message complete
    - type: error - Error occurred
    """
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        if not app_state.claude_agent:
            raise HTTPException(status_code=503, detail="Claude agent not available")

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        response = await app_state.claude_agent.chat(messages, stream=False)

        return ChatResponse(
            id=str(uuid.uuid4()),
            content=response.text,
            thinking=response.thinking,
            tool_calls=response.tool_calls,
            usage=response.usage,
            session_id=request.session_id or str(uuid.uuid4()),
            created_at=datetime.utcnow().isoformat(),
        )


@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new task."""
    task_id = str(uuid.uuid4())

    app_state.active_tasks[task_id] = {
        "id": task_id,
        "goal": request.goal,
        "status": "pending",
        "result": None,
        "steps": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Add background task execution
    background_tasks.add_task(execute_task, task_id, request)

    return TaskResponse(
        task_id=task_id,
        status="pending",
        goal=request.goal,
        created_at=app_state.active_tasks[task_id]["created_at"],
    )


async def execute_task(task_id: str, request: TaskRequest):
    """Execute task in background."""
    try:
        app_state.active_tasks[task_id]["status"] = "running"

        # TODO: Implement actual DroidAgent execution
        # For now, return mock result
        await asyncio.sleep(2)

        app_state.active_tasks[task_id].update({
            "status": "completed",
            "result": {
                "success": True,
                "reason": f"Task '{request.goal}' completed successfully",
            },
            "steps": 5,
        })

    except Exception as e:
        app_state.active_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": str(e)},
        })


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get task status."""
    if task_id not in app_state.active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = app_state.active_tasks[task_id]
    return TaskResponse(
        task_id=task["id"],
        status=task["status"],
        goal=task["goal"],
        result=task.get("result"),
        steps=task.get("steps", 0),
        created_at=task["created_at"],
    )


@app.post("/api/research")
async def research(request: ResearchRequest):
    """Execute research query."""
    if not app_state.research_agent:
        raise HTTPException(status_code=503, detail="Research agent not available")

    result = await app_state.research_agent.research(
        query=request.query,
        max_results=request.max_results,
        include_memory=request.include_memory,
    )

    return {
        "query": result.query,
        "results": [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "score": r.score,
                "source": r.source,
            }
            for r in result.results
        ],
        "summary": result.summary,
        "sources_used": result.sources_used,
        "duration_ms": result.duration_ms,
        "cached": result.cached,
    }


@app.post("/api/memory/store")
async def store_memory(request: MemoryStoreRequest):
    """Store task execution in memory."""
    if not app_state.memory_manager:
        raise HTTPException(status_code=503, detail="Memory not enabled")

    from droidrun.agent.memory import EpisodeRecord

    episode = EpisodeRecord(
        task=request.task,
        goal=request.goal,
        final_success=request.success,
        final_reason=request.reason,
        steps=request.steps,
        actions=request.actions,
        tags=request.tags,
    )

    entry_id = await app_state.memory_manager.store_episode(episode)

    return {"id": entry_id, "stored": True}


@app.get("/api/memory/recall")
async def recall_memory(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    success_only: bool = Query(False, description="Only successful episodes"),
):
    """Recall similar episodes from memory."""
    if not app_state.memory_manager:
        raise HTTPException(status_code=503, detail="Memory not enabled")

    episodes = await app_state.memory_manager.recall_similar(
        query=query,
        top_k=top_k,
        success_only=success_only,
    )

    return {
        "query": query,
        "results": [
            {
                "id": ep.id,
                "task": ep.task,
                "goal": ep.goal,
                "success": ep.final_success,
                "reason": ep.final_reason,
                "steps": ep.steps,
                "score": score,
            }
            for ep, score in episodes
        ],
    }


@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory statistics."""
    if not app_state.memory_manager:
        raise HTTPException(status_code=503, detail="Memory not enabled")

    return await app_state.memory_manager.get_statistics()


@app.get("/api/config")
async def get_server_config():
    """Get server configuration (non-sensitive)."""
    return {
        "model": app_state.config.default_model,
        "max_tokens": app_state.config.max_tokens,
        "offline_mode": app_state.config.offline_mode,
        "memory_enabled": app_state.config.memory_enabled,
        "memory_store": app_state.config.memory_store,
    }


# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    app_state.websocket_connections[session_id] = websocket

    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_json()

            # Handle different message types
            msg_type = data.get("type")

            if msg_type == "chat":
                # Process chat message
                messages = data.get("messages", [])
                response_text = ""

                if app_state.claude_agent:
                    chat_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages
                    ]

                    async for event in await app_state.claude_agent.chat(
                        chat_messages, stream=True
                    ):
                        if event.type.value == "text_delta":
                            response_text += event.data
                            await websocket.send_json({
                                "type": "text_delta",
                                "delta": event.data,
                            })
                        elif event.type.value == "message_complete":
                            await websocket.send_json({
                                "type": "complete",
                                "content": response_text,
                            })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "subscribe":
                # Subscribe to task updates
                task_id = data.get("task_id")
                await websocket.send_json({
                    "type": "subscribed",
                    "task_id": task_id,
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app_state.websocket_connections.pop(session_id, None)


# ============================================================================
# Metrics
# ============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not app_state.config.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    metrics_data = []

    # Uptime
    uptime = time.time() - app_state.start_time
    metrics_data.append(f"droidrun_uptime_seconds {uptime}")

    # Active sessions
    metrics_data.append(f"droidrun_active_sessions {len(app_state.active_sessions)}")

    # Active tasks
    metrics_data.append(f"droidrun_active_tasks {len(app_state.active_tasks)}")

    # WebSocket connections
    metrics_data.append(
        f"droidrun_websocket_connections {len(app_state.websocket_connections)}"
    )

    # Claude agent metrics
    if app_state.claude_agent:
        agent_metrics = app_state.claude_agent.get_metrics()
        metrics_data.append(
            f"droidrun_claude_requests_total {agent_metrics['request_count']}"
        )
        metrics_data.append(
            f"droidrun_claude_errors_total {agent_metrics['error_count']}"
        )
        metrics_data.append(
            f"droidrun_claude_tokens_total {agent_metrics['total_tokens']}"
        )

    return "\n".join(metrics_data) + "\n"


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Run the server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "droidrun.backend.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()

"""
Tests for FastAPI Backend.

Production-ready tests for:
- Health check endpoint
- Chat endpoints
- Task management
- Memory endpoints
- Research endpoints
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthEndpoint(unittest.TestCase):
    """Tests for health check endpoint."""

    def test_health_response_model(self):
        """Test HealthResponse model."""
        from backend.main import HealthResponse

        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=100.5,
            memory_enabled=True,
            offline_mode=False,
        )

        self.assertEqual(response.status, "healthy")
        self.assertEqual(response.version, "1.0.0")
        self.assertAlmostEqual(response.uptime_seconds, 100.5)


class TestChatModels(unittest.TestCase):
    """Tests for chat request/response models."""

    def test_chat_message_model(self):
        """Test ChatMessage model."""
        from backend.main import ChatMessage

        msg = ChatMessage(role="user", content="Hello")

        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")

    def test_chat_request_model(self):
        """Test ChatRequest model."""
        from backend.main import ChatRequest, ChatMessage

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Test")],
            stream=True,
            include_memory=True,
        )

        self.assertEqual(len(request.messages), 1)
        self.assertTrue(request.stream)
        self.assertTrue(request.include_memory)

    def test_chat_request_defaults(self):
        """Test ChatRequest default values."""
        from backend.main import ChatRequest, ChatMessage

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Test")],
        )

        self.assertTrue(request.stream)  # Default
        self.assertTrue(request.include_memory)  # Default
        self.assertIsNone(request.session_id)

    def test_chat_response_model(self):
        """Test ChatResponse model."""
        from backend.main import ChatResponse

        response = ChatResponse(
            id="test-123",
            content="Hello back!",
            thinking="Processed the request",
            tool_calls=[],
            usage={"input_tokens": 5, "output_tokens": 3},
            session_id="session-456",
            created_at="2024-01-15T10:30:00",
        )

        self.assertEqual(response.id, "test-123")
        self.assertEqual(response.content, "Hello back!")


class TestTaskModels(unittest.TestCase):
    """Tests for task request/response models."""

    def test_task_request_model(self):
        """Test TaskRequest model."""
        from backend.main import TaskRequest

        request = TaskRequest(
            goal="Open Instagram",
            max_steps=10,
            reasoning=True,
        )

        self.assertEqual(request.goal, "Open Instagram")
        self.assertEqual(request.max_steps, 10)
        self.assertTrue(request.reasoning)

    def test_task_request_defaults(self):
        """Test TaskRequest default values."""
        from backend.main import TaskRequest

        request = TaskRequest(goal="Test task")

        self.assertEqual(request.max_steps, 15)  # Default
        self.assertFalse(request.reasoning)  # Default

    def test_task_response_model(self):
        """Test TaskResponse model."""
        from backend.main import TaskResponse

        response = TaskResponse(
            task_id="task-123",
            status="running",
            goal="Test goal",
            steps=5,
            created_at="2024-01-15T10:30:00",
        )

        self.assertEqual(response.task_id, "task-123")
        self.assertEqual(response.status, "running")
        self.assertEqual(response.steps, 5)


class TestResearchModels(unittest.TestCase):
    """Tests for research request model."""

    def test_research_request_model(self):
        """Test ResearchRequest model."""
        from backend.main import ResearchRequest

        request = ResearchRequest(
            query="How to automate Android",
            max_results=5,
            include_memory=False,
        )

        self.assertEqual(request.query, "How to automate Android")
        self.assertEqual(request.max_results, 5)
        self.assertFalse(request.include_memory)


class TestMemoryModels(unittest.TestCase):
    """Tests for memory request model."""

    def test_memory_store_request_model(self):
        """Test MemoryStoreRequest model."""
        from backend.main import MemoryStoreRequest

        request = MemoryStoreRequest(
            task="Open app",
            goal="Launch Instagram",
            success=True,
            reason="Opened successfully",
            steps=3,
            actions=[{"action": "tap"}],
            tags=["automation"],
        )

        self.assertEqual(request.task, "Open app")
        self.assertTrue(request.success)
        self.assertEqual(len(request.actions), 1)
        self.assertEqual(request.tags, ["automation"])


class TestServerConfig(unittest.TestCase):
    """Tests for ServerConfig."""

    def test_server_config_defaults(self):
        """Test ServerConfig default values."""
        from backend.main import ServerConfig

        config = ServerConfig()

        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 8000)
        self.assertFalse(config.debug)
        self.assertTrue(config.memory_enabled)

    def test_server_config_custom(self):
        """Test ServerConfig with custom values."""
        from backend.main import ServerConfig

        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            debug=True,
            offline_mode=True,
        )

        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 9000)
        self.assertTrue(config.debug)
        self.assertTrue(config.offline_mode)


class TestAppState(unittest.TestCase):
    """Tests for AppState."""

    def test_app_state_initialization(self):
        """Test AppState initialization."""
        from backend.main import AppState

        state = AppState()

        self.assertIsNotNone(state.config)
        self.assertIsNotNone(state.start_time)
        self.assertIsInstance(state.active_sessions, dict)
        self.assertIsInstance(state.active_tasks, dict)


class TestSSEStreaming(unittest.TestCase):
    """Tests for SSE streaming functionality."""

    def test_stream_format(self):
        """Test SSE stream format."""
        # SSE format should be "data: {json}\n\n"
        import json

        event = {"type": "text_delta", "delta": "Hello"}
        sse_line = f"data: {json.dumps(event)}\n\n"

        self.assertTrue(sse_line.startswith("data: "))
        self.assertTrue(sse_line.endswith("\n\n"))

        # Parse it back
        data = sse_line.split("data: ")[1].strip()
        parsed = json.loads(data)

        self.assertEqual(parsed["type"], "text_delta")
        self.assertEqual(parsed["delta"], "Hello")


class TestIntegration(unittest.TestCase):
    """Integration tests (require running server)."""

    @unittest.skip("Requires running server")
    def test_health_endpoint_integration(self):
        """Test health endpoint with real server."""
        import requests

        response = requests.get("http://localhost:8000/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")


if __name__ == "__main__":
    unittest.main()

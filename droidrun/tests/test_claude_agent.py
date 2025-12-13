"""
Tests for Claude Code Agent.

Production-ready tests for:
- ClaudeCodeAgent
- Streaming handler
- Offline mode
- Retry logic
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

# Add claude module path directly to bypass llama_index dependency
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_parent_dir, 'droidrun', 'agent', 'claude'))


class TestStreamEventType(unittest.TestCase):
    """Tests for StreamEventType enum."""

    def test_event_types_exist(self):
        """Test that all required event types exist."""
        from streaming import StreamEventType

        required_types = [
            "TEXT_START",
            "TEXT_DELTA",
            "TEXT_COMPLETE",
            "TOOL_USE_START",
            "TOOL_USE_DELTA",
            "TOOL_USE_COMPLETE",
            "MESSAGE_START",
            "MESSAGE_COMPLETE",
            "ERROR",
            "RETRY",
        ]

        for event_type in required_types:
            self.assertTrue(hasattr(StreamEventType, event_type))


class TestStreamEvent(unittest.TestCase):
    """Tests for StreamEvent dataclass."""

    def test_stream_event_creation(self):
        """Test StreamEvent creation."""
        from streaming import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data="Hello",
            metadata={"key": "value"},
        )

        self.assertEqual(event.type, StreamEventType.TEXT_DELTA)
        self.assertEqual(event.data, "Hello")
        self.assertEqual(event.metadata["key"], "value")
        self.assertIsInstance(event.timestamp, float)

    def test_stream_event_to_dict(self):
        """Test StreamEvent serialization."""
        from streaming import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            data={"text": "Done"},
        )

        data = event.to_dict()

        self.assertEqual(data["type"], "message_complete")
        self.assertEqual(data["data"]["text"], "Done")
        self.assertIn("timestamp", data)


class TestStreamingHandler(unittest.TestCase):
    """Tests for StreamingHandler."""

    def test_handler_initialization(self):
        """Test StreamingHandler initialization."""
        from streaming import StreamingHandler

        handler = StreamingHandler(buffer_size=50)

        self.assertEqual(handler.buffer_size, 50)
        self.assertEqual(handler.text, "")
        self.assertEqual(handler.thinking, "")
        self.assertEqual(len(handler.tool_calls), 0)

    def test_handler_text_accumulation(self):
        """Test text accumulation in handler."""
        from streaming import StreamingHandler

        handler = StreamingHandler()
        handler.start()

        handler.on_text_start()
        handler.on_text_delta("Hello ")
        handler.on_text_delta("World")
        handler.on_text_complete()

        self.assertEqual(handler.text, "Hello World")

    def test_handler_thinking_accumulation(self):
        """Test thinking accumulation in handler."""
        from streaming import StreamingHandler

        handler = StreamingHandler()
        handler.start()

        handler.on_thinking_start()
        handler.on_thinking_delta("Let me think...")
        handler.on_thinking_complete()

        self.assertEqual(handler.thinking, "Let me think...")

    def test_handler_tool_calls(self):
        """Test tool call handling."""
        from streaming import StreamingHandler

        handler = StreamingHandler()
        handler.start()

        handler.on_tool_use_start("read_file", "tool-1")
        handler.on_tool_use_delta("tool-1", '{"path":')
        handler.on_tool_use_delta("tool-1", '"/test"}')
        handler.on_tool_use_complete("tool-1", "read_file", {"path": "/test"})

        self.assertEqual(len(handler.tool_calls), 1)
        self.assertEqual(handler.tool_calls[0]["name"], "read_file")

    def test_handler_callback(self):
        """Test event callback."""
        from streaming import StreamingHandler, StreamEventType

        events_received = []

        def on_event(event):
            events_received.append(event)

        handler = StreamingHandler(on_event=on_event)
        handler.start()
        handler.on_text_delta("Test")

        self.assertGreater(len(events_received), 0)

    def test_handler_complete_returns_metrics(self):
        """Test that complete() returns metrics."""
        from streaming import StreamingHandler

        handler = StreamingHandler(emit_latency_reports=True)
        handler.start()
        handler.on_text_delta("Test")

        result = handler.complete()

        self.assertIn("text", result)
        self.assertIn("metrics", result)
        self.assertIn("total_duration_ms", result["metrics"])

    def test_handler_events_buffer(self):
        """Test events buffer."""
        from streaming import StreamingHandler

        handler = StreamingHandler(buffer_size=10)
        handler.start()

        for i in range(20):
            handler.on_text_delta(f"chunk{i}")

        # Buffer should be limited
        self.assertLessEqual(len(handler.events), 10)


class TestClaudeAgentConfig(unittest.TestCase):
    """Tests for ClaudeAgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from claude_agent import ClaudeAgentConfig

        config = ClaudeAgentConfig()

        self.assertIn("claude", config.model.lower())
        self.assertEqual(config.max_retries, 3)
        self.assertTrue(config.stream)
        self.assertFalse(config.offline_mode)

    def test_custom_config(self):
        """Test custom configuration."""
        from claude_agent import ClaudeAgentConfig

        config = ClaudeAgentConfig(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            temperature=0.5,
            offline_mode=True,
        )

        self.assertEqual(config.model, "claude-opus-4-20250514")
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.temperature, 0.5)
        self.assertTrue(config.offline_mode)


class TestClaudeResponse(unittest.TestCase):
    """Tests for ClaudeResponse."""

    def test_response_creation(self):
        """Test ClaudeResponse creation."""
        from claude_agent import ClaudeResponse

        response = ClaudeResponse(
            text="Hello, I'm Claude!",
            thinking="Processing the request...",
            tool_calls=[{"name": "test", "input": {}}],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 20},
            metrics={"latency_ms": 500},
        )

        self.assertEqual(response.text, "Hello, I'm Claude!")
        self.assertEqual(response.thinking, "Processing the request...")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.usage["output_tokens"], 20)


class TestClaudeCodeAgent(unittest.TestCase):
    """Tests for ClaudeCodeAgent."""

    def test_offline_mode_returns_mock(self):
        """Test that offline mode returns mock response."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(offline_mode=True)
        agent = ClaudeCodeAgent(config=config)

        response = asyncio.run(agent.chat([
            {"role": "user", "content": "Hello"}
        ]))

        self.assertIn("OFFLINE MODE", response.text)
        self.assertEqual(response.usage["input_tokens"], 0)

    def test_get_metrics(self):
        """Test metrics retrieval."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(offline_mode=True)
        agent = ClaudeCodeAgent(config=config)

        metrics = agent.get_metrics()

        self.assertIn("request_count", metrics)
        self.assertIn("error_count", metrics)
        self.assertIn("model", metrics)

    def test_reset_metrics(self):
        """Test metrics reset."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(offline_mode=True)
        agent = ClaudeCodeAgent(config=config)

        # Make some requests
        asyncio.run(agent.chat([{"role": "user", "content": "Test"}]))

        agent.reset_metrics()

        metrics = agent.get_metrics()
        self.assertEqual(metrics["request_count"], 0)

    def test_complete_convenience_method(self):
        """Test complete() convenience method."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(offline_mode=True)
        agent = ClaudeCodeAgent(config=config)

        response = asyncio.run(agent.complete("Tell me a joke"))

        self.assertIsInstance(response.text, str)

    def test_system_prompt_building(self):
        """Test system prompt building with CLAUDE.md."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(
            offline_mode=True,
            system_prompt="Custom system prompt",
            safe_mode=True,
        )
        agent = ClaudeCodeAgent(config=config)

        system_prompt = agent._build_system_prompt()

        self.assertIn("Custom system prompt", system_prompt)
        self.assertIn("Safety Guidelines", system_prompt)

    def test_tool_filtering_blocked(self):
        """Test tool filtering with blocked list."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(
            offline_mode=True,
            blocked_tools=["dangerous_tool"],
        )
        agent = ClaudeCodeAgent(config=config)

        tools = [
            {"name": "safe_tool", "description": "Safe"},
            {"name": "dangerous_tool", "description": "Dangerous"},
        ]

        filtered = agent._filter_tools(tools)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["name"], "safe_tool")

    def test_tool_filtering_allowed(self):
        """Test tool filtering with allowed list."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(
            offline_mode=True,
            allowed_tools=["allowed_tool"],
        )
        agent = ClaudeCodeAgent(config=config)

        tools = [
            {"name": "allowed_tool", "description": "Allowed"},
            {"name": "other_tool", "description": "Other"},
        ]

        filtered = agent._filter_tools(tools)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["name"], "allowed_tool")


class TestRetryLogic(unittest.TestCase):
    """Tests for retry logic."""

    def test_retry_delay_calculation(self):
        """Test exponential backoff calculation."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(
            offline_mode=True,
            initial_retry_delay=1.0,
            retry_multiplier=2.0,
            max_retry_delay=30.0,
        )
        agent = ClaudeCodeAgent(config=config)

        self.assertEqual(agent._get_retry_delay(0), 1.0)
        self.assertEqual(agent._get_retry_delay(1), 2.0)
        self.assertEqual(agent._get_retry_delay(2), 4.0)
        self.assertEqual(agent._get_retry_delay(10), 30.0)  # Capped at max

    def test_should_retry_rate_limit(self):
        """Test should_retry for rate limit errors."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(offline_mode=True)
        agent = ClaudeCodeAgent(config=config)

        # Use a class name with underscores to match retry_on_errors patterns
        class rate_limit_error(Exception):
            pass

        self.assertTrue(agent._should_retry(rate_limit_error("rate limit exceeded")))

    def test_should_not_retry_auth_error(self):
        """Test should_retry for auth errors."""
        from claude_agent import ClaudeCodeAgent, ClaudeAgentConfig

        config = ClaudeAgentConfig(offline_mode=True)
        agent = ClaudeCodeAgent(config=config)

        class AuthenticationError(Exception):
            pass

        self.assertFalse(agent._should_retry(AuthenticationError("invalid api key")))


if __name__ == "__main__":
    unittest.main()

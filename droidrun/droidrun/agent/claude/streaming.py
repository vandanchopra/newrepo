"""
Streaming Handler for Claude Responses.

Production-ready streaming with event types and callbacks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
import asyncio
import logging
import time

logger = logging.getLogger("droidrun.claude")


class StreamEventType(Enum):
    """Types of streaming events."""

    # Content events
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"

    # Tool events
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_COMPLETE = "tool_use_complete"

    # Message events
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_COMPLETE = "message_complete"

    # Control events
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_COMPLETE = "thinking_complete"

    # Error events
    ERROR = "error"
    RETRY = "retry"

    # Metrics events
    LATENCY_REPORT = "latency_report"


@dataclass
class StreamEvent:
    """Single streaming event."""

    type: StreamEventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class StreamingHandler:
    """
    Handler for streaming Claude responses.

    Features:
    - Event-based streaming with typed events
    - Callback support for UI updates
    - Buffering and batching
    - Latency tracking
    """

    def __init__(
        self,
        on_event: Optional[Callable[[StreamEvent], None]] = None,
        buffer_size: int = 100,
        emit_latency_reports: bool = True,
    ):
        self.on_event = on_event
        self.buffer_size = buffer_size
        self.emit_latency_reports = emit_latency_reports

        self._buffer: List[StreamEvent] = []
        self._complete_text: str = ""
        self._complete_thinking: str = ""
        self._tool_calls: List[Dict[str, Any]] = []
        self._start_time: float = 0
        self._first_token_time: float = 0
        self._tokens_received: int = 0

    def start(self):
        """Start tracking a new stream."""
        self._buffer.clear()
        self._complete_text = ""
        self._complete_thinking = ""
        self._tool_calls = []
        self._start_time = time.time()
        self._first_token_time = 0
        self._tokens_received = 0

        self._emit(StreamEvent(type=StreamEventType.MESSAGE_START))

    def _emit(self, event: StreamEvent):
        """Emit an event to callback and buffer."""
        self._buffer.append(event)

        # Trim buffer if needed
        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer[-self.buffer_size:]

        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error(f"Error in stream callback: {e}")

    def on_text_start(self):
        """Handle start of text content."""
        self._emit(StreamEvent(type=StreamEventType.TEXT_START))

    def on_text_delta(self, delta: str):
        """Handle text content delta."""
        if not self._first_token_time and delta.strip():
            self._first_token_time = time.time()

        self._complete_text += delta
        self._tokens_received += 1

        self._emit(StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data=delta,
            metadata={"total_length": len(self._complete_text)},
        ))

    def on_text_complete(self):
        """Handle text content complete."""
        self._emit(StreamEvent(
            type=StreamEventType.TEXT_COMPLETE,
            data=self._complete_text,
        ))

    def on_thinking_start(self):
        """Handle start of thinking content."""
        self._emit(StreamEvent(type=StreamEventType.THINKING_START))

    def on_thinking_delta(self, delta: str):
        """Handle thinking content delta."""
        self._complete_thinking += delta

        self._emit(StreamEvent(
            type=StreamEventType.THINKING_DELTA,
            data=delta,
            metadata={"total_length": len(self._complete_thinking)},
        ))

    def on_thinking_complete(self):
        """Handle thinking content complete."""
        self._emit(StreamEvent(
            type=StreamEventType.THINKING_COMPLETE,
            data=self._complete_thinking,
        ))

    def on_tool_use_start(self, tool_name: str, tool_id: str):
        """Handle start of tool use."""
        self._emit(StreamEvent(
            type=StreamEventType.TOOL_USE_START,
            data={"name": tool_name, "id": tool_id},
        ))

    def on_tool_use_delta(self, tool_id: str, delta: str):
        """Handle tool use input delta."""
        self._emit(StreamEvent(
            type=StreamEventType.TOOL_USE_DELTA,
            data={"id": tool_id, "delta": delta},
        ))

    def on_tool_use_complete(self, tool_id: str, tool_name: str, input_json: Dict):
        """Handle tool use complete."""
        tool_call = {
            "id": tool_id,
            "name": tool_name,
            "input": input_json,
        }
        self._tool_calls.append(tool_call)

        self._emit(StreamEvent(
            type=StreamEventType.TOOL_USE_COMPLETE,
            data=tool_call,
        ))

    def on_error(self, error: Exception, retry_count: int = 0):
        """Handle error event."""
        self._emit(StreamEvent(
            type=StreamEventType.ERROR,
            data=str(error),
            metadata={"retry_count": retry_count, "error_type": type(error).__name__},
        ))

    def on_retry(self, attempt: int, max_attempts: int, delay: float):
        """Handle retry event."""
        self._emit(StreamEvent(
            type=StreamEventType.RETRY,
            data={
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay": delay,
            },
        ))

    def complete(self) -> Dict[str, Any]:
        """Complete the stream and return final data."""
        end_time = time.time()
        total_duration = end_time - self._start_time
        ttft = (self._first_token_time - self._start_time) if self._first_token_time else 0

        # Emit latency report
        if self.emit_latency_reports:
            self._emit(StreamEvent(
                type=StreamEventType.LATENCY_REPORT,
                data={
                    "total_duration_ms": total_duration * 1000,
                    "time_to_first_token_ms": ttft * 1000,
                    "tokens_received": self._tokens_received,
                    "tokens_per_second": self._tokens_received / total_duration if total_duration > 0 else 0,
                },
            ))

        self._emit(StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            data={
                "text": self._complete_text,
                "thinking": self._complete_thinking,
                "tool_calls": self._tool_calls,
            },
        ))

        return {
            "text": self._complete_text,
            "thinking": self._complete_thinking,
            "tool_calls": self._tool_calls,
            "metrics": {
                "total_duration_ms": total_duration * 1000,
                "time_to_first_token_ms": ttft * 1000,
                "tokens_received": self._tokens_received,
            },
        }

    @property
    def text(self) -> str:
        """Get accumulated text."""
        return self._complete_text

    @property
    def thinking(self) -> str:
        """Get accumulated thinking."""
        return self._complete_thinking

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """Get tool calls."""
        return self._tool_calls.copy()

    @property
    def events(self) -> List[StreamEvent]:
        """Get buffered events."""
        return self._buffer.copy()


class AsyncStreamingHandler(StreamingHandler):
    """Async version of StreamingHandler with queue support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._closed = False

    def _emit(self, event: StreamEvent):
        """Emit event to queue and callback."""
        super()._emit(event)
        if not self._closed:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Stream event queue full, dropping event")

    async def events_async(self) -> AsyncIterator[StreamEvent]:
        """Async iterator for streaming events."""
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=30.0)
                yield event
                if event.type == StreamEventType.MESSAGE_COMPLETE:
                    break
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def close(self):
        """Close the stream."""
        self._closed = True

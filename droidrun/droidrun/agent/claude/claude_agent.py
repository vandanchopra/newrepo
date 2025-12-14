"""
Claude Code Agent.

Production-ready Claude integration with:
- Streaming responses
- Resilient retries with exponential backoff
- Offline safety mode
- CLAUDE.md guidance support
- Prometheus metrics integration
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
import asyncio
import json
import logging
import os
import time
from pathlib import Path

try:
    from .streaming import (
        AsyncStreamingHandler,
        StreamEvent,
        StreamEventType,
        StreamingHandler,
    )
except ImportError:
    # Fallback for standalone testing
    from streaming import (
        AsyncStreamingHandler,
        StreamEvent,
        StreamEventType,
        StreamingHandler,
    )

logger = logging.getLogger("droidrun.claude")


@dataclass
class ClaudeAgentConfig:
    """Configuration for Claude Code Agent."""

    # Execution Mode: "api" (Anthropic API) or "cli" (Claude Code CLI)
    execution_mode: str = "cli"  # Default to CLI mode (uses subscription)
    cli_path: Optional[str] = None  # Path to claude executable
    cli_timeout: float = 300.0  # CLI timeout in seconds

    # API Configuration (only used if execution_mode="api")
    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.7

    # Retry Configuration
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    retry_multiplier: float = 2.0
    retry_on_errors: List[str] = field(default_factory=lambda: [
        "overloaded_error",
        "api_error",
        "rate_limit_error",
        "timeout",
    ])

    # Safety Configuration
    offline_mode: bool = False
    safe_mode: bool = True
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None

    # CLAUDE.md Configuration
    claude_md_path: Optional[str] = None
    auto_load_claude_md: bool = True

    # Streaming Configuration
    stream: bool = True
    buffer_size: int = 100

    # Metrics Configuration
    enable_metrics: bool = True
    metrics_prefix: str = "droidrun_claude"

    # Context Configuration
    system_prompt: Optional[str] = None
    max_context_tokens: int = 100000


@dataclass
class ClaudeResponse:
    """Response from Claude agent."""

    text: str
    thinking: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_response: Optional[Any] = None


class ClaudeCodeAgent:
    """
    Production-ready Claude Code Agent.

    Features:
    - Streaming responses with event callbacks
    - Resilient retries with exponential backoff
    - Offline safety mode (returns mock responses when offline)
    - CLAUDE.md guidance support
    - Prometheus metrics integration
    - Tool use support
    """

    def __init__(
        self,
        config: Optional[ClaudeAgentConfig] = None,
        on_stream_event: Optional[Callable[[StreamEvent], None]] = None,
    ):
        self.config = config or ClaudeAgentConfig()
        self.on_stream_event = on_stream_event

        # Initialize API client (for API mode)
        self._client = None
        self._async_client = None
        self._claude_md_content: Optional[str] = None

        # Initialize CLI wrapper (for CLI mode)
        self._cli_wrapper = None

        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._total_tokens = 0
        self._total_latency = 0.0

        self._initialize()

    def _initialize(self):
        """Initialize the Claude client based on execution mode."""
        # CLI Mode: Use Claude Code CLI (subscription-based)
        if self.config.execution_mode == "cli":
            self._initialize_cli()
            return

        # API Mode: Use Anthropic API directly
        self._initialize_api()

    def _initialize_cli(self):
        """Initialize Claude Code CLI wrapper."""
        try:
            from .cli_wrapper import ClaudeCodeCLI, CLIConfig

            cli_config = CLIConfig(
                cli_path=self.config.cli_path or "claude",
                timeout_seconds=self.config.cli_timeout,
            )
            self._cli_wrapper = ClaudeCodeCLI(config=cli_config)

            if self._cli_wrapper.is_available():
                version = self._cli_wrapper.get_version()
                logger.info(f"🚀 Claude Code CLI initialized (version: {version})")
                logger.info("Using your Claude Code subscription - no API key needed!")
            else:
                logger.warning(
                    "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code\n"
                    "Or set execution_mode='api' to use Anthropic API instead."
                )
        except ImportError as e:
            logger.error(f"Failed to import CLI wrapper: {e}")

        # Load CLAUDE.md if configured
        if self.config.auto_load_claude_md:
            self._load_claude_md()

    def _initialize_api(self):
        """Initialize Anthropic API client."""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key and not self.config.offline_mode:
            logger.warning(
                "No ANTHROPIC_API_KEY found. Set offline_mode=True for mock responses, "
                "or set execution_mode='cli' to use Claude Code CLI with your subscription."
            )

        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
                logger.info(f"Claude API client initialized with model: {self.config.model}")
            except ImportError:
                logger.error("anthropic package not installed. Install: pip install anthropic")
                raise

        # Load CLAUDE.md if configured
        if self.config.auto_load_claude_md:
            self._load_claude_md()

    def _load_claude_md(self):
        """Load CLAUDE.md guidance file."""
        paths_to_try = [
            self.config.claude_md_path,
            "CLAUDE.md",
            ".claude/CLAUDE.md",
            os.path.expanduser("~/.claude/CLAUDE.md"),
        ]

        for path in paths_to_try:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        self._claude_md_content = f.read()
                    logger.info(f"Loaded CLAUDE.md from: {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load CLAUDE.md from {path}: {e}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt including CLAUDE.md guidance."""
        parts = []

        # Add CLAUDE.md content if available
        if self._claude_md_content:
            parts.append("# Project Guidelines (CLAUDE.md)\n")
            parts.append(self._claude_md_content)
            parts.append("\n---\n")

        # Add custom system prompt
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)

        # Add safety instructions if in safe mode
        if self.config.safe_mode:
            parts.append(
                "\n\n# Safety Guidelines\n"
                "- Always verify actions before execution\n"
                "- Never modify or delete data without explicit confirmation\n"
                "- Report any potentially dangerous operations\n"
                "- Prefer read-only operations when possible\n"
            )

        return "\n".join(parts) if parts else ""

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = self.config.initial_retry_delay * (
            self.config.retry_multiplier ** attempt
        )
        return min(delay, self.config.max_retry_delay)

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        error_type = type(error).__name__.lower()
        error_str = str(error).lower()

        for retry_error in self.config.retry_on_errors:
            if retry_error.lower() in error_type or retry_error.lower() in error_str:
                return True

        return False

    def _create_mock_response(self, messages: List[Dict]) -> ClaudeResponse:
        """Create a mock response for offline mode."""
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "")

        return ClaudeResponse(
            text=(
                f"[OFFLINE MODE] I received your message about: "
                f"{content[:100]}... "
                "This is a mock response because the agent is running in offline mode. "
                "No actual API call was made."
            ),
            thinking="[OFFLINE] Mock thinking process",
            stop_reason="end_turn",
            usage={"input_tokens": 0, "output_tokens": 0},
            metrics={"latency_ms": 0},
        )

    async def _cli_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
    ) -> ClaudeResponse:
        """
        Execute chat via Claude Code CLI.

        Uses the locally installed `claude` CLI command with your subscription.
        """
        start_time = time.time()

        if not self._cli_wrapper:
            return ClaudeResponse(
                text="[CLI ERROR] Claude Code CLI wrapper not initialized",
                stop_reason="error",
                usage={},
                metrics={},
            )

        # Build prompt from messages
        prompt_parts = []

        # Add system prompt if configured
        system_prompt = self._build_system_prompt()
        if system_prompt:
            prompt_parts.append(f"System context:\n{system_prompt}\n\n---\n")

        # Add conversation history
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            elif role == "system":
                prompt_parts.append(f"System: {content}\n")

        # Add tool context if provided
        if tools:
            tool_names = [t.get("name", "") for t in tools]
            prompt_parts.append(f"\nAvailable tools: {', '.join(tool_names)}\n")

        full_prompt = "\n".join(prompt_parts)

        # Execute via CLI
        try:
            self._request_count += 1

            cli_response = await self._cli_wrapper.execute(
                prompt=full_prompt,
                timeout=self.config.cli_timeout,
            )

            latency = (time.time() - start_time) * 1000
            self._total_latency += latency

            if cli_response.success:
                logger.debug(f"CLI response received in {latency:.0f}ms")

                return ClaudeResponse(
                    text=cli_response.text,
                    thinking="",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage={
                        "input_tokens": len(full_prompt) // 4,  # Rough estimate
                        "output_tokens": len(cli_response.text) // 4,
                    },
                    metrics={
                        "latency_ms": latency,
                        "cli_duration_s": cli_response.duration_seconds,
                    },
                    raw_response=cli_response.raw_output,
                )
            else:
                self._error_count += 1
                logger.error(f"CLI execution failed: {cli_response.error}")

                return ClaudeResponse(
                    text=f"[CLI ERROR] {cli_response.error}",
                    stop_reason="error",
                    usage={},
                    metrics={"latency_ms": latency},
                )

        except Exception as e:
            self._error_count += 1
            latency = (time.time() - start_time) * 1000
            logger.error(f"CLI chat error: {e}")

            return ClaudeResponse(
                text=f"[CLI ERROR] {str(e)}",
                stop_reason="error",
                usage={},
                metrics={"latency_ms": latency},
            )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: Optional[bool] = None,
    ) -> Union[ClaudeResponse, AsyncIterator[StreamEvent]]:
        """
        Send a chat request to Claude.

        Args:
            messages: List of message dicts with "role" and "content"
            tools: Optional list of tool definitions
            stream: Override streaming setting

        Returns:
            ClaudeResponse or AsyncIterator of StreamEvents if streaming
        """
        should_stream = stream if stream is not None else self.config.stream

        # CLI Mode: Use Claude Code CLI
        if self.config.execution_mode == "cli" and self._cli_wrapper:
            return await self._cli_chat(messages, tools)

        # Handle offline mode
        if self.config.offline_mode or not self._async_client:
            logger.info("Running in offline mode, returning mock response")
            return self._create_mock_response(messages)

        # Build request
        system_prompt = self._build_system_prompt()

        request_kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            # Filter tools based on config
            filtered_tools = self._filter_tools(tools)
            if filtered_tools:
                request_kwargs["tools"] = filtered_tools

        if should_stream:
            return self._stream_chat(request_kwargs)
        else:
            return await self._single_chat(request_kwargs)

    def _filter_tools(self, tools: List[Dict]) -> List[Dict]:
        """Filter tools based on allowed/blocked lists."""
        filtered = []

        for tool in tools:
            tool_name = tool.get("name", "")

            # Check blocked list
            if self.config.blocked_tools:
                if tool_name in self.config.blocked_tools:
                    continue

            # Check allowed list
            if self.config.allowed_tools:
                if tool_name not in self.config.allowed_tools:
                    continue

            filtered.append(tool)

        return filtered

    async def _single_chat(
        self,
        request_kwargs: Dict[str, Any],
    ) -> ClaudeResponse:
        """Non-streaming chat request with retries."""
        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                self._request_count += 1

                response = await self._async_client.messages.create(**request_kwargs)

                # Parse response
                text = ""
                thinking = ""
                tool_calls = []

                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                latency = (time.time() - start_time) * 1000
                self._total_latency += latency
                self._total_tokens += response.usage.input_tokens + response.usage.output_tokens

                return ClaudeResponse(
                    text=text,
                    thinking=thinking,
                    tool_calls=tool_calls,
                    stop_reason=response.stop_reason,
                    usage={
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    metrics={"latency_ms": latency},
                    raw_response=response,
                )

            except Exception as e:
                last_error = e
                self._error_count += 1

                if attempt < self.config.max_retries and self._should_retry(e):
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Claude request failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        raise last_error or RuntimeError("Unknown error in Claude request")

    async def _stream_chat(
        self,
        request_kwargs: Dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """Streaming chat request with retries."""
        handler = AsyncStreamingHandler(
            on_event=self.on_stream_event,
            buffer_size=self.config.buffer_size,
        )

        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                self._request_count += 1
                handler.start()

                async with self._async_client.messages.stream(**request_kwargs) as stream:
                    current_tool_id = None
                    current_tool_name = None
                    current_tool_input = ""

                    async for event in stream:
                        if hasattr(event, "type"):
                            if event.type == "content_block_start":
                                block = event.content_block
                                if hasattr(block, "type"):
                                    if block.type == "text":
                                        handler.on_text_start()
                                    elif block.type == "tool_use":
                                        current_tool_id = block.id
                                        current_tool_name = block.name
                                        current_tool_input = ""
                                        handler.on_tool_use_start(block.name, block.id)

                            elif event.type == "content_block_delta":
                                delta = event.delta
                                if hasattr(delta, "type"):
                                    if delta.type == "text_delta":
                                        handler.on_text_delta(delta.text)
                                    elif delta.type == "input_json_delta":
                                        current_tool_input += delta.partial_json
                                        handler.on_tool_use_delta(
                                            current_tool_id, delta.partial_json
                                        )
                                    elif delta.type == "thinking_delta":
                                        handler.on_thinking_delta(delta.thinking)

                            elif event.type == "content_block_stop":
                                if current_tool_id and current_tool_name:
                                    try:
                                        input_json = json.loads(current_tool_input) if current_tool_input else {}
                                    except json.JSONDecodeError:
                                        input_json = {"raw": current_tool_input}

                                    handler.on_tool_use_complete(
                                        current_tool_id,
                                        current_tool_name,
                                        input_json,
                                    )
                                    current_tool_id = None
                                    current_tool_name = None
                                    current_tool_input = ""
                                else:
                                    handler.on_text_complete()

                # Complete and yield final event
                final_data = handler.complete()

                # Update metrics
                self._total_latency += final_data["metrics"]["total_duration_ms"]

                # Yield all events
                for event in handler.events:
                    yield event

                return

            except Exception as e:
                last_error = e
                self._error_count += 1
                handler.on_error(e, attempt)

                if attempt < self.config.max_retries and self._should_retry(e):
                    delay = self._get_retry_delay(attempt)
                    handler.on_retry(attempt + 1, self.config.max_retries, delay)
                    logger.warning(
                        f"Claude stream failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        # If we get here, all retries failed
        yield StreamEvent(
            type=StreamEventType.ERROR,
            data=str(last_error),
            metadata={"final": True},
        )

    async def complete(
        self,
        prompt: str,
        **kwargs,
    ) -> ClaudeResponse:
        """
        Simple completion interface.

        Args:
            prompt: The prompt text
            **kwargs: Additional arguments for chat()

        Returns:
            ClaudeResponse
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, stream=False, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics for monitoring."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "total_tokens": self._total_tokens,
            "avg_latency_ms": self._total_latency / max(1, self._request_count),
            "model": self.config.model,
            "offline_mode": self.config.offline_mode,
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self._request_count = 0
        self._error_count = 0
        self._total_tokens = 0
        self._total_latency = 0.0


# Prometheus metrics integration
class PrometheusMetrics:
    """Prometheus metrics for Claude agent."""

    def __init__(self, prefix: str = "droidrun_claude"):
        self.prefix = prefix
        self._initialized = False
        self._request_counter = None
        self._error_counter = None
        self._latency_histogram = None
        self._token_counter = None

    def initialize(self):
        """Initialize Prometheus metrics."""
        if self._initialized:
            return

        try:
            from prometheus_client import Counter, Histogram

            self._request_counter = Counter(
                f"{self.prefix}_requests_total",
                "Total Claude API requests",
                ["model", "status"],
            )

            self._error_counter = Counter(
                f"{self.prefix}_errors_total",
                "Total Claude API errors",
                ["model", "error_type"],
            )

            self._latency_histogram = Histogram(
                f"{self.prefix}_request_latency_seconds",
                "Claude API request latency",
                ["model"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            )

            self._token_counter = Counter(
                f"{self.prefix}_tokens_total",
                "Total tokens used",
                ["model", "type"],
            )

            self._initialized = True
            logger.info("Prometheus metrics initialized")

        except ImportError:
            logger.warning(
                "prometheus_client not installed. "
                "Install: pip install prometheus-client"
            )

    def record_request(self, model: str, status: str, latency: float):
        """Record a request."""
        if not self._initialized:
            return

        self._request_counter.labels(model=model, status=status).inc()
        self._latency_histogram.labels(model=model).observe(latency)

    def record_error(self, model: str, error_type: str):
        """Record an error."""
        if not self._initialized:
            return

        self._error_counter.labels(model=model, error_type=error_type).inc()

    def record_tokens(self, model: str, input_tokens: int, output_tokens: int):
        """Record token usage."""
        if not self._initialized:
            return

        self._token_counter.labels(model=model, type="input").inc(input_tokens)
        self._token_counter.labels(model=model, type="output").inc(output_tokens)


# Global metrics instance
prometheus_metrics = PrometheusMetrics()


class ClaudeCodeCLI:
    """
    Integration with Claude Code CLI for actual coding tasks.

    Uses the `claude` CLI command to execute prompts and handle
    file operations through Claude's native tool use.
    """

    def __init__(
        self,
        working_directory: Optional[str] = None,
        timeout_seconds: float = 300.0,
        print_output: bool = False,
    ):
        """
        Initialize Claude Code CLI wrapper.

        Args:
            working_directory: Directory to run claude commands in
            timeout_seconds: Maximum time to wait for response
            print_output: Whether to print output in real-time
        """
        self.working_directory = working_directory or os.getcwd()
        self.timeout_seconds = timeout_seconds
        self.print_output = print_output
        self._cli_available = None

    async def check_cli_available(self) -> bool:
        """
        Check if Claude CLI is installed and available.

        Returns:
            True if claude CLI is available
        """
        if self._cli_available is not None:
            return self._cli_available

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)

            self._cli_available = proc.returncode == 0
            if self._cli_available:
                version = stdout.decode().strip()
                logger.info(f"Claude CLI available: {version}")
            return self._cli_available

        except (FileNotFoundError, asyncio.TimeoutError):
            self._cli_available = False
            logger.warning("Claude CLI not available")
            return False

    async def execute(
        self,
        prompt: str,
        continue_conversation: bool = False,
        allowed_tools: Optional[List[str]] = None,
        json_output: bool = True,
    ) -> ClaudeResponse:
        """
        Execute a prompt using Claude Code CLI.

        Args:
            prompt: The prompt to send
            continue_conversation: Whether to continue previous conversation
            allowed_tools: List of tools to allow
            json_output: Whether to request JSON output

        Returns:
            ClaudeResponse with result
        """
        if not await self.check_cli_available():
            return ClaudeResponse(
                text="[ERROR] Claude CLI is not available. Install with: npm install -g @anthropic-ai/claude-code",
                stop_reason="error",
            )

        try:
            # Build command
            cmd = ["claude"]

            if continue_conversation:
                cmd.append("--continue")

            if json_output:
                cmd.extend(["--output-format", "json"])

            if allowed_tools:
                cmd.extend(["--allowedTools", ",".join(allowed_tools)])

            # Add the prompt via --print flag for non-interactive execution
            cmd.extend(["--print", prompt])

            logger.info(f"Executing Claude CLI: {' '.join(cmd[:3])}...")

            # Execute
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )

            output = stdout.decode()
            error = stderr.decode()

            if self.print_output:
                print(output)

            if proc.returncode != 0:
                return ClaudeResponse(
                    text=f"[ERROR] Claude CLI failed: {error or output}",
                    stop_reason="error",
                )

            # Parse response
            return self._parse_cli_output(output, json_output)

        except asyncio.TimeoutError:
            return ClaudeResponse(
                text=f"[ERROR] Claude CLI timed out after {self.timeout_seconds}s",
                stop_reason="timeout",
            )

        except Exception as e:
            return ClaudeResponse(
                text=f"[ERROR] Claude CLI execution failed: {e}",
                stop_reason="error",
            )

    def _parse_cli_output(self, output: str, is_json: bool) -> ClaudeResponse:
        """Parse output from Claude CLI."""
        if is_json:
            try:
                data = json.loads(output)

                # Extract text from response
                text = ""
                tool_calls = []
                thinking = ""

                if isinstance(data, dict):
                    # Handle JSON output format
                    text = data.get("result", data.get("response", str(data)))
                    tool_calls = data.get("tool_calls", [])
                    thinking = data.get("thinking", "")

                elif isinstance(data, list):
                    # Handle streaming JSON blocks
                    for item in data:
                        if item.get("type") == "text":
                            text += item.get("text", "")
                        elif item.get("type") == "tool_use":
                            tool_calls.append(item)

                return ClaudeResponse(
                    text=text,
                    thinking=thinking,
                    tool_calls=tool_calls,
                    stop_reason="end_turn",
                    raw_response=data,
                )

            except json.JSONDecodeError:
                # Fall back to text output
                return ClaudeResponse(
                    text=output,
                    stop_reason="end_turn",
                )
        else:
            return ClaudeResponse(
                text=output,
                stop_reason="end_turn",
            )

    async def edit_file(
        self,
        file_path: str,
        instruction: str,
    ) -> ClaudeResponse:
        """
        Edit a file using Claude.

        Args:
            file_path: Path to the file to edit
            instruction: What changes to make

        Returns:
            ClaudeResponse with result
        """
        prompt = f"Edit the file {file_path}: {instruction}"
        return await self.execute(prompt, allowed_tools=["Edit", "Read"])

    async def write_file(
        self,
        file_path: str,
        description: str,
    ) -> ClaudeResponse:
        """
        Write a new file using Claude.

        Args:
            file_path: Path for the new file
            description: What the file should contain

        Returns:
            ClaudeResponse with result
        """
        prompt = f"Create a new file at {file_path} with the following content: {description}"
        return await self.execute(prompt, allowed_tools=["Write"])

    async def run_command(
        self,
        command: str,
        description: str = "",
    ) -> ClaudeResponse:
        """
        Run a shell command using Claude.

        Args:
            command: Command to run
            description: Context for why

        Returns:
            ClaudeResponse with result
        """
        prompt = f"Run this command: {command}"
        if description:
            prompt += f" ({description})"
        return await self.execute(prompt, allowed_tools=["Bash"])

    async def code_review(
        self,
        file_path: str,
    ) -> ClaudeResponse:
        """
        Review code in a file.

        Args:
            file_path: Path to the file to review

        Returns:
            ClaudeResponse with review
        """
        prompt = f"Review the code in {file_path} and suggest improvements."
        return await self.execute(prompt, allowed_tools=["Read"])


def create_claude_cli(
    working_directory: Optional[str] = None,
    timeout_seconds: float = 300.0,
) -> ClaudeCodeCLI:
    """
    Create a Claude Code CLI instance.

    Args:
        working_directory: Directory to run claude commands in
        timeout_seconds: Maximum time to wait

    Returns:
        ClaudeCodeCLI instance
    """
    return ClaudeCodeCLI(
        working_directory=working_directory,
        timeout_seconds=timeout_seconds,
    )

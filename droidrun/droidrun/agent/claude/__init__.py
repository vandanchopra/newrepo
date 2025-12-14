"""
Claude Code Agent for DroidRun.

Production-ready Claude integration with:
- CLI Mode: Uses Claude Code CLI (subscription-based, no API key needed)
- API Mode: Direct Anthropic API calls
- Streaming responses
- Resilient retries with exponential backoff
- Offline safety mode
- CLAUDE.md guidance support
- Prometheus metrics
"""

from .claude_agent import (
    ClaudeCodeAgent,
    ClaudeAgentConfig,
    ClaudeResponse,
)
from .streaming import (
    StreamingHandler,
    StreamEvent,
    StreamEventType,
)
from .cli_wrapper import (
    ClaudeCodeCLI,
    CLIConfig,
    CLIResponse,
    create_cli_wrapper,
    check_cli_available,
)

__all__ = [
    # Main agent
    "ClaudeCodeAgent",
    "ClaudeAgentConfig",
    "ClaudeResponse",
    # Streaming
    "StreamingHandler",
    "StreamEvent",
    "StreamEventType",
    # CLI wrapper
    "ClaudeCodeCLI",
    "CLIConfig",
    "CLIResponse",
    "create_cli_wrapper",
    "check_cli_available",
]

"""
Claude Code CLI Wrapper.

Executes Claude Code CLI commands via subprocess, allowing use of
Claude Code subscription instead of direct Anthropic API.

This enables:
- Using Claude Code with your subscription (no API key needed)
- Running via Termux on Android
- Full Claude Code capabilities (tools, file access, etc.)
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("droidrun.claude.cli")


@dataclass
class CLIConfig:
    """Configuration for Claude Code CLI execution."""

    # CLI executable
    cli_path: str = "claude"  # Assumes 'claude' is in PATH

    # Execution mode
    use_json_output: bool = True
    timeout_seconds: float = 300.0  # 5 minutes default

    # Working directory for Claude Code
    working_dir: Optional[str] = None

    # Model override (if supported by CLI)
    model: Optional[str] = None

    # Output mode
    print_output: bool = False  # Print raw CLI output for debugging

    # Termux-specific settings
    termux_mode: bool = False
    termux_prefix: str = ""  # e.g., "termux-" for Termux-specific commands


@dataclass
class CLIResponse:
    """Response from Claude Code CLI execution."""

    text: str
    success: bool
    exit_code: int
    raw_output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaudeCodeCLI:
    """
    Wrapper for Claude Code CLI execution.

    Usage:
        cli = ClaudeCodeCLI()

        # Check if CLI is available
        if cli.is_available():
            # Execute a prompt
            response = await cli.execute("What is 2+2?")
            print(response.text)

            # Execute with conversation context
            response = await cli.execute(
                "Continue from where we left off",
                conversation_id="my-session"
            )
    """

    def __init__(self, config: Optional[CLIConfig] = None):
        self.config = config or CLIConfig()
        self._cli_path = self._find_cli()
        self._conversation_files: Dict[str, Path] = {}

    def _find_cli(self) -> Optional[str]:
        """Find the Claude Code CLI executable."""
        # Check configured path first
        if self.config.cli_path and shutil.which(self.config.cli_path):
            return self.config.cli_path

        # Common locations
        paths_to_check = [
            "claude",
            "/usr/local/bin/claude",
            "/usr/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
            os.path.expanduser("~/bin/claude"),
            # Termux paths
            "/data/data/com.termux/files/usr/bin/claude",
            os.path.expanduser("~/../usr/bin/claude"),
        ]

        # Also check npm global installs
        npm_paths = [
            os.path.expanduser("~/.npm-global/bin/claude"),
            "/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude",
        ]
        paths_to_check.extend(npm_paths)

        for path in paths_to_check:
            if shutil.which(path) or (os.path.isfile(path) and os.access(path, os.X_OK)):
                logger.info(f"Found Claude Code CLI at: {path}")
                return path

        return None

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available."""
        if not self._cli_path:
            return False

        try:
            result = subprocess.run(
                [self._cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"Claude Code CLI version: {result.stdout.strip()}")
                return True
        except Exception as e:
            logger.debug(f"CLI check failed: {e}")

        return False

    def get_version(self) -> Optional[str]:
        """Get Claude Code CLI version."""
        if not self._cli_path:
            return None

        try:
            result = subprocess.run(
                [self._cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    async def execute(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> CLIResponse:
        """
        Execute a prompt via Claude Code CLI.

        Args:
            prompt: The prompt to send to Claude
            conversation_id: Optional ID for conversation continuity
            system_prompt: Optional system prompt override
            working_dir: Optional working directory for execution
            timeout: Optional timeout override in seconds

        Returns:
            CLIResponse with the result
        """
        import time
        start_time = time.time()

        if not self._cli_path:
            return CLIResponse(
                text="",
                success=False,
                exit_code=-1,
                raw_output="",
                error="Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
                duration_seconds=0,
            )

        # Build command
        cmd = [self._cli_path]

        # Add prompt via --print flag for non-interactive mode
        cmd.extend(["--print", prompt])

        # Add output format if supported
        if self.config.use_json_output:
            cmd.append("--output-format")
            cmd.append("json")

        # Prepare environment
        env = os.environ.copy()

        # Set working directory
        cwd = working_dir or self.config.working_dir or os.getcwd()

        # Execute
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            timeout_val = timeout or self.config.timeout_seconds
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_val,
            )

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            duration = time.time() - start_time

            if self.config.print_output:
                logger.info(f"CLI stdout: {stdout_text}")
                if stderr_text:
                    logger.warning(f"CLI stderr: {stderr_text}")

            # Parse response
            if process.returncode == 0:
                response_text = self._parse_output(stdout_text)
                return CLIResponse(
                    text=response_text,
                    success=True,
                    exit_code=process.returncode,
                    raw_output=stdout_text,
                    duration_seconds=duration,
                    metadata={"stderr": stderr_text} if stderr_text else {},
                )
            else:
                return CLIResponse(
                    text="",
                    success=False,
                    exit_code=process.returncode,
                    raw_output=stdout_text,
                    error=stderr_text or f"CLI exited with code {process.returncode}",
                    duration_seconds=duration,
                )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return CLIResponse(
                text="",
                success=False,
                exit_code=-1,
                raw_output="",
                error=f"CLI execution timed out after {timeout_val}s",
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return CLIResponse(
                text="",
                success=False,
                exit_code=-1,
                raw_output="",
                error=f"CLI execution failed: {str(e)}",
                duration_seconds=duration,
            )

    def _parse_output(self, output: str) -> str:
        """Parse CLI output, handling JSON or plain text."""
        output = output.strip()

        if not output:
            return ""

        # Try JSON parsing first
        if self.config.use_json_output and output.startswith("{"):
            try:
                data = json.loads(output)
                # Extract text from various possible JSON structures
                if isinstance(data, dict):
                    if "result" in data:
                        return str(data["result"])
                    if "text" in data:
                        return str(data["text"])
                    if "content" in data:
                        return str(data["content"])
                    if "response" in data:
                        return str(data["response"])
                return output
            except json.JSONDecodeError:
                pass

        return output

    async def execute_with_files(
        self,
        prompt: str,
        files: List[str],
        working_dir: Optional[str] = None,
    ) -> CLIResponse:
        """
        Execute prompt with file context.

        Args:
            prompt: The prompt to send
            files: List of file paths to include as context
            working_dir: Working directory for execution
        """
        # Build prompt with file references
        file_context = []
        for file_path in files:
            if os.path.exists(file_path):
                file_context.append(f"File: {file_path}")

        if file_context:
            full_prompt = f"{prompt}\n\nRelevant files:\n" + "\n".join(file_context)
        else:
            full_prompt = prompt

        return await self.execute(full_prompt, working_dir=working_dir)


def create_cli_wrapper(
    cli_path: Optional[str] = None,
    timeout: float = 300.0,
    termux_mode: bool = False,
) -> ClaudeCodeCLI:
    """
    Create a Claude Code CLI wrapper with common settings.

    Args:
        cli_path: Path to claude executable (auto-detected if None)
        timeout: Execution timeout in seconds
        termux_mode: Enable Termux-specific settings

    Returns:
        Configured ClaudeCodeCLI instance
    """
    config = CLIConfig(
        cli_path=cli_path or "claude",
        timeout_seconds=timeout,
        termux_mode=termux_mode,
    )
    return ClaudeCodeCLI(config=config)


# Quick check function
def check_cli_available() -> Tuple[bool, Optional[str]]:
    """
    Check if Claude Code CLI is available.

    Returns:
        Tuple of (is_available, version_or_error)
    """
    cli = ClaudeCodeCLI()
    if cli.is_available():
        return True, cli.get_version()
    return False, "Claude Code CLI not found"

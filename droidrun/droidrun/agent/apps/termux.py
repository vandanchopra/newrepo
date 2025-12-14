"""
Termux Integration for DroidRun.

Provides specialized actions for Termux terminal emulator:
- Execute shell commands
- Read command output
- Send input to running processes
- Parse terminal content
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import register_app_handler
except ImportError:
    # Standalone testing - create dummy decorator
    def register_app_handler(package_name):
        def decorator(cls):
            return cls
        return decorator

logger = logging.getLogger("droidrun.apps.termux")

# Termux package name
TERMUX_PACKAGE = "com.termux"


@dataclass
class CommandResult:
    """Result of a terminal command execution."""
    command: str
    output: str
    exit_code: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0


@register_app_handler(TERMUX_PACKAGE)
class TermuxHandler:
    """
    Handler for Termux terminal app.

    Provides high-level actions for executing commands,
    reading output, and interacting with the terminal.
    """

    def __init__(self, tools_instance: Any = None):
        """
        Initialize Termux handler.

        Args:
            tools_instance: AdbTools instance for device interaction
        """
        self.tools = tools_instance
        self._last_screen_text = ""
        self._command_history: List[CommandResult] = []

    async def ensure_termux_open(self) -> bool:
        """
        Ensure Termux is open and ready.

        Returns:
            True if Termux is ready, False otherwise
        """
        if not self.tools:
            logger.warning("No tools instance available")
            return False

        try:
            # Launch Termux
            await self.tools.launch_app(TERMUX_PACKAGE)
            await asyncio.sleep(1.0)  # Wait for app to load

            # Verify we're in Termux
            ui_state = await self.tools.get_ui_state()
            current_package = ui_state.get("package_name", "")

            if current_package == TERMUX_PACKAGE:
                logger.info("✅ Termux is ready")
                return True
            else:
                logger.warning(f"Not in Termux, current package: {current_package}")
                return False

        except Exception as e:
            logger.error(f"Failed to open Termux: {e}")
            return False

    async def execute_command(
        self,
        command: str,
        wait_for_output: bool = True,
        timeout_seconds: float = 30.0,
    ) -> CommandResult:
        """
        Execute a shell command in Termux.

        Args:
            command: Command to execute
            wait_for_output: Whether to wait for command completion
            timeout_seconds: Maximum time to wait for output

        Returns:
            CommandResult with output and status
        """
        start_time = time.time()
        result = CommandResult(command=command, output="")

        if not self.tools:
            result.success = False
            result.error = "No tools instance available"
            return result

        try:
            # Ensure Termux is open
            if not await self.ensure_termux_open():
                result.success = False
                result.error = "Failed to open Termux"
                return result

            # Clear any existing input and type the command
            await self.tools.input_text(command)
            await asyncio.sleep(0.2)

            # Press Enter to execute
            await self.tools.press_key("ENTER")

            if wait_for_output:
                # Wait for command to complete
                output = await self._wait_for_output(
                    timeout_seconds=timeout_seconds,
                    command=command,
                )
                result.output = output

                # Try to detect exit code from output
                result.exit_code = self._extract_exit_code(output)
                result.success = result.exit_code == 0 if result.exit_code is not None else True

            result.duration_ms = (time.time() - start_time) * 1000

            # Store in history
            self._command_history.append(result)

            logger.info(f"Command executed: {command[:50]}... (success={result.success})")
            return result

        except asyncio.TimeoutError:
            result.success = False
            result.error = f"Command timed out after {timeout_seconds}s"
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_ms = (time.time() - start_time) * 1000
            return result

    async def _wait_for_output(
        self,
        timeout_seconds: float,
        command: str,
        poll_interval: float = 0.5,
    ) -> str:
        """
        Wait for command output to appear in terminal.

        Args:
            timeout_seconds: Maximum time to wait
            command: The command that was executed (for context)
            poll_interval: How often to check for new output

        Returns:
            Terminal output text
        """
        start_time = time.time()
        previous_text = ""
        stable_count = 0
        required_stable = 2  # Number of stable reads to consider output complete

        while time.time() - start_time < timeout_seconds:
            # Get current screen text
            current_text = await self.read_terminal_output()

            if current_text == previous_text:
                stable_count += 1
                if stable_count >= required_stable:
                    # Output is stable, command likely complete
                    return current_text
            else:
                stable_count = 0
                previous_text = current_text

            await asyncio.sleep(poll_interval)

        # Timeout - return whatever we have
        return previous_text

    async def read_terminal_output(self) -> str:
        """
        Read current terminal output.

        Returns:
            Text content visible on the terminal screen
        """
        if not self.tools:
            return ""

        try:
            # Get UI state which includes text content
            ui_state = await self.tools.get_ui_state()

            # Extract text from accessibility tree
            text_content = []

            def extract_text(node):
                if isinstance(node, dict):
                    text = node.get("text", "")
                    if text:
                        text_content.append(text)
                    for child in node.get("children", []):
                        extract_text(child)
                elif isinstance(node, list):
                    for item in node:
                        extract_text(item)

            a11y_tree = ui_state.get("a11y_tree", [])
            extract_text(a11y_tree)

            output = "\n".join(text_content)
            self._last_screen_text = output
            return output

        except Exception as e:
            logger.error(f"Failed to read terminal: {e}")
            return self._last_screen_text

    async def send_input(self, text: str) -> bool:
        """
        Send text input to the terminal.

        Args:
            text: Text to send (does NOT press Enter)

        Returns:
            True if successful
        """
        if not self.tools:
            return False

        try:
            await self.tools.input_text(text)
            return True
        except Exception as e:
            logger.error(f"Failed to send input: {e}")
            return False

    async def send_ctrl_c(self) -> bool:
        """Send Ctrl+C to interrupt current process."""
        if not self.tools:
            return False

        try:
            # Send Ctrl+C key combination
            # Using ADB shell input keyevent
            await self.tools.press_key("CTRL_C")
            return True
        except Exception as e:
            logger.error(f"Failed to send Ctrl+C: {e}")
            return False

    async def clear_terminal(self) -> bool:
        """Clear the terminal screen."""
        result = await self.execute_command("clear", wait_for_output=False)
        return result.success

    def _extract_exit_code(self, output: str) -> Optional[int]:
        """
        Try to extract exit code from terminal output.

        Args:
            output: Terminal output text

        Returns:
            Exit code if detected, None otherwise
        """
        # Look for common exit code patterns
        patterns = [
            r"exit code[:\s]+(\d+)",
            r"returned (\d+)",
            r"\[(\d+)\]$",  # Shell prompt often shows exit code
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def get_command_history(self) -> List[CommandResult]:
        """Get command execution history."""
        return self._command_history.copy()

    # High-level convenience methods

    async def run_python_script(self, script_content: str) -> CommandResult:
        """
        Run a Python script in Termux.

        Args:
            script_content: Python code to execute

        Returns:
            CommandResult with output
        """
        # Create a temporary script
        escaped = script_content.replace("'", "'\"'\"'")
        create_cmd = f"echo '{escaped}' > /tmp/script.py"
        await self.execute_command(create_cmd, wait_for_output=True)

        # Run the script
        return await self.execute_command("python /tmp/script.py")

    async def install_package(self, package: str) -> CommandResult:
        """
        Install a package using pkg.

        Args:
            package: Package name to install

        Returns:
            CommandResult with installation output
        """
        return await self.execute_command(
            f"pkg install -y {package}",
            timeout_seconds=120.0,  # Package installation can take time
        )

    async def git_clone(self, repo_url: str, directory: str = None) -> CommandResult:
        """
        Clone a git repository.

        Args:
            repo_url: Repository URL
            directory: Target directory (optional)

        Returns:
            CommandResult with output
        """
        cmd = f"git clone {repo_url}"
        if directory:
            cmd += f" {directory}"

        return await self.execute_command(cmd, timeout_seconds=120.0)

    async def run_tests(self, test_command: str = "pytest") -> CommandResult:
        """
        Run tests in the current directory.

        Args:
            test_command: Test command to run

        Returns:
            CommandResult with test output
        """
        return await self.execute_command(
            test_command,
            timeout_seconds=300.0,  # Tests can take a while
        )


# Convenience function for creating handler
def create_termux_handler(tools_instance: Any = None) -> TermuxHandler:
    """Create a Termux handler instance."""
    return TermuxHandler(tools_instance=tools_instance)

"""
Coding Workflow Orchestration for DroidRun.

Provides high-level coding operations that combine:
- Termux for shell commands and environment
- Claude Code CLI for writing/editing code
- Automated testing and error fixing
- Git operations for version control

This enables autonomous coding tasks like:
- "Build a web scraper for news sites"
- "Create a REST API with Flask"
- "Fix the failing tests in this project"
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("droidrun.workflows.coding")


class CodingTaskStatus(Enum):
    """Status of a coding task."""
    PENDING = "pending"
    SETTING_UP = "setting_up"
    CODING = "coding"
    TESTING = "testing"
    FIXING = "fixing"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CodingResult:
    """Result of a coding task."""
    success: bool
    task_description: str
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    tests_passed: bool = False
    test_output: str = ""
    commits_made: List[str] = field(default_factory=list)
    error: Optional[str] = None
    steps_taken: List[str] = field(default_factory=list)


@dataclass
class CodingWorkflowConfig:
    """Configuration for coding workflow."""

    # Environment
    working_directory: str = "/data/data/com.termux/files/home"
    python_path: str = "python"
    pip_path: str = "pip"
    git_enabled: bool = True

    # Testing
    test_command: str = "pytest -v"
    max_fix_attempts: int = 3

    # Claude Code
    claude_timeout_seconds: float = 120.0

    # Git
    auto_commit: bool = True
    commit_message_prefix: str = "🤖 Auto: "


class CodingWorkflow:
    """
    Orchestrates coding tasks end-to-end.

    Example workflow for "Build a web scraper":
    1. Setup: Create project directory, virtual env
    2. Code: Use Claude Code to write scraper
    3. Test: Run the scraper, check output
    4. Fix: If errors, ask Claude Code to fix
    5. Commit: Git commit the working code
    """

    def __init__(
        self,
        tools_instance: Any = None,
        config: CodingWorkflowConfig = None,
    ):
        """
        Initialize coding workflow.

        Args:
            tools_instance: AdbTools instance for device control
            config: Workflow configuration
        """
        self.tools = tools_instance
        self.config = config or CodingWorkflowConfig()
        self._status = CodingTaskStatus.PENDING
        self._current_project_dir = ""
        self._termux_handler = None
        self._claude_cli = None

    async def _ensure_termux(self):
        """Ensure Termux handler is available."""
        if self._termux_handler is None:
            from droidrun.agent.apps.termux import TermuxHandler
            self._termux_handler = TermuxHandler(tools_instance=self.tools)
        return self._termux_handler

    async def _ensure_claude_cli(self):
        """Ensure Claude CLI is available."""
        if self._claude_cli is None:
            try:
                from droidrun.agent.claude.claude_agent import create_claude_cli
                self._claude_cli = create_claude_cli(
                    working_directory=self._current_project_dir or self.config.working_directory
                )
            except ImportError:
                logger.warning("Claude Code CLI not available")
                return None
        return self._claude_cli

    async def setup_project(
        self,
        project_name: str,
        project_type: str = "python",
        dependencies: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Set up a new coding project.

        Args:
            project_name: Name of the project
            project_type: Type (python, node, etc.)
            dependencies: List of dependencies to install

        Returns:
            Setup result with project path
        """
        self._status = CodingTaskStatus.SETTING_UP
        termux = await self._ensure_termux()

        result = {
            "success": False,
            "project_path": "",
            "steps": [],
        }

        try:
            # Create project directory
            project_dir = f"{self.config.working_directory}/{project_name}"
            self._current_project_dir = project_dir

            await termux.execute_command(f"mkdir -p {project_dir}")
            result["steps"].append(f"Created directory: {project_dir}")

            if project_type == "python":
                # Create virtual environment
                await termux.execute_command(
                    f"cd {project_dir} && python -m venv venv",
                    timeout_seconds=60.0,
                )
                result["steps"].append("Created Python virtual environment")

                # Install dependencies
                if dependencies:
                    deps_str = " ".join(dependencies)
                    await termux.execute_command(
                        f"cd {project_dir} && source venv/bin/activate && pip install {deps_str}",
                        timeout_seconds=120.0,
                    )
                    result["steps"].append(f"Installed dependencies: {deps_str}")

            elif project_type == "node":
                # Initialize Node project
                await termux.execute_command(
                    f"cd {project_dir} && npm init -y",
                    timeout_seconds=30.0,
                )
                result["steps"].append("Initialized Node.js project")

                if dependencies:
                    deps_str = " ".join(dependencies)
                    await termux.execute_command(
                        f"cd {project_dir} && npm install {deps_str}",
                        timeout_seconds=120.0,
                    )
                    result["steps"].append(f"Installed dependencies: {deps_str}")

            # Initialize git if enabled
            if self.config.git_enabled:
                await termux.execute_command(f"cd {project_dir} && git init")
                result["steps"].append("Initialized git repository")

            result["success"] = True
            result["project_path"] = project_dir
            logger.info(f"✅ Project setup complete: {project_dir}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Project setup failed: {e}")

        return result

    async def write_code(
        self,
        description: str,
        file_path: str = None,
    ) -> Dict[str, Any]:
        """
        Write code using Claude Code.

        Args:
            description: Description of what to write
            file_path: Optional specific file to create

        Returns:
            Result with created files
        """
        self._status = CodingTaskStatus.CODING
        cli = await self._ensure_claude_cli()

        result = {
            "success": False,
            "files_created": [],
            "response": "",
        }

        if cli is None:
            result["error"] = "Claude Code CLI not available"
            return result

        try:
            if file_path:
                response = await cli.write_file(file_path, description)
                result["files_created"].append(file_path)
            else:
                response = await cli.execute(description)

            result["success"] = True
            result["response"] = response.text
            logger.info(f"✅ Code written successfully")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Code writing failed: {e}")

        return result

    async def run_tests(
        self,
        test_command: str = None,
    ) -> Dict[str, Any]:
        """
        Run tests in the project.

        Args:
            test_command: Custom test command (default from config)

        Returns:
            Test results
        """
        self._status = CodingTaskStatus.TESTING
        termux = await self._ensure_termux()

        cmd = test_command or self.config.test_command

        result = {
            "success": False,
            "passed": False,
            "output": "",
            "error_count": 0,
        }

        try:
            # Run tests
            test_result = await termux.execute_command(
                f"cd {self._current_project_dir} && source venv/bin/activate && {cmd}",
                timeout_seconds=120.0,
            )

            result["output"] = test_result.output
            result["success"] = True

            # Check if tests passed
            output_lower = test_result.output.lower()
            if "passed" in output_lower and "failed" not in output_lower:
                result["passed"] = True
            elif "error" in output_lower or "failed" in output_lower:
                result["passed"] = False
                # Count errors/failures
                import re
                errors = re.findall(r"(\d+) (?:error|failed)", output_lower)
                if errors:
                    result["error_count"] = sum(int(e) for e in errors)

            logger.info(f"Tests {'passed' if result['passed'] else 'failed'}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Test execution failed: {e}")

        return result

    async def fix_errors(
        self,
        error_output: str,
    ) -> Dict[str, Any]:
        """
        Ask Claude Code to fix errors.

        Args:
            error_output: The error message/output to fix

        Returns:
            Fix result
        """
        self._status = CodingTaskStatus.FIXING
        cli = await self._ensure_claude_cli()

        result = {
            "success": False,
            "response": "",
            "files_modified": [],
        }

        if cli is None:
            result["error"] = "Claude Code CLI not available"
            return result

        try:
            prompt = f"""Fix the following errors in the project:

{error_output}

Analyze the errors, find the root cause, and fix them.
"""
            response = await cli.execute(prompt, continue_conversation=True)

            result["success"] = True
            result["response"] = response.text
            logger.info("✅ Fix attempt completed")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Fix failed: {e}")

        return result

    async def commit_changes(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Commit changes to git.

        Args:
            message: Commit message

        Returns:
            Commit result
        """
        self._status = CodingTaskStatus.COMMITTING
        termux = await self._ensure_termux()

        result = {
            "success": False,
            "commit_hash": "",
        }

        if not self.config.git_enabled:
            result["error"] = "Git not enabled"
            return result

        try:
            full_message = f"{self.config.commit_message_prefix}{message}"

            # Add all files
            await termux.execute_command(
                f"cd {self._current_project_dir} && git add -A"
            )

            # Commit
            commit_result = await termux.execute_command(
                f'cd {self._current_project_dir} && git commit -m "{full_message}"'
            )

            result["success"] = True
            result["output"] = commit_result.output
            logger.info(f"✅ Changes committed")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Commit failed: {e}")

        return result

    async def execute_coding_task(
        self,
        task_description: str,
        project_name: str = "auto_project",
        dependencies: List[str] = None,
    ) -> CodingResult:
        """
        Execute a complete coding task end-to-end.

        This is the main entry point for autonomous coding.

        Args:
            task_description: What to build/create
            project_name: Name for the project
            dependencies: Required dependencies

        Returns:
            CodingResult with full execution details
        """
        result = CodingResult(
            success=False,
            task_description=task_description,
        )

        try:
            # Step 1: Setup project
            logger.info(f"📁 Setting up project: {project_name}")
            setup = await self.setup_project(
                project_name=project_name,
                project_type="python",
                dependencies=dependencies or ["pytest"],
            )
            result.steps_taken.extend(setup.get("steps", []))

            if not setup["success"]:
                result.error = f"Setup failed: {setup.get('error')}"
                return result

            # Step 2: Write code
            logger.info(f"💻 Writing code for: {task_description}")
            code_result = await self.write_code(task_description)
            result.steps_taken.append("Wrote initial code")

            if not code_result["success"]:
                result.error = f"Coding failed: {code_result.get('error')}"
                return result

            result.files_created = code_result.get("files_created", [])

            # Step 3: Run tests
            logger.info("🧪 Running tests...")
            test_result = await self.run_tests()
            result.steps_taken.append("Ran tests")
            result.test_output = test_result.get("output", "")

            # Step 4: Fix errors if needed
            fix_attempts = 0
            while not test_result.get("passed", False) and fix_attempts < self.config.max_fix_attempts:
                fix_attempts += 1
                logger.info(f"🔧 Fixing errors (attempt {fix_attempts})...")

                fix_result = await self.fix_errors(test_result.get("output", ""))
                result.steps_taken.append(f"Fix attempt {fix_attempts}")

                if fix_result["success"]:
                    result.files_modified.extend(fix_result.get("files_modified", []))

                # Re-run tests
                test_result = await self.run_tests()
                result.test_output = test_result.get("output", "")

            result.tests_passed = test_result.get("passed", False)

            # Step 5: Commit if successful
            if result.tests_passed and self.config.auto_commit:
                logger.info("📦 Committing changes...")
                commit_result = await self.commit_changes(
                    f"Implemented: {task_description[:50]}"
                )
                result.steps_taken.append("Committed to git")
                if commit_result["success"]:
                    result.commits_made.append(commit_result.get("commit_hash", ""))

            result.success = result.tests_passed
            self._status = CodingTaskStatus.COMPLETED if result.success else CodingTaskStatus.FAILED

            logger.info(f"{'✅' if result.success else '❌'} Coding task {'completed' if result.success else 'failed'}")

        except Exception as e:
            result.error = str(e)
            result.success = False
            self._status = CodingTaskStatus.FAILED
            logger.error(f"Coding task failed: {e}")

        return result

    @property
    def status(self) -> CodingTaskStatus:
        """Get current workflow status."""
        return self._status


# Convenience function
def create_coding_workflow(
    tools_instance: Any = None,
    working_directory: str = None,
) -> CodingWorkflow:
    """
    Create a coding workflow instance.

    Args:
        tools_instance: AdbTools instance
        working_directory: Base directory for projects

    Returns:
        Configured CodingWorkflow
    """
    config = CodingWorkflowConfig()
    if working_directory:
        config.working_directory = working_directory

    return CodingWorkflow(tools_instance=tools_instance, config=config)

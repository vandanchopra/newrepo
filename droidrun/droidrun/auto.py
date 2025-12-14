#!/usr/bin/env python3
"""
DroidRun Auto - Single Command Autonomous Execution.

Run from Termux with one command:
    droidrun-auto "open calculator and compute 25 * 4"
    droidrun-auto "search google for weather today"
    droidrun-auto "open whatsapp and send hello to mom"

Features:
- Auto-detects Termux environment
- Uses Claude Code CLI (your subscription, no API key)
- Connects to localhost ADB when running on device
- Full memory system (Titans + ReMe)
- Fast execution mode
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("droidrun.auto")


def is_termux() -> bool:
    """Check if running in Termux environment."""
    return (
        os.path.exists("/data/data/com.termux") or
        "com.termux" in os.environ.get("PREFIX", "") or
        os.environ.get("TERMUX_VERSION") is not None
    )


def setup_termux_adb():
    """Setup ADB for Termux (localhost connection)."""
    # On Termux, ADB connects to the device itself via localhost
    os.environ.setdefault("ADB_HOST", "localhost")
    os.environ.setdefault("ADB_PORT", "5037")

    # Ensure ADB is running
    try:
        import subprocess
        # Try to start ADB server if not running
        subprocess.run(
            ["adb", "start-server"],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass  # ADB might already be running


def check_prerequisites() -> tuple[bool, list[str]]:
    """Check all prerequisites are met."""
    issues = []

    # Check Claude Code CLI
    import shutil
    if not shutil.which("claude"):
        issues.append("Claude Code CLI not found. Run: npm install -g @anthropic-ai/claude-code")

    # Check ADB
    if not shutil.which("adb"):
        issues.append("ADB not found. Install android-tools in Termux: pkg install android-tools")

    return len(issues) == 0, issues


async def run_autonomous(
    task: str,
    max_steps: int = 30,
    fast_mode: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run a task autonomously.

    Args:
        task: The task/goal to accomplish
        max_steps: Maximum steps to take
        fast_mode: Enable fast execution (less waiting)
        verbose: Enable verbose logging

    Returns:
        Result dict with success status and details
    """
    start_time = time.time()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"\n🚀 DroidRun Auto")
    print(f"📱 Task: {task}")
    print(f"⚡ Mode: {'Fast' if fast_mode else 'Standard'}")
    print("-" * 50)

    # Detect environment FIRST - before any heavy imports
    on_termux = is_termux()
    if on_termux:
        print("📍 Running on Termux (on-device)")
        print("📱 Using Claude Code CLI (no API key needed)")
        setup_termux_adb()

        # Check prerequisites
        ok, issues = check_prerequisites()
        if not ok:
            print("\n❌ Prerequisites not met:")
            for issue in issues:
                print(f"   - {issue}")
            return {"success": False, "error": "Prerequisites not met", "issues": issues}
        print("✅ Prerequisites OK")

        # Import ONLY the lightweight CLI wrapper - no DroidAgent, no LLMs
        sys.path.insert(0, str(Path(__file__).parent))
        from agent.claude.cli_wrapper import ClaudeCodeCLI, CLIConfig

        cli = ClaudeCodeCLI(CLIConfig(timeout_seconds=120))
        if not cli.is_available():
            print("❌ Claude Code CLI not available")
            return {"success": False, "error": "Claude Code CLI not available"}

        print(f"✅ Claude Code CLI ready (v{cli.get_version()})")
        print("-" * 50)

        # Go directly to lightweight mode - NO DroidAgent, NO LLM configs
        return await run_lightweight_mode(task, cli, max_steps)

    # Non-Termux path (desktop with API keys)
    print("📍 Running on external machine")

    # Check prerequisites
    ok, issues = check_prerequisites()
    if not ok:
        print("\n❌ Prerequisites not met:")
        for issue in issues:
            print(f"   - {issue}")
        return {"success": False, "error": "Prerequisites not met", "issues": issues}

    print("✅ Prerequisites OK")

    # Import DroidRun components (only on desktop)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from agent.claude.cli_wrapper import ClaudeCodeCLI, CLIConfig

        cli = ClaudeCodeCLI(CLIConfig(timeout_seconds=120))
        if not cli.is_available():
            print("❌ Claude Code CLI not available")
            return {"success": False, "error": "Claude Code CLI not available"}

        print(f"✅ Claude Code CLI ready (v{cli.get_version()})")

    except ImportError as e:
        print(f"⚠️  Import warning: {e}")

    # Try to use full DroidAgent if available (only on desktop with API keys)
    try:
        from droidrun.agent.droid import DroidAgent
        from droidrun.config_manager.config_manager import (
            DroidrunConfig,
            AgentConfig,
            DeviceConfig,
            MemoryConfig,
        )

        print("✅ Full DroidAgent available")

        # Build config optimized for speed
        config = DroidrunConfig(
            agent=AgentConfig(
                max_steps=max_steps,
                reasoning=True,  # Use Manager/Executor for better results
            ),
            device=DeviceConfig(
                serial=None,  # Auto-detect
                use_tcp=on_termux,  # Use TCP on Termux
            ),
            memory=MemoryConfig(
                enabled=True,
            ),
        )

        # Create and run agent
        print("\n▶️  Starting autonomous execution...")
        print("-" * 50)

        agent = DroidAgent(
            goal=task,
            config=config,
        )

        step_count = 0
        handler = agent.run()

        async for event in handler.stream_events():
            event_type = type(event).__name__

            if event_type == "ManagerPlanEvent":
                print(f"📋 Planning...")
            elif event_type == "ExecutorResultEvent":
                step_count += 1
                action = getattr(event, 'action', 'unknown')
                print(f"⚡ Step {step_count}: {action}")
            elif event_type == "ScreenshotEvent":
                print(f"📸 Screenshot taken")

        result = await handler

        duration = time.time() - start_time
        print("-" * 50)

        if result.success:
            print(f"✅ Task completed in {duration:.1f}s ({step_count} steps)")
            print(f"📝 Result: {result.reason}")
        else:
            print(f"❌ Task failed after {duration:.1f}s")
            print(f"📝 Reason: {result.reason}")

        return {
            "success": result.success,
            "reason": result.reason,
            "steps": step_count,
            "duration_seconds": duration,
        }

    except ImportError as e:
        # Fallback: Use lightweight Claude-only mode
        print(f"⚠️  Full agent not available: {e}")
        print("🔄 Using lightweight Claude Code mode...")

        return await run_lightweight_mode(task, cli, max_steps)


async def run_lightweight_mode(
    task: str,
    cli: "ClaudeCodeCLI",
    max_steps: int,
) -> dict:
    """
    Lightweight mode using only Claude Code CLI + basic ADB.

    This mode works even without llama_index installed.
    """
    import subprocess

    start_time = time.time()

    # Get initial screen state
    print("\n📱 Getting device state...")

    try:
        # Get UI hierarchy via ADB
        ui_dump = subprocess.run(
            ["adb", "shell", "uiautomator", "dump", "/dev/tty"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        screen_state = ui_dump.stdout[:2000] if ui_dump.returncode == 0 else "Unable to get UI state"
    except Exception as e:
        screen_state = f"Error getting UI state: {e}"

    # Build prompt for Claude
    prompt = f"""You are an Android automation agent. Execute this task step by step.

TASK: {task}

CURRENT SCREEN STATE:
{screen_state}

AVAILABLE ACTIONS (use adb shell commands):
- Tap: adb shell input tap X Y
- Type: adb shell input text "text"
- Swipe: adb shell input swipe X1 Y1 X2 Y2
- Back: adb shell input keyevent 4
- Home: adb shell input keyevent 3
- Open app: adb shell am start -n package/activity
- Get UI: adb shell uiautomator dump /dev/tty

Respond with the EXACT adb command to execute next, or "DONE" if task is complete.
Just the command, nothing else."""

    print("🤖 Asking Claude for action...")

    for step in range(max_steps):
        # Get action from Claude
        response = await cli.execute(prompt)

        if not response.success:
            print(f"❌ Claude error: {response.error}")
            break

        action = response.text.strip()

        if "DONE" in action.upper():
            duration = time.time() - start_time
            print(f"\n✅ Task completed in {duration:.1f}s ({step} steps)")
            return {
                "success": True,
                "reason": "Task completed",
                "steps": step,
                "duration_seconds": duration,
            }

        # Execute the ADB command
        if action.startswith("adb "):
            print(f"⚡ Step {step + 1}: {action[:60]}...")
            try:
                result = subprocess.run(
                    action.split(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    print(f"   ⚠️  Command failed: {result.stderr[:100]}")
            except Exception as e:
                print(f"   ⚠️  Execution error: {e}")

        # Get new screen state
        await asyncio.sleep(0.5)  # Brief wait for UI to update
        try:
            ui_dump = subprocess.run(
                ["adb", "shell", "uiautomator", "dump", "/dev/tty"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            screen_state = ui_dump.stdout[:2000] if ui_dump.returncode == 0 else "Unable to get UI state"
        except Exception:
            pass

        # Update prompt with new state
        prompt = f"""Continue the task. Previous action: {action}

TASK: {task}

CURRENT SCREEN STATE:
{screen_state}

What's the next adb command? Or "DONE" if complete."""

    duration = time.time() - start_time
    print(f"\n⚠️  Max steps reached after {duration:.1f}s")
    return {
        "success": False,
        "reason": "Max steps reached",
        "steps": max_steps,
        "duration_seconds": duration,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DroidRun Auto - Single Command Autonomous Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  droidrun-auto "open calculator"
  droidrun-auto "search google for python tutorials"
  droidrun-auto "open whatsapp and send hi to mom"
  droidrun-auto --fast "take a screenshot"
        """,
    )

    parser.add_argument(
        "task",
        help="The task to accomplish",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=30,
        help="Maximum steps (default: 30)",
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        default=True,
        help="Fast mode with minimal delays (default: on)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Run the task
    try:
        result = asyncio.run(
            run_autonomous(
                task=args.task,
                max_steps=args.steps,
                fast_mode=args.fast,
                verbose=args.verbose,
            )
        )

        # Exit with appropriate code
        sys.exit(0 if result.get("success") else 1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

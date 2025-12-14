"""
DroidRun Daemon - Continuous Autonomous Operation.

Production-ready daemon for:
- Long-running autonomous task execution
- Scheduled task processing
- Device health monitoring
- Automatic recovery from failures
"""

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("droidrun.daemon")


@dataclass
class DaemonConfig:
    """Configuration for the daemon."""

    # Scheduler settings
    scheduler_enabled: bool = True
    scheduler_interval: float = 5.0  # Check for tasks every N seconds
    task_storage_path: str = "daemon_tasks.json"
    max_concurrent_tasks: int = 1

    # Device settings
    device_serial: Optional[str] = None
    device_check_interval: float = 30.0  # Check device connection every N seconds

    # Health monitoring
    health_check_interval: float = 60.0
    max_consecutive_failures: int = 3
    restart_on_failure: bool = True

    # Agent settings
    llm_provider: str = "anthropic"
    model_name: Optional[str] = None
    max_steps_per_task: int = 50
    memory_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Persistence
    state_file: str = "daemon_state.json"
    checkpoint_interval: float = 300.0  # Save state every 5 minutes


class DroidRunDaemon:
    """
    Main daemon process for autonomous DroidRun operation.

    This daemon:
    1. Runs continuously in the background
    2. Processes scheduled tasks from the queue
    3. Monitors device connection health
    4. Auto-recovers from failures
    5. Persists state for restart recovery
    """

    def __init__(self, config: DaemonConfig = None):
        self.config = config or DaemonConfig()
        self._running = False
        self._scheduler = None
        self._device_connected = False
        self._consecutive_failures = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._start_time = None
        self._last_health_check = None
        self._shutdown_event = asyncio.Event()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the daemon."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            logging.getLogger().addHandler(file_handler)

    async def start(self):
        """Start the daemon."""
        logger.info("🚀 Starting DroidRun Daemon...")

        self._running = True
        self._start_time = time.time()

        # Setup signal handlers
        self._setup_signal_handlers()

        # Initialize scheduler
        if self.config.scheduler_enabled:
            await self._init_scheduler()

        # Load previous state
        await self._load_state()

        # Main daemon loop
        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Daemon cancelled")
        finally:
            await self._cleanup()

    async def stop(self):
        """Stop the daemon gracefully."""
        logger.info("🛑 Stopping DroidRun Daemon...")
        self._running = False
        self._shutdown_event.set()

    async def _main_loop(self):
        """Main daemon loop."""
        logger.info("📍 Daemon main loop started")

        # Create concurrent tasks
        tasks = [
            asyncio.create_task(self._scheduler_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._checkpoint_loop()),
        ]

        # Wait for shutdown or error
        try:
            await self._shutdown_event.wait()
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _scheduler_loop(self):
        """Process scheduled tasks."""
        if not self.config.scheduler_enabled or not self._scheduler:
            return

        logger.info("📅 Scheduler loop started")

        while self._running:
            try:
                # Check for due tasks
                pending = self._scheduler.get_pending_tasks()

                for task in pending:
                    if task.scheduled_time <= time.time():
                        logger.info(f"▶️ Executing scheduled task: {task.task_id}")

                        try:
                            # The scheduler handles execution via the executor
                            await self._scheduler.execute_due_tasks()
                            self._tasks_completed += 1
                            self._consecutive_failures = 0

                        except Exception as e:
                            logger.error(f"Task execution failed: {e}")
                            self._tasks_failed += 1
                            self._consecutive_failures += 1

                            if self._consecutive_failures >= self.config.max_consecutive_failures:
                                logger.warning(
                                    f"⚠️ {self._consecutive_failures} consecutive failures - "
                                    f"checking system health"
                                )
                                await self._handle_failure_threshold()

                # Wait before next check
                await asyncio.sleep(self.config.scheduler_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.config.scheduler_interval)

    async def _health_check_loop(self):
        """Monitor system and device health."""
        logger.info("💓 Health check loop started")

        while self._running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _perform_health_check(self):
        """Perform a health check."""
        self._last_health_check = time.time()

        # Check device connection
        device_ok = await self._check_device_connection()

        if not device_ok and self._device_connected:
            logger.warning("⚠️ Device connection lost")
            self._device_connected = False
        elif device_ok and not self._device_connected:
            logger.info("✅ Device connected")
            self._device_connected = True

        # Log stats
        uptime = time.time() - self._start_time if self._start_time else 0
        logger.debug(
            f"Health: uptime={uptime:.0f}s, completed={self._tasks_completed}, "
            f"failed={self._tasks_failed}, device={'OK' if self._device_connected else 'DISCONNECTED'}"
        )

    async def _check_device_connection(self) -> bool:
        """Check if Android device is connected."""
        try:
            # Try to list devices via adb
            proc = await asyncio.create_subprocess_exec(
                "adb", "devices",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            output = stdout.decode()
            # Check if any device is connected
            lines = output.strip().split("\n")
            for line in lines[1:]:  # Skip header
                if "\tdevice" in line:
                    return True

            return False

        except Exception as e:
            logger.debug(f"Device check failed: {e}")
            return False

    async def _checkpoint_loop(self):
        """Periodically save daemon state."""
        logger.info("💾 Checkpoint loop started")

        while self._running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval)
                await self._save_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")

    async def _handle_failure_threshold(self):
        """Handle reaching the failure threshold."""
        if self.config.restart_on_failure:
            logger.warning("🔄 Restarting components due to repeated failures...")

            # Re-initialize scheduler
            if self.config.scheduler_enabled:
                await self._init_scheduler()

            # Reset failure counter
            self._consecutive_failures = 0

    async def _init_scheduler(self):
        """Initialize the task scheduler."""
        try:
            from droidrun.scheduler import (
                create_droid_scheduler,
                TaskScheduler,
                SchedulerConfig,
            )

            self._scheduler = create_droid_scheduler(
                device_serial=self.config.device_serial,
                llm_provider=self.config.llm_provider,
                model_name=self.config.model_name,
                max_steps=self.config.max_steps_per_task,
                storage_path=self.config.task_storage_path,
                max_concurrent=self.config.max_concurrent_tasks,
                memory_enabled=self.config.memory_enabled,
            )

            logger.info("✅ Scheduler initialized")

        except ImportError as e:
            logger.warning(f"Scheduler not available: {e}")
            self._scheduler = None

    async def _save_state(self):
        """Save daemon state for recovery."""
        import json

        state = {
            "start_time": self._start_time,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "device_connected": self._device_connected,
            "saved_at": time.time(),
        }

        try:
            Path(self.config.state_file).write_text(json.dumps(state, indent=2))
            logger.debug(f"State saved to {self.config.state_file}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    async def _load_state(self):
        """Load previous daemon state."""
        import json

        try:
            if Path(self.config.state_file).exists():
                state = json.loads(Path(self.config.state_file).read_text())
                self._tasks_completed = state.get("tasks_completed", 0)
                self._tasks_failed = state.get("tasks_failed", 0)
                logger.info(
                    f"📂 Loaded previous state: {self._tasks_completed} completed, "
                    f"{self._tasks_failed} failed"
                )
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self.stop())
            )

    async def _cleanup(self):
        """Cleanup before shutdown."""
        logger.info("🧹 Cleaning up...")

        # Save final state
        await self._save_state()

        # Stop scheduler
        if self._scheduler:
            await self._scheduler.stop()

        logger.info("👋 DroidRun Daemon stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "device_connected": self._device_connected,
            "consecutive_failures": self._consecutive_failures,
            "last_health_check": self._last_health_check,
            "scheduler_enabled": self.config.scheduler_enabled,
        }

    # Public API for task management
    def schedule_task(
        self,
        goal: str,
        delay_seconds: float = 0,
        priority: str = "normal",
        **config
    ) -> Optional[str]:
        """
        Schedule a task for execution.

        Args:
            goal: Task goal/instruction
            delay_seconds: Delay before execution
            priority: Task priority (low, normal, high, critical)
            **config: Additional task configuration

        Returns:
            Task ID if scheduled, None if scheduler unavailable
        """
        if not self._scheduler:
            logger.warning("Scheduler not available")
            return None

        from droidrun.scheduler import TaskPriority

        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }

        return self._scheduler.schedule_task(
            goal=goal,
            delay_seconds=delay_seconds,
            priority=priority_map.get(priority, TaskPriority.NORMAL),
            config=config,
        )


async def run_daemon(config: DaemonConfig = None):
    """Run the daemon (entry point)."""
    daemon = DroidRunDaemon(config)
    await daemon.start()


def main():
    """CLI entry point for the daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="DroidRun Daemon")
    parser.add_argument("--device", "-d", help="Device serial")
    parser.add_argument("--provider", "-p", default="anthropic", help="LLM provider")
    parser.add_argument("--log-level", "-l", default="INFO", help="Log level")
    parser.add_argument("--log-file", "-f", help="Log file path")
    parser.add_argument("--task-file", default="daemon_tasks.json", help="Task storage")

    args = parser.parse_args()

    config = DaemonConfig(
        device_serial=args.device,
        llm_provider=args.provider,
        log_level=args.log_level,
        log_file=args.log_file,
        task_storage_path=args.task_file,
    )

    print("🤖 DroidRun Daemon starting...")
    print(f"   Provider: {config.llm_provider}")
    print(f"   Device: {config.device_serial or 'auto-detect'}")
    print(f"   Task file: {config.task_storage_path}")
    print("")

    asyncio.run(run_daemon(config))


if __name__ == "__main__":
    main()

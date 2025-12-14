"""
DroidRun Daemon - Continuous Autonomous Operation.

Production-ready daemon for:
- Long-running autonomous task execution
- Scheduled task processing
- Device health monitoring
- Automatic recovery from failures
- Memory consolidation (Titans + ReMe)
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

# Memory system (Titans + ReMe)
try:
    from droidrun.agent.memory import (
        UnifiedMemorySystem,
        UnifiedMemoryConfig,
        create_unified_memory,
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False
    UnifiedMemorySystem = None
    UnifiedMemoryConfig = None
    create_unified_memory = None

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

    # Memory system (Titans + ReMe)
    memory_persist_dir: str = ".droidrun/memory"
    memory_consolidation_interval: float = 3600.0  # Consolidate every hour

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Persistence
    state_file: str = "daemon_state.json"
    checkpoint_interval: float = 300.0  # Save state every 5 minutes

    # Resource monitoring
    resource_monitoring_enabled: bool = True
    min_battery_percent: int = 20
    resource_check_interval: float = 60.0

    # Human escalation
    escalation_enabled: bool = True
    escalation_storage_path: str = "escalation_queue.json"


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

        # Resource monitoring and escalation
        self._resource_monitor = None
        self._escalation_queue = None
        self._paused_for_resources = False

        # ADB tools for resource monitoring
        self._adb_tools = None

        # Memory system (Titans + ReMe)
        self._unified_memory = None
        self._last_memory_consolidation = None

        # Task watchdog tracking
        self._task_start_times: Dict[str, float] = {}
        self._task_timeout_seconds: float = 3600.0  # 1 hour default

        # Initialization status tracking (for detecting silent failures)
        self._init_status = {
            "scheduler": False,
            "resource_monitor": False,
            "escalation_queue": False,
            "adb_tools": False,
            "memory": False,
        }

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

        # Initialize ADB tools first (needed for resource monitoring)
        await self._init_adb_tools()

        # Initialize scheduler
        if self.config.scheduler_enabled:
            await self._init_scheduler()

        # Initialize resource monitor (requires ADB tools)
        if self.config.resource_monitoring_enabled:
            await self._init_resource_monitor()

        # Initialize escalation queue
        if self.config.escalation_enabled:
            self._init_escalation_queue()

        # Initialize memory system (Titans + ReMe)
        if self.config.memory_enabled:
            await self._init_memory()

        # Load previous state
        await self._load_state()

        # Log initialization status
        self._log_init_status()

        # Main daemon loop
        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Daemon cancelled")
        finally:
            await self._cleanup()

    def _log_init_status(self):
        """Log the initialization status of all components."""
        logger.info("=" * 50)
        logger.info("📊 Daemon Initialization Status:")
        for component, status in self._init_status.items():
            emoji = "✅" if status else "❌"
            logger.info(f"   {emoji} {component}: {'OK' if status else 'FAILED'}")

        # Warn if critical components failed
        if not self._init_status["scheduler"]:
            logger.error("⚠️ SCHEDULER NOT INITIALIZED - No tasks will execute!")
        if not self._init_status["adb_tools"]:
            logger.warning("⚠️ ADB TOOLS NOT INITIALIZED - Resource monitoring limited")
        logger.info("=" * 50)

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
            asyncio.create_task(self._watchdog_loop()),  # NEW: Task watchdog
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

    async def _watchdog_loop(self):
        """Monitor running tasks for timeouts and hung states."""
        logger.info("🐕 Watchdog loop started")

        while self._running:
            try:
                await asyncio.sleep(60.0)  # Check every minute

                if not self._scheduler:
                    continue

                # Check for stale/hung tasks
                stale_tasks = await self._scheduler.check_stale_tasks()
                if stale_tasks:
                    for task in stale_tasks:
                        logger.warning(f"🚨 Watchdog detected stale task: {task.task_id}")

                        # Escalate stale tasks
                        if self._escalation_queue:
                            self._escalation_queue.escalate(
                                task_id=task.task_id,
                                reason=f"Task timed out or became unresponsive",
                                context={
                                    "last_action": task.last_action,
                                    "progress_percent": task.progress_percent,
                                    "current_step": task.current_step,
                                },
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    async def _scheduler_loop(self):
        """Process scheduled tasks."""
        # CRITICAL: Verify scheduler is initialized
        if not self.config.scheduler_enabled:
            logger.info("Scheduler disabled in config")
            return

        if not self._scheduler:
            logger.error("❌ SCHEDULER NOT INITIALIZED - Cannot process tasks!")
            logger.error("   Check initialization logs for errors")
            return

        logger.info("📅 Scheduler loop started")

        while self._running:
            try:
                # Double-check scheduler still exists (could have been cleared on error)
                if not self._scheduler:
                    logger.error("Scheduler became None - attempting reinitialization")
                    await self._init_scheduler()
                    if not self._scheduler:
                        logger.error("Reinitialization failed - sleeping")
                        await asyncio.sleep(30.0)
                        continue

                # Check if paused for resources (low battery, thermal, etc.)
                if self._paused_for_resources:
                    logger.debug("Tasks paused - waiting for resources")
                    await asyncio.sleep(self.config.scheduler_interval)
                    continue

                # Also check resource monitor directly if available
                if self._resource_monitor and self._resource_monitor.is_paused:
                    logger.debug("Resource monitor paused - skipping execution")
                    await asyncio.sleep(self.config.scheduler_interval)
                    continue

                # Execute due tasks through scheduler
                try:
                    await self._scheduler.execute_due_tasks()
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    self._tasks_failed += 1
                    self._consecutive_failures += 1

                    # Escalate if we hit max failures
                    if self._consecutive_failures >= self.config.max_consecutive_failures:
                        logger.warning(
                            f"⚠️ {self._consecutive_failures} consecutive failures - "
                            f"checking system health"
                        )
                        await self._handle_failure_threshold()

                        # Escalate to human if escalation queue is available
                        if self._escalation_queue:
                            self._escalation_queue.escalate(
                                task_id="scheduler_failures",
                                reason=f"Scheduler hit {self._consecutive_failures} consecutive failures",
                                context={
                                    "last_error": str(e),
                                    "consecutive_failures": self._consecutive_failures,
                                },
                            )

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

                # Periodic cleanup (every checkpoint interval)
                await self._run_cleanup()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")

    async def _run_cleanup(self):
        """Run cleanup to prevent unbounded growth."""
        try:
            # Cleanup old tasks from scheduler
            if self._scheduler:
                removed_tasks = self._scheduler.cleanup_old_tasks(
                    max_completed_tasks=1000,
                    max_age_days=30,
                )
                if removed_tasks > 0:
                    logger.debug(f"Cleaned up {removed_tasks} old tasks")

            # Cleanup old escalations
            if self._escalation_queue:
                removed_escalations = self._escalation_queue.cleanup_old_items(
                    max_resolved_items=500,
                    max_age_days=30,
                )
                if removed_escalations > 0:
                    logger.debug(f"Cleaned up {removed_escalations} old escalations")

            # Memory consolidation (Titans + ReMe)
            if self._unified_memory and self._last_memory_consolidation:
                time_since_consolidation = time.time() - self._last_memory_consolidation
                if time_since_consolidation >= self.config.memory_consolidation_interval:
                    logger.info("🧠 Running memory consolidation...")
                    try:
                        consolidation_result = await self._unified_memory.consolidate()
                        self._last_memory_consolidation = time.time()
                        logger.info(f"🧠 Memory consolidation complete: {consolidation_result}")
                    except Exception as e:
                        logger.warning(f"Memory consolidation failed: {e}")

        except Exception as e:
            logger.warning(f"Cleanup error (non-fatal): {e}")

    async def _handle_failure_threshold(self):
        """Handle reaching the failure threshold."""
        if self.config.restart_on_failure:
            logger.warning("🔄 Restarting components due to repeated failures...")

            # Re-initialize scheduler
            if self.config.scheduler_enabled:
                await self._init_scheduler()

            # Reset failure counter
            self._consecutive_failures = 0

    async def _init_adb_tools(self):
        """Initialize ADB tools for device communication."""
        try:
            from droidrun.tools.adb import AdbTools

            self._adb_tools = AdbTools(serial=self.config.device_serial)

            # Try to connect to verify device is available
            try:
                await self._adb_tools.connect()
                self._device_connected = True
                self._init_status["adb_tools"] = True
                logger.info("✅ ADB tools initialized and device connected")
            except Exception as e:
                logger.warning(f"ADB tools initialized but device not connected: {e}")
                self._init_status["adb_tools"] = True  # Tools are ready, device may connect later
                self._device_connected = False

        except ImportError as e:
            logger.error(f"ADB tools not available: {e}")
            self._adb_tools = None
            self._init_status["adb_tools"] = False
        except Exception as e:
            logger.error(f"Failed to initialize ADB tools: {e}")
            self._adb_tools = None
            self._init_status["adb_tools"] = False

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

            self._init_status["scheduler"] = True
            logger.info("✅ Scheduler initialized")

        except ImportError as e:
            logger.error(f"Scheduler not available: {e}")
            self._scheduler = None
            self._init_status["scheduler"] = False
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            self._scheduler = None
            self._init_status["scheduler"] = False

    async def _init_resource_monitor(self):
        """Initialize resource monitoring."""
        try:
            from droidrun.agent.resources import (
                ResourceMonitor,
                ResourceConfig,
            )

            config = ResourceConfig(
                enabled=True,
                min_battery_percent=self.config.min_battery_percent,
                check_interval_seconds=self.config.resource_check_interval,
                pause_on_low_battery=True,
                resume_on_charging=True,
                on_low_battery=self._on_low_battery,
                on_resources_ok=self._on_resources_ok,
            )

            # Pass ADB tools for actual device resource monitoring
            self._resource_monitor = ResourceMonitor(
                config=config,
                tools_instance=self._adb_tools,
            )
            await self._resource_monitor.start()

            self._init_status["resource_monitor"] = True
            if self._adb_tools:
                logger.info("✅ Resource monitor initialized with device access")
            else:
                logger.warning("⚠️ Resource monitor initialized WITHOUT device access (limited functionality)")

        except ImportError as e:
            logger.error(f"Resource monitor not available: {e}")
            self._resource_monitor = None
            self._init_status["resource_monitor"] = False
        except Exception as e:
            logger.error(f"Failed to initialize resource monitor: {e}")
            self._resource_monitor = None
            self._init_status["resource_monitor"] = False

    def _init_escalation_queue(self):
        """Initialize human escalation queue."""
        try:
            from droidrun.agent.resources import HumanEscalationQueue

            self._escalation_queue = HumanEscalationQueue(
                storage_path=self.config.escalation_storage_path
            )

            # Register callback for escalations
            self._escalation_queue.on_escalation(self._on_escalation)

            self._init_status["escalation_queue"] = True
            logger.info("✅ Escalation queue initialized")

        except ImportError as e:
            logger.error(f"Escalation queue not available: {e}")
            self._escalation_queue = None
            self._init_status["escalation_queue"] = False
        except Exception as e:
            logger.error(f"Failed to initialize escalation queue: {e}")
            self._escalation_queue = None
            self._init_status["escalation_queue"] = False

    async def _init_memory(self):
        """Initialize Unified Memory System (Titans + ReMe)."""
        if not UNIFIED_MEMORY_AVAILABLE:
            logger.warning("Unified memory system not available (missing imports)")
            self._init_status["memory"] = False
            return

        try:
            # Create persist directory
            persist_dir = Path(self.config.memory_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            # Initialize unified memory (Titans + ReMe)
            config = UnifiedMemoryConfig(
                titans_persist_path=str(persist_dir / "titans_state.pkl"),
                reme_persist_path=str(persist_dir / "reme_experiences.json"),
                use_titans=True,
                use_reme=True,
                titans_memory_size=384,
                titans_num_slots=64,
                reme_max_experiences=5000,
            )

            self._unified_memory = UnifiedMemorySystem(config=config)
            await self._unified_memory.start()

            self._last_memory_consolidation = time.time()
            self._init_status["memory"] = True
            logger.info("✅ Unified Memory System initialized (Titans + ReMe)")

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            self._unified_memory = None
            self._init_status["memory"] = False

    def _on_low_battery(self, resources):
        """Callback when battery is low."""
        self._paused_for_resources = True
        logger.warning(
            f"🔋 Low battery ({resources.battery_level}%) - pausing task execution"
        )

    def _on_resources_ok(self, resources):
        """Callback when resources are OK again."""
        self._paused_for_resources = False
        logger.info(
            f"✅ Resources OK (battery: {resources.battery_level}%) - resuming"
        )

    def _on_escalation(self, item):
        """Callback when a new escalation is created."""
        logger.warning(
            f"🚨 Human escalation needed: {item.reason} (task: {item.task_id})"
        )
        # Could add notification here (webhook, email, etc.)

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

        # Stop memory system (saves state)
        if self._unified_memory:
            try:
                await self._unified_memory.stop()
                logger.info("🧠 Memory system stopped and saved")
            except Exception as e:
                logger.warning(f"Error stopping memory system: {e}")

        # Stop resource monitor
        if self._resource_monitor:
            await self._resource_monitor.stop()

        # Stop scheduler
        if self._scheduler:
            await self._scheduler.stop()

        logger.info("👋 DroidRun Daemon stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        uptime = time.time() - self._start_time if self._start_time else 0

        status = {
            "running": self._running,
            "uptime_seconds": uptime,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "device_connected": self._device_connected,
            "consecutive_failures": self._consecutive_failures,
            "last_health_check": self._last_health_check,
            "scheduler_enabled": self.config.scheduler_enabled,
            "paused_for_resources": self._paused_for_resources,
        }

        # Add resource monitor stats if available
        if self._resource_monitor:
            status["resources"] = self._resource_monitor.get_statistics()

        # Add escalation queue stats if available
        if self._escalation_queue:
            status["escalations"] = self._escalation_queue.get_statistics()

        # Add memory system stats if available
        if self._unified_memory:
            status["memory"] = self._unified_memory.get_statistics()
            status["last_memory_consolidation"] = self._last_memory_consolidation

        return status

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

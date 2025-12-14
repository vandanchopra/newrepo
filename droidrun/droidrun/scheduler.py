"""
Task Scheduler for DroidRun Agent.

Production-ready task scheduling with:
- Priority-based task queue
- Cron-like scheduling
- Persistent task storage
- Retry with exponential backoff
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import heapq
import uuid

logger = logging.getLogger("droidrun.scheduler")


class TaskStatus(Enum):
    """Status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskPriority(Enum):
    """Priority levels for tasks."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ScheduledTask:
    """A task scheduled for execution."""

    task_id: str
    goal: str
    scheduled_time: float  # Unix timestamp
    priority: int = TaskPriority.NORMAL.value
    status: str = TaskStatus.PENDING.value
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    cron_expression: Optional[str] = None  # For recurring tasks

    # NEW: Execution timeout and progress tracking
    execution_timeout_seconds: float = 3600.0  # 1 hour default
    last_heartbeat: Optional[float] = None
    progress_percent: float = 0.0
    current_step: int = 0
    last_action: str = ""

    # NEW: Task dependencies for multi-day goals
    depends_on: List[str] = field(default_factory=list)  # Task IDs this depends on
    parent_task_id: Optional[str] = None  # If this is a subtask
    subtask_ids: List[str] = field(default_factory=list)  # Child tasks

    def __lt__(self, other: "ScheduledTask") -> bool:
        """Compare for priority queue (higher priority first, then earlier time)."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.scheduled_time < other.scheduled_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledTask":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class NotificationConfig:
    """Configuration for task notifications."""

    enabled: bool = False
    webhook_url: Optional[str] = None  # HTTP webhook for notifications
    notify_on_failure: bool = True
    notify_on_success: bool = False
    notify_on_timeout: bool = True
    notify_on_max_retries: bool = True
    # Escalation settings
    escalation_after_failures: int = 3  # Escalate after N consecutive failures
    escalation_webhook_url: Optional[str] = None  # Different webhook for escalation


@dataclass
class SchedulerConfig:
    """Configuration for the task scheduler."""

    enabled: bool = True
    storage_path: str = "scheduler_tasks.json"
    poll_interval_seconds: float = 1.0
    max_concurrent_tasks: int = 1
    default_retry_delay_seconds: float = 60.0
    retry_multiplier: float = 2.0
    max_retry_delay_seconds: float = 3600.0

    # NEW: Timeout and stale task detection
    default_execution_timeout_seconds: float = 3600.0  # 1 hour
    stale_task_check_interval_seconds: float = 300.0  # Check every 5 min
    stale_task_timeout_seconds: float = 7200.0  # 2 hours without heartbeat = stale

    # NEW: Notification configuration
    notifications: NotificationConfig = field(default_factory=NotificationConfig)


class TaskScheduler:
    """
    Schedules and executes tasks for DroidAgent.

    Features:
    - Priority queue for task ordering
    - Scheduled execution at specific times
    - Recurring tasks with cron expressions
    - Persistent storage for task queue
    - Automatic retry with exponential backoff
    """

    def __init__(
        self,
        config: SchedulerConfig = None,
        task_executor: Callable[[ScheduledTask], Any] = None,
    ):
        """
        Initialize the task scheduler.

        Args:
            config: Scheduler configuration
            task_executor: Async callable to execute tasks
        """
        self.config = config or SchedulerConfig()
        self.task_executor = task_executor
        self._task_queue: List[ScheduledTask] = []
        self._tasks_by_id: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._current_tasks: Dict[str, asyncio.Task] = {}

        # Load persisted tasks
        if self.config.enabled:
            self._load_tasks()

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{uuid.uuid4().hex[:12]}"

    def _get_retry_delay(self, retry_count: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = self.config.default_retry_delay_seconds * (
            self.config.retry_multiplier ** retry_count
        )
        return min(delay, self.config.max_retry_delay_seconds)

    def schedule_task(
        self,
        goal: str,
        scheduled_time: datetime = None,
        delay_seconds: float = 0,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        config: Dict[str, Any] = None,
        tags: List[str] = None,
        cron_expression: str = None,
    ) -> str:
        """
        Schedule a task for execution.

        Args:
            goal: The goal/instruction for the task
            scheduled_time: When to execute (default: now)
            delay_seconds: Alternative to scheduled_time - delay from now
            priority: Task priority
            max_retries: Maximum retry attempts
            config: Additional configuration for the task
            tags: Tags for categorization
            cron_expression: Cron expression for recurring tasks

        Returns:
            Task ID
        """
        task_id = self._generate_task_id()

        if scheduled_time:
            exec_time = scheduled_time.timestamp()
        elif delay_seconds > 0:
            exec_time = time.time() + delay_seconds
        else:
            exec_time = time.time()

        task = ScheduledTask(
            task_id=task_id,
            goal=goal,
            scheduled_time=exec_time,
            priority=priority.value,
            max_retries=max_retries,
            config=config or {},
            tags=tags or [],
            cron_expression=cron_expression,
        )

        heapq.heappush(self._task_queue, task)
        self._tasks_by_id[task_id] = task

        self._save_tasks()

        logger.info(f"📅 Task scheduled: {task_id} - {goal[:50]}...")

        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled, False if not found
        """
        if task_id not in self._tasks_by_id:
            return False

        task = self._tasks_by_id[task_id]

        if task.status in [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]:
            return False

        task.status = TaskStatus.CANCELLED.value

        # Cancel running task if applicable
        if task_id in self._current_tasks:
            self._current_tasks[task_id].cancel()

        self._save_tasks()

        logger.info(f"🚫 Task cancelled: {task_id}")

        return True

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self._tasks_by_id.get(task_id)

    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending tasks sorted by priority and time."""
        return sorted(
            [t for t in self._tasks_by_id.values() if t.status == TaskStatus.PENDING.value]
        )

    def get_task_history(
        self,
        limit: int = 100,
        status: TaskStatus = None,
    ) -> List[ScheduledTask]:
        """
        Get task execution history.

        Args:
            limit: Maximum number of tasks to return
            status: Filter by status (None for all)

        Returns:
            List of tasks sorted by completion time (newest first)
        """
        tasks = list(self._tasks_by_id.values())

        if status:
            tasks = [t for t in tasks if t.status == status.value]

        # Sort by completion time, then by creation time
        tasks.sort(
            key=lambda t: (t.completed_at or t.created_at, t.created_at),
            reverse=True,
        )

        return tasks[:limit]

    async def start(self):
        """Start the scheduler."""
        if not self.config.enabled:
            logger.info("Scheduler is disabled")
            return

        self._running = True

        async def scheduler_loop():
            while self._running:
                try:
                    await self._process_due_tasks()
                    await asyncio.sleep(self.config.poll_interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    await asyncio.sleep(self.config.poll_interval_seconds)

        self._scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info("⏰ Task scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False

        # Cancel all running tasks
        for task_id, task in list(self._current_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        self._save_tasks()

        logger.info("⏹️ Task scheduler stopped")

    async def _process_due_tasks(self):
        """Process tasks that are due for execution."""
        current_time = time.time()

        while self._task_queue:
            # Peek at the next task
            if self._task_queue[0].scheduled_time > current_time:
                break  # No more due tasks

            if len(self._current_tasks) >= self.config.max_concurrent_tasks:
                break  # At capacity

            # Pop the next due task
            task = heapq.heappop(self._task_queue)

            if task.status != TaskStatus.PENDING.value:
                continue  # Skip non-pending tasks

            # Execute the task
            await self._execute_task(task)

    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task."""
        if not self.task_executor:
            logger.warning(f"No executor configured for task: {task.task_id}")
            task.status = TaskStatus.FAILED.value
            task.error = "No task executor configured"
            self._save_tasks()
            return

        task.status = TaskStatus.RUNNING.value
        task.started_at = time.time()
        self._save_tasks()

        async def run_task():
            try:
                logger.info(f"▶️ Executing task: {task.task_id} - {task.goal[:50]}...")

                result = await self.task_executor(task)

                task.status = TaskStatus.COMPLETED.value
                task.completed_at = time.time()
                task.result = str(result) if result else "Completed"

                logger.info(f"✅ Task completed: {task.task_id}")

                # Schedule next occurrence if recurring
                if task.cron_expression:
                    self._schedule_next_occurrence(task)

            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED.value
                task.completed_at = time.time()
                raise

            except Exception as e:
                logger.error(f"❌ Task failed: {task.task_id} - {e}")
                task.error = str(e)

                if task.retry_count < task.max_retries:
                    # Schedule retry
                    task.retry_count += 1
                    task.status = TaskStatus.RETRY.value
                    delay = self._get_retry_delay(task.retry_count)
                    task.scheduled_time = time.time() + delay

                    logger.info(
                        f"🔄 Task scheduled for retry {task.retry_count}/{task.max_retries} "
                        f"in {delay:.1f}s: {task.task_id}"
                    )

                    # Re-add to queue
                    task.status = TaskStatus.PENDING.value
                    heapq.heappush(self._task_queue, task)
                else:
                    task.status = TaskStatus.FAILED.value
                    task.completed_at = time.time()

            finally:
                self._current_tasks.pop(task.task_id, None)
                self._save_tasks()

        self._current_tasks[task.task_id] = asyncio.create_task(run_task())

    def _schedule_next_occurrence(self, task: ScheduledTask):
        """Schedule the next occurrence of a recurring task."""
        if not task.cron_expression:
            return

        # Simple cron parsing for common patterns
        # Format: "minute hour day month weekday" or keywords
        next_time = self._parse_cron_next(task.cron_expression)
        if next_time:
            self.schedule_task(
                goal=task.goal,
                scheduled_time=next_time,
                priority=TaskPriority(task.priority),
                max_retries=task.max_retries,
                config=task.config,
                tags=task.tags,
                cron_expression=task.cron_expression,
            )

    def _parse_cron_next(self, expression: str) -> Optional[datetime]:
        """Parse cron expression and get next execution time."""
        now = datetime.now()

        # Handle simple keywords
        if expression == "@hourly":
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif expression == "@daily":
            return now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        elif expression == "@weekly":
            days_ahead = 7 - now.weekday()  # Monday = 0
            return now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_ahead)

        # For complex cron expressions, would need a proper parser
        # Return None for unsupported expressions
        logger.warning(f"Unsupported cron expression: {expression}")
        return None

    def _save_tasks(self):
        """Save tasks to persistent storage."""
        if not self.config.storage_path:
            return

        try:
            data = {
                task_id: task.to_dict()
                for task_id, task in self._tasks_by_id.items()
            }

            with open(self.config.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def _load_tasks(self):
        """Load tasks from persistent storage."""
        if not self.config.storage_path or not os.path.exists(self.config.storage_path):
            return

        try:
            with open(self.config.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for task_id, task_data in data.items():
                task = ScheduledTask.from_dict(task_data)
                self._tasks_by_id[task_id] = task

                # Re-add pending tasks to queue
                if task.status == TaskStatus.PENDING.value:
                    heapq.heappush(self._task_queue, task)

            logger.info(f"📂 Loaded {len(self._tasks_by_id)} tasks from storage")

        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        by_status = {}
        for task in self._tasks_by_id.values():
            by_status[task.status] = by_status.get(task.status, 0) + 1

        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "total_tasks": len(self._tasks_by_id),
            "pending_tasks": len(self._task_queue),
            "running_tasks": len(self._current_tasks),
            "tasks_by_status": by_status,
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
        }

    # ---- Progress Tracking & Heartbeat ----

    def update_task_progress(
        self,
        task_id: str,
        current_step: int = None,
        progress_percent: float = None,
        last_action: str = None,
    ) -> bool:
        """
        Update progress on a running task.

        Call this periodically during task execution to:
        1. Update heartbeat (prevents stale detection)
        2. Track progress for monitoring
        3. Enable resume from last known state

        Args:
            task_id: Task ID to update
            current_step: Current step number
            progress_percent: Progress 0-100
            last_action: Description of last action

        Returns:
            True if updated, False if task not found
        """
        task = self._tasks_by_id.get(task_id)
        if not task:
            return False

        task.last_heartbeat = time.time()

        if current_step is not None:
            task.current_step = current_step
        if progress_percent is not None:
            task.progress_percent = min(100.0, max(0.0, progress_percent))
        if last_action is not None:
            task.last_action = last_action

        self._save_tasks()
        return True

    async def check_stale_tasks(self) -> List[ScheduledTask]:
        """
        Find and handle stale tasks (running but no heartbeat).

        A task is stale if:
        - Status is RUNNING
        - last_heartbeat is older than stale_task_timeout_seconds
        - OR started_at + execution_timeout_seconds has passed

        Returns:
            List of stale tasks found
        """
        stale_tasks = []
        current_time = time.time()

        for task in self._tasks_by_id.values():
            if task.status != TaskStatus.RUNNING.value:
                continue

            # Check heartbeat timeout
            heartbeat_time = task.last_heartbeat or task.started_at or 0
            heartbeat_age = current_time - heartbeat_time

            # Check execution timeout
            started = task.started_at or current_time
            execution_time = current_time - started

            is_stale = (
                heartbeat_age > self.config.stale_task_timeout_seconds or
                execution_time > task.execution_timeout_seconds
            )

            if is_stale:
                stale_tasks.append(task)
                logger.warning(
                    f"⚠️ Stale task detected: {task.task_id} "
                    f"(heartbeat_age: {heartbeat_age:.0f}s, "
                    f"execution_time: {execution_time:.0f}s)"
                )

                # Mark as failed due to timeout
                task.status = TaskStatus.FAILED.value
                task.completed_at = current_time
                task.error = f"Task timed out (heartbeat: {heartbeat_age:.0f}s, execution: {execution_time:.0f}s)"

                # Send notification
                await self._send_notification(
                    task=task,
                    event_type="timeout",
                    message=f"Task timed out: {task.goal[:50]}",
                )

        if stale_tasks:
            self._save_tasks()

        return stale_tasks

    # ---- Notification System ----

    async def _send_notification(
        self,
        task: ScheduledTask,
        event_type: str,
        message: str,
        escalate: bool = False,
    ) -> bool:
        """
        Send notification about task event.

        Args:
            task: The task this notification is about
            event_type: Type of event (failure, success, timeout, escalation)
            message: Human-readable message
            escalate: Whether this is an escalation

        Returns:
            True if notification sent successfully
        """
        notif_config = self.config.notifications
        if not notif_config.enabled:
            return False

        # Determine if we should send based on event type
        should_send = (
            (event_type == "failure" and notif_config.notify_on_failure) or
            (event_type == "success" and notif_config.notify_on_success) or
            (event_type == "timeout" and notif_config.notify_on_timeout) or
            (event_type == "max_retries" and notif_config.notify_on_max_retries) or
            escalate
        )

        if not should_send:
            return False

        # Choose webhook URL
        webhook_url = (
            notif_config.escalation_webhook_url if escalate
            else notif_config.webhook_url
        )

        if not webhook_url:
            logger.debug(f"No webhook URL configured for {event_type}")
            return False

        # Build notification payload
        payload = {
            "event_type": event_type,
            "escalation": escalate,
            "timestamp": time.time(),
            "message": message,
            "task": {
                "task_id": task.task_id,
                "goal": task.goal,
                "status": task.status,
                "priority": task.priority,
                "retry_count": task.retry_count,
                "progress_percent": task.progress_percent,
                "current_step": task.current_step,
                "last_action": task.last_action,
                "error": task.error,
            },
        }

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.info(f"📤 Notification sent: {event_type}")
                        return True
                    else:
                        logger.warning(
                            f"Notification failed: {response.status}"
                        )
                        return False

        except ImportError:
            logger.warning("aiohttp not installed, notifications disabled")
            return False
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
            return False

    # ---- Task Dependencies ----

    def schedule_dependent_task(
        self,
        goal: str,
        depends_on: List[str],
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs,
    ) -> str:
        """
        Schedule a task that depends on other tasks completing.

        The task will only execute after all dependencies have
        completed successfully.

        Args:
            goal: Task goal
            depends_on: List of task IDs this task depends on
            priority: Task priority
            **kwargs: Additional arguments passed to schedule_task

        Returns:
            Task ID
        """
        task_id = self.schedule_task(
            goal=goal,
            priority=priority,
            **kwargs,
        )

        # Set dependencies
        task = self._tasks_by_id[task_id]
        task.depends_on = list(depends_on)
        self._save_tasks()

        logger.info(
            f"📋 Task {task_id} scheduled with {len(depends_on)} dependencies"
        )

        return task_id

    def are_dependencies_met(self, task_id: str) -> bool:
        """
        Check if all dependencies for a task are satisfied.

        Args:
            task_id: Task to check

        Returns:
            True if all dependencies completed successfully
        """
        task = self._tasks_by_id.get(task_id)
        if not task or not task.depends_on:
            return True

        for dep_id in task.depends_on:
            dep_task = self._tasks_by_id.get(dep_id)
            if not dep_task:
                logger.warning(f"Dependency {dep_id} not found")
                return False

            if dep_task.status != TaskStatus.COMPLETED.value:
                return False

        return True

    def create_task_chain(
        self,
        goals: List[str],
        base_priority: TaskPriority = TaskPriority.NORMAL,
        delay_between_seconds: float = 0,
    ) -> List[str]:
        """
        Create a chain of dependent tasks for multi-day goals.

        Each task depends on the previous one completing.

        Args:
            goals: List of goal strings in order
            base_priority: Priority for the chain
            delay_between_seconds: Delay between tasks

        Returns:
            List of created task IDs in order
        """
        task_ids = []
        previous_id = None

        for i, goal in enumerate(goals):
            depends_on = [previous_id] if previous_id else []

            task_id = self.schedule_task(
                goal=goal,
                priority=base_priority,
                delay_seconds=delay_between_seconds * i,
            )

            task = self._tasks_by_id[task_id]
            task.depends_on = depends_on
            if previous_id:
                parent = self._tasks_by_id[previous_id]
                parent.subtask_ids.append(task_id)
                task.parent_task_id = task_ids[0]  # First task is parent

            task_ids.append(task_id)
            previous_id = task_id

        self._save_tasks()

        logger.info(f"📋 Created task chain with {len(task_ids)} tasks")
        return task_ids


# Convenience function
def create_scheduler(
    task_executor: Callable[[ScheduledTask], Any] = None,
    storage_path: str = "scheduler_tasks.json",
    max_concurrent: int = 1,
) -> TaskScheduler:
    """
    Create a task scheduler with common settings.

    Args:
        task_executor: Async callable to execute tasks
        storage_path: Path for persistent task storage
        max_concurrent: Maximum concurrent tasks

    Returns:
        Configured TaskScheduler instance
    """
    config = SchedulerConfig(
        enabled=True,
        storage_path=storage_path,
        max_concurrent_tasks=max_concurrent,
    )
    return TaskScheduler(config=config, task_executor=task_executor)


class DroidAgentExecutor:
    """
    Executor that runs scheduled tasks using DroidAgent.

    This bridges the scheduler with actual device control.
    """

    def __init__(
        self,
        device_serial: Optional[str] = None,
        llm_provider: str = "anthropic",
        model_name: Optional[str] = None,
        max_steps: int = 50,
        reasoning: bool = True,
        memory_enabled: bool = True,
    ):
        """
        Initialize the DroidAgent executor.

        Args:
            device_serial: Android device serial (or None for auto-detect)
            llm_provider: LLM provider to use
            model_name: Model name (optional, uses provider default)
            max_steps: Maximum steps per task
            reasoning: Whether to use Manager/Executor reasoning mode
            memory_enabled: Whether to enable memory system
        """
        self.device_serial = device_serial
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.max_steps = max_steps
        self.reasoning = reasoning
        self.memory_enabled = memory_enabled
        self._agent = None

    async def __call__(self, task: ScheduledTask) -> Dict[str, Any]:
        """
        Execute a scheduled task using DroidAgent.

        Args:
            task: The scheduled task to execute

        Returns:
            Dictionary with execution results
        """
        try:
            # Import DroidAgent lazily to avoid circular imports
            from droidrun.agent.droid import DroidAgent
            from droidrun.config_manager.config_manager import (
                DroidrunConfig,
                AgentConfig,
                DeviceConfig,
                MemoryConfig,
            )

            # Build config from task
            config = DroidrunConfig(
                agent=AgentConfig(
                    max_steps=task.config.get("max_steps", self.max_steps),
                    reasoning=task.config.get("reasoning", self.reasoning),
                ),
                device=DeviceConfig(
                    serial=task.config.get("device_serial", self.device_serial),
                ),
                memory=MemoryConfig(
                    enabled=task.config.get("memory_enabled", self.memory_enabled),
                ),
            )

            # Create agent
            agent = DroidAgent(
                instruction=task.goal,
                config=config,
            )

            # Run the agent
            logger.info(f"🤖 Starting DroidAgent for task: {task.goal[:50]}...")

            result = {"success": False, "reason": "", "steps": 0}

            handler = agent.run()
            async for event in handler.stream_events():
                # Log significant events
                event_type = type(event).__name__
                if event_type in ("ManagerPlanEvent", "ExecutorResultEvent", "FinalizeEvent"):
                    logger.debug(f"  Event: {event_type}")

            # Get final result
            final_result = await handler

            result = {
                "success": final_result.success,
                "reason": final_result.reason or "",
                "steps": final_result.steps,
            }

            logger.info(
                f"🏁 DroidAgent completed: success={result['success']}, "
                f"steps={result['steps']}"
            )

            return result

        except ImportError as e:
            logger.error(f"DroidAgent not available: {e}")
            return {
                "success": False,
                "reason": f"DroidAgent not available: {e}",
                "steps": 0,
            }

        except Exception as e:
            logger.error(f"DroidAgent execution failed: {e}")
            raise  # Let scheduler handle retry logic


def create_droid_scheduler(
    device_serial: Optional[str] = None,
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None,
    max_steps: int = 50,
    storage_path: str = "scheduler_tasks.json",
    max_concurrent: int = 1,
    memory_enabled: bool = True,
) -> TaskScheduler:
    """
    Create a scheduler configured with DroidAgent executor.

    This is the main entry point for autonomous task scheduling.

    Args:
        device_serial: Android device serial
        llm_provider: LLM provider to use
        model_name: Model name
        max_steps: Maximum steps per task
        storage_path: Path for task persistence
        max_concurrent: Maximum concurrent tasks
        memory_enabled: Whether to enable memory

    Returns:
        TaskScheduler configured with DroidAgentExecutor
    """
    executor = DroidAgentExecutor(
        device_serial=device_serial,
        llm_provider=llm_provider,
        model_name=model_name,
        max_steps=max_steps,
        memory_enabled=memory_enabled,
    )

    config = SchedulerConfig(
        enabled=True,
        storage_path=storage_path,
        max_concurrent_tasks=max_concurrent,
    )

    return TaskScheduler(config=config, task_executor=executor)

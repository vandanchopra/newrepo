"""
Tests for Task Scheduler.

Production-ready tests for:
- TaskScheduler
- Task scheduling and execution
- Priority handling
- Retry logic
"""

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import sys

# Add scheduler module path directly
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_parent_dir, 'droidrun'))


class TestTaskStatus(unittest.TestCase):
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test that all status values exist."""
        from scheduler import TaskStatus

        required_statuses = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "RETRY"]

        for status in required_statuses:
            self.assertTrue(hasattr(TaskStatus, status))


class TestTaskPriority(unittest.TestCase):
    """Tests for TaskPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        from scheduler import TaskPriority

        self.assertLess(TaskPriority.LOW.value, TaskPriority.NORMAL.value)
        self.assertLess(TaskPriority.NORMAL.value, TaskPriority.HIGH.value)
        self.assertLess(TaskPriority.HIGH.value, TaskPriority.CRITICAL.value)


class TestScheduledTask(unittest.TestCase):
    """Tests for ScheduledTask dataclass."""

    def test_task_creation(self):
        """Test task creation."""
        from scheduler import ScheduledTask

        task = ScheduledTask(
            task_id="test_task",
            goal="Open Instagram",
            scheduled_time=1000.0,
        )

        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.goal, "Open Instagram")
        self.assertEqual(task.scheduled_time, 1000.0)

    def test_task_serialization(self):
        """Test task to/from dict."""
        from scheduler import ScheduledTask

        original = ScheduledTask(
            task_id="test_task",
            goal="Test goal",
            scheduled_time=1000.0,
            priority=2,
            tags=["test", "automation"],
        )

        data = original.to_dict()
        restored = ScheduledTask.from_dict(data)

        self.assertEqual(original.task_id, restored.task_id)
        self.assertEqual(original.goal, restored.goal)
        self.assertEqual(original.tags, restored.tags)

    def test_task_comparison(self):
        """Test task comparison for priority queue."""
        from scheduler import ScheduledTask, TaskPriority

        high_priority = ScheduledTask(
            task_id="high",
            goal="High priority",
            scheduled_time=2000.0,
            priority=TaskPriority.HIGH.value,
        )

        low_priority = ScheduledTask(
            task_id="low",
            goal="Low priority",
            scheduled_time=1000.0,  # Earlier time but lower priority
            priority=TaskPriority.LOW.value,
        )

        # High priority should come first despite later time
        self.assertLess(high_priority, low_priority)


class TestSchedulerConfig(unittest.TestCase):
    """Tests for SchedulerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from scheduler import SchedulerConfig

        config = SchedulerConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.max_concurrent_tasks, 1)
        self.assertEqual(config.poll_interval_seconds, 1.0)

    def test_custom_config(self):
        """Test custom configuration."""
        from scheduler import SchedulerConfig

        config = SchedulerConfig(
            enabled=True,
            max_concurrent_tasks=5,
            poll_interval_seconds=0.5,
        )

        self.assertEqual(config.max_concurrent_tasks, 5)
        self.assertEqual(config.poll_interval_seconds, 0.5)


class TestTaskScheduler(unittest.TestCase):
    """Tests for TaskScheduler."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()

        from scheduler import TaskScheduler, SchedulerConfig

        self.config = SchedulerConfig(
            enabled=True,
            storage_path=self.temp_file.name,
            poll_interval_seconds=0.1,
        )

        self.executed_tasks = []

        async def mock_executor(task):
            self.executed_tasks.append(task)
            return f"Executed: {task.goal}"

        self.scheduler = TaskScheduler(config=self.config, task_executor=mock_executor)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_schedule_task(self):
        """Test scheduling a task."""
        from scheduler import TaskPriority

        task_id = self.scheduler.schedule_task(
            goal="Open Instagram",
            priority=TaskPriority.HIGH,
            tags=["test"],
        )

        self.assertIsNotNone(task_id)
        self.assertTrue(task_id.startswith("task_"))

        task = self.scheduler.get_task(task_id)
        self.assertEqual(task.goal, "Open Instagram")
        self.assertEqual(task.priority, TaskPriority.HIGH.value)

    def test_schedule_delayed_task(self):
        """Test scheduling a task with delay."""
        import time

        current_time = time.time()

        task_id = self.scheduler.schedule_task(
            goal="Delayed task",
            delay_seconds=60,
        )

        task = self.scheduler.get_task(task_id)
        self.assertGreater(task.scheduled_time, current_time + 55)

    def test_schedule_task_at_time(self):
        """Test scheduling a task at specific time."""
        future_time = datetime.now() + timedelta(hours=1)

        task_id = self.scheduler.schedule_task(
            goal="Future task",
            scheduled_time=future_time,
        )

        task = self.scheduler.get_task(task_id)
        self.assertAlmostEqual(
            task.scheduled_time,
            future_time.timestamp(),
            delta=1.0,
        )

    def test_cancel_task(self):
        """Test cancelling a task."""
        from scheduler import TaskStatus

        task_id = self.scheduler.schedule_task(
            goal="Task to cancel",
            delay_seconds=3600,  # Far in future
        )

        result = self.scheduler.cancel_task(task_id)
        self.assertTrue(result)

        task = self.scheduler.get_task(task_id)
        self.assertEqual(task.status, TaskStatus.CANCELLED.value)

    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        self.scheduler.schedule_task(goal="Task 1", delay_seconds=10)
        self.scheduler.schedule_task(goal="Task 2", delay_seconds=20)
        self.scheduler.schedule_task(goal="Task 3", delay_seconds=30)

        pending = self.scheduler.get_pending_tasks()
        self.assertEqual(len(pending), 3)

    def test_get_statistics(self):
        """Test getting scheduler statistics."""
        self.scheduler.schedule_task(goal="Test task")

        stats = self.scheduler.get_statistics()

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["total_tasks"], 1)
        self.assertEqual(stats["pending_tasks"], 1)

    def test_task_execution(self):
        """Test that tasks are executed."""
        # Schedule task for immediate execution
        task_id = self.scheduler.schedule_task(goal="Execute me")

        async def run_test():
            await self.scheduler.start()
            await asyncio.sleep(0.3)  # Give time for execution
            await self.scheduler.stop()

        asyncio.run(run_test())

        self.assertEqual(len(self.executed_tasks), 1)
        self.assertEqual(self.executed_tasks[0].goal, "Execute me")


class TestSchedulerPersistence(unittest.TestCase):
    """Tests for scheduler persistence."""

    def test_tasks_persisted_and_loaded(self):
        """Test that tasks are saved and can be reloaded."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_file.close()

        try:
            from scheduler import TaskScheduler, SchedulerConfig

            # Create scheduler and add tasks
            config = SchedulerConfig(
                enabled=True,
                storage_path=temp_file.name,
            )
            scheduler1 = TaskScheduler(config=config)

            scheduler1.schedule_task(goal="Persistent task 1", delay_seconds=3600)
            scheduler1.schedule_task(goal="Persistent task 2", delay_seconds=3600)

            # Create new scheduler with same storage
            scheduler2 = TaskScheduler(config=config)

            # Should have loaded the tasks
            self.assertEqual(len(scheduler2.get_pending_tasks()), 2)

        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


class TestCreateScheduler(unittest.TestCase):
    """Tests for the convenience function."""

    def test_create_scheduler(self):
        """Test creating a scheduler with convenience function."""
        from scheduler import create_scheduler

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_file.close()

        try:
            scheduler = create_scheduler(
                storage_path=temp_file.name,
                max_concurrent=3,
            )

            self.assertTrue(scheduler.config.enabled)
            self.assertEqual(scheduler.config.max_concurrent_tasks, 3)

        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


if __name__ == "__main__":
    unittest.main()

"""
Integration Tests for DroidRun Daemon.

These tests verify that daemon components work together correctly.
They don't require a real device but test the integration points.
"""

import asyncio
import importlib.util
import os
import shutil
import tempfile
import unittest
import time

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent_dir)


def load_module(name, path):
    """Load a module directly from file path to avoid import chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules directly to avoid droidrun.__init__.py import chain
daemon_module = load_module('daemon', os.path.join(_parent_dir, 'droidrun/daemon.py'))
scheduler_module = load_module('scheduler', os.path.join(_parent_dir, 'droidrun/scheduler.py'))
resources_module = load_module('resources', os.path.join(_parent_dir, 'droidrun/agent/resources.py'))


class TestDaemonInitialization(unittest.TestCase):
    """Tests for daemon initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.test_dir, "tasks.json")
        self.state_file = os.path.join(self.test_dir, "state.json")
        self.escalation_file = os.path.join(self.test_dir, "escalations.json")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_daemon_config_defaults(self):
        """Test daemon config has correct defaults."""
        DaemonConfig = daemon_module.DaemonConfig

        config = DaemonConfig()

        self.assertTrue(config.scheduler_enabled)
        self.assertTrue(config.resource_monitoring_enabled)
        self.assertTrue(config.escalation_enabled)
        self.assertEqual(config.min_battery_percent, 20)
        self.assertEqual(config.max_consecutive_failures, 3)

    def test_daemon_creation(self):
        """Test daemon can be created."""
        DroidRunDaemon = daemon_module.DroidRunDaemon
        DaemonConfig = daemon_module.DaemonConfig

        config = DaemonConfig(
            task_storage_path=self.task_file,
            state_file=self.state_file,
            escalation_storage_path=self.escalation_file,
        )

        daemon = DroidRunDaemon(config=config)

        self.assertIsNotNone(daemon)
        self.assertFalse(daemon._running)
        self.assertIsNone(daemon._scheduler)

    def test_daemon_init_status_tracking(self):
        """Test that daemon tracks initialization status."""
        DroidRunDaemon = daemon_module.DroidRunDaemon
        DaemonConfig = daemon_module.DaemonConfig

        config = DaemonConfig()
        daemon = DroidRunDaemon(config=config)

        # All should be False initially
        self.assertFalse(daemon._init_status["scheduler"])
        self.assertFalse(daemon._init_status["resource_monitor"])
        self.assertFalse(daemon._init_status["escalation_queue"])
        self.assertFalse(daemon._init_status["adb_tools"])

    def test_daemon_start_stop(self):
        """Test daemon can start and stop."""
        DroidRunDaemon = daemon_module.DroidRunDaemon
        DaemonConfig = daemon_module.DaemonConfig

        config = DaemonConfig(
            task_storage_path=self.task_file,
            state_file=self.state_file,
            escalation_storage_path=self.escalation_file,
            scheduler_enabled=False,  # Disable to avoid dependencies
            resource_monitoring_enabled=False,
            escalation_enabled=False,
        )

        daemon = DroidRunDaemon(config=config)

        async def run_test():
            # Start daemon in background
            start_task = asyncio.create_task(daemon.start())

            # Let it run briefly
            await asyncio.sleep(0.5)

            # Stop daemon
            await daemon.stop()

            # Wait for start task to complete
            try:
                await asyncio.wait_for(start_task, timeout=2.0)
            except asyncio.TimeoutError:
                start_task.cancel()

            return daemon._running

        is_running = asyncio.run(run_test())
        self.assertFalse(is_running)


class TestSchedulerIntegration(unittest.TestCase):
    """Tests for scheduler integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.test_dir, "tasks.json")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_scheduler_has_execute_due_tasks(self):
        """Test scheduler has execute_due_tasks method."""
        TaskScheduler = scheduler_module.TaskScheduler

        scheduler = TaskScheduler()
        self.assertTrue(hasattr(scheduler, 'execute_due_tasks'))
        self.assertTrue(asyncio.iscoroutinefunction(scheduler.execute_due_tasks))

    def test_scheduler_has_cleanup_old_tasks(self):
        """Test scheduler has cleanup_old_tasks method."""
        TaskScheduler = scheduler_module.TaskScheduler

        scheduler = TaskScheduler()
        self.assertTrue(hasattr(scheduler, 'cleanup_old_tasks'))

    def test_scheduler_dependency_check(self):
        """Test scheduler checks dependencies."""
        TaskScheduler = scheduler_module.TaskScheduler
        SchedulerConfig = scheduler_module.SchedulerConfig
        TaskPriority = scheduler_module.TaskPriority

        config = SchedulerConfig(
            storage_path=self.task_file,
            enabled=True,
        )
        scheduler = TaskScheduler(config=config)

        # Create parent task
        parent_id = scheduler.schedule_task(
            goal="Parent task",
            priority=TaskPriority.NORMAL,
        )

        # Create dependent task
        child_id = scheduler.schedule_dependent_task(
            goal="Child task",
            depends_on=[parent_id],
            priority=TaskPriority.NORMAL,
        )

        # Dependency should not be met (parent not completed)
        self.assertFalse(scheduler.are_dependencies_met(child_id))

        # Mark parent as completed
        parent_task = scheduler.get_task(parent_id)
        parent_task.status = "completed"

        # Now dependency should be met
        self.assertTrue(scheduler.are_dependencies_met(child_id))

    def test_scheduler_cleanup(self):
        """Test scheduler cleanup_old_tasks."""
        TaskScheduler = scheduler_module.TaskScheduler
        SchedulerConfig = scheduler_module.SchedulerConfig
        TaskPriority = scheduler_module.TaskPriority

        config = SchedulerConfig(
            storage_path=self.task_file,
            enabled=True,
        )
        scheduler = TaskScheduler(config=config)

        # Create and complete some tasks
        for i in range(10):
            task_id = scheduler.schedule_task(
                goal=f"Task {i}",
                priority=TaskPriority.NORMAL,
            )
            task = scheduler.get_task(task_id)
            task.status = "completed"
            task.completed_at = time.time() - (40 * 24 * 60 * 60)  # 40 days ago

        # Cleanup tasks older than 30 days
        removed = scheduler.cleanup_old_tasks(max_age_days=30)

        self.assertEqual(removed, 10)


class TestResourceMonitorIntegration(unittest.TestCase):
    """Tests for resource monitor integration."""

    def test_resource_monitor_without_tools(self):
        """Test resource monitor works without tools (limited functionality)."""
        ResourceMonitor = resources_module.ResourceMonitor
        ResourceConfig = resources_module.ResourceConfig

        config = ResourceConfig(enabled=True)
        monitor = ResourceMonitor(config=config, tools_instance=None)

        self.assertFalse(monitor.is_paused)

    def test_device_resources_can_execute(self):
        """Test device resources can_execute_tasks."""
        DeviceResources = resources_module.DeviceResources
        ResourceConfig = resources_module.ResourceConfig

        config = ResourceConfig(min_battery_percent=20)

        # Good resources (need to set storage too - default is 0 which fails)
        resources = DeviceResources(battery_level=80, is_charging=False, storage_available_mb=1000)
        can_execute, reason = resources.can_execute_tasks(config)
        self.assertTrue(can_execute)
        self.assertEqual(reason, "OK")

        # Low battery
        resources = DeviceResources(battery_level=10, is_charging=False, storage_available_mb=1000)
        can_execute, reason = resources.can_execute_tasks(config)
        self.assertFalse(can_execute)
        self.assertIn("Battery", reason)

        # Low battery but charging - should be OK
        resources = DeviceResources(battery_level=10, is_charging=True, storage_available_mb=1000)
        can_execute, reason = resources.can_execute_tasks(config)
        self.assertTrue(can_execute)


class TestEscalationQueueIntegration(unittest.TestCase):
    """Tests for escalation queue integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.escalation_file = os.path.join(self.test_dir, "escalations.json")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_escalation_queue_escalate_resolve(self):
        """Test escalation queue escalate and resolve."""
        HumanEscalationQueue = resources_module.HumanEscalationQueue

        queue = HumanEscalationQueue(storage_path=self.escalation_file)

        # Escalate
        escalation_id = queue.escalate(
            task_id="test_task",
            reason="Test escalation",
            context={"test": True},
        )

        self.assertIsNotNone(escalation_id)

        # Check pending
        pending = queue.get_pending()
        self.assertEqual(len(pending), 1)

        # Resolve
        resolved = queue.resolve(escalation_id, "Fixed it")
        self.assertTrue(resolved)

        # Check pending again
        pending = queue.get_pending()
        self.assertEqual(len(pending), 0)

    def test_escalation_queue_cleanup(self):
        """Test escalation queue cleanup."""
        HumanEscalationQueue = resources_module.HumanEscalationQueue

        queue = HumanEscalationQueue(storage_path=self.escalation_file)

        # Create and resolve some escalations
        for i in range(10):
            esc_id = queue.escalate(
                task_id=f"task_{i}",
                reason=f"Test {i}",
            )
            queue.resolve(esc_id, "Fixed")

            # Manually set old resolved_at time
            for item in queue._queue:
                if item.escalation_id == esc_id:
                    item.resolved_at = time.time() - (40 * 24 * 60 * 60)  # 40 days ago

        # Cleanup items older than 30 days
        removed = queue.cleanup_old_items(max_age_days=30)

        self.assertEqual(removed, 10)


class TestStateManagement(unittest.TestCase):
    """Tests for state management."""

    def test_prune_history_exists(self):
        """Test prune_history method exists on DroidAgentState."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            'state',
            os.path.join(_parent_dir, 'droidrun/agent/droid/state.py')
        )
        state_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_module)

        state = state_module.DroidAgentState(instruction="Test")
        self.assertTrue(hasattr(state, 'prune_history'))

    def test_prune_history_works(self):
        """Test prune_history actually prunes."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            'state',
            os.path.join(_parent_dir, 'droidrun/agent/droid/state.py')
        )
        state_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_module)

        state = state_module.DroidAgentState(instruction="Test")

        # Add lots of history
        for i in range(200):
            state.action_history.append(f"action_{i}")

        # Prune
        pruned = state.prune_history()

        # Should have pruned to MAX_ACTION_HISTORY
        self.assertLessEqual(len(state.action_history), state_module.MAX_ACTION_HISTORY)
        self.assertGreater(pruned["action_history"], 0)


class TestEndToEndFlow(unittest.TestCase):
    """End-to-end flow tests (without real device)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.test_dir, "tasks.json")
        self.state_file = os.path.join(self.test_dir, "state.json")
        self.escalation_file = os.path.join(self.test_dir, "escalations.json")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_daemon_logs_init_status(self):
        """Test daemon logs initialization status correctly."""
        DroidRunDaemon = daemon_module.DroidRunDaemon
        DaemonConfig = daemon_module.DaemonConfig
        import io
        import logging

        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        logging.getLogger("droidrun.daemon").addHandler(handler)

        config = DaemonConfig(
            task_storage_path=self.task_file,
            state_file=self.state_file,
            escalation_storage_path=self.escalation_file,
            scheduler_enabled=False,
            resource_monitoring_enabled=False,
            escalation_enabled=False,
        )

        daemon = DroidRunDaemon(config=config)
        daemon._log_init_status()

        log_output = log_capture.getvalue()

        # Should log status or warnings for components
        # When components aren't initialized, it logs warnings
        self.assertTrue(
            "Daemon Initialization Status" in log_output or
            "SCHEDULER NOT INITIALIZED" in log_output or
            "ADB TOOLS NOT INITIALIZED" in log_output
        )

    def test_scheduler_task_chain(self):
        """Test creating a task chain works."""
        TaskScheduler = scheduler_module.TaskScheduler
        SchedulerConfig = scheduler_module.SchedulerConfig

        config = SchedulerConfig(
            storage_path=self.task_file,
            enabled=True,
        )
        scheduler = TaskScheduler(config=config)

        # Create a chain of 3 tasks
        task_ids = scheduler.create_task_chain(
            goals=["Step 1", "Step 2", "Step 3"],
        )

        self.assertEqual(len(task_ids), 3)

        # Verify dependencies
        task1 = scheduler.get_task(task_ids[0])
        task2 = scheduler.get_task(task_ids[1])
        task3 = scheduler.get_task(task_ids[2])

        self.assertEqual(task1.depends_on, [])
        self.assertEqual(task2.depends_on, [task_ids[0]])
        self.assertEqual(task3.depends_on, [task_ids[1]])


class TestExecutorHeartbeat(unittest.TestCase):
    """Tests for executor heartbeat functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.test_dir, "tasks.json")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_droid_scheduler_wires_executor_to_scheduler(self):
        """Test that create_droid_scheduler wires executor back to scheduler."""
        create_droid_scheduler = scheduler_module.create_droid_scheduler

        scheduler = create_droid_scheduler(
            storage_path=self.task_file,
            max_concurrent=1,
        )

        # Verify executor has scheduler reference
        self.assertIsNotNone(scheduler.task_executor)
        self.assertIsNotNone(scheduler.task_executor._scheduler)
        self.assertIs(scheduler.task_executor._scheduler, scheduler)

    def test_update_task_progress_updates_heartbeat(self):
        """Test that update_task_progress updates the heartbeat."""
        TaskScheduler = scheduler_module.TaskScheduler
        SchedulerConfig = scheduler_module.SchedulerConfig
        TaskPriority = scheduler_module.TaskPriority

        config = SchedulerConfig(
            storage_path=self.task_file,
            enabled=True,
        )
        scheduler = TaskScheduler(config=config)

        # Schedule a task
        task_id = scheduler.schedule_task(
            goal="Test heartbeat",
            priority=TaskPriority.NORMAL,
        )

        # Update progress
        scheduler.update_task_progress(
            task_id=task_id,
            current_step=5,
            progress_percent=50.0,
            last_action="Test action",
        )

        # Verify heartbeat was updated
        task = scheduler.get_task(task_id)
        self.assertIsNotNone(task.last_heartbeat)
        self.assertEqual(task.current_step, 5)
        self.assertEqual(task.progress_percent, 50.0)
        self.assertEqual(task.last_action, "Test action")

    def test_stale_task_detection_respects_heartbeat(self):
        """Test that stale task detection respects recent heartbeats."""
        TaskScheduler = scheduler_module.TaskScheduler
        SchedulerConfig = scheduler_module.SchedulerConfig
        TaskPriority = scheduler_module.TaskPriority

        config = SchedulerConfig(
            storage_path=self.task_file,
            enabled=True,
            stale_task_timeout_seconds=1.0,  # Very short for testing
        )
        scheduler = TaskScheduler(config=config)

        # Schedule and mark as running
        task_id = scheduler.schedule_task(
            goal="Test stale detection",
            priority=TaskPriority.NORMAL,
        )
        task = scheduler.get_task(task_id)
        task.status = "running"
        task.started_at = time.time()

        # Immediately update heartbeat
        scheduler.update_task_progress(
            task_id=task_id,
            last_action="Still running",
        )

        # Check stale - should not be stale since heartbeat is fresh
        async def check():
            stale = await scheduler.check_stale_tasks()
            return stale

        stale_tasks = asyncio.run(check())
        self.assertEqual(len(stale_tasks), 0)


if __name__ == "__main__":
    unittest.main()

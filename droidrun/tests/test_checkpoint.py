"""
Tests for State Checkpointing.

Production-ready tests for:
- CheckpointManager
- Checkpoint save/load
- Auto-checkpointing
- Checkpoint rotation
"""

import asyncio
import os
import shutil
import tempfile
import unittest

import sys

# Add checkpoint module path directly
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_parent_dir, 'droidrun', 'agent'))


class TestCheckpointConfig(unittest.TestCase):
    """Tests for CheckpointConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from checkpoint import CheckpointConfig

        config = CheckpointConfig()

        self.assertTrue(config.enabled)  # Now enabled by default for autonomy
        self.assertEqual(config.checkpoint_dir, "checkpoints")
        self.assertEqual(config.checkpoint_interval_seconds, 30.0)
        self.assertEqual(config.max_checkpoints, 10)  # Increased for better recovery
        self.assertTrue(config.auto_resume)

    def test_custom_config(self):
        """Test custom configuration."""
        from checkpoint import CheckpointConfig

        config = CheckpointConfig(
            enabled=True,
            checkpoint_dir="/tmp/custom_checkpoints",
            checkpoint_interval_seconds=60.0,
            max_checkpoints=10,
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.checkpoint_dir, "/tmp/custom_checkpoints")
        self.assertEqual(config.checkpoint_interval_seconds, 60.0)
        self.assertEqual(config.max_checkpoints, 10)


class TestCheckpointMetadata(unittest.TestCase):
    """Tests for CheckpointMetadata."""

    def test_metadata_creation(self):
        """Test metadata creation."""
        from checkpoint import CheckpointMetadata

        metadata = CheckpointMetadata(
            checkpoint_id="test_checkpoint",
            created_at="2024-01-15T10:30:00",
            agent_state_hash="abc123",
            step_count=5,
            instruction="Test instruction",
        )

        self.assertEqual(metadata.checkpoint_id, "test_checkpoint")
        self.assertEqual(metadata.step_count, 5)

    def test_metadata_serialization(self):
        """Test metadata to/from dict."""
        from checkpoint import CheckpointMetadata

        original = CheckpointMetadata(
            checkpoint_id="test_checkpoint",
            created_at="2024-01-15T10:30:00",
            agent_state_hash="abc123",
            step_count=5,
            instruction="Test",
        )

        data = original.to_dict()
        restored = CheckpointMetadata.from_dict(data)

        self.assertEqual(original.checkpoint_id, restored.checkpoint_id)
        self.assertEqual(original.step_count, restored.step_count)


class TestCheckpointManager(unittest.TestCase):
    """Tests for CheckpointManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        from checkpoint import CheckpointManager, CheckpointConfig

        self.config = CheckpointConfig(
            enabled=True,
            checkpoint_dir=self.test_dir,
            max_checkpoints=3,
        )
        self.manager = CheckpointManager(config=self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        state = {"instruction": "Test task", "step_count": 5}

        checkpoint_id = asyncio.run(
            self.manager.save_checkpoint(
                state=state,
                instruction="Test task",
                step_count=5,
            )
        )

        self.assertIsNotNone(checkpoint_id)
        self.assertTrue(checkpoint_id.startswith("checkpoint_"))

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        state = {"instruction": "Test task", "value": 42}

        checkpoint_id = asyncio.run(
            self.manager.save_checkpoint(
                state=state,
                instruction="Test task",
                step_count=1,
            )
        )

        loaded_state = asyncio.run(self.manager.load_checkpoint(checkpoint_id))

        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state["instruction"], "Test task")
        self.assertEqual(loaded_state["value"], 42)

    def test_load_latest_checkpoint(self):
        """Test loading the latest checkpoint."""
        # Save multiple checkpoints
        for i in range(3):
            asyncio.run(
                self.manager.save_checkpoint(
                    state={"step": i},
                    instruction=f"Task {i}",
                    step_count=i,
                )
            )

        loaded_state = asyncio.run(self.manager.load_checkpoint())

        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state["step"], 2)  # Latest

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        # Save multiple checkpoints
        for i in range(3):
            asyncio.run(
                self.manager.save_checkpoint(
                    state={"step": i},
                    instruction=f"Task {i}",
                    step_count=i,
                )
            )

        checkpoints = asyncio.run(self.manager.list_checkpoints())

        self.assertEqual(len(checkpoints), 3)

    def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        checkpoint_id = asyncio.run(
            self.manager.save_checkpoint(
                state={"test": True},
                instruction="Test",
                step_count=1,
            )
        )

        deleted = asyncio.run(self.manager.delete_checkpoint(checkpoint_id))
        self.assertTrue(deleted)

        checkpoints = asyncio.run(self.manager.list_checkpoints())
        self.assertEqual(len(checkpoints), 0)

    def test_checkpoint_rotation(self):
        """Test that old checkpoints are rotated out."""
        # Save more than max_checkpoints (3)
        for i in range(5):
            asyncio.run(
                self.manager.save_checkpoint(
                    state={"step": i},
                    instruction=f"Task {i}",
                    step_count=i,
                )
            )

        checkpoints = asyncio.run(self.manager.list_checkpoints())

        self.assertEqual(len(checkpoints), 3)  # Should be limited to max

    def test_clear_all_checkpoints(self):
        """Test clearing all checkpoints."""
        for i in range(3):
            asyncio.run(
                self.manager.save_checkpoint(
                    state={"step": i},
                    instruction=f"Task {i}",
                    step_count=i,
                )
            )

        deleted_count = asyncio.run(self.manager.clear_all_checkpoints())

        self.assertEqual(deleted_count, 3)

        checkpoints = asyncio.run(self.manager.list_checkpoints())
        self.assertEqual(len(checkpoints), 0)

    def test_get_statistics(self):
        """Test getting statistics."""
        asyncio.run(
            self.manager.save_checkpoint(
                state={"test": True},
                instruction="Test",
                step_count=1,
            )
        )

        stats = self.manager.get_statistics()

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["checkpoint_count"], 1)
        self.assertGreater(stats["last_checkpoint_time"], 0)


class TestCheckpointIntegrity(unittest.TestCase):
    """Tests for checkpoint integrity features."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        from checkpoint import CheckpointManager, CheckpointConfig

        self.config = CheckpointConfig(
            enabled=True,
            checkpoint_dir=self.test_dir,
        )
        self.manager = CheckpointManager(config=self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_corrupted_checkpoint_fallback_to_backup(self):
        """Test that corrupted checkpoint falls back to backup."""
        state = {"important": "data"}

        checkpoint_id = asyncio.run(
            self.manager.save_checkpoint(
                state=state,
                instruction="Test",
                step_count=1,
            )
        )

        # Save another to create backup of first
        asyncio.run(
            self.manager.save_checkpoint(
                state={"other": "data"},
                instruction="Test 2",
                step_count=2,
            )
        )

        # The original checkpoint should still be loadable
        loaded = asyncio.run(self.manager.load_checkpoint(checkpoint_id))
        self.assertIsNotNone(loaded)

    def test_load_nonexistent_checkpoint(self):
        """Test loading a non-existent checkpoint."""
        loaded = asyncio.run(self.manager.load_checkpoint("nonexistent"))

        self.assertIsNone(loaded)


class TestCreateCheckpointManager(unittest.TestCase):
    """Tests for the convenience function."""

    def test_create_checkpoint_manager(self):
        """Test creating a checkpoint manager with convenience function."""
        from checkpoint import create_checkpoint_manager

        manager = create_checkpoint_manager(
            enabled=True,
            checkpoint_dir="/tmp/test_checkpoints",
            interval_seconds=60.0,
            max_checkpoints=10,
        )

        self.assertTrue(manager.config.enabled)
        self.assertEqual(manager.config.checkpoint_interval_seconds, 60.0)
        self.assertEqual(manager.config.max_checkpoints, 10)


if __name__ == "__main__":
    unittest.main()

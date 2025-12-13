"""
Tests for Memory System.

Production-ready tests for:
- EpisodeRecord
- MemoryManager
- InMemoryStore
- Embedding providers
"""

import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

# Add memory module path directly to bypass llama_index dependency
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_parent_dir, 'droidrun', 'agent', 'memory'))


class TestEpisodeRecord(unittest.TestCase):
    """Tests for EpisodeRecord dataclass."""

    def test_episode_record_creation(self):
        """Test basic episode record creation."""
        from memory_manager import EpisodeRecord

        episode = EpisodeRecord(
            task="Open Instagram",
            goal="Post a photo",
            final_success=True,
            final_reason="Successfully posted",
            steps=5,
        )

        self.assertEqual(episode.task, "Open Instagram")
        self.assertEqual(episode.goal, "Post a photo")
        self.assertTrue(episode.final_success)
        self.assertEqual(episode.steps, 5)
        self.assertIsInstance(episode.id, str)
        self.assertIsInstance(episode.timestamp, datetime)

    def test_episode_record_to_dict(self):
        """Test episode record serialization."""
        from memory_manager import EpisodeRecord

        episode = EpisodeRecord(
            task="Test task",
            goal="Test goal",
            final_success=True,
            final_reason="Done",
            steps=3,
            tags=["test", "automation"],
        )

        data = episode.to_dict()

        self.assertIn("id", data)
        self.assertEqual(data["task"], "Test task")
        self.assertEqual(data["goal"], "Test goal")
        self.assertTrue(data["final_success"])
        self.assertEqual(data["tags"], ["test", "automation"])
        self.assertIsInstance(data["timestamp"], str)

    def test_episode_record_from_dict(self):
        """Test episode record deserialization."""
        from memory_manager import EpisodeRecord

        data = {
            "id": "test-id-123",
            "task": "Test task",
            "goal": "Test goal",
            "actions": [],
            "outcomes": [True],
            "final_success": True,
            "final_reason": "Done",
            "steps": 3,
            "duration_seconds": 10.5,
            "device_state": {},
            "learned_patterns": ["pattern1"],
            "errors": [],
            "timestamp": "2024-01-15T10:30:00",
            "tags": ["test"],
        }

        episode = EpisodeRecord.from_dict(data)

        self.assertEqual(episode.id, "test-id-123")
        self.assertEqual(episode.task, "Test task")
        self.assertTrue(episode.final_success)
        self.assertEqual(episode.learned_patterns, ["pattern1"])

    def test_episode_record_to_summary(self):
        """Test episode summary generation."""
        from memory_manager import EpisodeRecord

        episode = EpisodeRecord(
            task="Open app and login",
            goal="Access account",
            final_success=True,
            final_reason="Login successful",
            steps=4,
            actions=[
                {"action": "tap"},
                {"action": "input_text"},
                {"action": "tap"},
            ],
            learned_patterns=["Use tap for buttons"],
            errors=[],
        )

        summary = episode.to_summary()

        self.assertIn("Open app and login", summary)
        self.assertIn("succeeded", summary)
        self.assertIn("4 steps", summary)


class TestInMemoryStore(unittest.TestCase):
    """Tests for InMemoryStore."""

    def setUp(self):
        """Set up test fixtures."""
        from stores import InMemoryStore, MemoryEntry

        self.store = InMemoryStore(max_entries=100)
        self.MemoryEntry = MemoryEntry

    def test_add_entry(self):
        """Test adding a memory entry."""
        entry = self.MemoryEntry(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test"},
        )

        result = asyncio.run(self.store.add(entry))

        self.assertEqual(result, "test-1")

    def test_get_entry(self):
        """Test retrieving a memory entry."""
        entry = self.MemoryEntry(
            id="test-2",
            content="Test content 2",
            embedding=[0.1, 0.2, 0.3],
        )

        asyncio.run(self.store.add(entry))
        retrieved = asyncio.run(self.store.get("test-2"))

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Test content 2")

    def test_get_nonexistent_entry(self):
        """Test retrieving a non-existent entry."""
        retrieved = asyncio.run(self.store.get("nonexistent"))

        self.assertIsNone(retrieved)

    def test_delete_entry(self):
        """Test deleting a memory entry."""
        entry = self.MemoryEntry(
            id="test-3",
            content="Test content 3",
            embedding=[0.1, 0.2, 0.3],
        )

        asyncio.run(self.store.add(entry))
        deleted = asyncio.run(self.store.delete("test-3"))

        self.assertTrue(deleted)

        retrieved = asyncio.run(self.store.get("test-3"))
        self.assertIsNone(retrieved)

    def test_search_entries(self):
        """Test searching for similar entries."""
        # Add multiple entries with different embeddings
        entries = [
            self.MemoryEntry(
                id=f"search-{i}",
                content=f"Content {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],
            )
            for i in range(1, 5)
        ]

        for entry in entries:
            asyncio.run(self.store.add(entry))

        # Search
        results = asyncio.run(self.store.search(
            query_embedding=[0.3, 0.6, 0.9],
            top_k=2,
        ))

        self.assertEqual(len(results), 2)
        # Results should be sorted by similarity

    def test_count_entries(self):
        """Test counting entries."""
        for i in range(5):
            entry = self.MemoryEntry(
                id=f"count-{i}",
                content=f"Content {i}",
                embedding=[0.1, 0.2, 0.3],
            )
            asyncio.run(self.store.add(entry))

        count = asyncio.run(self.store.count())

        self.assertEqual(count, 5)

    def test_clear_entries(self):
        """Test clearing all entries."""
        for i in range(3):
            entry = self.MemoryEntry(
                id=f"clear-{i}",
                content=f"Content {i}",
                embedding=[0.1, 0.2, 0.3],
            )
            asyncio.run(self.store.add(entry))

        deleted_count = asyncio.run(self.store.clear())

        self.assertEqual(deleted_count, 3)

        count = asyncio.run(self.store.count())
        self.assertEqual(count, 0)

    def test_max_entries_enforcement(self):
        """Test that max_entries limit is enforced."""
        small_store = self.store.__class__(max_entries=3)

        for i in range(5):
            entry = self.MemoryEntry(
                id=f"limit-{i}",
                content=f"Content {i}",
                embedding=[0.1, 0.2, 0.3],
            )
            asyncio.run(small_store.add(entry))

        count = asyncio.run(small_store.count())

        self.assertEqual(count, 3)


class TestLocalEmbeddingProvider(unittest.TestCase):
    """Tests for LocalEmbeddingProvider."""

    def test_fallback_embedding(self):
        """Test fallback hash-based embedding."""
        from embeddings import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(fallback_dimension=128)
        provider._use_fallback = True  # Force fallback mode
        provider._dimension = 128  # Reset dimension to fallback value

        # Even if model loading fails, should work with fallback
        embeddings = provider.embed_sync(["test text"])

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 128)

    def test_deterministic_fallback(self):
        """Test that fallback embeddings are deterministic."""
        from embeddings import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(fallback_dimension=64)
        provider._use_fallback = True  # Force fallback

        emb1 = provider.embed_sync(["same text"])
        emb2 = provider.embed_sync(["same text"])

        self.assertEqual(emb1, emb2)

    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        from embeddings import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(fallback_dimension=64)
        provider._use_fallback = True

        emb1 = provider.embed_sync(["text one"])
        emb2 = provider.embed_sync(["text two"])

        self.assertNotEqual(emb1, emb2)


class TestMemoryManager(unittest.TestCase):
    """Tests for MemoryManager."""

    def setUp(self):
        """Set up test fixtures."""
        from memory_manager import MemoryManager, MemoryConfig

        self.config = MemoryConfig(
            store_type="in_memory",
            use_local_embeddings=True,
            similarity_threshold=0.0,  # Accept all for testing
        )
        self.manager = MemoryManager(config=self.config)

    def test_store_episode(self):
        """Test storing an episode."""
        from memory_manager import EpisodeRecord

        episode = EpisodeRecord(
            task="Test task",
            goal="Test goal",
            final_success=True,
            final_reason="Done",
            steps=2,
        )

        entry_id = asyncio.run(self.manager.store_episode(episode))

        self.assertIsInstance(entry_id, str)
        self.assertTrue(len(entry_id) > 0)

    def test_recall_similar(self):
        """Test recalling similar episodes."""
        from memory_manager import EpisodeRecord

        # Store some episodes
        episodes = [
            EpisodeRecord(
                task="Open Instagram app",
                goal="Post a photo",
                final_success=True,
                final_reason="Posted successfully",
                steps=5,
            ),
            EpisodeRecord(
                task="Open Twitter app",
                goal="Tweet something",
                final_success=True,
                final_reason="Tweeted",
                steps=3,
            ),
        ]

        for ep in episodes:
            asyncio.run(self.manager.store_episode(ep))

        # Recall similar
        results = asyncio.run(self.manager.recall_similar(
            query="Instagram photo posting",
            top_k=5,
        ))

        self.assertGreater(len(results), 0)

    def test_get_context_for_task(self):
        """Test getting context for a new task."""
        from memory_manager import EpisodeRecord

        # Store an episode
        episode = EpisodeRecord(
            task="Send email",
            goal="Compose and send",
            final_success=True,
            final_reason="Email sent",
            steps=4,
            learned_patterns=["Use compose button"],
        )

        asyncio.run(self.manager.store_episode(episode))

        # Get context
        context = asyncio.run(self.manager.get_context_for_task(
            task="Send another email",
            goal="Email composition",
        ))

        # Context might be empty if similarity threshold not met
        self.assertIsInstance(context, str)

    def test_learn_pattern(self):
        """Test learning a pattern."""
        entry_id = asyncio.run(self.manager.learn_pattern(
            pattern="Always tap the blue button for confirmation",
            source_task="Button testing",
            confidence=0.9,
            tags=["ui", "buttons"],
        ))

        self.assertIsInstance(entry_id, str)

    def test_get_statistics(self):
        """Test getting memory statistics."""
        stats = asyncio.run(self.manager.get_statistics())

        self.assertIn("total_memories", stats)
        self.assertIn("store_type", stats)
        self.assertIn("embedding_dimension", stats)


class TestMemoryManagerOffline(unittest.TestCase):
    """Tests for MemoryManager in offline mode."""

    def test_offline_configuration(self):
        """Test offline configuration."""
        from memory_manager import MemoryManager, MemoryConfig

        config = MemoryConfig(
            store_type="in_memory",
            use_local_embeddings=True,
        )

        manager = MemoryManager(config=config)

        # Should work without any external dependencies
        self.assertIsNotNone(manager._store)
        self.assertIsNotNone(manager.embedder)


if __name__ == "__main__":
    unittest.main()

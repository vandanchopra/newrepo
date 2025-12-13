"""
Memory Manager for DroidRun Agent.

Production-ready episodic memory system with:
- Long-term persistence across sessions
- Semantic retrieval for context-aware decisions
- Memory summarization for multi-week operations
- Configurable retention policies
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import logging
import uuid

try:
    from .stores import (
        BaseMemoryStore,
        InMemoryStore,
        MemoryEntry,
    )
    from .embeddings import (
        EmbeddingProvider,
        LocalEmbeddingProvider,
    )
except ImportError:
    # Fallback for standalone testing
    from stores import (
        BaseMemoryStore,
        InMemoryStore,
        MemoryEntry,
    )
    from embeddings import (
        EmbeddingProvider,
        LocalEmbeddingProvider,
    )

logger = logging.getLogger("droidrun.memory")


@dataclass
class EpisodeRecord:
    """
    Record of a single agent episode/task execution.

    Captures task, actions, outcomes, and learned patterns
    for future reference and learning.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    goal: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: List[bool] = field(default_factory=list)
    final_success: bool = False
    final_reason: str = ""
    steps: int = 0
    duration_seconds: float = 0.0
    device_state: Dict[str, Any] = field(default_factory=dict)
    learned_patterns: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "goal": self.goal,
            "actions": self.actions,
            "outcomes": self.outcomes,
            "final_success": self.final_success,
            "final_reason": self.final_reason,
            "steps": self.steps,
            "duration_seconds": self.duration_seconds,
            "device_state": self.device_state,
            "learned_patterns": self.learned_patterns,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeRecord":
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def to_summary(self) -> str:
        """Generate a text summary for embedding."""
        success_str = "succeeded" if self.final_success else "failed"
        actions_str = ", ".join(
            a.get("action", "unknown") for a in self.actions[-5:]
        )

        summary = (
            f"Task: {self.task}. "
            f"Goal: {self.goal}. "
            f"Outcome: {success_str} in {self.steps} steps. "
            f"Reason: {self.final_reason}. "
            f"Key actions: {actions_str}. "
        )

        if self.learned_patterns:
            summary += f"Patterns learned: {', '.join(self.learned_patterns[:3])}. "

        if self.errors:
            summary += f"Errors encountered: {', '.join(self.errors[:3])}. "

        return summary


@dataclass
class MemoryConfig:
    """Configuration for the Memory Manager."""

    # Store configuration
    store_type: str = "in_memory"  # "in_memory" or "qdrant"
    persist_path: str = ".droidrun/memory"
    max_entries: int = 10000

    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    collection_name: str = "droidrun_memory"

    # Embedding configuration
    use_local_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Retrieval configuration
    default_top_k: int = 5
    similarity_threshold: float = 0.5

    # Retention configuration
    max_age_days: int = 90
    auto_cleanup: bool = True
    cleanup_interval_hours: int = 24


class MemoryManager:
    """
    Production-ready memory manager for autonomous agent operation.

    Features:
    - Episodic memory storage and retrieval
    - Semantic search using vector embeddings
    - Automatic memory summarization
    - Long-term persistence
    - Memory retention policies
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        store: Optional[BaseMemoryStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.config = config or MemoryConfig()

        # Initialize embedding provider
        if embedding_provider:
            self.embedder = embedding_provider
        elif self.config.use_local_embeddings:
            self.embedder = LocalEmbeddingProvider(
                model_name=self.config.embedding_model,
                fallback_dimension=self.config.embedding_dimension,
            )
        else:
            self.embedder = LocalEmbeddingProvider()

        # Initialize store
        if store:
            self._store = store
        elif self.config.store_type == "qdrant":
            try:
                from .stores import QdrantMemoryStore
            except ImportError:
                from stores import QdrantMemoryStore
            self._store = QdrantMemoryStore(
                collection_name=self.config.collection_name,
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                vector_size=self.embedder.dimension,
            )
        else:
            self._store = InMemoryStore(
                persist_path=f"{self.config.persist_path}/memories.pkl",
                max_entries=self.config.max_entries,
            )

        self._cleanup_task: Optional[asyncio.Task] = None
        logger.info(f"MemoryManager initialized with {self.config.store_type} store")

    async def start(self):
        """Start background tasks (cleanup, etc.)."""
        if self.config.auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """Periodic cleanup of old memories."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                await self._cleanup_old_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")

    async def _cleanup_old_memories(self):
        """Remove memories older than max_age_days."""
        cutoff = datetime.utcnow() - timedelta(days=self.config.max_age_days)
        count = await self._store.count()

        # For in-memory store, we can iterate
        if isinstance(self._store, InMemoryStore):
            to_delete = []
            for entry_id, entry in self._store._entries.items():
                if entry.timestamp < cutoff:
                    to_delete.append(entry_id)

            for entry_id in to_delete:
                await self._store.delete(entry_id)

            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old memories")

    async def store_episode(self, episode: EpisodeRecord) -> str:
        """
        Store an episode record in memory.

        Args:
            episode: The episode record to store

        Returns:
            The memory entry ID
        """
        # Generate summary and embedding
        summary = episode.to_summary()
        embeddings = await self.embedder.embed([summary])

        # Create memory entry
        entry = MemoryEntry(
            id=episode.id,
            content=summary,
            embedding=embeddings[0],
            metadata={
                "type": "episode",
                "task": episode.task,
                "goal": episode.goal,
                "success": episode.final_success,
                "steps": episode.steps,
                "tags": episode.tags,
                "episode_data": episode.to_dict(),
            },
            timestamp=episode.timestamp,
        )

        entry_id = await self._store.add(entry)
        logger.debug(f"Stored episode {entry_id}: {episode.task[:50]}...")
        return entry_id

    async def recall_similar(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        success_only: bool = False,
    ) -> List[Tuple[EpisodeRecord, float]]:
        """
        Recall episodes similar to the query.

        Args:
            query: Text to search for similar episodes
            top_k: Number of results to return
            filter_tags: Only return episodes with these tags
            success_only: Only return successful episodes

        Returns:
            List of (episode, similarity_score) tuples
        """
        top_k = top_k or self.config.default_top_k

        # Generate query embedding
        embeddings = await self.embedder.embed([query])
        query_embedding = embeddings[0]

        # Build metadata filter
        filter_metadata = {"type": "episode"}
        if success_only:
            filter_metadata["success"] = True

        # Search
        results = await self._store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get extra for filtering
            filter_metadata=filter_metadata,
        )

        # Post-filter and convert to episodes
        episodes = []
        for entry, score in results:
            if score < self.config.similarity_threshold:
                continue

            # Tag filtering
            if filter_tags:
                entry_tags = entry.metadata.get("tags", [])
                if not any(t in entry_tags for t in filter_tags):
                    continue

            # Reconstruct episode
            episode_data = entry.metadata.get("episode_data", {})
            if episode_data:
                episode = EpisodeRecord.from_dict(episode_data)
                episodes.append((episode, score))

        return episodes[:top_k]

    async def get_context_for_task(
        self,
        task: str,
        goal: str,
        max_context_length: int = 2000,
    ) -> str:
        """
        Get relevant context from memory for a new task.

        This summarizes past experiences relevant to the current task
        for inclusion in the agent's context window.

        Args:
            task: Current task description
            goal: Overall goal
            max_context_length: Maximum characters for context

        Returns:
            Formatted context string
        """
        # Search for similar past tasks
        query = f"{task} {goal}"
        similar_episodes = await self.recall_similar(query, top_k=5)

        if not similar_episodes:
            return ""

        # Build context
        context_parts = ["## Relevant Past Experiences\n"]
        current_length = len(context_parts[0])

        for episode, score in similar_episodes:
            entry = (
                f"\n### Similar Task (relevance: {score:.2f})\n"
                f"- Task: {episode.task}\n"
                f"- Outcome: {'Success' if episode.final_success else 'Failed'}\n"
                f"- Steps: {episode.steps}\n"
            )

            if episode.learned_patterns:
                entry += f"- Learned: {', '.join(episode.learned_patterns[:2])}\n"

            if episode.errors and not episode.final_success:
                entry += f"- Errors to avoid: {', '.join(episode.errors[:2])}\n"

            if current_length + len(entry) > max_context_length:
                break

            context_parts.append(entry)
            current_length += len(entry)

        return "".join(context_parts)

    async def learn_pattern(
        self,
        pattern: str,
        source_task: str,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a learned pattern for future reference.

        Args:
            pattern: The pattern or insight learned
            source_task: The task from which it was learned
            confidence: Confidence level (0-1)
            tags: Tags for categorization

        Returns:
            Memory entry ID
        """
        embeddings = await self.embedder.embed([pattern])

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=pattern,
            embedding=embeddings[0],
            metadata={
                "type": "pattern",
                "source_task": source_task,
                "confidence": confidence,
                "tags": tags or [],
            },
        )

        entry_id = await self._store.add(entry)
        logger.info(f"Learned pattern: {pattern[:50]}...")
        return entry_id

    async def get_patterns_for_context(
        self,
        context: str,
        top_k: int = 5,
    ) -> List[str]:
        """
        Retrieve relevant learned patterns for a context.

        Args:
            context: Current context/task description
            top_k: Number of patterns to retrieve

        Returns:
            List of relevant patterns
        """
        embeddings = await self.embedder.embed([context])

        results = await self._store.search(
            query_embedding=embeddings[0],
            top_k=top_k,
            filter_metadata={"type": "pattern"},
        )

        patterns = []
        for entry, score in results:
            if score >= self.config.similarity_threshold:
                patterns.append(entry.content)

        return patterns

    async def summarize_session(
        self,
        session_id: str,
        episodes: List[EpisodeRecord],
    ) -> str:
        """
        Create a summary of a session's episodes.

        Useful for long-running operations spanning days/weeks.

        Args:
            session_id: Session identifier
            episodes: Episodes from the session

        Returns:
            Session summary text
        """
        if not episodes:
            return "No episodes in session."

        total_steps = sum(e.steps for e in episodes)
        successes = sum(1 for e in episodes if e.final_success)
        total_duration = sum(e.duration_seconds for e in episodes)

        # Collect unique patterns and errors
        all_patterns = set()
        all_errors = set()
        for ep in episodes:
            all_patterns.update(ep.learned_patterns)
            all_errors.update(ep.errors)

        summary = (
            f"Session {session_id} Summary:\n"
            f"- Episodes: {len(episodes)} ({successes} successful)\n"
            f"- Total Steps: {total_steps}\n"
            f"- Duration: {total_duration/3600:.1f} hours\n"
            f"- Success Rate: {successes/len(episodes)*100:.1f}%\n"
        )

        if all_patterns:
            summary += f"- Key Patterns: {', '.join(list(all_patterns)[:5])}\n"

        if all_errors:
            summary += f"- Common Issues: {', '.join(list(all_errors)[:5])}\n"

        # Store summary as a memory
        embeddings = await self.embedder.embed([summary])
        entry = MemoryEntry(
            id=f"session_{session_id}",
            content=summary,
            embedding=embeddings[0],
            metadata={
                "type": "session_summary",
                "session_id": session_id,
                "episode_count": len(episodes),
                "success_count": successes,
                "total_steps": total_steps,
            },
        )
        await self._store.add(entry)

        return summary

    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        count = await self._store.count()

        return {
            "total_memories": count,
            "store_type": self.config.store_type,
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": self.embedder.dimension,
            "max_entries": self.config.max_entries,
            "max_age_days": self.config.max_age_days,
        }


# Convenience function for quick setup
def create_memory_manager(
    use_qdrant: bool = False,
    persist_path: str = ".droidrun/memory",
    **kwargs,
) -> MemoryManager:
    """
    Create a memory manager with sensible defaults.

    Args:
        use_qdrant: Use Qdrant instead of in-memory store
        persist_path: Path for persistent storage
        **kwargs: Additional MemoryConfig parameters

    Returns:
        Configured MemoryManager instance
    """
    config = MemoryConfig(
        store_type="qdrant" if use_qdrant else "in_memory",
        persist_path=persist_path,
        **kwargs,
    )
    return MemoryManager(config=config)

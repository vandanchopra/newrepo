"""
Unified Memory System: Titans + ReMe Integration.

This module combines two complementary memory approaches:

1. TITANS (Neural Long-Term Memory)
   - Internal to the agent's "brain"
   - Compresses sequence information into fixed-size memory state
   - Fast test-time learning and retrieval
   - Good for: Pattern recognition, context compression, fast recall

2. REME (Procedural Memory)
   - External experience database
   - Stores "how-to" knowledge from past executions
   - Dynamic refinement with utility-based pruning
   - Good for: Tool-using procedures, success/failure patterns, transferable knowledge

Together they provide:
- Short-term: Attention/context window
- Medium-term: Titans neural memory
- Long-term: ReMe procedural memory

The stack works like human memory:
- Working memory (context) - What you're thinking about now
- Episodic memory (Titans) - Compressed experiences
- Procedural memory (ReMe) - How to do things
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .titans_memory import TitansMemoryModule, TitansConfig, create_titans_memory
    from .reme_memory import (
        ReMeMemory,
        ReMeConfig,
        Experience,
        ExperienceType,
        Trajectory,
        create_reme_memory,
    )
except ImportError:
    # Fallback for standalone testing
    from titans_memory import TitansMemoryModule, TitansConfig, create_titans_memory
    from reme_memory import (
        ReMeMemory,
        ReMeConfig,
        Experience,
        ExperienceType,
        Trajectory,
        create_reme_memory,
    )

logger = logging.getLogger("droidrun.memory.unified")


@dataclass
class UnifiedMemoryConfig:
    """Configuration for unified memory system."""

    # Titans config
    titans_memory_size: int = 384  # Match embedding dimension
    titans_num_slots: int = 64
    titans_persist_path: str = ".droidrun/memory/titans_state.pkl"

    # ReMe config
    reme_persist_path: str = ".droidrun/memory/reme_experiences.json"
    reme_max_experiences: int = 5000

    # Integration settings
    use_titans: bool = True
    use_reme: bool = True

    # Retrieval blending
    titans_weight: float = 0.4  # Weight for Titans in combined retrieval
    reme_weight: float = 0.6  # Weight for ReMe in combined retrieval

    # Embedding
    embedding_dimension: int = 384


class UnifiedMemorySystem:
    """
    Unified Memory combining Titans (neural) and ReMe (procedural).

    This provides a complete memory solution for autonomous agents:
    - Titans handles fast, compressed, internal memory
    - ReMe handles structured procedural knowledge

    Usage:
        memory = UnifiedMemorySystem()
        await memory.start()

        # During task execution
        context = await memory.get_context_for_task(task, goal)

        # After task completion
        await memory.store_trajectory(trajectory)

        # Periodic maintenance
        await memory.consolidate()
    """

    def __init__(
        self,
        config: Optional[UnifiedMemoryConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.config = config or UnifiedMemoryConfig()
        self._embedding_fn = embedding_fn

        # Initialize Titans neural memory
        if self.config.use_titans:
            titans_config = TitansConfig(
                memory_size=self.config.titans_memory_size,
                num_memory_slots=self.config.titans_num_slots,
                persist_path=self.config.titans_persist_path,
            )
            self._titans = TitansMemoryModule(config=titans_config)
        else:
            self._titans = None

        # Initialize ReMe procedural memory
        if self.config.use_reme:
            reme_config = ReMeConfig(
                persist_path=self.config.reme_persist_path,
                max_experiences=self.config.reme_max_experiences,
                embedding_dimension=self.config.embedding_dimension,
            )

            # Wrap embedding function for ReMe (it expects synchronous)
            def sync_embed(text: str) -> List[float]:
                if self._embedding_fn:
                    return self._embedding_fn(text)
                # Fallback: simple hash-based pseudo-embedding
                return self._simple_embed(text)

            self._reme = ReMeMemory(config=reme_config, embedding_fn=sync_embed)
        else:
            self._reme = None

        # Track memory usage
        self._query_count = 0
        self._store_count = 0

    def _simple_embed(self, text: str) -> List[float]:
        """Simple fallback embedding using character hashing."""
        dim = self.config.embedding_dimension
        embedding = [0.0] * dim

        words = text.lower().split()
        for word in words:
            for i, char in enumerate(word):
                idx = (hash(char) + i) % dim
                embedding[idx] += 1.0

        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    async def start(self):
        """Start memory system background tasks."""
        if self._reme:
            await self._reme.start()
        logger.info("Unified memory system started")

    async def stop(self):
        """Stop memory system and save state."""
        if self._reme:
            await self._reme.stop()
        if self._titans:
            self._titans._save_state()
        logger.info("Unified memory system stopped")

    # =========================================================================
    # Main Interface
    # =========================================================================

    async def get_context_for_task(
        self,
        task: str,
        goal: str,
        include_titans: bool = True,
        include_reme: bool = True,
        max_context_length: int = 2000,
    ) -> str:
        """
        Get combined memory context for a new task.

        This is the primary interface for agents to access memory.
        It combines:
        - Titans: Fast neural recall of similar patterns
        - ReMe: Structured procedural knowledge

        Args:
            task: Current task description
            goal: Overall goal
            include_titans: Include Titans neural memory
            include_reme: Include ReMe procedural memory
            max_context_length: Maximum context length

        Returns:
            Formatted context string for inclusion in agent prompt
        """
        context_parts = []
        current_length = 0

        # Get Titans context (compressed neural recall)
        if include_titans and self._titans:
            titans_context = await self._get_titans_context(task, goal)
            if titans_context:
                context_parts.append(titans_context)
                current_length += len(titans_context)

        # Get ReMe context (procedural knowledge)
        if include_reme and self._reme:
            remaining = max_context_length - current_length
            reme_context = await self._reme.get_context_for_task(
                task, goal, max_context_length=remaining
            )
            if reme_context:
                context_parts.append(reme_context)

        self._query_count += 1

        if not context_parts:
            return ""

        return "\n".join(context_parts)

    async def _get_titans_context(self, task: str, goal: str) -> str:
        """Get context from Titans neural memory."""
        if not self._titans:
            return ""

        # Create query embedding
        query_text = f"{task} {goal}"
        query_embedding = np.array(self._simple_embed(query_text), dtype=np.float32)

        # Read from Titans memory
        retrieved, info = await self._titans.read(query_embedding, top_k=3)

        # If we got meaningful retrieval, format it
        if np.sum(np.abs(retrieved)) > 0.1:  # Non-zero retrieval
            # Convert back to interpretable form (this is a limitation)
            # In a full implementation, you'd store metadata alongside embeddings
            return (
                "## Neural Memory Activation\n"
                f"Retrieved pattern from {len(info.get('top_slots', []))} memory slots "
                f"(weights: {info.get('top_weights', [])[:3]})\n"
                "Note: Similar task patterns detected in neural memory.\n"
            )

        return ""

    async def store_trajectory(
        self,
        trajectory: Trajectory,
    ) -> Dict[str, Any]:
        """
        Store a task trajectory in both memory systems.

        This should be called after task completion to learn from experience.

        Args:
            trajectory: The execution trajectory

        Returns:
            Statistics about what was stored
        """
        stats = {
            "titans_stored": False,
            "reme_experiences": 0,
        }

        # Store in Titans (compressed representation)
        if self._titans:
            # Create embedding from trajectory summary
            summary = self._trajectory_to_summary(trajectory)
            embedding = np.array(self._simple_embed(summary), dtype=np.float32)

            write_info = await self._titans.write(
                content=embedding,
                key=embedding,
            )
            stats["titans_stored"] = write_info.get("did_write", False)
            stats["titans_surprise"] = write_info.get("surprise", 0.0)

        # Store in ReMe (structured experiences)
        if self._reme:
            experiences = await self._reme.distill_trajectory(trajectory)
            stats["reme_experiences"] = len(experiences)
            stats["reme_experience_ids"] = [e.id for e in experiences]

        self._store_count += 1
        return stats

    def _trajectory_to_summary(self, trajectory: Trajectory) -> str:
        """Convert trajectory to text summary for Titans."""
        actions = [s.get("action", "")[:50] for s in trajectory.steps[-5:]]
        return (
            f"Task: {trajectory.task}. "
            f"Goal: {trajectory.goal}. "
            f"Outcome: {'success' if trajectory.final_success else 'failure'}. "
            f"Actions: {', '.join(actions)}. "
            f"Reason: {trajectory.final_reason[:100]}."
        )

    async def store_pattern(
        self,
        pattern: str,
        context: str,
        success: bool,
        tags: List[str] = None,
    ):
        """
        Store a learned pattern in both memory systems.

        Use this for explicit pattern learning (not from trajectories).
        """
        # Store in Titans
        if self._titans:
            embedding = np.array(
                self._simple_embed(f"{pattern} {context}"),
                dtype=np.float32
            )
            await self._titans.write(content=embedding, force_write=True)

        # Store in ReMe as procedural knowledge
        if self._reme:
            exp = Experience(
                type=ExperienceType.PROCEDURAL_KNOWLEDGE,
                description=f"Learned pattern: {pattern[:100]}",
                knowledge=pattern,
                context=context,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                tags=tags or [],
            )
            await self._reme._store_experience(exp)

    async def update_experience_outcome(
        self,
        experience_id: str,
        success: bool,
    ):
        """Update ReMe experience based on outcome."""
        if self._reme:
            await self._reme.update_experience_outcome(experience_id, success)

    async def consolidate(self):
        """
        Consolidate both memory systems.

        Should be called periodically (e.g., daily) for maintenance:
        - Titans: Merge similar memory slots
        - ReMe: Prune low-utility experiences
        """
        results = {}

        if self._titans:
            titans_result = await self._titans.consolidate()
            results["titans"] = titans_result

        if self._reme:
            reme_result = await self._reme.refine()
            results["reme"] = reme_result

        logger.info(f"Memory consolidation complete: {results}")
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from both memory systems."""
        stats = {
            "query_count": self._query_count,
            "store_count": self._store_count,
            "titans_enabled": self._titans is not None,
            "reme_enabled": self._reme is not None,
        }

        if self._titans:
            stats["titans"] = self._titans.get_statistics()

        if self._reme:
            stats["reme"] = self._reme.get_statistics()

        return stats

    # =========================================================================
    # Advanced Retrieval
    # =========================================================================

    async def hybrid_retrieval(
        self,
        query: str,
        context_queries: List[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform hybrid retrieval from both memory systems.

        Returns structured results from both Titans and ReMe.
        """
        results = {
            "query": query,
            "titans_retrieval": None,
            "reme_experiences": [],
        }

        # Titans retrieval
        if self._titans:
            query_embedding = np.array(
                self._simple_embed(query),
                dtype=np.float32
            )

            if context_queries:
                context_embeddings = [
                    np.array(self._simple_embed(q), dtype=np.float32)
                    for q in context_queries
                ]
                retrieved, info = await self._titans.read_with_context(
                    query_embedding, context_embeddings
                )
            else:
                retrieved, info = await self._titans.read(query_embedding, top_k=3)

            results["titans_retrieval"] = {
                "info": info,
                "has_activation": bool(np.sum(np.abs(retrieved)) > 0.1),
            }

        # ReMe retrieval
        if self._reme:
            experiences = await self._reme.retrieve(
                task=query,
                context="",
                top_k=top_k,
            )
            results["reme_experiences"] = [
                {
                    "id": exp.id,
                    "type": exp.type.value,
                    "description": exp.description,
                    "knowledge": exp.knowledge[:200],
                    "score": score,
                    "utility": exp.utility_score,
                }
                for exp, score in experiences
            ]

        return results


# Convenience functions
def create_unified_memory(
    persist_dir: str = ".droidrun/memory",
    embedding_fn: Optional[Callable] = None,
    **kwargs
) -> UnifiedMemorySystem:
    """Create a unified memory system with common settings."""
    config = UnifiedMemoryConfig(
        titans_persist_path=f"{persist_dir}/titans_state.pkl",
        reme_persist_path=f"{persist_dir}/reme_experiences.json",
        **kwargs
    )
    return UnifiedMemorySystem(config=config, embedding_fn=embedding_fn)


# For backwards compatibility with existing MemoryManager interface
class EnhancedMemoryManager:
    """
    Enhanced Memory Manager that wraps UnifiedMemorySystem.

    Provides backwards-compatible interface while using Titans + ReMe internally.
    """

    def __init__(
        self,
        config: Optional[UnifiedMemoryConfig] = None,
        embedding_fn: Optional[Callable] = None,
    ):
        self._unified = UnifiedMemorySystem(config=config, embedding_fn=embedding_fn)

    async def start(self):
        await self._unified.start()

    async def stop(self):
        await self._unified.stop()

    async def get_context_for_task(
        self,
        task: str,
        goal: str,
        max_results: int = 5,
    ) -> str:
        return await self._unified.get_context_for_task(task, goal)

    async def store_episode(self, episode) -> str:
        """Convert episode to trajectory and store."""
        trajectory = Trajectory(
            task=episode.task,
            goal=episode.goal,
            steps=[{"action": a.get("action", str(a))} for a in episode.actions],
            final_success=episode.final_success,
            final_reason=episode.final_reason,
            duration_seconds=episode.duration_seconds,
        )
        stats = await self._unified.store_trajectory(trajectory)
        return episode.id

    async def recall_similar(
        self,
        query: str,
        top_k: int = 5,
        success_only: bool = False,
    ):
        if self._unified._reme:
            return await self._unified._reme.retrieve(
                task=query,
                context="",
                top_k=top_k,
                success_only=success_only,
            )
        return []

    async def get_statistics(self) -> Dict[str, Any]:
        return self._unified.get_statistics()

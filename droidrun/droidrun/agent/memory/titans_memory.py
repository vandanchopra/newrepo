"""
Titans-style Neural Long-Term Memory Module.

Based on: "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)

This module provides internal neural memory that:
- Learns to memorize at test time (not just training)
- Acts as long-term memory (complements attention's short-term memory)
- Supports fast parallelizable operations
- Compresses sequential information into fixed-size memory state

Key Concepts from Titans:
- Attention = Short-term memory (accurate but limited context)
- Neural Memory = Long-term memory (compressed, persistent)
- Memory is updated as new information arrives
- Retrieval happens through learned associative mechanisms
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("droidrun.memory.titans")


@dataclass
class TitansConfig:
    """Configuration for Titans Neural Memory."""

    # Memory dimensions
    memory_size: int = 512  # Size of each memory vector
    num_memory_slots: int = 64  # Number of memory slots (like hidden states)

    # Update parameters
    learning_rate: float = 0.1  # How fast to update memory
    momentum: float = 0.9  # Momentum for memory updates
    decay_rate: float = 0.01  # Gradual decay of old memories

    # Retrieval parameters
    num_heads: int = 4  # Multi-head attention for retrieval
    temperature: float = 1.0  # Softmax temperature

    # Surprise-based gating (from Titans paper)
    surprise_threshold: float = 0.5  # Only update on surprising inputs
    use_surprise_gating: bool = True

    # Persistence
    persist_path: Optional[str] = None
    auto_save_interval: int = 100  # Save every N updates


class TitansMemoryModule:
    """
    Neural Long-Term Memory inspired by Titans architecture.

    This provides a differentiable memory that:
    1. Compresses information into fixed-size memory state
    2. Updates memory based on "surprise" (unexpected inputs)
    3. Retrieves relevant memories via attention-like mechanism
    4. Persists across sessions for long-term learning

    The memory acts as a complement to the context window:
    - Context window (attention): Short-term, accurate, limited
    - Neural memory (this): Long-term, compressed, persistent
    """

    def __init__(self, config: Optional[TitansConfig] = None):
        self.config = config or TitansConfig()

        # Initialize memory state
        # M: Main memory matrix [num_slots x memory_size]
        self._memory = np.zeros(
            (self.config.num_memory_slots, self.config.memory_size),
            dtype=np.float32
        )

        # Memory momentum (for stable updates)
        self._momentum = np.zeros_like(self._memory)

        # Memory keys for associative retrieval
        # K: Key matrix for content-based addressing
        self._keys = np.random.randn(
            self.config.num_memory_slots, self.config.memory_size
        ).astype(np.float32) * 0.1

        # Usage statistics for each memory slot
        self._usage = np.zeros(self.config.num_memory_slots, dtype=np.float32)
        self._last_access = np.zeros(self.config.num_memory_slots, dtype=np.float32)

        # Statistics
        self._update_count = 0
        self._retrieval_count = 0
        self._surprise_sum = 0.0

        # Load persisted state if available
        if self.config.persist_path:
            self._load_state()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """L2 normalize vectors."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-8)

    def _compute_surprise(self, query: np.ndarray, retrieved: np.ndarray) -> float:
        """
        Compute surprise score (how unexpected the input is).

        High surprise = input doesn't match existing memories well
        Low surprise = input is similar to what's already memorized

        From Titans: Only update memory on surprising inputs to prevent
        overwriting useful information with redundant data.
        """
        # Cosine similarity between query and retrieved
        query_norm = self._normalize(query.reshape(1, -1))
        retrieved_norm = self._normalize(retrieved.reshape(1, -1))
        similarity = np.dot(query_norm, retrieved_norm.T).item()

        # Surprise is inverse of similarity
        surprise = 1.0 - max(0.0, similarity)
        return surprise

    def _attention_weights(self, query: np.ndarray) -> np.ndarray:
        """
        Compute attention weights over memory slots.

        Uses scaled dot-product attention like in Transformers,
        but over memory slots instead of sequence positions.
        """
        # Query-key attention
        query_norm = self._normalize(query.reshape(1, -1))
        keys_norm = self._normalize(self._keys)

        # Scaled dot-product attention
        scores = np.dot(query_norm, keys_norm.T).squeeze()
        scores = scores / math.sqrt(self.config.memory_size)
        scores = scores / self.config.temperature

        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)

        return weights

    async def write(
        self,
        content: np.ndarray,
        key: Optional[np.ndarray] = None,
        force_write: bool = False,
    ) -> Dict[str, Any]:
        """
        Write information to neural memory.

        The write operation:
        1. Computes surprise to determine if update is needed
        2. Finds least-used memory slot (or most similar for update)
        3. Updates memory with momentum for stability
        4. Updates key for future retrieval

        Args:
            content: Vector to memorize [memory_size]
            key: Optional key for retrieval (defaults to content)
            force_write: Bypass surprise gating

        Returns:
            Dict with write statistics
        """
        # Ensure correct shape
        content = np.asarray(content, dtype=np.float32).flatten()
        if len(content) != self.config.memory_size:
            # Resize if needed (simple truncation/padding)
            if len(content) > self.config.memory_size:
                content = content[:self.config.memory_size]
            else:
                content = np.pad(content, (0, self.config.memory_size - len(content)))

        key = key if key is not None else content
        key = np.asarray(key, dtype=np.float32).flatten()
        if len(key) != self.config.memory_size:
            if len(key) > self.config.memory_size:
                key = key[:self.config.memory_size]
            else:
                key = np.pad(key, (0, self.config.memory_size - len(key)))

        # Retrieve current memory state for surprise computation
        retrieved, _ = await self.read(key)

        # Compute surprise
        surprise = self._compute_surprise(content, retrieved)
        self._surprise_sum += surprise

        # Surprise gating: only update if input is surprising enough
        should_write = force_write or not self.config.use_surprise_gating
        if self.config.use_surprise_gating and surprise > self.config.surprise_threshold:
            should_write = True

        write_info = {
            "surprise": surprise,
            "threshold": self.config.surprise_threshold,
            "did_write": should_write,
        }

        if not should_write:
            return write_info

        # Find slot to write to
        # Strategy: Use least recently accessed slot with low usage
        current_time = time.time()
        slot_scores = (
            -self._usage +  # Prefer unused slots
            0.1 * (current_time - self._last_access)  # Prefer old slots
        )

        # Also consider similarity to existing keys (for updates)
        key_sim = np.dot(self._normalize(self._keys), self._normalize(key.reshape(-1)))
        high_sim_mask = key_sim > 0.9  # Very similar = update existing
        slot_scores[high_sim_mask] += 1.0  # Prefer updating similar slots

        target_slot = np.argmax(slot_scores)

        # Update memory with momentum
        delta = content - self._memory[target_slot]
        self._momentum[target_slot] = (
            self.config.momentum * self._momentum[target_slot] +
            self.config.learning_rate * delta
        )
        self._memory[target_slot] += self._momentum[target_slot]

        # Update key
        key_delta = key - self._keys[target_slot]
        self._keys[target_slot] += self.config.learning_rate * key_delta

        # Update usage stats
        self._usage[target_slot] += 1.0
        self._last_access[target_slot] = current_time

        # Apply decay to all memories
        self._memory *= (1.0 - self.config.decay_rate)

        self._update_count += 1
        write_info["slot"] = int(target_slot)
        write_info["update_count"] = self._update_count

        # Auto-save
        if (self.config.persist_path and
            self._update_count % self.config.auto_save_interval == 0):
            self._save_state()

        return write_info

    async def read(
        self,
        query: np.ndarray,
        top_k: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read from neural memory using attention-based retrieval.

        The read operation:
        1. Computes attention weights over memory slots
        2. Returns weighted combination of memory contents
        3. Updates access statistics

        Args:
            query: Query vector [memory_size]
            top_k: Number of top slots to consider

        Returns:
            Tuple of (retrieved_memory, read_info)
        """
        # Ensure correct shape
        query = np.asarray(query, dtype=np.float32).flatten()
        if len(query) != self.config.memory_size:
            if len(query) > self.config.memory_size:
                query = query[:self.config.memory_size]
            else:
                query = np.pad(query, (0, self.config.memory_size - len(query)))

        # Compute attention weights
        weights = self._attention_weights(query)

        # Get top-k slots
        top_indices = np.argsort(weights)[-top_k:]
        top_weights = weights[top_indices]
        top_weights = top_weights / (np.sum(top_weights) + 1e-8)

        # Weighted combination of memories
        retrieved = np.zeros(self.config.memory_size, dtype=np.float32)
        for idx, w in zip(top_indices, top_weights):
            retrieved += w * self._memory[idx]

        # Update access times
        current_time = time.time()
        for idx in top_indices:
            self._last_access[idx] = current_time

        self._retrieval_count += 1

        read_info = {
            "top_slots": top_indices.tolist(),
            "top_weights": top_weights.tolist(),
            "retrieval_count": self._retrieval_count,
        }

        return retrieved, read_info

    async def read_with_context(
        self,
        query: np.ndarray,
        context_queries: List[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read with multi-query context (like multi-head attention).

        Aggregates retrievals from multiple related queries
        for richer context retrieval.
        """
        all_retrieved = []
        all_weights = []

        # Primary query
        retrieved, info = await self.read(query, top_k=3)
        all_retrieved.append(retrieved)
        all_weights.append(1.0)

        # Context queries with lower weight
        for ctx_query in context_queries[:self.config.num_heads - 1]:
            ctx_retrieved, _ = await self.read(ctx_query, top_k=2)
            all_retrieved.append(ctx_retrieved)
            all_weights.append(0.5)

        # Weighted combination
        total_weight = sum(all_weights)
        combined = np.zeros(self.config.memory_size, dtype=np.float32)
        for r, w in zip(all_retrieved, all_weights):
            combined += (w / total_weight) * r

        info["context_queries_used"] = len(context_queries)
        return combined, info

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        active_slots = np.sum(self._usage > 0)
        avg_surprise = (
            self._surprise_sum / self._update_count
            if self._update_count > 0 else 0.0
        )

        return {
            "memory_size": self.config.memory_size,
            "num_slots": self.config.num_memory_slots,
            "active_slots": int(active_slots),
            "utilization": float(active_slots / self.config.num_memory_slots),
            "update_count": self._update_count,
            "retrieval_count": self._retrieval_count,
            "avg_surprise": avg_surprise,
            "total_usage": float(np.sum(self._usage)),
        }

    def _save_state(self):
        """Save memory state to disk."""
        if not self.config.persist_path:
            return

        import os
        import pickle

        os.makedirs(os.path.dirname(self.config.persist_path), exist_ok=True)

        state = {
            "memory": self._memory,
            "momentum": self._momentum,
            "keys": self._keys,
            "usage": self._usage,
            "last_access": self._last_access,
            "update_count": self._update_count,
            "retrieval_count": self._retrieval_count,
            "surprise_sum": self._surprise_sum,
        }

        with open(self.config.persist_path, 'wb') as f:
            pickle.dump(state, f)

        logger.debug(f"Saved Titans memory state to {self.config.persist_path}")

    def _load_state(self):
        """Load memory state from disk."""
        if not self.config.persist_path:
            return

        import os
        import pickle

        if not os.path.exists(self.config.persist_path):
            return

        try:
            with open(self.config.persist_path, 'rb') as f:
                state = pickle.load(f)

            self._memory = state["memory"]
            self._momentum = state["momentum"]
            self._keys = state["keys"]
            self._usage = state["usage"]
            self._last_access = state["last_access"]
            self._update_count = state["update_count"]
            self._retrieval_count = state["retrieval_count"]
            self._surprise_sum = state["surprise_sum"]

            logger.info(
                f"Loaded Titans memory: {self._update_count} updates, "
                f"{int(np.sum(self._usage > 0))} active slots"
            )
        except Exception as e:
            logger.warning(f"Failed to load Titans memory state: {e}")

    async def consolidate(self):
        """
        Consolidate memories (like sleep-based memory consolidation).

        This merges similar memories and removes low-value ones,
        making room for new information.
        """
        # Find similar memory pairs
        keys_norm = self._normalize(self._keys)
        similarity_matrix = np.dot(keys_norm, keys_norm.T)

        # Merge very similar memories (except diagonal)
        np.fill_diagonal(similarity_matrix, 0)

        merged_count = 0
        for i in range(self.config.num_memory_slots):
            for j in range(i + 1, self.config.num_memory_slots):
                if similarity_matrix[i, j] > 0.95:  # Very similar
                    # Merge j into i
                    total_usage = self._usage[i] + self._usage[j]
                    if total_usage > 0:
                        weight_i = self._usage[i] / total_usage
                        weight_j = self._usage[j] / total_usage
                        self._memory[i] = (
                            weight_i * self._memory[i] +
                            weight_j * self._memory[j]
                        )
                        self._keys[i] = (
                            weight_i * self._keys[i] +
                            weight_j * self._keys[j]
                        )

                    # Clear slot j
                    self._memory[j] = 0
                    self._keys[j] = np.random.randn(self.config.memory_size) * 0.1
                    self._usage[j] = 0

                    merged_count += 1

        logger.info(f"Memory consolidation: merged {merged_count} slot pairs")
        return {"merged_count": merged_count}


# Convenience function
def create_titans_memory(
    memory_size: int = 512,
    num_slots: int = 64,
    persist_path: Optional[str] = None,
    **kwargs
) -> TitansMemoryModule:
    """Create a Titans memory module with common settings."""
    config = TitansConfig(
        memory_size=memory_size,
        num_memory_slots=num_slots,
        persist_path=persist_path,
        **kwargs
    )
    return TitansMemoryModule(config=config)

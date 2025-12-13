"""
Memory Store Implementations.

Production-ready storage backends for episodic memory including:
- InMemoryStore: Fast, no dependencies, ideal for development
- QdrantMemoryStore: Scalable vector database for production
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger("droidrun.memory")


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""

    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class BaseMemoryStore(ABC):
    """Abstract base class for memory stores."""

    @abstractmethod
    async def add(self, entry: MemoryEntry) -> str:
        """Add a memory entry, return its ID."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories, return entries with scores."""
        pass

    @abstractmethod
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all memories, return count deleted."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return total number of memories."""
        pass


class InMemoryStore(BaseMemoryStore):
    """
    In-memory store with optional persistence.

    Features:
    - Zero external dependencies
    - Optional disk persistence
    - Fast cosine similarity search
    - Ideal for development and testing
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        auto_persist: bool = True,
        max_entries: int = 10000,
    ):
        self.persist_path = persist_path
        self.auto_persist = auto_persist
        self.max_entries = max_entries
        self._entries: Dict[str, MemoryEntry] = {}

        if persist_path and os.path.exists(persist_path):
            self._load()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot_product = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def _load(self):
        """Load memories from disk."""
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self._entries = {
                    k: MemoryEntry.from_dict(v) if isinstance(v, dict) else v
                    for k, v in data.items()
                }
            logger.info(f"Loaded {len(self._entries)} memories from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load memories: {e}")

    def _save(self):
        """Save memories to disk."""
        if not self.persist_path:
            return
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'wb') as f:
                data = {k: v.to_dict() for k, v in self._entries.items()}
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

    async def add(self, entry: MemoryEntry) -> str:
        """Add a memory entry."""
        # Enforce max entries (FIFO)
        if len(self._entries) >= self.max_entries:
            oldest_id = min(self._entries.keys(), key=lambda k: self._entries[k].timestamp)
            del self._entries[oldest_id]

        self._entries[entry.id] = entry

        if self.auto_persist:
            self._save()

        return entry.id

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories using cosine similarity."""
        results = []

        for entry in self._entries.values():
            # Apply metadata filter
            if filter_metadata:
                if not all(
                    entry.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                ):
                    continue

            score = self._cosine_similarity(query_embedding, entry.embedding)
            results.append((entry, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        return self._entries.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            if self.auto_persist:
                self._save()
            return True
        return False

    async def clear(self) -> int:
        """Clear all memories."""
        count = len(self._entries)
        self._entries.clear()
        if self.auto_persist:
            self._save()
        return count

    async def count(self) -> int:
        """Return total number of memories."""
        return len(self._entries)


class QdrantMemoryStore(BaseMemoryStore):
    """
    Qdrant vector database store for production use.

    Features:
    - Scalable vector search
    - Persistent storage
    - Metadata filtering
    - Works with Qdrant Cloud or local instance
    """

    def __init__(
        self,
        collection_name: str = "droidrun_memory",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 384,
        distance: str = "Cosine",
        on_disk: bool = True,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.vector_size = vector_size
        self.distance = distance
        self.on_disk = on_disk

        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            if self.url:
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                )
            else:
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                )

            # Create collection if not exists
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance[self.distance.upper()],
                        on_disk=self.on_disk,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")

        except ImportError:
            raise ImportError(
                "qdrant-client required. Install: pip install qdrant-client"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    async def add(self, entry: MemoryEntry) -> str:
        """Add a memory entry to Qdrant."""
        from qdrant_client.http import models

        point = models.PointStruct(
            id=entry.id,
            vector=entry.embedding,
            payload={
                "content": entry.content,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp.isoformat(),
            },
        )

        self._client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

        return entry.id

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search Qdrant for similar memories."""
        from qdrant_client.http import models

        # Build filter if provided
        qdrant_filter = None
        if filter_metadata:
            conditions = [
                models.FieldCondition(
                    key=f"metadata.{k}",
                    match=models.MatchValue(value=v),
                )
                for k, v in filter_metadata.items()
            ]
            qdrant_filter = models.Filter(must=conditions)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        entries = []
        for result in results:
            entry = MemoryEntry(
                id=str(result.id),
                content=result.payload["content"],
                embedding=result.vector if result.vector else [],
                metadata=result.payload.get("metadata", {}),
                timestamp=datetime.fromisoformat(result.payload["timestamp"]),
            )
            entries.append((entry, result.score))

        return entries

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory from Qdrant."""
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[entry_id],
            with_vectors=True,
        )

        if not results:
            return None

        result = results[0]
        return MemoryEntry(
            id=str(result.id),
            content=result.payload["content"],
            embedding=result.vector if result.vector else [],
            metadata=result.payload.get("metadata", {}),
            timestamp=datetime.fromisoformat(result.payload["timestamp"]),
        )

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory from Qdrant."""
        from qdrant_client.http import models

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[entry_id]),
        )
        return True

    async def clear(self) -> int:
        """Clear all memories from collection."""
        count = await self.count()

        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._initialize_client()

        return count

    async def count(self) -> int:
        """Return total number of memories."""
        info = self._client.get_collection(self.collection_name)
        return info.points_count


# Re-export LocalEmbeddingProvider for convenience
try:
    from .embeddings import LocalEmbeddingProvider
except ImportError:
    # Fallback for standalone testing
    LocalEmbeddingProvider = None

__all__ = [
    "MemoryEntry",
    "BaseMemoryStore",
    "InMemoryStore",
    "QdrantMemoryStore",
    "LocalEmbeddingProvider",
]

"""
Memory System for DroidRun Agent.

Production-ready memory with:
- Episodic memory: Vector store for past experiences
- Titans memory: Neural long-term memory (test-time learning)
- ReMe memory: Procedural memory for "how-to" knowledge

Architecture:
- Short-term: Context window / attention
- Medium-term: Titans neural memory (compressed patterns)
- Long-term: ReMe procedural memory (structured experiences)

References:
- Titans: "Learning to Memorize at Test Time" (arXiv:2501.00663)
- ReMe: "Remember Me, Refine Me" (arXiv:2512.10696)
"""

from .memory_manager import (
    MemoryManager,
    MemoryConfig,
    EpisodeRecord,
)
from .stores import (
    BaseMemoryStore,
    InMemoryStore,
    QdrantMemoryStore,
    LocalEmbeddingProvider,
)
from .embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    OllamaEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)

# Titans: Neural Long-Term Memory
from .titans_memory import (
    TitansMemoryModule,
    TitansConfig,
    create_titans_memory,
)

# ReMe: Procedural Memory
from .reme_memory import (
    ReMeMemory,
    ReMeConfig,
    Experience,
    ExperienceType,
    Trajectory,
    create_reme_memory,
)

# Unified: Combined Titans + ReMe
from .unified_memory import (
    UnifiedMemorySystem,
    UnifiedMemoryConfig,
    EnhancedMemoryManager,
    create_unified_memory,
)

__all__ = [
    # Original episodic memory
    "MemoryManager",
    "MemoryConfig",
    "EpisodeRecord",
    # Stores
    "BaseMemoryStore",
    "InMemoryStore",
    "QdrantMemoryStore",
    "LocalEmbeddingProvider",
    # Embeddings
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    # Titans (Neural Long-Term Memory)
    "TitansMemoryModule",
    "TitansConfig",
    "create_titans_memory",
    # ReMe (Procedural Memory)
    "ReMeMemory",
    "ReMeConfig",
    "Experience",
    "ExperienceType",
    "Trajectory",
    "create_reme_memory",
    # Unified (Titans + ReMe)
    "UnifiedMemorySystem",
    "UnifiedMemoryConfig",
    "EnhancedMemoryManager",
    "create_unified_memory",
]

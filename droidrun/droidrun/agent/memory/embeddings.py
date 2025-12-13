"""
Embedding Providers for Memory System.

Supports multiple embedding backends with offline-first design.
All providers work without external API dependencies when needed.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import hashlib
import logging
import numpy as np

logger = logging.getLogger("droidrun.memory")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    Falls back to deterministic hash-based embeddings if model loading fails.

    This provider works completely offline with no API dependencies.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        fallback_dimension: int = 384,
    ):
        self.model_name = model_name
        self.device = device
        self._dimension = fallback_dimension
        self._model = None
        self._use_fallback = False

        self._initialize_model()

    def _initialize_model(self):
        """Try to load sentence-transformers model, fallback to hash-based if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence-transformer model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, using hash-based fallback embeddings. "
                "Install with: pip install sentence-transformers"
            )
            self._use_fallback = True
        except Exception as e:
            logger.warning(f"Failed to load model {self.model_name}: {e}, using fallback")
            self._use_fallback = True

    def _hash_embed(self, text: str) -> List[float]:
        """Generate deterministic embedding from text hash."""
        # Create deterministic seed from text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')

        # Generate deterministic vector
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self._dimension).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings synchronously."""
        if self._use_fallback or self._model is None:
            return [self._hash_embed(text) for text in texts]

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider (requires API key)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._dimension = dimension
        self._client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        if not self._client:
            raise RuntimeError("OpenAI client not initialized. Check API key.")

        response = self._client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider (local, no API key needed)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
    ):
        self.model = model
        self.base_url = base_url
        self._dimension = dimension

    async def embed(self, texts: List[str]) -> List[List[float]]:
        import aiohttp

        embeddings = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings.append(data["embedding"])
                    else:
                        raise RuntimeError(f"Ollama embedding failed: {response.status}")
        return embeddings

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        import requests

        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            if response.status_code == 200:
                embeddings.append(response.json()["embedding"])
            else:
                raise RuntimeError(f"Ollama embedding failed: {response.status_code}")
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Direct sentence-transformers provider with full control."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dimension = 768

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

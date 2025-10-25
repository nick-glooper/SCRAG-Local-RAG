"""
Embedding handler for SecureRAG.
Supports local and cloud embedding models.
"""

import logging
from typing import List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingHandler:
    """
    Handle text embeddings using local or cloud models.
    Supports batching for efficiency.
    """

    def __init__(self, config):
        """
        Initialize embedding model based on config.

        Args:
            config: SecureRAGConfig instance or embeddings config dict
        """
        # Extract embeddings config
        if hasattr(config, 'embeddings'):
            self.config = config.embeddings
        else:
            self.config = config

        self.provider = self.config.provider
        self.model_name = self.config.model
        self.batch_size = self.config.batch_size
        self.device = self._get_device(self.config.device)

        logger.info(f"Initializing {self.provider} embedding handler")
        logger.info(f"Model: {self.model_name}, Device: {self.device}")

        # Initialize model
        if self.provider == "local":
            self._init_local_model()
        elif self.provider == "openai":
            self._init_openai_client()
        elif self.provider == "anthropic":
            self._init_anthropic_client()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_device(self, device_config: str) -> str:
        """Determine the best device to use"""
        if device_config != "auto":
            return device_config

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _init_local_model(self):
        """Initialize local sentence-transformer model"""
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded local model: {self.model_name}")
            logger.info(f"Vector size: {self.vector_size}")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            api_key = self.config.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            self.client = openai.OpenAI(api_key=api_key)

            # Set vector size based on model
            if "text-embedding-3-large" in self.model_name:
                self.vector_size = 3072
            elif "text-embedding-3-small" in self.model_name:
                self.vector_size = 1536
            else:
                self.vector_size = 1536  # Default

            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
            logger.warning("⚠️  Using cloud embeddings - data will be sent to OpenAI")

        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def _init_anthropic_client(self):
        """Initialize Anthropic/Voyage client"""
        try:
            import anthropic
            api_key = self.config.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not provided")

            self.client = anthropic.Anthropic(api_key=api_key)
            self.vector_size = 1024  # Voyage embeddings

            logger.info(f"Initialized Anthropic client")
            logger.warning("⚠️  Using cloud embeddings - data will be sent to Anthropic")

        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Embed single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text")
            return [0.0] * self.vector_size

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        filtered_texts = [text.strip() if text else "" for text in texts]

        if self.provider == "local":
            return self._embed_batch_local(filtered_texts)
        elif self.provider == "openai":
            return self._embed_batch_openai(filtered_texts)
        elif self.provider == "anthropic":
            return self._embed_batch_anthropic(filtered_texts)

    def _embed_batch_local(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using local model"""
        try:
            # Process in batches to manage memory
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                all_embeddings.extend(embeddings.tolist())

            return all_embeddings

        except Exception as e:
            logger.error(f"Error embedding batch locally: {e}")
            raise

    def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using OpenAI API"""
        try:
            all_embeddings = []

            # OpenAI has rate limits, process in smaller batches
            batch_size = min(self.batch_size, 100)

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error embedding batch with OpenAI: {e}")
            raise

    def _embed_batch_anthropic(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using Anthropic/Voyage API"""
        try:
            # Note: This is a placeholder - Anthropic's actual embedding API may differ
            # You would need to use the Voyage AI API directly
            logger.error("Anthropic/Voyage embeddings not yet implemented")
            raise NotImplementedError("Anthropic/Voyage embeddings coming soon")

        except Exception as e:
            logger.error(f"Error embedding batch with Anthropic: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed query text.
        Some models have separate encoding for queries vs documents.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        # For most models, query and document embedding are the same
        # BGE models can use instruction prefixes, but we'll keep it simple
        return self.embed_text(query)

    def get_vector_size(self) -> int:
        """Get the dimension of embedding vectors"""
        return self.vector_size


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


if __name__ == "__main__":
    # Test embedding handler
    from src.config import load_config

    config = load_config()
    handler = EmbeddingHandler(config)

    # Test single embedding
    text = "This is a test document about artificial intelligence."
    embedding = handler.embed_text(text)
    print(f"Single embedding size: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Test batch embedding
    texts = [
        "Machine learning is a subset of AI.",
        "Natural language processing enables computers to understand text.",
        "Deep learning uses neural networks."
    ]
    embeddings = handler.embed_batch(texts)
    print(f"\nBatch embeddings: {len(embeddings)} texts embedded")

    # Test similarity
    sim = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two texts: {sim:.4f}")

"""
Reranking system for SecureRAG.
Improves search accuracy using cross-encoder models.
"""

import logging
from typing import List, Dict
import torch
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """
    Rerank search results using cross-encoder model for better accuracy.
    """

    def __init__(self, config):
        """
        Initialize reranking model.

        Args:
            config: SecureRAGConfig instance or reranking config
        """
        # Extract reranking config
        if hasattr(config, 'reranking'):
            self.config = config.reranking
        else:
            self.config = config

        self.enabled = self.config.enabled
        self.model_name = self.config.model
        self.top_n_rerank = self.config.top_n_rerank
        self.device = self._get_device(self.config.device)

        if not self.enabled:
            logger.info("Reranking is disabled")
            self.model = None
            return

        logger.info(f"Initializing reranker: {self.model_name}")
        logger.info(f"Device: {self.device}")

        try:
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512
            )
            logger.info("Reranker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.enabled = False
            self.model = None

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

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query text
            documents: List of document dicts (must have 'text' field)
            top_k: Number of top results to return

        Returns:
            Reranked documents with updated confidence scores
        """
        if not self.enabled or self.model is None:
            # Return documents as-is, limited to top_k
            return documents[:top_k]

        if not documents:
            return []

        try:
            # Prepare pairs for cross-encoder
            # Use raw_text if available, otherwise use text
            pairs = [
                [query, doc.get('raw_text', doc.get('text', ''))]
                for doc in documents
            ]

            # Get relevance scores
            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Normalize scores to 0-1 range using sigmoid
            scores = 1 / (1 + np.exp(-scores))

            # Update documents with new scores
            for i, doc in enumerate(documents):
                doc['confidence'] = float(scores[i])
                doc['original_score'] = doc.get('score', 0.0)

            # Sort by new confidence scores
            reranked = sorted(
                documents,
                key=lambda x: x['confidence'],
                reverse=True
            )

            # Return top_k
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fall back to original ranking
            return documents[:top_k]


def rerank_results(
    query: str,
    results: List[Dict],
    config,
    top_k: int = 5
) -> List[Dict]:
    """
    Convenience function to rerank search results.

    Args:
        query: Query text
        results: Search results to rerank
        config: Reranking configuration
        top_k: Number of results to return

    Returns:
        Reranked results
    """
    reranker = Reranker(config)
    return reranker.rerank(query, results, top_k)


if __name__ == "__main__":
    # Test reranker
    from src.config import load_config

    config = load_config()
    reranker = Reranker(config)

    if reranker.enabled:
        # Test documents
        query = "What is machine learning?"
        documents = [
            {
                "text": "Machine learning is a subset of artificial intelligence.",
                "score": 0.8
            },
            {
                "text": "Deep learning uses neural networks with multiple layers.",
                "score": 0.75
            },
            {
                "text": "Natural language processing helps computers understand text.",
                "score": 0.7
            },
            {
                "text": "Computer vision enables machines to interpret images.",
                "score": 0.65
            }
        ]

        reranked = reranker.rerank(query, documents, top_k=3)

        print(f"Query: {query}\n")
        print("Reranked results:")
        for i, doc in enumerate(reranked):
            print(f"\n{i+1}. [Confidence: {doc['confidence']:.3f}] "
                  f"(Original: {doc['original_score']:.3f})")
            print(f"   {doc['text']}")
    else:
        print("Reranking is disabled or model failed to load")

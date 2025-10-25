"""
Tests for embeddings handler
"""

import pytest
from src.embeddings import EmbeddingHandler, cosine_similarity
from src.config import load_config


@pytest.fixture
def embedding_handler():
    """Create embedding handler (may be slow on first run)"""
    config = load_config()
    return EmbeddingHandler(config)


def test_embed_single_text(embedding_handler):
    """Test embedding single text"""
    text = "This is a test sentence"

    embedding = embedding_handler.embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) == embedding_handler.get_vector_size()
    assert all(isinstance(x, float) for x in embedding)


def test_embed_batch(embedding_handler):
    """Test batch embedding"""
    texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence"
    ]

    embeddings = embedding_handler.embed_batch(texts)

    assert len(embeddings) == len(texts)
    assert all(len(emb) == embedding_handler.get_vector_size() for emb in embeddings)


def test_embed_empty_text(embedding_handler):
    """Test embedding empty text"""
    embedding = embedding_handler.embed_text("")

    assert isinstance(embedding, list)
    assert len(embedding) == embedding_handler.get_vector_size()


def test_cosine_similarity():
    """Test cosine similarity calculation"""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    sim = cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 0.001  # Should be very similar

    vec3 = [0.0, 1.0, 0.0]
    sim = cosine_similarity(vec1, vec3)
    assert abs(sim) < 0.001  # Should be orthogonal


def test_embed_query(embedding_handler):
    """Test query embedding"""
    query = "What is the capital of France?"

    embedding = embedding_handler.embed_query(query)

    assert isinstance(embedding, list)
    assert len(embedding) == embedding_handler.get_vector_size()

"""
Tests for vector store
"""

import pytest
import tempfile
from pathlib import Path
from src.vector_store import VectorStore
from src.config import load_config


@pytest.fixture
def vector_store():
    """Create a vector store with temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config()
        config.vector_db.path = tmpdir

        store = VectorStore(config)
        yield store


def test_create_collection(vector_store):
    """Test creating a collection"""
    result = vector_store.create_collection("test_collection")

    assert result["success"] is True
    assert result["name"] == "test_collection"


def test_list_collections(vector_store):
    """Test listing collections"""
    # Create a few collections
    vector_store.create_collection("col1")
    vector_store.create_collection("col2")

    collections = vector_store.list_collections()

    assert len(collections) >= 2
    names = [c["name"] for c in collections]
    assert "col1" in names
    assert "col2" in names


def test_delete_collection(vector_store):
    """Test deleting a collection"""
    vector_store.create_collection("test_delete")

    success = vector_store.delete_collection("test_delete")
    assert success is True

    collections = vector_store.list_collections()
    names = [c["name"] for c in collections]
    assert "test_delete" not in names


def test_add_documents(vector_store):
    """Test adding documents to collection"""
    vector_store.create_collection("test_docs")

    chunks = [
        {
            "text": "Test chunk 1",
            "raw_text": "Test chunk 1",
            "embedding": [0.1] * 1024,
            "metadata": {"page": 1}
        },
        {
            "text": "Test chunk 2",
            "raw_text": "Test chunk 2",
            "embedding": [0.2] * 1024,
            "metadata": {"page": 2}
        }
    ]

    count = vector_store.add_documents(
        "test_docs",
        chunks,
        "doc123",
        "v1"
    )

    assert count == 2


def test_search(vector_store):
    """Test searching collection"""
    vector_store.create_collection("test_search")

    # Add some documents
    chunks = [
        {
            "text": "Machine learning is great",
            "raw_text": "Machine learning is great",
            "embedding": [0.5] * 1024,
            "metadata": {}
        }
    ]

    vector_store.add_documents("test_search", chunks, "doc1", "v1")

    # Search
    query_vector = [0.5] * 1024
    results = vector_store.search("test_search", query_vector, limit=10)

    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]


def test_get_collection_stats(vector_store):
    """Test getting collection statistics"""
    vector_store.create_collection("test_stats")

    stats = vector_store.get_collection_stats("test_stats")

    assert "name" in stats
    assert "points_count" in stats
    assert stats["name"] == "test_stats"

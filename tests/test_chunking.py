"""
Tests for chunking system
"""

import pytest
from src.chunking import SemanticChunker
from src.config import load_config


@pytest.fixture
def chunker():
    """Create a chunker instance"""
    config = load_config()
    return SemanticChunker(config)


def test_chunk_simple_text(chunker):
    """Test chunking simple text"""
    text = "This is a test document. It has multiple sentences. " * 50

    chunks = chunker.chunk_document(text, {}, "test.txt")

    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("raw_text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_chunk_with_sections(chunker):
    """Test chunking text with sections"""
    text = """
# Introduction

This is the introduction.

## Background

This is the background section.

## Methodology

This describes our methodology.
"""

    chunks = chunker.chunk_document(text, {}, "test.md")

    assert len(chunks) > 0


def test_chunk_respects_size_limits(chunker):
    """Test that chunks respect size limits"""
    text = "Word " * 1000

    chunks = chunker.chunk_document(text, {}, "test.txt")

    for chunk in chunks:
        raw_length = len(chunk["raw_text"])
        assert raw_length <= chunker.max_chunk_size * 1.1  # Allow 10% tolerance


def test_chunk_with_metadata(chunker):
    """Test chunking with metadata"""
    text = "Test document content"
    metadata = {"author": "Test", "date": "2024-03-15"}

    chunks = chunker.chunk_document(text, metadata, "test.txt")

    assert len(chunks) > 0
    for chunk in chunks:
        assert "author" in chunk["metadata"]
        assert chunk["metadata"]["author"] == "Test"

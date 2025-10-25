"""
Tests for configuration system
"""

import pytest
from pathlib import Path
from src.config import (
    load_config,
    expand_env_vars,
    expand_path,
    SecureRAGConfig
)


def test_load_default_config():
    """Test loading default configuration"""
    config = load_config()

    assert config is not None
    assert isinstance(config, SecureRAGConfig)
    assert config.embeddings.model == "BAAI/bge-large-en-v1.5"
    assert config.vector_db.type == "qdrant"
    assert config.reranking.enabled is True


def test_expand_env_vars():
    """Test environment variable expansion"""
    import os

    os.environ["TEST_VAR"] = "test_value"

    result = expand_env_vars("prefix_${TEST_VAR}_suffix")
    assert result == "prefix_test_value_suffix"

    # Test with missing var
    result = expand_env_vars("prefix_${MISSING_VAR}_suffix")
    assert result == "prefix__suffix"


def test_expand_path():
    """Test path expansion"""
    # Test ~ expansion
    path = expand_path("~/test")
    assert "~" not in str(path)

    # Test absolute path
    path = expand_path("/absolute/path")
    assert path.is_absolute()


def test_config_validation():
    """Test configuration validation"""
    config = SecureRAGConfig()

    # Check defaults
    assert config.embeddings.provider == "local"
    assert config.chunking.strategy == "semantic"
    assert config.search.default_top_k == 5

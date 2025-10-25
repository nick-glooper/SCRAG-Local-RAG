"""
Configuration system for SecureRAG.
Loads and validates configuration from YAML file with Pydantic models.
"""

import os
import re
from pathlib import Path
from typing import Optional, Literal
import yaml
from pydantic import BaseModel, Field, field_validator


class VectorDBConfig(BaseModel):
    """Vector database configuration"""
    type: Literal["qdrant"] = "qdrant"
    path: str = "~/.secure-rag/vector_db"
    collection_settings: dict = Field(default_factory=lambda: {
        "vector_size": 1024,
        "distance": "cosine"
    })


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration"""
    provider: Literal["local", "openai", "anthropic"] = "local"
    model: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 32
    device: str = "auto"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


class RerankingConfig(BaseModel):
    """Reranking configuration"""
    enabled: bool = True
    model: str = "BAAI/bge-reranker-large"
    top_n_rerank: int = 20
    device: str = "auto"


class ChunkingConfig(BaseModel):
    """Chunking configuration"""
    strategy: Literal["semantic", "fixed"] = "semantic"
    min_chunk_size: int = 256
    max_chunk_size: int = 1024
    chunk_overlap: int = 100
    include_context: bool = True


class SearchConfig(BaseModel):
    """Search configuration"""
    default_top_k: int = 5
    min_confidence: float = 0.0


class SecurityConfig(BaseModel):
    """Security configuration"""
    encryption_enabled: bool = True
    backup_encryption: bool = True
    pbkdf2_iterations: int = 100000


class StorageConfig(BaseModel):
    """Storage configuration"""
    data_dir: str = "~/.secure-rag"
    backup_dir: str = "~/.secure-rag/backups"
    max_backup_age_days: int = 90
    models_dir: str = "~/.secure-rag/models"


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str = "~/.secure-rag/logs/secure-rag.log"
    query_history: bool = True
    max_history_entries: int = 10000
    max_log_file_size_mb: int = 100


class SecureRAGConfig(BaseModel):
    """Complete SecureRAG configuration"""
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def expand_env_vars(value: str) -> str:
    """
    Expand environment variables in string.
    Supports ${VAR_NAME} syntax.
    """
    if not isinstance(value, str):
        return value

    # Find all ${VAR_NAME} patterns
    pattern = r'\$\{([^}]+)\}'
    matches = re.findall(pattern, value)

    for var_name in matches:
        env_value = os.getenv(var_name, "")
        value = value.replace(f"${{{var_name}}}", env_value)

    return value


def expand_env_vars_recursive(data: dict) -> dict:
    """Recursively expand environment variables in dictionary"""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = expand_env_vars(value)
        elif isinstance(value, dict):
            result[key] = expand_env_vars_recursive(value)
        elif isinstance(value, list):
            result[key] = [
                expand_env_vars(item) if isinstance(item, str)
                else expand_env_vars_recursive(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path"""
    path = expand_env_vars(path)
    return Path(path).expanduser().resolve()


def load_config(config_path: Optional[str] = None) -> SecureRAGConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for:
                    1. CONFIG_PATH environment variable
                    2. ~/.secure-rag/config.yaml
                    3. Falls back to default config

    Returns:
        SecureRAGConfig instance
    """
    # Determine config path
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH")

    if config_path is None:
        default_path = Path.home() / ".secure-rag" / "config.yaml"
        if default_path.exists():
            config_path = str(default_path)

    # Load config data
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Expand environment variables
        config_data = expand_env_vars_recursive(config_data)
    else:
        # Use default config
        config_data = {}

    # Create config object with validation
    config = SecureRAGConfig(**config_data)

    # Create necessary directories
    _create_directories(config)

    return config


def _create_directories(config: SecureRAGConfig) -> None:
    """Create necessary directories if they don't exist"""
    directories = [
        config.storage.data_dir,
        config.storage.backup_dir,
        config.storage.models_dir,
        config.vector_db.path,
    ]

    # Extract log directory
    log_file_path = expand_path(config.logging.log_file)
    directories.append(str(log_file_path.parent))

    for directory in directories:
        dir_path = expand_path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)


def save_config(config: SecureRAGConfig, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: SecureRAGConfig instance
        output_path: Path to save config file
    """
    output_path = expand_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def get_db_path(config: SecureRAGConfig) -> Path:
    """Get path to query history database"""
    return expand_path(config.storage.data_dir) / "query_history.db"


def get_vector_db_path(config: SecureRAGConfig) -> Path:
    """Get path to vector database"""
    return expand_path(config.vector_db.path)


def get_models_dir(config: SecureRAGConfig) -> Path:
    """Get path to models directory"""
    return expand_path(config.storage.models_dir)


def get_backup_dir(config: SecureRAGConfig) -> Path:
    """Get path to backups directory"""
    return expand_path(config.storage.backup_dir)


def get_log_file(config: SecureRAGConfig) -> Path:
    """Get path to log file"""
    return expand_path(config.logging.log_file)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Data directory: {expand_path(config.storage.data_dir)}")
    print(f"Vector DB path: {get_vector_db_path(config)}")
    print(f"Embedding model: {config.embeddings.model}")
    print(f"Reranking enabled: {config.reranking.enabled}")

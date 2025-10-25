"""
SecureRAG MCP Server
Main FastMCP server exposing all RAG tools to Claude Desktop.
"""

import logging
import time
from typing import Optional, Dict, List
from pathlib import Path

from fastmcp import FastMCP

# Import all components
from src.config import load_config, get_db_path
from src.vector_store import VectorStore
from src.embeddings import EmbeddingHandler
from src.reranker import Reranker
from src.chunking import SemanticChunker
from src.document_processor import DocumentProcessor
from src.collections import CollectionsManager
from src.query_history import QueryHistory
from src.kb_mode import get_kb_mode_manager
from src.backup import BackupManager
from src.versioning import list_document_versions, compare_versions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("SecureRAG", dependencies=["qdrant-client", "sentence-transformers"])

# Global components (initialized on first use)
_config = None
_vector_store = None
_embedding_handler = None
_reranker = None
_chunker = None
_doc_processor = None
_collections_manager = None
_query_history = None
_backup_manager = None
_kb_mode = None


def _init_components():
    """Initialize all components lazily"""
    global _config, _vector_store, _embedding_handler, _reranker, _chunker
    global _doc_processor, _collections_manager, _query_history, _backup_manager, _kb_mode

    if _config is not None:
        return

    logger.info("Initializing SecureRAG components...")

    # Load configuration
    _config = load_config()

    # Initialize components
    _vector_store = VectorStore(_config)
    _embedding_handler = EmbeddingHandler(_config)
    _reranker = Reranker(_config)
    _chunker = SemanticChunker(_config)

    # Initialize document processor
    _doc_processor = DocumentProcessor(
        _config,
        _vector_store,
        _embedding_handler,
        _chunker
    )

    # Initialize collections manager
    db_path = get_db_path(_config)
    _collections_manager = CollectionsManager(_vector_store, str(db_path))

    # Initialize query history
    _query_history = QueryHistory(str(db_path))

    # Initialize backup manager
    _backup_manager = BackupManager(_config)

    # Get KB mode manager
    _kb_mode = get_kb_mode_manager()

    logger.info("✓ All components initialized successfully")


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def create_collection(name: str, description: str = "", metadata: dict = None) -> dict:
    """
    Create a new document collection.

    Args:
        name: Collection name (alphanumeric + underscores)
        description: Optional description
        metadata: Optional metadata dict

    Returns:
        Collection creation result
    """
    _init_components()

    logger.info(f"Creating collection: {name}")

    return _collections_manager.create_collection(name, description, metadata)


@mcp.tool()
def list_collections() -> list:
    """
    List all collections with statistics.

    Returns:
        List of collection info dicts
    """
    _init_components()

    return _collections_manager.list_collections()


@mcp.tool()
def delete_collection(name: str, confirm: bool = False) -> dict:
    """
    Delete a collection and all its documents.

    Args:
        name: Collection name
        confirm: Must be True to actually delete

    Returns:
        Deletion result
    """
    _init_components()

    logger.info(f"Deleting collection: {name} (confirm={confirm})")

    return _collections_manager.delete_collection(name, confirm)


@mcp.tool()
def ingest_document(
    filepath: str,
    collection_name: str,
    metadata: dict = None,
    document_id: str = None,
    version: str = "v1"
) -> dict:
    """
    Ingest a document into a collection.

    Supports: PDF, DOCX, TXT, MD files

    Args:
        filepath: Path to document file
        collection_name: Target collection
        metadata: Optional metadata (e.g., {"client": "IBM", "date": "2024-03-15"})
        document_id: Optional document ID (auto-generated if not provided)
        version: Version string (default: "v1")

    Returns:
        Ingestion result with statistics
    """
    _init_components()

    logger.info(f"Ingesting document: {filepath} -> {collection_name}")

    result = _doc_processor.ingest_document(
        filepath,
        collection_name,
        metadata,
        document_id,
        version
    )

    # Record document in collections manager if successful
    if result.get("success"):
        _collections_manager.add_document(
            collection_name=collection_name,
            document_id=result["document_id"],
            version=version,
            filename=result["filename"],
            filepath=filepath,
            chunk_count=result["chunks_created"],
            page_count=result.get("pages", 0),
            metadata=metadata
        )

    return result


@mcp.tool()
def search_kb(
    query: str,
    collection_name: str = None,
    filters: dict = None,
    top_k: int = 5,
    min_confidence: float = 0.0,
    rerank: bool = True
) -> dict:
    """
    Search the knowledge base with advanced retrieval.

    Args:
        query: Search query
        collection_name: Optional collection to search (None = all collections)
        filters: Optional filters (e.g., {"document_id": "doc123", "client": "IBM"})
        top_k: Number of results to return (default: 5)
        min_confidence: Minimum confidence score (0-1)
        rerank: Whether to rerank results (default: True)

    Returns:
        Search results with sources and confidence scores
    """
    _init_components()

    start_time = time.time()

    try:
        # Check KB mode
        if _kb_mode.is_enabled() and collection_name is None:
            collection_name = _kb_mode.get_active_collection()

        if not collection_name:
            # Search all collections
            collections = _collections_manager.list_collections()
            if not collections:
                return {
                    "success": False,
                    "error": "No collections available",
                    "error_type": "NoCollectionsError"
                }

            # Use first collection or combine results
            collection_name = collections[0]["name"]

        logger.info(f"Searching '{collection_name}' for: {query}")

        # Embed query
        query_vector = _embedding_handler.embed_query(query)

        # Determine limit for initial retrieval
        initial_limit = max(top_k * 3, 20) if rerank else top_k

        # Search vector store
        results = _vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            filters=filters,
            limit=initial_limit
        )

        # Rerank if enabled
        if rerank and _reranker.enabled and results:
            results = _reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # Filter by confidence
        if min_confidence > 0:
            results = [
                r for r in results
                if r.get('confidence', r.get('score', 0)) >= min_confidence
            ]

        # Format results with citations
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "text": result["text"],
                "confidence": result.get("confidence", result.get("score", 0.0)),
                "source": {
                    "document_id": result["document_id"],
                    "version": result["version"],
                    "chunk_index": result["chunk_index"],
                    "metadata": result.get("metadata", {})
                }
            })

        search_time_ms = int((time.time() - start_time) * 1000)

        # Log query
        _query_history.log_query(
            query=query,
            collection_name=collection_name,
            filters=filters,
            results=results,
            search_time_ms=search_time_ms
        )

        return {
            "success": True,
            "query": query,
            "collection": collection_name,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "search_time_ms": search_time_ms,
            "kb_mode_enabled": _kb_mode.is_enabled()
        }

    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
def enable_kb_mode(collection_name: str = None) -> dict:
    """
    Enable KB-only mode where Claude only uses knowledge base.

    Args:
        collection_name: Optional collection to restrict searches to

    Returns:
        Mode status
    """
    _init_components()

    return _kb_mode.enable(collection_name)


@mcp.tool()
def disable_kb_mode() -> dict:
    """
    Disable KB-only mode. Claude will use both KB and general knowledge.

    Returns:
        Mode status
    """
    _init_components()

    return _kb_mode.disable()


@mcp.tool()
def get_kb_mode_status() -> dict:
    """
    Get current KB mode status.

    Returns:
        Current mode status
    """
    _init_components()

    return _kb_mode.get_status()


@mcp.tool()
def export_collection(
    collection_name: str,
    output_path: str,
    password: str = None,
    include_embeddings: bool = True
) -> dict:
    """
    Export collection to encrypted archive.

    Args:
        collection_name: Collection to export
        output_path: Path to save archive (.tar.gz)
        password: Password for encryption (required if encryption enabled)
        include_embeddings: Whether to include embedding vectors

    Returns:
        Export result
    """
    _init_components()

    logger.info(f"Exporting collection: {collection_name}")

    return _backup_manager.export_collection(
        collection_name,
        output_path,
        password,
        include_embeddings,
        _vector_store,
        str(get_db_path(_config))
    )


@mcp.tool()
def import_collection(
    archive_path: str,
    password: str = None,
    new_name: str = None
) -> dict:
    """
    Import collection from encrypted archive.

    Args:
        archive_path: Path to archive file
        password: Password for decryption
        new_name: Optional new name for collection

    Returns:
        Import result
    """
    _init_components()

    logger.info(f"Importing collection from: {archive_path}")

    return _backup_manager.import_collection(
        archive_path,
        password,
        new_name,
        _vector_store,
        str(get_db_path(_config))
    )


@mcp.tool()
def list_documents(collection_name: str, filters: dict = None) -> list:
    """
    List documents in a collection.

    Args:
        collection_name: Collection name
        filters: Optional filters

    Returns:
        List of documents with metadata
    """
    _init_components()

    return _vector_store.list_documents(collection_name, filters)


@mcp.tool()
def delete_document(
    document_id: str,
    collection_name: str,
    version: str = None
) -> dict:
    """
    Delete a document or specific version.

    Args:
        document_id: Document ID
        collection_name: Collection name
        version: Optional specific version to delete (None = all versions)

    Returns:
        Deletion result
    """
    _init_components()

    logger.info(f"Deleting document: {document_id} (version={version}) from {collection_name}")

    success = _vector_store.delete_document(collection_name, document_id, version)

    return {
        "success": success,
        "document_id": document_id,
        "collection_name": collection_name,
        "version": version
    }


@mcp.tool()
def list_document_versions(document_id: str, collection_name: str) -> list:
    """
    List all versions of a document.

    Args:
        document_id: Document ID
        collection_name: Collection name

    Returns:
        List of version info
    """
    _init_components()

    return list_document_versions(
        collection_name,
        document_id,
        _vector_store,
        str(get_db_path(_config))
    )


@mcp.tool()
def compare_document_versions(
    document_id: str,
    collection_name: str,
    version1: str,
    version2: str
) -> dict:
    """
    Compare two versions of a document.

    Args:
        document_id: Document ID
        collection_name: Collection name
        version1: First version
        version2: Second version

    Returns:
        Comparison results
    """
    _init_components()

    logger.info(f"Comparing versions: {version1} vs {version2}")

    return compare_versions(
        collection_name,
        document_id,
        version1,
        version2,
        _vector_store,
        _embedding_handler
    )


@mcp.tool()
def get_query_history(collection_name: str = None, limit: int = 20) -> list:
    """
    Get search query history.

    Args:
        collection_name: Optional collection filter
        limit: Maximum results (default: 20)

    Returns:
        List of query history entries
    """
    _init_components()

    return _query_history.get_query_history(collection_name, limit)


@mcp.tool()
def get_collection_info(name: str) -> dict:
    """
    Get detailed information about a collection.

    Args:
        name: Collection name

    Returns:
        Collection information
    """
    _init_components()

    info = _collections_manager.get_collection_info(name)

    if info is None:
        return {
            "success": False,
            "error": f"Collection '{name}' not found",
            "error_type": "CollectionNotFoundError"
        }

    return {
        "success": True,
        **info
    }


@mcp.tool()
def get_stats() -> dict:
    """
    Get overall system statistics.

    Returns:
        System statistics
    """
    _init_components()

    try:
        collections = _collections_manager.list_collections()
        query_stats = _query_history.get_stats()

        total_docs = sum(c.get('doc_count', 0) for c in collections)
        total_chunks = sum(c.get('points_count', 0) for c in collections)

        return {
            "success": True,
            "collections_count": len(collections),
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "query_history": query_stats,
            "kb_mode_enabled": _kb_mode.is_enabled(),
            "embedding_model": _config.embeddings.model,
            "reranking_enabled": _config.reranking.enabled
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
def health_check() -> dict:
    """
    Verify system health and readiness.

    Returns:
        Health status
    """
    try:
        _init_components()

        checks = {
            "config_loaded": _config is not None,
            "vector_store_ready": _vector_store is not None,
            "embeddings_ready": _embedding_handler is not None,
            "reranker_ready": _reranker is not None and (_reranker.enabled == False or _reranker.model is not None),
            "collections_manager_ready": _collections_manager is not None,
        }

        all_healthy = all(checks.values())

        return {
            "success": True,
            "healthy": all_healthy,
            "checks": checks,
            "version": "1.0.0",
            "embedding_model": _config.embeddings.model,
            "vector_size": _embedding_handler.get_vector_size()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "healthy": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    logger.info("🚀 Starting SecureRAG MCP Server...")

    try:
        # Run FastMCP server
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

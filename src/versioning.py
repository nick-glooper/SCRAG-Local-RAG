"""
Document versioning system for SecureRAG.
Handles document versions and comparisons.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from src.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


def list_document_versions(
    collection_name: str,
    document_id: str,
    vector_store,
    db
) -> List[Dict]:
    """
    List all versions of a document.

    Args:
        collection_name: Collection name
        document_id: Document ID
        vector_store: VectorStore instance
        db: SQLite connection or database path

    Returns:
        List of version info dicts
    """
    try:
        import sqlite3

        # Get from metadata database
        if isinstance(db, str):
            conn = sqlite3.connect(db)
            should_close = True
        else:
            conn = db
            should_close = False

        try:
            conn.row_factory = sqlite3.Row

            # Get collection ID
            cursor = conn.execute(
                "SELECT id FROM collections WHERE name = ?",
                (collection_name,)
            )
            row = cursor.fetchone()

            if not row:
                logger.error(f"Collection '{collection_name}' not found")
                return []

            collection_id = row[0]

            # Get all versions of document
            cursor = conn.execute("""
                SELECT version, filename, ingested_at, chunk_count, page_count, metadata
                FROM documents
                WHERE collection_id = ? AND document_id = ?
                ORDER BY ingested_at DESC
            """, (collection_id, document_id))

            versions = []
            for row in cursor:
                version = dict(row)
                # Parse metadata if needed
                if version.get('metadata'):
                    import json
                    try:
                        version['metadata'] = json.loads(version['metadata'])
                    except:
                        version['metadata'] = {}
                versions.append(version)

            return versions

        finally:
            if should_close:
                conn.close()

    except Exception as e:
        logger.error(f"Error listing document versions: {e}")
        return []


def compare_versions(
    collection_name: str,
    document_id: str,
    version1: str,
    version2: str,
    vector_store,
    embedding_handler
) -> Dict:
    """
    Compare two versions of a document using semantic similarity.

    Args:
        collection_name: Collection name
        document_id: Document ID
        version1: First version
        version2: Second version
        vector_store: VectorStore instance
        embedding_handler: EmbeddingHandler instance

    Returns:
        Dict with comparison results
    """
    try:
        # Get chunks for both versions
        v1_docs = vector_store.list_documents(
            collection_name,
            filters={"document_id": document_id, "version": version1}
        )

        v2_docs = vector_store.list_documents(
            collection_name,
            filters={"document_id": document_id, "version": version2}
        )

        if not v1_docs or not v2_docs:
            return {
                "success": False,
                "error": "One or both versions not found",
                "error_type": "VersionNotFoundError"
            }

        # Get actual chunks from vector store
        v1_results = vector_store.search(
            collection_name,
            query_vector=[0.0] * vector_store.vector_size,  # Dummy query
            filters={"document_id": document_id, "version": version1},
            limit=10000
        )

        v2_results = vector_store.search(
            collection_name,
            query_vector=[0.0] * vector_store.vector_size,  # Dummy query
            filters={"document_id": document_id, "version": version2},
            limit=10000
        )

        # Sort by chunk index
        v1_chunks = sorted(v1_results, key=lambda x: x.get('chunk_index', 0))
        v2_chunks = sorted(v2_results, key=lambda x: x.get('chunk_index', 0))

        # Extract texts
        v1_texts = [chunk['raw_text'] for chunk in v1_chunks]
        v2_texts = [chunk['raw_text'] for chunk in v2_chunks]

        # Calculate overall similarity
        v1_full_text = " ".join(v1_texts)
        v2_full_text = " ".join(v2_texts)

        # Embed full texts
        v1_embedding = embedding_handler.embed_text(v1_full_text[:5000])  # Limit length
        v2_embedding = embedding_handler.embed_text(v2_full_text[:5000])

        overall_similarity = cosine_similarity(v1_embedding, v2_embedding)

        # Find added and removed content
        added_content = []
        removed_content = []
        modified_content = []

        # Simple diff: compare chunk counts and content
        v1_set = set(v1_texts)
        v2_set = set(v2_texts)

        # Removed chunks (in v1 but not v2)
        for text in v1_set - v2_set:
            if len(text) > 50:  # Only significant chunks
                removed_content.append(text[:200])  # Limit length

        # Added chunks (in v2 but not v1)
        for text in v2_set - v1_set:
            if len(text) > 50:
                added_content.append(text[:200])

        # For modified content, do pairwise comparison
        # (This is simplified - could be more sophisticated)
        if len(v1_chunks) == len(v2_chunks):
            for c1, c2 in zip(v1_chunks, v2_chunks):
                if c1['raw_text'] != c2['raw_text']:
                    # Calculate similarity
                    sim = cosine_similarity(
                        embedding_handler.embed_text(c1['raw_text']),
                        embedding_handler.embed_text(c2['raw_text'])
                    )

                    if 0.5 < sim < 0.95:  # Likely modified, not completely different
                        modified_content.append({
                            "v1_text": c1['raw_text'][:200],
                            "v2_text": c2['raw_text'][:200],
                            "similarity": round(sim, 3)
                        })

        return {
            "success": True,
            "version1": version1,
            "version2": version2,
            "similarity_score": round(overall_similarity, 3),
            "v1_chunks": len(v1_chunks),
            "v2_chunks": len(v2_chunks),
            "added_content": added_content[:10],  # Limit to 10
            "removed_content": removed_content[:10],
            "modified_content": modified_content[:10],
            "summary": _generate_comparison_summary(
                overall_similarity,
                len(v1_chunks),
                len(v2_chunks),
                len(added_content),
                len(removed_content),
                len(modified_content)
            )
        }

    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def _generate_comparison_summary(
    similarity: float,
    v1_chunks: int,
    v2_chunks: int,
    added: int,
    removed: int,
    modified: int
) -> str:
    """Generate human-readable comparison summary"""
    summary_parts = []

    if similarity > 0.95:
        summary_parts.append("Documents are very similar")
    elif similarity > 0.8:
        summary_parts.append("Documents are mostly similar with some changes")
    elif similarity > 0.6:
        summary_parts.append("Documents have moderate differences")
    else:
        summary_parts.append("Documents are significantly different")

    if added > 0:
        summary_parts.append(f"{added} chunks added")

    if removed > 0:
        summary_parts.append(f"{removed} chunks removed")

    if modified > 0:
        summary_parts.append(f"{modified} chunks modified")

    chunk_diff = v2_chunks - v1_chunks
    if chunk_diff > 0:
        summary_parts.append(f"increased by {chunk_diff} chunks")
    elif chunk_diff < 0:
        summary_parts.append(f"decreased by {abs(chunk_diff)} chunks")

    return ". ".join(summary_parts) + "."


if __name__ == "__main__":
    # Test versioning
    from src.config import load_config, get_db_path
    from src.vector_store import VectorStore
    from src.embeddings import EmbeddingHandler

    config = load_config()
    vector_store = VectorStore(config)
    embedding_handler = EmbeddingHandler(config)
    db_path = get_db_path(config)

    print("Versioning system test initialized")
    print("Use list_document_versions() and compare_versions() functions")

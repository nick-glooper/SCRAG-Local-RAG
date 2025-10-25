"""
Vector store wrapper for SecureRAG.
Wraps Qdrant for all vector database operations.
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around Qdrant for vector operations.
    Uses embedded/local mode (no separate server needed).
    """

    def __init__(self, config):
        """
        Initialize Qdrant client in local mode.

        Args:
            config: SecureRAGConfig instance or vector_db config
        """
        # Extract vector DB config
        if hasattr(config, 'vector_db'):
            self.config = config.vector_db
            self.full_config = config
        else:
            self.config = config
            self.full_config = None

        # Expand path
        from src.config import expand_path
        self.db_path = expand_path(self.config.path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant client in embedded mode
        logger.info(f"Initializing Qdrant client at: {self.db_path}")
        self.client = QdrantClient(path=str(self.db_path))

        # Get vector size and distance metric
        self.vector_size = self.config.collection_settings.get("vector_size", 1024)
        self.distance = self._get_distance_metric(
            self.config.collection_settings.get("distance", "cosine")
        )

        logger.info(f"Vector store initialized (size={self.vector_size}, distance={self.distance})")

    def _get_distance_metric(self, distance_name: str) -> Distance:
        """Convert distance name to Qdrant Distance enum"""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        return distance_map.get(distance_name.lower(), Distance.COSINE)

    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a new collection.

        Args:
            name: Collection name
            metadata: Optional metadata to store

        Returns:
            Dict with collection info
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == name for c in collections):
                logger.warning(f"Collection '{name}' already exists")
                return {
                    "success": False,
                    "error": f"Collection '{name}' already exists",
                    "error_type": "CollectionExistsError"
                }

            # Create collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )

            logger.info(f"Created collection: {name}")

            return {
                "success": True,
                "name": name,
                "vector_size": self.vector_size,
                "distance": str(self.distance),
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def list_collections(self) -> List[Dict]:
        """
        List all collections with statistics.

        Returns:
            List of collection info dicts
        """
        try:
            collections = self.client.get_collections().collections

            result = []
            for collection in collections:
                # Get collection info
                info = self.client.get_collection(collection.name)

                result.append({
                    "name": collection.name,
                    "vector_size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                    "points_count": info.points_count,
                    "segments_count": info.segments_count,
                })

            return result

        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(collection_name=name)
            logger.info(f"Deleted collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {e}")
            return False

    def add_documents(
        self,
        collection_name: str,
        chunks: List[Dict],
        document_id: str,
        version: str
    ) -> int:
        """
        Add document chunks to collection.

        Args:
            collection_name: Name of collection
            chunks: List of chunk dicts with 'text', 'embedding', 'metadata'
            document_id: Document identifier
            version: Document version

        Returns:
            Number of chunks added
        """
        try:
            points = []

            for i, chunk in enumerate(chunks):
                # Generate unique point ID
                point_id = str(uuid.uuid4())

                # Prepare payload (metadata)
                payload = {
                    "document_id": document_id,
                    "version": version,
                    "chunk_index": i,
                    "text": chunk["text"],
                    "raw_text": chunk.get("raw_text", chunk["text"]),
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", len(chunk["text"])),
                }

                # Add chunk metadata
                if "metadata" in chunk:
                    payload.update(chunk["metadata"])

                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload=payload
                )
                points.append(point)

            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )

            logger.info(f"Added {len(chunks)} chunks to '{collection_name}'")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding documents to '{collection_name}': {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        filters: Optional[Dict] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search collection for similar vectors.

        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            filters: Optional filters (e.g., {"document_id": "doc123"})
            limit: Maximum results to return

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Build filter if provided
            query_filter = None
            if filters:
                query_filter = self._build_filter(filters)

            # Execute search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": float(result.score),
                    "text": result.payload.get("text", ""),
                    "raw_text": result.payload.get("raw_text", ""),
                    "document_id": result.payload.get("document_id"),
                    "version": result.payload.get("version"),
                    "chunk_index": result.payload.get("chunk_index"),
                    "metadata": {
                        k: v for k, v in result.payload.items()
                        if k not in ["text", "raw_text", "document_id", "version", "chunk_index"]
                    }
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {e}")
            raise

    def _build_filter(self, filters: Dict) -> Filter:
        """Build Qdrant filter from dict"""
        conditions = []

        for key, value in filters.items():
            if isinstance(value, (str, int, bool)):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            elif isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
                        )
                    )

        return Filter(must=conditions) if conditions else None

    def get_collection_stats(self, name: str) -> Dict:
        """
        Get collection statistics.

        Args:
            name: Collection name

        Returns:
            Dict with collection stats
        """
        try:
            info = self.client.get_collection(name)

            return {
                "name": name,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
                "status": info.status,
            }

        except Exception as e:
            logger.error(f"Error getting stats for '{name}': {e}")
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }

    def list_documents(
        self,
        collection_name: str,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        List all documents in collection.

        Args:
            collection_name: Collection name
            filters: Optional filters

        Returns:
            List of unique documents with metadata
        """
        try:
            # Scroll through all points
            query_filter = self._build_filter(filters) if filters else None

            # Get all points (in batches)
            offset = None
            all_points = []

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=query_filter
                )

                all_points.extend(results)

                if next_offset is None:
                    break
                offset = next_offset

            # Group by document_id and version
            documents = {}
            for point in all_points:
                doc_id = point.payload.get("document_id")
                version = point.payload.get("version")

                if not doc_id:
                    continue

                key = f"{doc_id}::{version}"

                if key not in documents:
                    documents[key] = {
                        "document_id": doc_id,
                        "version": version,
                        "chunk_count": 0,
                        "metadata": {}
                    }

                documents[key]["chunk_count"] += 1

                # Collect metadata (from first chunk)
                if not documents[key]["metadata"]:
                    documents[key]["metadata"] = {
                        k: v for k, v in point.payload.items()
                        if k not in ["text", "raw_text", "document_id", "version", "chunk_index"]
                    }

            return list(documents.values())

        except Exception as e:
            logger.error(f"Error listing documents in '{collection_name}': {e}")
            return []

    def delete_document(
        self,
        collection_name: str,
        document_id: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Delete a document or specific version.

        Args:
            collection_name: Collection name
            document_id: Document ID
            version: Optional specific version to delete

        Returns:
            True if successful
        """
        try:
            # Build filter
            filters = {"document_id": document_id}
            if version:
                filters["version"] = version

            query_filter = self._build_filter(filters)

            # Delete points matching filter
            self.client.delete(
                collection_name=collection_name,
                points_selector=query_filter
            )

            logger.info(f"Deleted document '{document_id}' (version={version}) from '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    def export_collection(self, collection_name: str, output_path: str) -> Dict:
        """
        Export collection to snapshot.

        Args:
            collection_name: Collection to export
            output_path: Path to save snapshot

        Returns:
            Dict with export info
        """
        try:
            # Create snapshot
            snapshot_name = self.client.create_snapshot(collection_name=collection_name)

            logger.info(f"Created snapshot: {snapshot_name}")

            return {
                "success": True,
                "snapshot_name": snapshot_name,
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def import_collection(
        self,
        snapshot_path: str,
        collection_name: str
    ) -> Dict:
        """
        Import collection from snapshot.

        Args:
            snapshot_path: Path to snapshot
            collection_name: Name for new collection

        Returns:
            Dict with import info
        """
        try:
            # Qdrant's snapshot recovery
            # Note: This would require the Qdrant server API
            # For embedded mode, we'd need to copy files directly

            logger.warning("Snapshot import for embedded mode requires manual file operations")

            return {
                "success": False,
                "error": "Import not yet implemented for embedded mode",
                "error_type": "NotImplementedError"
            }

        except Exception as e:
            logger.error(f"Error importing collection: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


if __name__ == "__main__":
    # Test vector store
    from src.config import load_config

    config = load_config()
    store = VectorStore(config)

    # Test create collection
    result = store.create_collection("test_collection")
    print(f"Create collection: {result}")

    # Test list collections
    collections = store.list_collections()
    print(f"\nCollections: {collections}")

    # Test delete collection
    # store.delete_collection("test_collection")

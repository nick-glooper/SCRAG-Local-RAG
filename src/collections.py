"""
Collections manager for SecureRAG.
Manages collections and their metadata using SQLite.
"""

import sqlite3
import logging
import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CollectionsManager:
    """
    High-level collection operations with metadata tracking.
    """

    def __init__(self, vector_store, db_path: str):
        """
        Initialize collections manager.

        Args:
            vector_store: VectorStore instance
            db_path: Path to SQLite database
        """
        self.vector_store = vector_store
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Collections manager initialized: {self.db_path}")

    def _init_database(self):
        """Create database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Collections table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    doc_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0
                )
            """)

            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    filename TEXT,
                    filepath TEXT,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    chunk_count INTEGER,
                    page_count INTEGER,
                    FOREIGN KEY (collection_id) REFERENCES collections(id),
                    UNIQUE(collection_id, document_id, version)
                )
            """)

            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection_name
                ON collections(name)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_collection
                ON documents(collection_id)
            """)

            conn.commit()

    def create_collection(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new collection in both vector store and metadata database.

        Args:
            name: Collection name
            description: Collection description
            metadata: Optional metadata

        Returns:
            Dict with collection info
        """
        try:
            # Create in vector store
            result = self.vector_store.create_collection(name, metadata)

            if not result.get("success"):
                return result

            # Create in metadata database
            collection_id = str(uuid.uuid4())
            metadata_json = json.dumps(metadata or {})

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO collections
                    (id, name, description, metadata)
                    VALUES (?, ?, ?, ?)
                """, (collection_id, name, description, metadata_json))
                conn.commit()

            logger.info(f"Created collection: {name}")

            return {
                "success": True,
                "id": collection_id,
                "name": name,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

        except sqlite3.IntegrityError:
            return {
                "success": False,
                "error": f"Collection '{name}' already exists",
                "error_type": "CollectionExistsError"
            }
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def list_collections(self) -> List[Dict]:
        """
        List all collections with full metadata.

        Returns:
            List of collection info dicts
        """
        try:
            # Get from vector store
            vs_collections = {
                c["name"]: c for c in self.vector_store.list_collections()
            }

            # Get from metadata database
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute("""
                    SELECT * FROM collections
                    ORDER BY created_at DESC
                """)

                collections = []
                for row in cursor:
                    collection = dict(row)

                    # Parse metadata JSON
                    if collection['metadata']:
                        try:
                            collection['metadata'] = json.loads(collection['metadata'])
                        except:
                            collection['metadata'] = {}

                    # Add vector store stats
                    vs_info = vs_collections.get(collection['name'])
                    if vs_info:
                        collection['points_count'] = vs_info.get('points_count', 0)
                        collection['vector_size'] = vs_info.get('vector_size')
                    else:
                        collection['points_count'] = 0

                    collections.append(collection)

                return collections

        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def get_collection_info(self, name: str) -> Optional[Dict]:
        """
        Get detailed collection information.

        Args:
            name: Collection name

        Returns:
            Dict with collection info or None
        """
        try:
            # Get vector store stats
            vs_stats = self.vector_store.get_collection_stats(name)

            # Get metadata
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute(
                    "SELECT * FROM collections WHERE name = ?",
                    (name,)
                )

                row = cursor.fetchone()
                if not row:
                    return None

                info = dict(row)

                # Parse metadata
                if info['metadata']:
                    try:
                        info['metadata'] = json.loads(info['metadata'])
                    except:
                        info['metadata'] = {}

                # Add vector store stats
                info.update(vs_stats)

                # Count documents
                doc_count = conn.execute("""
                    SELECT COUNT(DISTINCT document_id)
                    FROM documents
                    WHERE collection_id = ?
                """, (info['id'],)).fetchone()[0]

                info['doc_count'] = doc_count

                return info

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def delete_collection(self, name: str, confirm: bool = False) -> Dict:
        """
        Delete a collection from both stores.

        Args:
            name: Collection name
            confirm: Must be True to actually delete

        Returns:
            Dict with result
        """
        if not confirm:
            return {
                "success": False,
                "error": "Must confirm deletion by setting confirm=True",
                "error_type": "ConfirmationRequiredError"
            }

        try:
            # Delete from vector store
            success = self.vector_store.delete_collection(name)

            if not success:
                return {
                    "success": False,
                    "error": f"Failed to delete collection '{name}' from vector store",
                    "error_type": "VectorStoreDeletionError"
                }

            # Delete from metadata database
            with sqlite3.connect(self.db_path) as conn:
                # Get collection ID
                cursor = conn.execute(
                    "SELECT id FROM collections WHERE name = ?",
                    (name,)
                )
                row = cursor.fetchone()

                if row:
                    collection_id = row[0]

                    # Delete documents
                    conn.execute(
                        "DELETE FROM documents WHERE collection_id = ?",
                        (collection_id,)
                    )

                    # Delete collection
                    conn.execute(
                        "DELETE FROM collections WHERE id = ?",
                        (collection_id,)
                    )

                    conn.commit()

            logger.info(f"Deleted collection: {name}")

            return {
                "success": True,
                "name": name,
                "deleted_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def add_document(
        self,
        collection_name: str,
        document_id: str,
        version: str,
        filename: str,
        filepath: str,
        chunk_count: int,
        page_count: int,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Record a document in the metadata database.

        Args:
            collection_name: Collection name
            document_id: Document ID
            version: Document version
            filename: Original filename
            filepath: File path
            chunk_count: Number of chunks
            page_count: Number of pages
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            # Get collection ID
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM collections WHERE name = ?",
                    (collection_name,)
                )
                row = cursor.fetchone()

                if not row:
                    logger.error(f"Collection '{collection_name}' not found")
                    return False

                collection_id = row[0]

                # Insert document
                doc_id = str(uuid.uuid4())
                metadata_json = json.dumps(metadata or {})

                conn.execute("""
                    INSERT OR REPLACE INTO documents
                    (id, collection_id, document_id, version, filename, filepath,
                     metadata, chunk_count, page_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    collection_id,
                    document_id,
                    version,
                    filename,
                    filepath,
                    metadata_json,
                    chunk_count,
                    page_count
                ))

                # Update collection stats
                conn.execute("""
                    UPDATE collections
                    SET doc_count = (
                        SELECT COUNT(DISTINCT document_id)
                        FROM documents
                        WHERE collection_id = ?
                    ),
                    chunk_count = (
                        SELECT SUM(chunk_count)
                        FROM documents
                        WHERE collection_id = ?
                    )
                    WHERE id = ?
                """, (collection_id, collection_id, collection_id))

                conn.commit()

            return True

        except Exception as e:
            logger.error(f"Error adding document to metadata: {e}")
            return False


if __name__ == "__main__":
    # Test collections manager
    from src.config import load_config, get_db_path
    from src.vector_store import VectorStore

    config = load_config()
    vector_store = VectorStore(config)
    db_path = get_db_path(config)

    manager = CollectionsManager(vector_store, str(db_path))

    # Test create collection
    result = manager.create_collection(
        name="test_collection",
        description="Test collection for development",
        metadata={"author": "test", "type": "demo"}
    )
    print(f"Create: {result}")

    # Test list collections
    collections = manager.list_collections()
    print(f"\nCollections: {len(collections)}")
    for col in collections:
        print(f"  - {col['name']}: {col.get('description', 'No description')}")

    # Test get info
    info = manager.get_collection_info("test_collection")
    print(f"\nCollection info: {info}")

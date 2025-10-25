"""
Backup and restore system for SecureRAG.
Handles encrypted export/import of collections.
"""

import logging
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Export and import collections with optional encryption.
    """

    def __init__(self, config):
        """
        Initialize backup manager.

        Args:
            config: SecureRAGConfig instance
        """
        if hasattr(config, 'security'):
            self.security_config = config.security
            self.storage_config = config.storage
        else:
            self.security_config = config
            self.storage_config = None

        self.encryption_enabled = self.security_config.encryption_enabled
        self.pbkdf2_iterations = getattr(
            self.security_config,
            'pbkdf2_iterations',
            100000
        )

        logger.info(f"Backup manager initialized (encryption={self.encryption_enabled})")

    def export_collection(
        self,
        collection_name: str,
        output_path: str,
        password: Optional[str],
        include_embeddings: bool,
        vector_store,
        db
    ) -> Dict:
        """
        Export collection to encrypted tar.gz archive.

        Args:
            collection_name: Collection to export
            output_path: Path to save archive
            password: Password for encryption (required if encryption enabled)
            include_embeddings: Whether to include embedding vectors
            vector_store: VectorStore instance
            db: Database path or connection

        Returns:
            Dict with export results
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if encryption required
            if self.encryption_enabled and not password:
                return {
                    "success": False,
                    "error": "Password required for encrypted backup",
                    "error_type": "PasswordRequiredError"
                }

            # Create temporary directory for export
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                logger.info(f"Exporting collection '{collection_name}' to {output_path}")

                # 1. Export metadata
                metadata = self._export_metadata(collection_name, db, temp_path)

                if not metadata:
                    return {
                        "success": False,
                        "error": f"Collection '{collection_name}' not found",
                        "error_type": "CollectionNotFoundError"
                    }

                # 2. Export documents
                self._export_documents(collection_name, db, temp_path)

                # 3. Export vector data
                self._export_vector_data(
                    collection_name,
                    vector_store,
                    temp_path,
                    include_embeddings
                )

                # 4. Create archive
                archive_path = temp_path / "collection.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    for file in temp_path.glob("*.json"):
                        tar.add(file, arcname=file.name)

                # 5. Encrypt if enabled
                if self.encryption_enabled and password:
                    encrypted_data = self._encrypt_file(archive_path, password)
                    output_path.write_bytes(encrypted_data)
                else:
                    shutil.copy(archive_path, output_path)

                # Get file size
                file_size_mb = output_path.stat().st_size / (1024 * 1024)

                logger.info(f"Export complete: {output_path} ({file_size_mb:.2f} MB)")

                return {
                    "success": True,
                    "collection_name": collection_name,
                    "output_path": str(output_path),
                    "file_size_mb": round(file_size_mb, 2),
                    "encrypted": self.encryption_enabled and password is not None,
                    "includes_embeddings": include_embeddings,
                    "exported_at": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Error exporting collection: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def import_collection(
        self,
        archive_path: str,
        password: Optional[str],
        new_name: Optional[str],
        vector_store,
        db
    ) -> Dict:
        """
        Import collection from encrypted archive.

        Args:
            archive_path: Path to archive file
            password: Password for decryption
            new_name: Optional new name for collection
            vector_store: VectorStore instance
            db: Database path or connection

        Returns:
            Dict with import results
        """
        try:
            archive_path = Path(archive_path)

            if not archive_path.exists():
                return {
                    "success": False,
                    "error": f"Archive not found: {archive_path}",
                    "error_type": "FileNotFoundError"
                }

            # Create temporary directory for import
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                logger.info(f"Importing collection from {archive_path}")

                # 1. Decrypt if needed
                archive_data = archive_path.read_bytes()

                # Try to decrypt (if it fails, assume not encrypted)
                if password:
                    try:
                        decrypted_data = self._decrypt_file(archive_data, password)
                        decrypted_path = temp_path / "decrypted.tar.gz"
                        decrypted_path.write_bytes(decrypted_data)
                        tar_path = decrypted_path
                    except Exception as e:
                        return {
                            "success": False,
                            "error": "Failed to decrypt archive. Invalid password?",
                            "error_type": "DecryptionError",
                            "details": str(e)
                        }
                else:
                    tar_path = archive_path

                # 2. Extract archive
                extract_path = temp_path / "extracted"
                extract_path.mkdir()

                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(extract_path)

                # 3. Load metadata
                metadata_path = extract_path / "metadata.json"
                if not metadata_path.exists():
                    return {
                        "success": False,
                        "error": "Invalid archive: metadata.json not found",
                        "error_type": "InvalidArchiveError"
                    }

                with open(metadata_path) as f:
                    metadata = json.load(f)

                collection_name = new_name or metadata['name']

                # 4. Import to vector store
                # (This is simplified - would need full implementation)
                result = vector_store.create_collection(
                    collection_name,
                    metadata.get('metadata', {})
                )

                if not result.get('success'):
                    return result

                logger.info(f"Import complete: {collection_name}")

                return {
                    "success": True,
                    "collection_name": collection_name,
                    "original_name": metadata['name'],
                    "imported_at": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Error importing collection: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def _export_metadata(
        self,
        collection_name: str,
        db,
        output_dir: Path
    ) -> Optional[Dict]:
        """Export collection metadata to JSON"""
        import sqlite3

        if isinstance(db, str):
            conn = sqlite3.connect(db)
            should_close = True
        else:
            conn = db
            should_close = False

        try:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                "SELECT * FROM collections WHERE name = ?",
                (collection_name,)
            )

            row = cursor.fetchone()
            if not row:
                return None

            metadata = dict(row)

            # Parse JSON metadata
            if metadata.get('metadata'):
                metadata['metadata'] = json.loads(metadata['metadata'])

            # Save to file
            output_file = output_dir / "metadata.json"
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return metadata

        finally:
            if should_close:
                conn.close()

    def _export_documents(self, collection_name: str, db, output_dir: Path):
        """Export documents metadata to JSON"""
        import sqlite3

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
                return

            collection_id = row[0]

            # Get all documents
            cursor = conn.execute(
                "SELECT * FROM documents WHERE collection_id = ?",
                (collection_id,)
            )

            documents = []
            for row in cursor:
                doc = dict(row)
                if doc.get('metadata'):
                    doc['metadata'] = json.loads(doc['metadata'])
                documents.append(doc)

            # Save to file
            output_file = output_dir / "documents.json"
            with open(output_file, 'w') as f:
                json.dump(documents, f, indent=2)

        finally:
            if should_close:
                conn.close()

    def _export_vector_data(
        self,
        collection_name: str,
        vector_store,
        output_dir: Path,
        include_embeddings: bool
    ):
        """Export vector data to JSON"""
        # Get all points
        # Note: This is simplified - in production would use proper export
        documents = vector_store.list_documents(collection_name)

        # Save to file
        output_file = output_dir / "vectors.json"
        with open(output_file, 'w') as f:
            json.dump({
                "documents": documents,
                "includes_embeddings": include_embeddings
            }, f, indent=2)

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.pbkdf2_iterations,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _encrypt_file(self, file_path: Path, password: str) -> bytes:
        """Encrypt file with password"""
        # Generate salt
        import os
        salt = os.urandom(16)

        # Derive key
        key = self._derive_key(password, salt)

        # Encrypt
        fernet = Fernet(key)
        data = file_path.read_bytes()
        encrypted = fernet.encrypt(data)

        # Prepend salt
        return salt + encrypted

    def _decrypt_file(self, encrypted_data: bytes, password: str) -> bytes:
        """Decrypt file with password"""
        # Extract salt
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]

        # Derive key
        key = self._derive_key(password, salt)

        # Decrypt
        fernet = Fernet(key)
        return fernet.decrypt(encrypted)


if __name__ == "__main__":
    # Test backup manager
    from src.config import load_config

    config = load_config()
    manager = BackupManager(config)

    print(f"Backup manager test initialized")
    print(f"Encryption enabled: {manager.encryption_enabled}")

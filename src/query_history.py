"""
Query history tracking for SecureRAG.
Uses SQLite to store and retrieve search history.
"""

import sqlite3
import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryHistory:
    """
    Track and manage query history in SQLite database.
    """

    def __init__(self, db_path: str):
        """
        Initialize query history database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Query history initialized: {self.db_path}")

    def _init_database(self):
        """Create database schema if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    collection_name TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filters TEXT,
                    results_count INTEGER,
                    top_confidence FLOAT,
                    search_time_ms INTEGER
                )
            """)

            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON query_history(timestamp DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection
                ON query_history(collection_name)
            """)

            conn.commit()

    def log_query(
        self,
        query: str,
        collection_name: Optional[str],
        filters: Optional[Dict],
        results: List[Dict],
        search_time_ms: int
    ):
        """
        Log a search query to history.

        Args:
            query: Query text
            collection_name: Collection searched (None for all)
            filters: Query filters
            results: Search results
            search_time_ms: Search time in milliseconds
        """
        try:
            results_count = len(results)
            top_confidence = results[0].get("confidence", results[0].get("score", 0.0)) if results else 0.0

            filters_json = json.dumps(filters) if filters else None

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO query_history
                    (query, collection_name, filters, results_count, top_confidence, search_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    query,
                    collection_name,
                    filters_json,
                    results_count,
                    top_confidence,
                    search_time_ms
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Error logging query: {e}")

    def get_query_history(
        self,
        collection_name: Optional[str] = None,
        limit: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve query history with optional filters.

        Args:
            collection_name: Filter by collection (None for all)
            limit: Maximum results to return
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)

        Returns:
            List of query history entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Build query
                sql = "SELECT * FROM query_history WHERE 1=1"
                params = []

                if collection_name:
                    sql += " AND collection_name = ?"
                    params.append(collection_name)

                if start_date:
                    sql += " AND timestamp >= ?"
                    params.append(start_date)

                if end_date:
                    sql += " AND timestamp <= ?"
                    params.append(end_date)

                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                # Convert to dicts
                history = []
                for row in rows:
                    entry = dict(row)

                    # Parse filters JSON
                    if entry['filters']:
                        try:
                            entry['filters'] = json.loads(entry['filters'])
                        except:
                            entry['filters'] = None

                    history.append(entry)

                return history

        except Exception as e:
            logger.error(f"Error retrieving query history: {e}")
            return []

    def get_stats(self) -> Dict:
        """
        Get query history statistics.

        Returns:
            Dict with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total queries
                total = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]

                # Queries by collection
                by_collection = {}
                cursor = conn.execute("""
                    SELECT collection_name, COUNT(*) as count
                    FROM query_history
                    WHERE collection_name IS NOT NULL
                    GROUP BY collection_name
                """)

                for row in cursor:
                    by_collection[row[0]] = row[1]

                # Average search time
                avg_time = conn.execute(
                    "SELECT AVG(search_time_ms) FROM query_history"
                ).fetchone()[0] or 0

                # Most recent query
                recent = conn.execute("""
                    SELECT timestamp FROM query_history
                    ORDER BY timestamp DESC LIMIT 1
                """).fetchone()

                most_recent = recent[0] if recent else None

                return {
                    "total_queries": total,
                    "queries_by_collection": by_collection,
                    "avg_search_time_ms": round(avg_time, 2),
                    "most_recent_query": most_recent
                }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def clear_history(self, collection_name: Optional[str] = None) -> int:
        """
        Clear query history.

        Args:
            collection_name: Clear only this collection (None for all)

        Returns:
            Number of entries deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if collection_name:
                    cursor = conn.execute(
                        "DELETE FROM query_history WHERE collection_name = ?",
                        (collection_name,)
                    )
                else:
                    cursor = conn.execute("DELETE FROM query_history")

                conn.commit()
                deleted = cursor.rowcount

                logger.info(f"Cleared {deleted} query history entries")
                return deleted

        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return 0


def log_query(
    query: str,
    collection_name: Optional[str],
    filters: Optional[Dict],
    results: List[Dict],
    search_time_ms: int,
    db_path: str
):
    """
    Convenience function to log a query.
    """
    history = QueryHistory(db_path)
    history.log_query(query, collection_name, filters, results, search_time_ms)


def get_query_history(
    db_path: str,
    collection_name: Optional[str] = None,
    limit: int = 20
) -> List[Dict]:
    """
    Convenience function to get query history.
    """
    history = QueryHistory(db_path)
    return history.get_query_history(collection_name, limit)


if __name__ == "__main__":
    # Test query history
    from src.config import load_config, get_db_path

    config = load_config()
    db_path = get_db_path(config)

    history = QueryHistory(str(db_path))

    # Test logging a query
    history.log_query(
        query="test query",
        collection_name="test_collection",
        filters={"document_id": "doc123"},
        results=[{"text": "result", "score": 0.8}],
        search_time_ms=150
    )

    # Get history
    entries = history.get_query_history(limit=5)
    print(f"Query history ({len(entries)} entries):")
    for entry in entries:
        print(f"  [{entry['timestamp']}] {entry['query']} -> {entry['results_count']} results")

    # Get stats
    stats = history.get_stats()
    print(f"\nStats: {stats}")

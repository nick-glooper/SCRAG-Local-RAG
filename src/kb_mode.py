"""
KB-only mode manager for SecureRAG.
Manages the state of KB-only mode where Claude only uses knowledge base.
"""

import logging
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class KBModeManager:
    """
    Manage KB-only mode state.
    This is an in-memory state manager (per session).
    """

    def __init__(self):
        """Initialize KB mode manager"""
        self.enabled = False
        self.active_collection: Optional[str] = None
        self.enabled_at: Optional[str] = None

        logger.info("KB mode manager initialized")

    def enable(self, collection_name: Optional[str] = None) -> Dict:
        """
        Enable KB-only mode.

        Args:
            collection_name: Optional collection to restrict searches to

        Returns:
            Dict with mode status
        """
        self.enabled = True
        self.active_collection = collection_name
        self.enabled_at = datetime.utcnow().isoformat()

        logger.info(f"KB mode enabled (collection={collection_name})")

        return {
            "success": True,
            "enabled": True,
            "active_collection": collection_name,
            "enabled_at": self.enabled_at,
            "message": "KB-only mode enabled. Claude will only use knowledge base for responses."
        }

    def disable(self) -> Dict:
        """
        Disable KB-only mode.

        Returns:
            Dict with mode status
        """
        was_enabled = self.enabled

        self.enabled = False
        self.active_collection = None
        disabled_at = datetime.utcnow().isoformat()

        logger.info("KB mode disabled")

        return {
            "success": True,
            "enabled": False,
            "active_collection": None,
            "disabled_at": disabled_at,
            "message": "KB-only mode disabled. Claude will use both knowledge base and general knowledge."
        }

    def get_status(self) -> Dict:
        """
        Get current KB mode status.

        Returns:
            Dict with current status
        """
        return {
            "enabled": self.enabled,
            "active_collection": self.active_collection,
            "enabled_at": self.enabled_at
        }

    def is_enabled(self) -> bool:
        """
        Check if KB mode is enabled.

        Returns:
            True if enabled
        """
        return self.enabled

    def get_active_collection(self) -> Optional[str]:
        """
        Get the active collection for KB mode.

        Returns:
            Collection name or None
        """
        return self.active_collection if self.enabled else None


# Global singleton instance
_kb_mode_manager: Optional[KBModeManager] = None


def get_kb_mode_manager() -> KBModeManager:
    """
    Get or create global KB mode manager instance.

    Returns:
        KBModeManager singleton
    """
    global _kb_mode_manager

    if _kb_mode_manager is None:
        _kb_mode_manager = KBModeManager()

    return _kb_mode_manager


if __name__ == "__main__":
    # Test KB mode manager
    manager = KBModeManager()

    # Test enable
    result = manager.enable("my_collection")
    print(f"Enable: {result}")

    # Test status
    status = manager.get_status()
    print(f"Status: {status}")

    # Test disable
    result = manager.disable()
    print(f"Disable: {result}")

    # Test status again
    status = manager.get_status()
    print(f"Status after disable: {status}")

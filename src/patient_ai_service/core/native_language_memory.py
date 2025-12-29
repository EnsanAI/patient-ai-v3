"""
Native Language Memory System

Stores conversation in the user's original language for response generators.
This provides context for humanization and natural response generation.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from patient_ai_service.core.state_manager import StateBackend, InMemoryBackend, RedisBackend
from patient_ai_service.core.config import settings

logger = logging.getLogger(__name__)


class NativeLanguageTurn(BaseModel):
    """A turn in the user's native language."""
    role: str  # "user" or "assistant"
    content: str  # Original language content
    language: str  # "ar", "en", "es", etc.
    dialect: Optional[str] = None  # "ae", "eg", etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NativeLanguageMemory(BaseModel):
    """
    Stores conversation in user's original language.
    Used by response generators for tone/style matching.
    """
    recent_turns: List[NativeLanguageTurn] = Field(default_factory=list)
    summary: str = ""
    detected_language: str = "en"
    detected_dialect: Optional[str] = None
    turn_count: int = 0
    last_activity: datetime = Field(default_factory=datetime.utcnow)


class NativeLanguageMemoryManager:
    """
    Manages native language conversation history.

    This is a separate memory system that stores conversations in the user's
    original language. It's used by response generators to:
    - Match the user's tone and style
    - Provide context in the same language as the user
    - Enable more natural humanization

    For English-only conversations, this memory will be empty and response
    generators will fall back to the standard ConversationMemory.
    """

    def __init__(
        self,
        backend: Optional[StateBackend] = None,
        max_recent_turns: int = 6
    ):
        """
        Initialize native language memory manager.

        Args:
            backend: Storage backend (Redis or in-memory)
            max_recent_turns: Number of recent turns to keep
        """
        # Initialize backend
        if backend is None:
            if settings.redis_enabled and settings.redis_url:
                try:
                    backend = RedisBackend(settings.redis_url)
                    logger.info("Using Redis backend for native language memory")
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis, falling back to in-memory: {e}")
                    backend = InMemoryBackend()
            else:
                backend = InMemoryBackend()
                logger.info("Using in-memory backend for native language memory")

        self.backend = backend
        self.max_recent_turns = max_recent_turns

        logger.info(f"Initialized NativeLanguageMemoryManager (max_recent_turns={max_recent_turns})")

    def _make_key(self, session_id: str) -> str:
        """Generate storage key for session."""
        return f"native_memory:{session_id}"

    def get_memory(self, session_id: str) -> NativeLanguageMemory:
        """
        Get native language memory for a session.
        Creates new memory if doesn't exist.
        """
        key = self._make_key(session_id)
        data = self.backend.get(key)

        if data:
            try:
                memory_dict = json.loads(data)
                return NativeLanguageMemory(**memory_dict)
            except Exception as e:
                logger.error(f"Error deserializing native language memory: {e}")

        # Create new memory
        return NativeLanguageMemory()

    def _save_memory(self, session_id: str, memory: NativeLanguageMemory):
        """Save native language memory to backend."""
        key = self._make_key(session_id)
        memory.last_activity = datetime.utcnow()

        try:
            data = memory.model_dump_json()
            # Set TTL to 24 hours (86400 seconds)
            self.backend.set(key, data, ttl=86400)
        except Exception as e:
            logger.error(f"Error saving native language memory: {e}")

    def add_user_turn(
        self,
        session_id: str,
        content: str,
        language: str,
        dialect: Optional[str] = None
    ) -> NativeLanguageTurn:
        """
        Add a user message in original language.

        Args:
            session_id: Session identifier
            content: Original message content (not translated)
            language: ISO 639-1 language code (e.g., "ar", "en")
            dialect: Optional dialect code (e.g., "ae", "eg")

        Returns:
            The created turn
        """
        memory = self.get_memory(session_id)

        turn = NativeLanguageTurn(
            role="user",
            content=content,
            language=language,
            dialect=dialect,
            timestamp=datetime.utcnow()
        )

        memory.recent_turns.append(turn)
        memory.turn_count += 1
        memory.detected_language = language
        memory.detected_dialect = dialect

        # Trim to max recent turns
        if len(memory.recent_turns) > self.max_recent_turns:
            memory.recent_turns = memory.recent_turns[-self.max_recent_turns:]

        self._save_memory(session_id, memory)
        logger.debug(f"Added native user turn to session {session_id} ({language}): {content[:50]}...")

        return turn

    def add_assistant_turn(
        self,
        session_id: str,
        content: str
    ) -> NativeLanguageTurn:
        """
        Add an assistant response in user's language.

        Uses the detected language from the memory (set by add_user_turn).

        Args:
            session_id: Session identifier
            content: Response content in user's language

        Returns:
            The created turn
        """
        memory = self.get_memory(session_id)

        turn = NativeLanguageTurn(
            role="assistant",
            content=content,
            language=memory.detected_language,
            dialect=memory.detected_dialect,
            timestamp=datetime.utcnow()
        )

        memory.recent_turns.append(turn)
        memory.turn_count += 1

        # Trim to max recent turns
        if len(memory.recent_turns) > self.max_recent_turns:
            memory.recent_turns = memory.recent_turns[-self.max_recent_turns:]

        self._save_memory(session_id, memory)
        logger.debug(f"Added native assistant turn to session {session_id}: {content[:50]}...")

        return turn

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Get recent turns for response generators.

        Falls back to ConversationMemory if native memory is empty
        (for backward compatibility with legacy sessions).

        Args:
            session_id: Session identifier
            limit: Maximum number of turns to return

        Returns:
            List of turn dicts with role, content, language, dialect
        """
        memory = self.get_memory(session_id)

        if memory.recent_turns:
            # Return native language turns
            turns = memory.recent_turns[-limit:] if len(memory.recent_turns) > limit else memory.recent_turns
            return [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "language": turn.language,
                    "dialect": turn.dialect,
                    "timestamp": turn.timestamp.isoformat() if turn.timestamp else None
                }
                for turn in turns
            ]

        # Fallback to ConversationMemory (for backward compatibility with legacy sessions)
        logger.debug(f"No native memory for session {session_id}, falling back to ConversationMemory")
        from patient_ai_service.core.conversation_memory import get_conversation_memory_manager
        conv_memory_manager = get_conversation_memory_manager()
        conv_memory = conv_memory_manager.get_memory(session_id)

        if conv_memory.recent_turns:
            turns = conv_memory.recent_turns[-limit:] if len(conv_memory.recent_turns) > limit else conv_memory.recent_turns
            return [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "language": "en",
                    "dialect": None,
                    "timestamp": turn.timestamp.isoformat() if hasattr(turn, 'timestamp') and turn.timestamp else None
                }
                for turn in turns
            ]

        return []

    def has_native_memory(self, session_id: str) -> bool:
        """
        Check if session has non-English conversation stored.

        Args:
            session_id: Session identifier

        Returns:
            True if native memory exists with non-English content
        """
        memory = self.get_memory(session_id)
        return len(memory.recent_turns) > 0 and memory.detected_language != "en"

    def clear_session(self, session_id: str):
        """Clear all native language memory for a session."""
        key = self._make_key(session_id)
        self.backend.delete(key)
        logger.info(f"Cleared native language memory for session {session_id}")


# Global instance
_native_language_memory_manager: Optional[NativeLanguageMemoryManager] = None


def get_native_language_memory_manager() -> NativeLanguageMemoryManager:
    """Get or create the global native language memory manager instance."""
    global _native_language_memory_manager
    if _native_language_memory_manager is None:
        _native_language_memory_manager = NativeLanguageMemoryManager()
    return _native_language_memory_manager


def reset_native_language_memory_manager():
    """Reset the global native language memory manager (for testing)."""
    global _native_language_memory_manager
    _native_language_memory_manager = None

"""
Conversation Memory System

Provides semantic understanding, summarization, and context tracking
for natural, human-like conversations.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from patient_ai_service.core.state_manager import StateBackend, InMemoryBackend, RedisBackend
from patient_ai_service.core.llm import LLMClient, get_llm_client
from patient_ai_service.core.config import settings

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationMemory(BaseModel):
    """
    Optimized memory structure with persistent facts and smart summarization.
    """
    # Persistent facts (NEVER summarized - always preserved)
    user_facts: Dict[str, Any] = Field(default_factory=dict)
    # Example: {"name": "John", "phone": "+1234", "wants_cleaning": True}

    # Narrative summary (summarized when token count > threshold)
    summary: str = ""

    # Recent turns (ALWAYS raw, last 6 kept)
    recent_turns: List[ConversationTurn] = Field(default_factory=list)

    # System state tracking
    last_action: str = ""  # "proposed_registration", "asked_question"
    awaiting: str = ""  # "confirmation", "date_selection"

    # Metadata
    turn_count: int = 0
    last_activity: datetime = Field(default_factory=datetime.utcnow)


class ConversationMemoryManager:
    """
    Manages conversation history with semantic understanding.

    Features:
    - Stores conversation turns with metadata
    - Maintains persistent facts (never summarized)
    - Smart summarization based on token count
    - Conversation boundary detection
    - Redis persistence support
    """

    def __init__(
        self,
        backend: Optional[StateBackend] = None,
        llm_client: Optional[LLMClient] = None,
        max_recent_turns: int = 6,
        summarization_token_threshold: int = 2000
    ):
        """
        Initialize conversation memory manager.

        Args:
            backend: Storage backend (Redis or in-memory)
            llm_client: LLM client for summarization
            max_recent_turns: Number of recent turns to keep raw
            summarization_token_threshold: Token count to trigger summarization
        """
        # Initialize backend
        if backend is None:
            if settings.redis_enabled and settings.redis_url:
                try:
                    backend = RedisBackend(settings.redis_url)
                    logger.info("Using Redis backend for conversation memory")
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis, falling back to in-memory: {e}")
                    backend = InMemoryBackend()
            else:
                backend = InMemoryBackend()
                logger.info("Using in-memory backend for conversation memory")

        self.backend = backend
        self.llm_client = llm_client or get_llm_client()
        self.max_recent_turns = max_recent_turns
        self.summarization_token_threshold = summarization_token_threshold

        logger.info(f"Initialized ConversationMemoryManager (max_recent_turns={max_recent_turns}, "
                   f"token_threshold={summarization_token_threshold})")

    def _make_key(self, session_id: str) -> str:
        """Generate storage key for session."""
        return f"conversation_memory:{session_id}"

    def get_memory(self, session_id: str) -> ConversationMemory:
        """
        Get conversation memory for a session.
        Creates new memory if doesn't exist.
        """
        key = self._make_key(session_id)
        data = self.backend.get(key)

        if data:
            try:
                memory_dict = json.loads(data)
                return ConversationMemory(**memory_dict)
            except Exception as e:
                logger.error(f"Error deserializing conversation memory: {e}")

        # Create new memory
        logger.info(f"Creating new conversation memory for session: {session_id}")
        return ConversationMemory()

    def _save_memory(self, session_id: str, memory: ConversationMemory):
        """Save conversation memory to backend."""
        key = self._make_key(session_id)
        memory.last_activity = datetime.utcnow()

        try:
            data = memory.model_dump_json()
            # Set TTL to 24 hours (86400 seconds)
            self.backend.set(key, data, ttl=86400)
        except Exception as e:
            logger.error(f"Error saving conversation memory: {e}")

    def add_user_turn(self, session_id: str, content: str) -> ConversationTurn:
        """Add a user message to conversation memory."""
        memory = self.get_memory(session_id)

        turn = ConversationTurn(
            role="user",
            content=content,
            timestamp=datetime.utcnow()
        )

        memory.recent_turns.append(turn)
        memory.turn_count += 1

        # Trim to max recent turns
        if len(memory.recent_turns) > self.max_recent_turns:
            memory.recent_turns = memory.recent_turns[-self.max_recent_turns:]

        self._save_memory(session_id, memory)
        logger.debug(f"Added user turn to session {session_id}: {content[:50]}...")

        return turn

    def add_assistant_turn(self, session_id: str, content: str) -> ConversationTurn:
        """Add an assistant message to conversation memory."""
        memory = self.get_memory(session_id)

        turn = ConversationTurn(
            role="assistant",
            content=content,
            timestamp=datetime.utcnow()
        )

        memory.recent_turns.append(turn)
        memory.turn_count += 1

        # Trim to max recent turns
        if len(memory.recent_turns) > self.max_recent_turns:
            memory.recent_turns = memory.recent_turns[-self.max_recent_turns:]

        self._save_memory(session_id, memory)
        logger.debug(f"Added assistant turn to session {session_id}: {content[:50]}...")

        return turn

    def update_facts(self, session_id: str, new_facts: Dict[str, Any]):
        """
        Update user facts (persistent, never summarized).

        Args:
            session_id: Session identifier
            new_facts: Dict of new facts to merge
        """
        if not new_facts:
            return

        memory = self.get_memory(session_id)
        memory.user_facts.update(new_facts)
        self._save_memory(session_id, memory)

        logger.info(f"Updated facts for session {session_id}: {list(new_facts.keys())}")

    def update_system_state(self, session_id: str, last_action: str = None, awaiting: str = None):
        """
        Update system state tracking.

        Args:
            session_id: Session identifier
            last_action: What system just did
            awaiting: What system is waiting for
        """
        memory = self.get_memory(session_id)

        if last_action is not None:
            memory.last_action = last_action
        if awaiting is not None:
            memory.awaiting = awaiting

        self._save_memory(session_id, memory)
        logger.debug(f"Updated system state for session {session_id}: action={last_action}, awaiting={awaiting}")

    def should_summarize(self, session_id: str) -> bool:
        """
        Check if summarization needed based on token count.

        Args:
            session_id: Session identifier

        Returns:
            True if should summarize, False otherwise
        """
        memory = self.get_memory(session_id)

        # Need at least 4 turns to summarize
        if len(memory.recent_turns) < 4:
            return False

        # Calculate approximate tokens in recent turns (excluding last 6 which we always keep)
        # We only summarize older turns
        turns_to_check = memory.recent_turns[:-self.max_recent_turns] if len(memory.recent_turns) > self.max_recent_turns else []

        if not turns_to_check:
            return False

        recent_text = "\n".join([turn.content for turn in turns_to_check])
        estimated_tokens = len(recent_text) // 4  # Rough estimate: 4 chars per token

        return estimated_tokens > self.summarization_token_threshold

    async def summarize(self, session_id: str):
        """
        Summarize older turns while preserving:
        - user_facts (never summarized)
        - last N raw turns (always kept)

        Args:
            session_id: Session identifier
        """
        memory = self.get_memory(session_id)

        # Get turns to summarize (all except last max_recent_turns)
        if len(memory.recent_turns) <= self.max_recent_turns:
            logger.debug(f"Not enough turns to summarize for session {session_id}")
            return

        to_summarize = memory.recent_turns[:-self.max_recent_turns]

        if len(to_summarize) < 3:
            logger.debug(f"Not enough old turns to summarize for session {session_id}")
            return

        # Format turns for summarization
        formatted_turns = "\n".join([
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
            for turn in to_summarize
        ])

        prompt = f"""Summarize this conversation segment, preserving key information.

PREVIOUS SUMMARY:
{memory.summary if memory.summary else "None - this is the first summary."}

NEW CONVERSATION SEGMENT:
{formatted_turns}

Create an updated summary that captures:
1. What the user wants/needs
2. What was discussed and decided
3. Any pending items or unresolved questions

IMPORTANT:
- Specific facts (names, dates, phone numbers) are stored separately - don't include them
- Focus on the narrative flow and context
- Keep it concise but complete (2-4 sentences)

Write the summary in third person (e.g., "The user wants to book an appointment...")"""

        try:
            new_summary = self.llm_client.create_message(
                system="You are a conversation summarizer. Create concise, informative summaries.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            memory.summary = new_summary.strip()
            # Keep only the last max_recent_turns
            memory.recent_turns = memory.recent_turns[-self.max_recent_turns:]

            self._save_memory(session_id, memory)

            logger.info(f"Summarized conversation for session {session_id} (kept {len(memory.recent_turns)} recent turns)")

        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}", exc_info=True)

    def is_conversation_restart(self, session_id: str, message: str) -> bool:
        """
        Detect if conversation should restart.
        Prevents context pollution across unrelated conversations.

        Args:
            session_id: Session identifier
            message: User message

        Returns:
            True if conversation should restart, False otherwise
        """
        memory = self.get_memory(session_id)
        message_lower = message.lower().strip()

        # Heuristic 1: Time gap > 24 hours
        if memory.last_activity:
            gap = datetime.utcnow() - memory.last_activity
            if gap.total_seconds() > 86400:  # 24 hours
                logger.info(f"Conversation restart detected (time gap) for session {session_id}")
                return True

        # Heuristic 2: Greeting after completed task
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        is_greeting = any(g in message_lower for g in greetings)

        if is_greeting and memory.last_action in ["task_completed", "appointment_booked", "registration_complete"]:
            logger.info(f"Conversation restart detected (greeting after completion) for session {session_id}")
            return True

        # Heuristic 3: Explicit restart signals
        restart_signals = ["start over", "new conversation", "forget everything", "reset"]
        if any(s in message_lower for s in restart_signals):
            logger.info(f"Conversation restart detected (explicit signal) for session {session_id}")
            return True

        return False

    def archive_and_reset(self, session_id: str):
        """
        Archive old conversation and start fresh.

        Args:
            session_id: Session identifier
        """
        # Get current memory for optional archival
        memory = self.get_memory(session_id)

        # TODO: Optional - archive to database for analytics
        # await self.archive_conversation(session_id, memory)

        # Delete from backend
        key = self._make_key(session_id)
        self.backend.delete(key)

        logger.info(f"Archived and reset conversation for session {session_id}")

    def clear_session(self, session_id: str):
        """Clear all memory for a session."""
        key = self._make_key(session_id)
        self.backend.delete(key)
        logger.info(f"Cleared conversation memory for session {session_id}")


# Global instance
_conversation_memory_manager: Optional[ConversationMemoryManager] = None


def get_conversation_memory_manager() -> ConversationMemoryManager:
    """Get or create the global conversation memory manager instance."""
    global _conversation_memory_manager
    if _conversation_memory_manager is None:
        _conversation_memory_manager = ConversationMemoryManager()
    return _conversation_memory_manager


def reset_conversation_memory_manager():
    """Reset the global conversation memory manager (for testing)."""
    global _conversation_memory_manager
    _conversation_memory_manager = None

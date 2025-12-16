"""
Reasoning State - Persistent Understanding Across Turns

This module maintains what we've established about the conversation,
so we don't need to re-derive everything on each message.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReasoningState(BaseModel):
    """
    Persistent reasoning state within a conversation.
    
    This captures what we've established so far, so continuations
    don't need to re-derive basic understanding.
    """
    
    session_id: str
    
    # Established understanding
    established_intent: Optional[str] = None  # "book_appointment", "cancel", "inquire"
    intent_established_at: Optional[datetime] = None
    
    # Active workflow
    active_agent: Optional[str] = None
    workflow_step: Optional[str] = None  # "collecting_info", "confirming", "executing"
    
    # What we're waiting for
    awaiting: Optional[str] = None  # "time_selection", "confirmation", "doctor_choice"
    awaiting_since: Optional[datetime] = None
    
    # Last system action
    last_proposal: Optional[str] = None  # What we proposed to user
    presented_options: List[Any] = Field(default_factory=list)
    
    # Confidence tracking
    understanding_confidence: float = 1.0
    
    # Turn tracking
    turns_since_intent_established: int = 0
    turns_waiting: int = 0
    
    # Context summary
    conversation_summary: Optional[str] = None
    key_facts: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def update_awaiting(self, awaiting: str, options: List[Any] = None):
        """Update what we're waiting for."""
        logger.info(f"🎯 [ReasoningState] ⏸️  Setting awaiting: {awaiting}")
        if options:
            logger.info(f"🎯 [ReasoningState]   Options presented: {len(options)}")
        self.awaiting = awaiting
        self.awaiting_since = datetime.utcnow()
        self.turns_waiting = 0
        if options:
            self.presented_options = options
        self.last_updated_at = datetime.utcnow()
    
    def clear_awaiting(self):
        """Clear awaiting state when resolved."""
        if self.awaiting:
            logger.info(f"🎯 [ReasoningState] ✅ Clearing awaiting: {self.awaiting}")
        self.awaiting = None
        self.awaiting_since = None
        self.presented_options = []
        self.turns_waiting = 0
        self.last_updated_at = datetime.utcnow()
    
    def establish_intent(self, intent: str, agent: str = None):
        """Establish user's intent."""
        logger.info(f"🎯 [ReasoningState] 🎯 Establishing intent: {intent}")
        if agent:
            logger.info(f"🎯 [ReasoningState]   Active agent: {agent}")
        self.established_intent = intent
        self.intent_established_at = datetime.utcnow()
        self.turns_since_intent_established = 0
        if agent:
            self.active_agent = agent
        self.last_updated_at = datetime.utcnow()
    
    def increment_turn(self):
        """Increment turn counters."""
        self.turns_since_intent_established += 1
        if self.awaiting:
            self.turns_waiting += 1
            logger.debug(f"🎯 [ReasoningState] Turn {self.turns_since_intent_established}, waiting {self.turns_waiting} turns for: {self.awaiting}")
        self.last_updated_at = datetime.utcnow()
    
    def is_stale(self, max_turns: int = 10) -> bool:
        """Check if established intent is stale."""
        return self.turns_since_intent_established > max_turns
    
    def has_active_flow(self) -> bool:
        """Check if there's an active flow in progress."""
        return self.active_agent is not None and self.established_intent is not None
    
    def get_context_for_assessor(self) -> Dict[str, Any]:
        """Get context to include in assessor prompt."""
        return {
            "established_intent": self.established_intent,
            "active_agent": self.active_agent,
            "awaiting": self.awaiting,
            "presented_options": self.presented_options[:5] if self.presented_options else [],
            "last_proposal": self.last_proposal,
            "turns_waiting": self.turns_waiting
        }


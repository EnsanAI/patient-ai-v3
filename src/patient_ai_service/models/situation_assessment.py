"""
Situation Assessment Models

Defines the output structure from the Situation Assessor,
which determines reasoning depth for each message.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class SituationType(str, Enum):
    """Type of situation detected in user message."""
    
    # Continuations - user responding to our previous message
    DIRECT_CONTINUATION = "direct_continuation"      # "yes", "ok", "sure"
    SELECTION = "selection"                          # "the 3pm one", "Dr. Sarah"
    CONFIRMATION = "confirmation"                    # "yes, book it", "confirm"
    REJECTION = "rejection"                          # "no", "neither", "cancel"
    
    # Modifications - user adjusting previous request
    MODIFICATION = "modification"                    # "actually 4pm instead"
    PARTIAL_MODIFICATION = "partial_modification"    # "same doctor but different time"
    
    # New intents
    NEW_INTENT = "new_intent"                        # Completely new request
    TOPIC_SHIFT = "topic_shift"                      # Changing subjects mid-flow
    
    # Ambiguous
    AMBIGUOUS = "ambiguous"                          # Can't determine with confidence
    CLARIFICATION_RESPONSE = "clarification_response"  # User answering our question
    
    # Special
    GREETING = "greeting"                            # Hello, hi, good morning
    FAREWELL = "farewell"                            # Bye, goodbye
    THANKS = "thanks"                                # Thank you, thanks
    PLEASANTRY = "pleasantry"                       # How are you, etc. - Pure social exchange, NO info request
    EMERGENCY = "emergency"                          # Urgent medical situation
    
    # NEW types for better awaiting handling
    DIRECT_ANSWER = "direct_answer"  # User directly answers what we asked
    PIVOT_SAME_FLOW = "pivot_same_flow"  # Different question, same agent/goal


class ReasoningNeeds(str, Enum):
    """How much reasoning is needed for this message."""
    
    NONE = "none"                    # Can handle with simple logic (greeting)
    MINIMAL = "minimal"              # Just extract entities, no complex reasoning
    FOCUSED = "focused"              # Use focused prompts, skip full reasoning
    COMPREHENSIVE = "comprehensive"  # Full reasoning engine needed
    VALIDATION = "validation"        # Need to validate/verify something


class ExtractedEntities(BaseModel):
    """Entities extracted by Situation Assessor."""
    
    # Appointment-related
    doctor_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    procedure: Optional[str] = None
    
    # Selection from options
    selected_option: Optional[Any] = None
    selected_index: Optional[int] = None  # If user said "the first one"
    
    # Personal info
    name: Optional[str] = None
    phone: Optional[str] = None
    
    # Raw extraction (for anything else)
    raw: Dict[str, Any] = Field(default_factory=dict)


class EntityChanges(BaseModel):
    """Changes to entities detected in this message."""
    
    changed_fields: List[str] = Field(default_factory=list)
    # e.g., ["doctor_preference", "time_preference"]
    
    changes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # e.g., {"doctor_preference": {"from": "Dr. Sarah", "to": "Dr. Ahmed"}}


class SituationAssessment(BaseModel):
    """
    Complete assessment of a user message.
    
    This is the output of the Situation Assessor, used to determine
    how to route and process the message.
    """
    
    # Core classification
    situation_type: SituationType
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    
    # Understanding
    key_understanding: str  # What the user means in plain English
    user_sentiment: str = "neutral"  # "positive", "negative", "neutral", "frustrated", "confused"
    
    # Entity extraction
    extracted_entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    entity_changes: EntityChanges = Field(default_factory=EntityChanges)
    
    # Reasoning determination
    reasoning_needs: ReasoningNeeds = ReasoningNeeds.COMPREHENSIVE
    reasoning_reason: str = ""  # Why this level of reasoning
    
    # Routing hints
    suggested_agent: Optional[str] = None
    continue_with_active_agent: bool = False
    # Optional tone guidance for downstream responses
    suggested_tone: Optional[str] = None
    
    # Continuation context
    is_response_to_options: bool = False
    references_previous_context: bool = False
    
    # NEW: Track if this responds to what we were awaiting
    is_response_to_awaiting: bool = False
    awaiting_match_type: Optional[str] = None  # "exact", "partial", "none"
    
    # Timing
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    assessment_duration_ms: Optional[float] = None
    
    def needs_comprehensive_reasoning(self) -> bool:
        """Check if comprehensive reasoning is needed."""
        # Low confidence always triggers comprehensive
        if self.confidence < 0.7:
            return True
        
        # Certain situation types always need comprehensive
        if self.situation_type in [
            SituationType.NEW_INTENT,
            SituationType.TOPIC_SHIFT,
            SituationType.AMBIGUOUS,
            SituationType.EMERGENCY
        ]:
            return True
        
        # Explicit reasoning needs
        if self.reasoning_needs == ReasoningNeeds.COMPREHENSIVE:
            return True
        
        return False
    
    def can_use_focused_handling(self) -> bool:
        """Check if focused handling is sufficient."""
        return (
            self.confidence >= 0.7 and
            self.reasoning_needs in [ReasoningNeeds.NONE, ReasoningNeeds.MINIMAL, ReasoningNeeds.FOCUSED] and
            self.situation_type not in [
                SituationType.NEW_INTENT,
                SituationType.TOPIC_SHIFT,
                SituationType.AMBIGUOUS,
                SituationType.EMERGENCY
            ]
        )
    
    def to_routing_context(self) -> Dict[str, Any]:
        """Convert to context dict for routing decisions."""
        return {
            "situation_type": self.situation_type.value,
            "confidence": self.confidence,
            "key_understanding": self.key_understanding,
            "reasoning_needs": self.reasoning_needs.value,
            "continue_with_active_agent": self.continue_with_active_agent,
            "is_response_to_options": self.is_response_to_options,
            "extracted_entities": self.extracted_entities.model_dump(),
            "entity_changes": self.entity_changes.model_dump()
        }


"""
Situation Assessment Models - Enums Only

The SituationType enum is still used by UnifiedReasoning.
"""

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
    UNCLEAR_REQUEST = "unclear_request"             # Request is unclear or ambiguous, needs clarification
    EMERGENCY = "emergency"                          # Urgent medical situation
    
    # Additional types
    DIRECT_ANSWER = "direct_answer"
    PIVOT_SAME_FLOW = "pivot_same_flow"


class ReasoningNeeds(str, Enum):
    """How much reasoning is needed for this message."""
    NONE = "none"
    MINIMAL = "minimal"
    FOCUSED = "focused"
    COMPREHENSIVE = "comprehensive"
    VALIDATION = "validation"


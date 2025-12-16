"""
Enhanced Reasoning Engine - Continuation Handling

This file contains updates to reasoning.py to support:
1. Better continuation detection ("yeah", "4pm works", "the first one")
2. Passing blocked/continuation context to agents
3. Recognizing user selections from presented options

Merge these changes with your existing reasoning.py
"""

import json
import logging
import re
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED MODELS
# =============================================================================

class TaskContext(BaseModel):
    """
    Structured context for agent execution.
    """
    user_intent: str = ""
    entities: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    prior_context: Optional[str] = None
    
    # NEW: Continuation fields
    is_continuation: bool = False
    continuation_type: Optional[str] = None  # "selection", "confirmation", "clarification"
    selected_option: Optional[Any] = None  # The option user selected
    continuation_context: Dict[str, Any] = Field(default_factory=dict)


class ResponseGuidance(BaseModel):
    """Guidance for the selected agent's response."""
    tone: str = "helpful"
    task_context: TaskContext = Field(default_factory=TaskContext)
    minimal_context: Dict[str, Any] = Field(default_factory=dict)
    plan: str = ""  # Deprecated


# =============================================================================
# CONTINUATION DETECTION HELPERS
# =============================================================================

class ContinuationDetector:
    """
    Detects when a user message is a continuation/response to previous options.
    """
    
    # Affirmative responses
    AFFIRMATIVE_PATTERNS = [
        r"^(yes|yeah|yep|yup|sure|ok|okay|alright|sounds good|perfect|great|fine)\.?$",
        r"^(that works|that\'s fine|that\'s good|go ahead|please do|do it)\.?$",
        r"^(the first one|the second one|the third one|first|second|third)\.?$",
        r"^(option [123a-c]|[123a-c])\.?$"
    ]
    
    # Time selection patterns
    TIME_PATTERNS = [
        r"^(\d{1,2})(:\d{2})?\s*(am|pm)?\.?$",  # "3", "3pm", "3:00 pm"
        r"^(the )?\d{1,2}(:\d{2})?\s*(am|pm)?( one)?\.?$",  # "the 3pm one"
    ]
    
    # Negative responses
    NEGATIVE_PATTERNS = [
        r"^(no|nope|nah|not really|neither|none)\.?$",
        r"^(actually|wait|hold on|never ?mind)\.?",
    ]
    
    @classmethod
    def detect_continuation_type(
        cls,
        message: str,
        awaiting: Optional[str] = None,
        presented_options: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect if message is a continuation and what type.
        
        Returns:
            {
                "is_continuation": True/False,
                "continuation_type": "selection" | "confirmation" | "rejection" | "modification",
                "selected_option": The option user selected (if applicable),
                "confidence": 0.0-1.0
            }
        """
        message_lower = message.lower().strip()
        
        result = {
            "is_continuation": False,
            "continuation_type": None,
            "selected_option": None,
            "confidence": 0.0
        }
        
        # Check for affirmative response
        for pattern in cls.AFFIRMATIVE_PATTERNS:
            if re.match(pattern, message_lower, re.IGNORECASE):
                result["is_continuation"] = True
                result["continuation_type"] = "confirmation"
                result["confidence"] = 0.9
                
                # Check for ordinal selection
                if "first" in message_lower and presented_options:
                    result["continuation_type"] = "selection"
                    result["selected_option"] = presented_options[0] if presented_options else None
                elif "second" in message_lower and presented_options and len(presented_options) > 1:
                    result["continuation_type"] = "selection"
                    result["selected_option"] = presented_options[1]
                elif "third" in message_lower and presented_options and len(presented_options) > 2:
                    result["continuation_type"] = "selection"
                    result["selected_option"] = presented_options[2]
                
                return result
        
        # Check for time selection
        for pattern in cls.TIME_PATTERNS:
            match = re.match(pattern, message_lower, re.IGNORECASE)
            if match:
                result["is_continuation"] = True
                result["continuation_type"] = "selection"
                result["confidence"] = 0.85
                
                # Extract the time value
                time_value = cls._extract_time(message_lower)
                result["selected_option"] = time_value
                
                # Validate against presented options if available
                if presented_options:
                    matched = cls._match_time_to_options(time_value, presented_options)
                    if matched:
                        result["selected_option"] = matched
                        result["confidence"] = 0.95
                
                return result
        
        # Check for negative response
        for pattern in cls.NEGATIVE_PATTERNS:
            if re.match(pattern, message_lower, re.IGNORECASE):
                result["is_continuation"] = True
                result["continuation_type"] = "rejection"
                result["confidence"] = 0.85
                return result
        
        # Check if it matches one of the presented options directly
        if presented_options:
            for option in presented_options:
                option_str = str(option).lower()
                if option_str in message_lower or message_lower in option_str:
                    result["is_continuation"] = True
                    result["continuation_type"] = "selection"
                    result["selected_option"] = option
                    result["confidence"] = 0.95
                    return result
        
        # Check based on what we're awaiting
        if awaiting:
            if awaiting == "time_selection" and cls._looks_like_time(message_lower):
                result["is_continuation"] = True
                result["continuation_type"] = "selection"
                result["selected_option"] = cls._extract_time(message_lower)
                result["confidence"] = 0.8
                return result
            
            elif awaiting == "confirmation" and len(message_lower) < 20:
                # Short message when awaiting confirmation is likely a response
                result["is_continuation"] = True
                result["continuation_type"] = "clarification"
                result["confidence"] = 0.6
                return result
        
        return result
    
    @classmethod
    def _extract_time(cls, message: str) -> str:
        """Extract time value from message."""
        # Look for time patterns
        match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', message, re.IGNORECASE)
        if match:
            hour = match.group(1)
            minute = match.group(2) or "00"
            period = match.group(3)
            
            if period:
                return f"{hour}:{minute} {period}"
            return f"{hour}:{minute}"
        
        return message
    
    @classmethod
    def _looks_like_time(cls, message: str) -> bool:
        """Check if message looks like a time."""
        return bool(re.search(r'\d{1,2}(:\d{2})?\s*(am|pm)?', message, re.IGNORECASE))
    
    @classmethod
    def _match_time_to_options(cls, time_value: str, options: List[Any]) -> Optional[Any]:
        """Try to match a time value to one of the presented options."""
        # Normalize the time
        time_lower = time_value.lower().replace(" ", "")
        
        for option in options:
            option_str = str(option).lower().replace(" ", "")
            
            # Direct match
            if time_lower == option_str:
                return option
            
            # Extract hour and check
            time_match = re.search(r'(\d{1,2})', time_lower)
            option_match = re.search(r'(\d{1,2})', option_str)
            
            if time_match and option_match:
                if time_match.group(1) == option_match.group(1):
                    # Same hour - likely a match
                    return option
        
        return None


# =============================================================================
# ENHANCED REASONING PROMPT
# =============================================================================

def _build_reasoning_prompt_with_continuation(
    self,
    user_message: str,
    memory: Any,
    patient_info: Dict[str, Any],
    continuation_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build reasoning prompt with continuation awareness.
    """
    
    # Format recent turns
    recent_turns_formatted = "\n".join([
        f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
        for turn in memory.recent_turns
    ])
    
    # Continuation context section
    continuation_section = ""
    if continuation_context:
        continuation_section = f"""
═══════════════════════════════════════════════════════════════
CONTINUATION CONTEXT (Previous flow was interrupted)
═══════════════════════════════════════════════════════════════

The system is waiting for: {continuation_context.get('awaiting', 'user response')}
Options presented to user: {continuation_context.get('presented_options', [])}
Original request: {continuation_context.get('original_request', 'Unknown')}
Resolved so far: {json.dumps(continuation_context.get('resolved_entities', {}), indent=2)}

IMPORTANT: Check if user's message is a response to the above!
"""
    
    return f"""Analyze this conversation situation and respond with ONE complete JSON.

═══════════════════════════════════════════════════════════════
CONVERSATION STATE
═══════════════════════════════════════════════════════════════

**User Facts:**
{json.dumps(memory.user_facts, indent=2) if memory.user_facts else "None collected yet"}

**Conversation Summary:**
{memory.summary if memory.summary else "No previous conversation"}

**Recent Messages:**
{recent_turns_formatted if recent_turns_formatted else "No previous messages"}

**System State:**
- Last Action: {memory.last_action or "None"}
- Awaiting: {memory.awaiting or "Nothing specific"}
{continuation_section}

═══════════════════════════════════════════════════════════════
PATIENT INFORMATION
═══════════════════════════════════════════════════════════════

- Name: {patient_info.get('first_name', 'Unknown')}
- Registered: {'Yes' if patient_info.get('patient_id') else 'No'}
- Patient ID: {patient_info.get('patient_id', 'None')}

═══════════════════════════════════════════════════════════════
NEW USER MESSAGE
═══════════════════════════════════════════════════════════════

"{user_message}"

═══════════════════════════════════════════════════════════════
CONTINUATION DETECTION RULES
═══════════════════════════════════════════════════════════════

If user's message is short (1-3 words) AND system was awaiting a response:
- "yeah", "ok", "sure" → User CONFIRMS previous suggestion
- "3pm", "4pm", "2:30" → User SELECTS a time from options
- "the first one" → User selects first option
- "no", "neither" → User REJECTS options, needs alternatives
- "actually..." → User wants to CHANGE something

When detected as continuation:
- Set is_continuation: true
- Set continuation_type: "selection" | "confirmation" | "rejection"
- Extract selected_option if applicable
- Include resolved_entities from continuation context

═══════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════

Respond with JSON:
{{
    "understanding": {{
        "what_user_means": "Plain English explanation",
        "is_continuation": true/false,
        "continuation_type": "selection/confirmation/rejection/new_request/null",
        "selected_option": "value user selected or null",
        "sentiment": "affirmative/negative/neutral/unclear",
        "is_conversation_restart": false
    }},
    "routing": {{
        "agent": "appointment_manager/registration/general_assistant/medical_inquiry/emergency_response",
        "action": "High-level action",
        "urgency": "routine/urgent/emergency"
    }},
    "memory_updates": {{
        "new_facts": {{}},
        "system_action": "What system did",
        "awaiting": "What system awaits"
    }},
    "response_guidance": {{
        "tone": "helpful/empathetic/urgent/professional",
        "task_context": {{
            "user_intent": "What user wants (incorporate continuation context if resuming)",
            "entities": {{
                // Include entities from continuation_context.resolved_entities if resuming
                // Add any new entities from current message
            }},
            "success_criteria": [
                "Same criteria as before if resuming blocked flow"
            ],
            "constraints": [],
            "prior_context": "Relevant context including previous options",
            "is_continuation": true/false,
            "continuation_type": "selection/confirmation/rejection/null",
            "selected_option": "The option user selected",
            "continuation_context": {{
                // Copy from continuation_context if resuming
            }}
        }}
    }},
    "reasoning_chain": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
    ]
}}

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

Example 1: User selects from options
Previous: "3pm isn't available. Would 2pm or 4pm work?"
User: "4pm"
→ is_continuation: true, continuation_type: "selection", selected_option: "4pm"
→ entities should include resolved entities + selected time

Example 2: User confirms
Previous: "Should I book 2pm with Dr. Mohammed?"
User: "yeah"
→ is_continuation: true, continuation_type: "confirmation"
→ Proceed with the proposed action

Example 3: User rejects
Previous: "I have 2pm, 3pm, or 4pm available"
User: "none of those work"
→ is_continuation: true, continuation_type: "rejection"
→ Need to find more alternatives or ask for preferred time

Example 4: New request (not continuation)
Previous: "Your appointment is booked!"
User: "I also need a cleaning"
→ is_continuation: false (new request, not responding to options)

RESPOND WITH VALID JSON ONLY."""


# =============================================================================
# ENHANCED REASON METHOD
# =============================================================================

async def reason_with_continuation(
    self,
    session_id: str,
    user_message: str,
    patient_info: Dict[str, Any] = None
):
    """
    Perform reasoning with continuation awareness.
    
    This is the enhanced version of the reason() method.
    """
    
    # Get conversation memory
    memory = self.memory_manager.get_memory(session_id)
    
    # Check for continuation context in state
    continuation_context = self.state_manager.get_continuation_context(session_id)
    
    # Pre-detect continuation to help LLM
    continuation_detection = None
    if continuation_context:
        continuation_detection = ContinuationDetector.detect_continuation_type(
            user_message,
            awaiting=continuation_context.get("awaiting"),
            presented_options=continuation_context.get("presented_options")
        )
        
        if continuation_detection.get("is_continuation"):
            logger.info(
                f"Detected continuation: type={continuation_detection.get('continuation_type')}, "
                f"selected={continuation_detection.get('selected_option')}"
            )
    
    # Build prompt with continuation context
    prompt = self._build_reasoning_prompt_with_continuation(
        user_message,
        memory,
        patient_info or {},
        continuation_context
    )
    
    # Call LLM
    response = self.llm_client.create_message(
        system=self._get_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.reasoning_temperature
    )
    
    # Parse response
    output = self._parse_reasoning_response_with_continuation(
        response,
        user_message,
        memory,
        continuation_context,
        continuation_detection
    )
    
    # If this was a continuation, clear the continuation context
    if output.understanding.is_continuation:
        # Don't clear yet - let agent handle it
        # Agent will clear after successful processing
        pass
    
    return output


def _parse_reasoning_response_with_continuation(
    self,
    response: str,
    user_message: str,
    memory: Any,
    continuation_context: Optional[Dict[str, Any]],
    continuation_detection: Optional[Dict[str, Any]]
) -> 'ReasoningOutput':
    """
    Parse reasoning response with continuation handling.
    """
    try:
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")
        
        data = json.loads(json_match.group())
        
        # Parse nested structures
        understanding_data = data.get("understanding", {})
        routing_data = data.get("routing", {})
        memory_data = data.get("memory_updates", {})
        guidance_data = data.get("response_guidance", {})
        task_context_data = guidance_data.get("task_context", {})
        
        # Enhance with continuation detection if LLM missed it
        is_continuation = understanding_data.get("is_continuation", False)
        continuation_type = understanding_data.get("continuation_type")
        selected_option = understanding_data.get("selected_option")
        
        if continuation_detection and continuation_detection.get("is_continuation"):
            if not is_continuation:
                logger.info("LLM missed continuation, using detected values")
                is_continuation = True
                continuation_type = continuation_detection.get("continuation_type")
                selected_option = continuation_detection.get("selected_option")
        
        # If continuing, merge in resolved entities
        entities = task_context_data.get("entities", {})
        if is_continuation and continuation_context:
            resolved = continuation_context.get("resolved_entities", {})
            # Merge resolved entities (don't overwrite new ones)
            for key, value in resolved.items():
                if key not in entities:
                    entities[key] = value
            
            # Add selected option to entities
            if selected_option:
                if continuation_type == "selection":
                    # Figure out what was selected
                    awaiting = continuation_context.get("awaiting", "")
                    if "time" in awaiting.lower():
                        entities["time"] = selected_option
                    elif "doctor" in awaiting.lower():
                        entities["doctor_id"] = selected_option
                    else:
                        entities["selected_option"] = selected_option
        
        # Build TaskContext
        task_context = TaskContext(
            user_intent=task_context_data.get("user_intent", understanding_data.get("what_user_means", "")),
            entities=entities,
            success_criteria=task_context_data.get("success_criteria", []),
            constraints=task_context_data.get("constraints", []),
            prior_context=task_context_data.get("prior_context"),
            is_continuation=is_continuation,
            continuation_type=continuation_type,
            selected_option=selected_option,
            continuation_context=continuation_context or {}
        )
        
        # Build backward-compatible minimal_context
        minimal_context = guidance_data.get("minimal_context", {})
        minimal_context.update({
            "user_wants": task_context.user_intent,
            "action": routing_data.get("action", ""),
            "is_continuation": is_continuation,
            "continuation_type": continuation_type,
            "selected_option": selected_option
        })
        
        # Add continuation context for agent
        if continuation_context:
            minimal_context["continuation_context"] = continuation_context
        
        return ReasoningOutput(
            understanding=UnderstandingResult(
                what_user_means=understanding_data.get("what_user_means", user_message),
                is_continuation=is_continuation,
                sentiment=understanding_data.get("sentiment", "neutral"),
                is_conversation_restart=understanding_data.get("is_conversation_restart", False)
            ),
            routing=RoutingResult(
                agent=routing_data.get("agent", "general_assistant"),
                action=routing_data.get("action", "respond"),
                urgency=routing_data.get("urgency", "routine")
            ),
            memory_updates=MemoryUpdate(
                new_facts=memory_data.get("new_facts", {}),
                system_action=memory_data.get("system_action", ""),
                awaiting=memory_data.get("awaiting", "")
            ),
            response_guidance=ResponseGuidance(
                tone=guidance_data.get("tone", "helpful"),
                task_context=task_context,
                minimal_context=minimal_context
            ),
            reasoning_chain=data.get("reasoning_chain", [])
        )
    
    except Exception as e:
        logger.error(f"Error parsing reasoning response: {e}")
        return self._fallback_reasoning(user_message, memory, {})


# =============================================================================
# HELPER: Extract User Selection
# =============================================================================

def extract_user_selection(
    user_message: str,
    options: List[Any],
    option_type: str = "time"
) -> Optional[Any]:
    """
    Extract user's selection from their message.
    
    Args:
        user_message: The user's message
        options: List of options that were presented
        option_type: Type of options ("time", "doctor", "date", etc.)
    
    Returns:
        The selected option, or None if not found
    """
    message_lower = user_message.lower().strip()
    
    # Check for ordinal references
    ordinals = {
        "first": 0, "1st": 0, "one": 0, "1": 0,
        "second": 1, "2nd": 1, "two": 1, "2": 1,
        "third": 2, "3rd": 2, "three": 2, "3": 2,
    }
    
    for ordinal, index in ordinals.items():
        if ordinal in message_lower:
            if index < len(options):
                return options[index]
    
    # Check for direct match
    for option in options:
        option_str = str(option).lower()
        if option_str in message_lower or message_lower in option_str:
            return option
    
    # Type-specific extraction
    if option_type == "time":
        # Extract time from message
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', message_lower, re.IGNORECASE)
        if time_match:
            hour = time_match.group(1)
            for option in options:
                if hour in str(option):
                    return option
    
    return None

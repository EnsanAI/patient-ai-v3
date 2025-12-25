"""
Unified Reasoning Engine

Performs chain-of-thought reasoning in a single LLM call for:
- Context understanding
- Intent detection
- Agent routing
- Response guidance
- Memory updates
"""

import json
import logging
import re
import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from patient_ai_service.core.llm import LLMClient, get_llm_client
from patient_ai_service.core.config import settings
from patient_ai_service.core.llm_config import get_llm_config_manager
from patient_ai_service.core.observability import get_observability_logger
from patient_ai_service.core.conversation_memory import (
    ConversationMemoryManager,
    get_conversation_memory_manager
)
from patient_ai_service.core import get_state_manager
from patient_ai_service.models.validation import (
    ValidationResult,
    ExecutionLog,
    ToolExecution
)
from patient_ai_service.models.observability import TokenUsage

logger = logging.getLogger(__name__)


class UnderstandingResult(BaseModel):
    """What the reasoning engine understood about the user's message."""
    what_user_means: str
    is_continuation: bool = False
    continuation_type: Optional[str] = None  # "selection", "confirmation", "rejection", "modification", "clarification"
    selected_option: Optional[Any] = None  # The option user selected
    sentiment: str = "neutral"  # "affirmative", "negative", "neutral", "unclear"
    is_conversation_restart: bool = False


class RoutingResult(BaseModel):
    """Where to route the message."""
    agent: str  # "registration", "appointment_manager", etc.
    action: str
    urgency: str = "routine"  # "routine", "urgent", "emergency"


class MemoryUpdate(BaseModel):
    """Updates to apply to conversation memory."""
    new_facts: Dict[str, Any] = Field(default_factory=dict)
    system_action: str = ""
    awaiting: str = ""


class TaskContext(BaseModel):
    """
    Structured context for agent execution.
    """
    user_intent: str = ""
    objective: str = ""  # NEW: High-level goal for agent
    entities: Dict[str, Any] = Field(default_factory=dict)
    llm_entities: Dict[str, Any] = Field(default_factory=dict)  # NEW: LLM-only entity updates (ISOLATED)
    success_criteria: List[str] = Field(default_factory=list)  # Keep but will be outcome-based
    constraints: List[str] = Field(default_factory=list)
    prior_context: Optional[str] = None
    
    # Continuation fields
    is_continuation: bool = False
    continuation_type: Optional[str] = None  # "selection", "confirmation", "clarification"
    selected_option: Optional[Any] = None  # The option user selected
    continuation_context: Optional[Dict[str, Any]] = Field(default=None)

class ResponseGuidance(BaseModel):
    """Guidance for the selected agent's response."""
    tone: str = "helpful"  # "helpful", "empathetic", "urgent", "professional"
    task_context: TaskContext = Field(default_factory=TaskContext)
    minimal_context: Dict[str, Any] = Field(default_factory=dict)
    # plan: str = ""  # REMOVED - agents generate their own task plans


class ReasoningOutput(BaseModel):
    """Complete output from unified reasoning engine."""
    understanding: UnderstandingResult
    routing: RoutingResult
    memory_updates: MemoryUpdate
    response_guidance: ResponseGuidance
    reasoning_chain: List[str] = Field(default_factory=list)


# =============================================================================
# CONTINUATION DETECTION
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


class ReasoningEngine:
    """
    Unified chain-of-thought reasoning engine.

    Performs comprehensive reasoning in a single LLM call:
    - Understands user message in context
    - Determines routing and urgency
    - Extracts new facts
    - Provides response guidance
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        memory_manager: Optional[ConversationMemoryManager] = None,
        test_mode: bool = False
    ):
        """
        Initialize reasoning engine.

        Args:
            llm_client: LLM client for reasoning (defaults to GPT-4o mini or Claude Haiku)
            memory_manager: Conversation memory manager
            test_mode: Enable deterministic test mode
        """
        self.llm_client = llm_client or get_llm_client()
        self.memory_manager = memory_manager or get_conversation_memory_manager()
        self.state_manager = get_state_manager()
        self.test_mode = test_mode
        self.llm_config_manager = get_llm_config_manager()

        # For deterministic testing
        self._test_responses: Dict[str, ReasoningOutput] = {}

        logger.info(f"Initialized ReasoningEngine (test_mode={test_mode})")

    def set_test_response(self, message: str, response: ReasoningOutput):
        """
        Set a deterministic response for testing.

        Args:
            message: User message to match
            response: Predefined reasoning output to return
        """
        self._test_responses[message] = response
        logger.debug(f"Set test response for message: {message}")

    async def reason(
        self,
        session_id: str,
        user_message: str,
        patient_info: Dict[str, Any] = None
    ) -> ReasoningOutput:
        """
        Perform unified reasoning about the user's message.

        Single LLM call that performs:
        - Context analysis
        - Intent detection
        - Agent routing
        - Memory updates
        - Response guidance

        Args:
            session_id: Session identifier
            user_message: User's message
            patient_info: Patient information dict

        Returns:
            Complete reasoning output
        """
        # TEST MODE: Return mocked response
        if self.test_mode and user_message in self._test_responses:
            logger.info(f"Returning test response for: {user_message}")
            return self._test_responses[user_message]

        # Get conversation memory
        memory = self.memory_manager.get_memory(session_id)

        # Get continuation context from state manager
        continuation_context = None
        try:
            continuation_context = self.state_manager.get_continuation_context(session_id)
        except Exception as e:
            logger.debug(f"Could not get continuation context: {e}")
            continuation_context = None

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
                    f"Pre-detected continuation: type={continuation_detection.get('continuation_type')}, "
                    f"selected={continuation_detection.get('selected_option')}, "
                    f"confidence={continuation_detection.get('confidence')}"
                )

        # Check for conversation restart
        if self.memory_manager.is_conversation_restart(session_id, user_message):
            logger.info(f"Conversation restart detected for session {session_id}")
            return ReasoningOutput(
                understanding=UnderstandingResult(
                    what_user_means="Starting new conversation",
                    is_conversation_restart=True,
                    sentiment="neutral"
                ),
                routing=RoutingResult(
                    agent="general_assistant",
                    action="greet_user",
                    urgency="routine"
                ),
                memory_updates=MemoryUpdate(),
                response_guidance=ResponseGuidance(
                    tone="helpful",
                    minimal_context={}
                ),
                reasoning_chain=["Conversation restart detected", "Route to general assistant"]
            )

        # Build unified reasoning prompt with continuation context
        prompt = self._build_reasoning_prompt(
            user_message, 
            memory, 
            patient_info or {},
            continuation_context=continuation_context
        )
        logger.info(">" * 80)
        logger.info(f"🧠 Reasoning Prompt: {prompt}")
        logger.info(">" * 80)

        
        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        reasoning_tracker = obs_logger.reasoning_tracker if obs_logger else None

        try:
            # Record reasoning step
            if reasoning_tracker:
                reasoning_tracker.record_step(1, "Building reasoning prompt", {
                    "user_message": user_message[:100],
                    "memory_summary": memory.summary[:200] if memory.summary else None
                })
                # Broadcast reasoning step (fire and forget)
                if obs_logger and obs_logger._broadcaster:
                    try:
                        import asyncio
                        step_data = {
                            "step_number": 1,
                            "description": "Building reasoning prompt",
                            "context": {
                                "user_message": user_message[:100],
                                "memory_summary": memory.summary[:200] if memory.summary else None
                            }
                        }
                        try:
                            loop = asyncio.get_running_loop()
                            # Schedule task without awaiting
                            loop.create_task(obs_logger._broadcaster.broadcast_reasoning_step(step_data))
                        except RuntimeError:
                            # No running loop, create new one
                            asyncio.run(obs_logger._broadcaster.broadcast_reasoning_step(step_data))
                    except Exception as e:
                        logger.debug(f"Error broadcasting reasoning step: {e}")
            
            # Single LLM call does everything
            llm_start_time = time.time()
            # Get hierarchical LLM config for reasoning agent
            llm_config = self.llm_config_manager.get_config(agent_name="reasoning")
            llm_client = self.llm_config_manager.get_client(agent_name="reasoning")
            reasoning_temp = llm_config.temperature
            
            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=reasoning_temp,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=reasoning_temp,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call
            if obs_logger:
                llm_call = obs_logger.record_llm_call(
                    component="reasoning",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(self._get_system_prompt()),
                    messages_count=1,
                    temperature=reasoning_temp,
                    max_tokens=llm_config.max_tokens,
                    function_name="reason"
                )
            
            # Record reasoning step
            if reasoning_tracker:
                reasoning_tracker.record_step(2, "LLM reasoning call completed", {
                    "response_length": len(response),
                    "tokens_used": tokens.total_tokens
                })

            # Parse and validate response with continuation awareness
            output = self._parse_reasoning_response(
                response, 
                user_message, 
                memory,
                continuation_context=continuation_context,
                continuation_detection=continuation_detection
            )

            # [CRITICAL] Inject language context into minimal_context
            # This ensures ALL agents receive language awareness
            global_state = self.state_manager.get_global_state(session_id)
            language_context = global_state.language_context

            # Add language context to minimal_context
            output.response_guidance.minimal_context["current_language"] = language_context.current_language
            output.response_guidance.minimal_context["current_dialect"] = language_context.current_dialect

            logger.info(
                f"Injected language context into minimal_context: "
                f"{language_context.get_full_language_code()}"
            )

            # Record reasoning details
            if reasoning_tracker:
                reasoning_tracker.set_understanding({
                    "what_user_means": output.understanding.what_user_means,
                    "is_continuation": output.understanding.is_continuation,
                    "continuation_type": output.understanding.continuation_type,
                    "selected_option": output.understanding.selected_option,
                    "sentiment": output.understanding.sentiment,
                    "is_conversation_restart": output.understanding.is_conversation_restart
                })
                reasoning_tracker.set_routing({
                    "agent": output.routing.agent,
                    "action": output.routing.action,
                    "urgency": output.routing.urgency
                })
                reasoning_tracker.set_memory_updates({
                    "new_facts": output.memory_updates.new_facts,
                    "system_action": output.memory_updates.system_action,
                    "awaiting": output.memory_updates.awaiting
                })
                reasoning_tracker.set_response_guidance({
                    "tone": output.response_guidance.tone,
                    "minimal_context": output.response_guidance.minimal_context
                })
                
                # Record reasoning chain steps
                for i, step_desc in enumerate(output.reasoning_chain, start=3):
                    reasoning_tracker.record_step(i, step_desc, {})
                
                # Set LLM call in reasoning details
                reasoning_details = reasoning_tracker.get_details()
                if obs_logger and llm_call:
                    reasoning_details.llm_call = llm_call
                    obs_logger.reasoning_tracker = reasoning_tracker

            # Update memory with extracted facts
            if output.memory_updates.new_facts:
                self.memory_manager.update_facts(session_id, output.memory_updates.new_facts)

            # Update system state
            if output.memory_updates.system_action or output.memory_updates.awaiting:
                self.memory_manager.update_system_state(
                    session_id,
                    last_action=output.memory_updates.system_action or None,
                    awaiting=output.memory_updates.awaiting or None
                )

            # Check if summarization needed
            if self.memory_manager.should_summarize(session_id):
                logger.info(f"Triggering summarization for session {session_id}")
                await self.memory_manager.summarize(session_id)

            logger.info(f"Reasoning complete for session {session_id}: "
                       f"agent={output.routing.agent}, "
                       f"urgency={output.routing.urgency}, "
                       f"sentiment={output.understanding.sentiment}")

            # Print full reasoning output to terminal logs
            logger.info("=" * 80)
            logger.info("REASONING OUTPUT (Full Structure):")
            logger.info("=" * 80)
            # Commented out individual field logging - using raw output instead
            # logger.info(f"UNDERSTANDING:")
            # logger.info(f"  - what_user_means: {output.understanding.what_user_means}")
            # logger.info(f"  - is_continuation: {output.understanding.is_continuation}")
            # logger.info(f"  - sentiment: {output.understanding.sentiment}")
            # logger.info(f"  - is_conversation_restart: {output.understanding.is_conversation_restart}")
            # logger.info(f"ROUTING:")
            # logger.info(f"  - agent: {output.routing.agent}")
            # logger.info(f"  - action: {output.routing.action}")
            # logger.info(f"  - urgency: {output.routing.urgency}")
            # logger.info(f"MEMORY_UPDATES:")
            # logger.info(f"  - new_facts: {json.dumps(output.memory_updates.new_facts, indent=4) if output.memory_updates.new_facts else '(empty)'}")
            # logger.info(f"  - system_action: '{output.memory_updates.system_action}' {'(EMPTY - should be filled!)' if not output.memory_updates.system_action else ''}")
            # logger.info(f"  - awaiting: '{output.memory_updates.awaiting}' {'(EMPTY - OK if nothing needed)' if not output.memory_updates.awaiting else ''}")
            # logger.info(f"RESPONSE_GUIDANCE:")
            # logger.info(f"  - tone: {output.response_guidance.tone}")
            # logger.info(f"  - minimal_context: {json.dumps(output.response_guidance.minimal_context, indent=4)}")
            # logger.info(f"  - task_context.user_intent: {output.response_guidance.task_context.user_intent}")
            # logger.info(f"  - task_context.objective: {output.response_guidance.task_context.objective}")
            # logger.info(f"  - task_context.entities: {json.dumps(output.response_guidance.task_context.entities, indent=4)}")
            # logger.info(f"  - task_context.success_criteria: {output.response_guidance.task_context.success_criteria}")
            # logger.info(f"  - task_context.is_continuation: {output.response_guidance.task_context.is_continuation}")
            # logger.info(f"REASONING_CHAIN:")
            # for i, step in enumerate(output.reasoning_chain, 1):
            #     logger.info(f"  {i}. {step}")
            logger.info("=" * 80)

            return output

        except Exception as e:
            logger.error(f"Error in reasoning engine: {e}", exc_info=True)
            # Use smart fallback with full context
            return self._smart_fallback_reasoning(
                user_message, 
                memory, 
                patient_info or {},
                continuation_context=continuation_context,
                continuation_detection=continuation_detection
            )

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON from LLM response using multiple strategies.
        
        Handles:
        - Raw JSON
        - JSON wrapped in markdown code blocks
        - JSON with extra text before/after
        - Multiple JSON objects (takes the first complete one)
        """
        if not response or not response.strip():
            logger.warning("Empty response received from LLM")
            return None
        
        # Strategy 1: Check if response is already clean JSON
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            return stripped
        
        # Strategy 2: Extract from markdown code blocks
        # Pattern: ```json ... ``` or ``` ... ```
        code_block_patterns = [
            r'```json\s*\n?(.*?)\n?```',  # ```json ... ```
            r'```\s*\n?(.*?)\n?```',       # ``` ... ```
        ]
        for pattern in code_block_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted.startswith('{'):
                    logger.debug("Extracted JSON from markdown code block")
                    return extracted
        
        # Strategy 3: Find JSON object boundaries
        # Look for first { and matching last }
        first_brace = response.find('{')
        if first_brace == -1:
            logger.warning("No opening brace found in response")
            return None
        
        # Find the matching closing brace by counting
        brace_count = 0
        last_brace = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(response[first_brace:], start=first_brace):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_brace = i
                    break
        
        if last_brace > first_brace:
            extracted = response[first_brace:last_brace + 1]
            logger.debug(f"Extracted JSON using brace matching (chars {first_brace}-{last_brace})")
            return extracted
        
        # Strategy 4: Fallback to simple regex
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            logger.debug("Extracted JSON using simple regex fallback")
            return json_match.group()
        
        logger.warning("Could not extract JSON from response")
        return None

    def _get_system_prompt(self) -> str:
        """Get system prompt for reasoning engine."""
        return """You are the reasoning engine for a dental clinic AI assistant.

Your job is to THINK through each user message and understand:
1. What the user is really saying (in context)
2. What they actually need
3. How best to help them

CRITICAL JSON OUTPUT RULES:
- You MUST output ONLY valid JSON - nothing else
- NO comments (no // or # inside JSON)
- NO trailing commas
- Use lowercase: true, false, null (NOT True, False, None)
- All strings must use double quotes "like this"
- Empty values: use "" for strings, {} for objects, [] for arrays, null for missing

IMPORTANT PRINCIPLES:
- Short responses like "yeah", "ok", "sure" are usually responses to what the system said
- If system proposed something and user agrees, honor that
- Don't force users to repeat themselves or use specific keywords
- Consider emotional state - frustrated users need empathy
- Emergency situations take priority (severe pain, bleeding, knocked out teeth)
- The goal is to HELP, not to categorize

VALID JSON EXAMPLE:
{"understanding":{"what_user_means":"User wants to cancel appointments","is_continuation":false,"continuation_type":null,"selected_option":null,"sentiment":"neutral","is_conversation_restart":false},"routing":{"agent":"appointment_manager","action":"cancel_appointments","urgency":"routine"},"memory_updates":{"new_facts":{},"system_action":"","awaiting":""},"response_guidance":{"tone":"helpful","task_context":{"user_intent":"cancel appointments","objective":"Cancel the user's appointments","entities":{},"success_criteria":[],"constraints":[],"prior_context":null,"is_continuation":false,"continuation_type":null,"selected_option":null,"continuation_context":{}},"minimal_context":{"what_user_means":"User wants to cancel appointments","action":"cancel"}},"reasoning_chain":["User wants to cancel","Route to appointment_manager"]}

Output ONLY the JSON object. No markdown, no explanation, just JSON."""

    def _build_reasoning_prompt(
        self,
        user_message: str,
        memory: Any,
        patient_info: Dict[str, Any],
        continuation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the unified reasoning prompt with continuation awareness."""

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
Options presented to user: {json.dumps(continuation_context.get('presented_options', []), indent=2)}
Original request: {continuation_context.get('original_request', 'Unknown')}
Resolved so far: {json.dumps(continuation_context.get('entities', {}), indent=2)}

IMPORTANT: Check if user's message is a response to the above!
"""

        # Registration status section - ONLY when patient is NOT registered
        registration_status_section = ""
        is_registered = bool(patient_info.get('patient_id'))
        if not is_registered:
            registration_status_section = """
═══════════════════════════════════════════════════════════════
REGISTRATION STATUS - IMPORTANT
═══════════════════════════════════════════════════════════════

⚠️ PATIENT IS NOT REGISTERED

The patient can ONLY inquire about the clinic through the general_assistant agent.

For any other actions (booking appointments, managing appointments, etc.), 
the patient MUST register first through the registration agent.

ROUTING RULES:
- General inquiries about the clinic → general_assistant (allowed)
- Appointment booking, management, or any patient-specific actions → registration (must register first)
- Medical inquiries → registration (must register first)
- Emergency situations → registration (must register first)

"""

        return f"""Analyze this conversation situation and respond with ONE complete JSON.

{registration_status_section}
═══════════════════════════════════════════════════════════════
CONVERSATION STATE
═══════════════════════════════════════════════════════════════

**Recent Messages (last {len(memory.recent_turns)} turns):**
{recent_turns_formatted if recent_turns_formatted else "No previous messages"}

**System State:**
- Last Action: {memory.last_action or "None"}
- Awaiting Response: {memory.awaiting or "Nothing specific"}
{continuation_section}
═══════════════════════════════════════════════════════════════
PATIENT INFORMATION
═══════════════════════════════════════════════════════════════

- Name: {patient_info.get('first_name', 'Unknown')}
- Registered: {'Yes' if patient_info.get('patient_id') else 'No - NOT registered yet'}
- Patient ID: {patient_info.get('patient_id', 'None')}

═══════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════

Analyze and respond with ONLY a valid JSON object.

CRITICAL: Do NOT generate step-by-step plans. The agent will decide HOW to execute.
You only determine WHAT the user wants and WHERE to route.

JSON STRUCTURE:

{{
    "understanding": {{
        "what_user_means": "Plain English explanation of what user actually wants",
        "selected_option": "latest value user selected or null",
        "sentiment": "user mood or feeling"
    }},
    "routing": {{
        "agent": "appointment_manager/registration/general_assistant/medical_inquiry/emergency_response",
        "action": "process_request/book_appointment/register_patient/answer_inquiry/handle_emergency",
        "urgency": "routine/urgent/emergency"
    }},
    "memory_updates": {{
        "new_facts": {{}},
        "awaiting": "What system is waiting for (e.g., 'time_selection', 'confirmation') or empty string"
    }},
    "response_guidance": {{
        "tone": "decide tone for response generation based on user tone",
        "task_context": {{
            "user_intent": "What user wants in plain English",
            "objective": "REQUIRED - ALWAYS FILL THIS: Specific goal for agent. Examples: 'Book teeth cleaning with Dr. Sarah on Dec 15 at 2pm', 'Cancel appointment #123', 'Register new patient for future appointments', 'List available appointment times for Dr. Ahmed tomorrow'",
            "entities": {{
                "doctor_preference": "extracted doctor name or null",
                "date_preference": "extracted date or null",
                "time_preference": "extracted time or null",
                "procedure_preference": "extracted procedure or null"
            }},
            "constraints": ["any constraints like 'must be afternoon'"],
            "prior_context": "relevant prior context"
        }},
        "minimal_context": {{
            "what_user_means": "Brief what user means (should match understanding.what_user_means)",
            "action": "Suggested action"
        }}
}}

RESPOND WITH VALID VALID JSON ONLY - NO OTHER TEXT."""

    def _parse_reasoning_response(
        self,
        response: str,
        user_message: str,
        memory: Any,
        continuation_context: Optional[Dict[str, Any]] = None,
        continuation_detection: Optional[Dict[str, Any]] = None
    ) -> ReasoningOutput:
        """Parse LLM response into structured reasoning output with continuation awareness."""

        try:
            # Extract JSON from response - handle multiple formats
            json_str = self._extract_json_from_response(response)
            if not json_str:
                raise ValueError("No JSON found in response")

            # Try direct parse first
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed: {e}")
                
                # LAYER 1: Comprehensive JSON Repair (multiple passes)
                data = self._robust_json_repair(json_str, user_message, memory, continuation_context)
                
                if data is None:
                    # LAYER 2: Smart keyword-based fallback
                    logger.warning(f"JSON repair exhausted, using smart fallback for: {user_message[:50]}")
                    return self._smart_fallback_reasoning(
                        user_message, memory, {}, 
                        continuation_context, continuation_detection
                    )

            # Log complete raw output from LLM
            logger.info("👷🏿‍♂️" * 80)
            logger.info("RAW LLM OUTPUT (Complete JSON):")
            logger.info(json.dumps(data, indent=2, default=str))
            logger.info("👷🏿‍♂️" * 80)

            # Parse nested structures
            understanding_data = data.get("understanding", {})
            routing_data = data.get("routing", {})
            memory_data = data.get("memory_updates", {})
            guidance_data = data.get("response_guidance", {}) or {}
            task_context_data = guidance_data.get("task_context", {}) or {}
            reasoning_chain = data.get("reasoning_chain", [])

            # Enhance with continuation detection if LLM missed it
            is_continuation = understanding_data.get("is_continuation", False)
            continuation_type = understanding_data.get("continuation_type")
            selected_option = understanding_data.get("selected_option")

            # Use pre-detection if LLM didn't detect continuation
            if continuation_detection and continuation_detection.get("is_continuation"):
                if not is_continuation or continuation_detection.get("confidence", 0) > 0.8:
                    is_continuation = True
                    if not continuation_type:
                        continuation_type = continuation_detection.get("continuation_type")
                    if not selected_option:
                        selected_option = continuation_detection.get("selected_option")
                    logger.info(f"Using pre-detected continuation: {continuation_type}")

            understanding_data["is_continuation"] = is_continuation
            understanding_data["continuation_type"] = continuation_type
            understanding_data["selected_option"] = selected_option

            understanding = UnderstandingResult(**understanding_data)
            
            # Ensure routing has required fields
            if not routing_data.get("action"):
                # Infer action from agent if missing
                agent = routing_data.get("agent", "general_assistant")
                if agent == "appointment_manager":
                    routing_data["action"] = "process_request"
                elif agent == "registration":
                    routing_data["action"] = "register_patient"
                elif agent == "medical_inquiry":
                    routing_data["action"] = "answer_inquiry"
                elif agent == "emergency_response":
                    routing_data["action"] = "handle_emergency"
                else:
                    routing_data["action"] = "process_message"
                logger.warning(f"Missing 'action' in routing, inferred: {routing_data['action']}")
            
            routing = RoutingResult(**routing_data)
            memory_updates = MemoryUpdate(**memory_data)

            # Handle task_context - merge resolved entities from continuation context
            if continuation_context and is_continuation:
                entities = continuation_context.get("entities", {})
                if entities:
                    # Merge resolved entities into task_context entities
                    if "entities" not in task_context_data:
                        task_context_data["entities"] = {}
                    task_context_data["entities"].update(entities)
                    logger.info(f"🔄 [ReasoningEngine] Merged {len(entities)} resolved entities from continuation: {json.dumps(entities, default=str)}")

                # Add selected option to entities if applicable
                if selected_option and continuation_type == "selection":
                    if "entities" not in task_context_data:
                        task_context_data["entities"] = {}
                    # Add selected option based on what was awaited
                    awaiting = continuation_context.get("awaiting", "")
                    if "time" in awaiting.lower():
                        task_context_data["entities"]["selected_time"] = selected_option
                    elif "doctor" in awaiting.lower():
                        task_context_data["entities"]["selected_doctor"] = selected_option
                    else:
                        task_context_data["entities"]["selected_option"] = selected_option

            # Ensure task_context is created
            if not task_context_data:
                task_context_data = {}

            # Clean None values that should be empty dicts/lists
            # (LLM may return null instead of omitting the field)
            if task_context_data.get("continuation_context") is None:
                task_context_data["continuation_context"] = {}
            if task_context_data.get("entities") is None:
                task_context_data["entities"] = {}
            if task_context_data.get("success_criteria") is None:
                task_context_data["success_criteria"] = []
            if task_context_data.get("constraints") is None:
                task_context_data["constraints"] = []

            # Ensure objective is filled - CRITICAL for agent execution
            if not task_context_data.get("objective"):
                # Generate objective from user_intent and action
                user_intent = task_context_data.get("user_intent", understanding.what_user_means)
                action = routing_data.get("action", "process_request")
                
                # Create a specific objective based on action
                if action == "book_appointment":
                    task_context_data["objective"] = f"Book an appointment for: {user_intent}"
                elif action == "cancel_appointment":
                    task_context_data["objective"] = f"Cancel appointment: {user_intent}"
                elif action == "reschedule_appointment":
                    task_context_data["objective"] = f"Reschedule appointment: {user_intent}"
                elif action == "register_patient":
                    task_context_data["objective"] = f"Register new patient: {user_intent}"
                elif action == "check_availability":
                    task_context_data["objective"] = f"Check availability for: {user_intent}"
                elif action == "list_appointments":
                    task_context_data["objective"] = f"List appointments: {user_intent}"
                else:
                    task_context_data["objective"] = user_intent
                
                logger.warning(f"⚠️ [ReasoningEngine] Objective was empty, generated: {task_context_data['objective']}")
            
            # Create TaskContext
            task_context = TaskContext(**task_context_data)

            # Log entities extracted by reasoning engine
            entities = task_context_data.get("entities", {})
            if entities:
                logger.info(f"📊 [ReasoningEngine] Extracted entities from user message: {json.dumps(entities, default=str)}")
            else:
                logger.info(f"📊 [ReasoningEngine] No entities extracted from user message")

            # Create ResponseGuidance from guidance_data
            # Extract fields from guidance_data with safe defaults
            tone = guidance_data.get("tone", "helpful")
            minimal_context = guidance_data.get("minimal_context", {})
            # plan removed - agents generate their own task plans
            
            # Create ResponseGuidance object
            response_guidance = ResponseGuidance(
                tone=tone,
                task_context=task_context,
                minimal_context=minimal_context
            )

            # Create output
            output = ReasoningOutput(
                understanding=understanding,
                routing=routing,
                memory_updates=memory_updates,
                response_guidance=response_guidance,
                reasoning_chain=reasoning_chain
            )

            # Special case: affirmative response to registration proposal
            if (understanding.is_continuation and
                understanding.sentiment == "affirmative" and
                "registration" in memory.last_action.lower()):
                output.routing.agent = "registration"
                logger.info("Detected affirmative response to registration proposal")

            return output

        except Exception as e:
            logger.error(f"Error parsing reasoning response: {e}")
            logger.debug(f"Response was: {response}")
            raise

    def _robust_json_repair(
        self,
        json_str: str,
        user_message: str,
        memory: Any,
        continuation_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        LAYER 1: Comprehensive JSON repair with multiple strategies.
        
        Attempts multiple repair passes before giving up.
        Returns parsed dict on success, None on failure.
        """
        repair_strategies = [
            ("basic_cleanup", self._repair_basic_cleanup),
            ("fix_structure", self._repair_fix_structure),
            ("aggressive_repair", self._repair_aggressive),
            ("extract_partial", self._repair_extract_partial),
        ]
        
        logger.info(f"🔧 Starting JSON repair for message: {user_message[:50]}...")
        logger.debug(f"🔧 Raw JSON string (first 500 chars): {json_str[:500]}")
        
        for strategy_name, repair_func in repair_strategies:
            try:
                logger.debug(f"🔧 Trying repair strategy: {strategy_name}")
                repaired = repair_func(json_str)
                if repaired:
                    data = json.loads(repaired)
                    logger.info(f"✅ JSON repair succeeded with strategy: {strategy_name}")
                    # Log what was recovered
                    routing = data.get("routing", {})
                    logger.info(f"✅ Recovered routing: agent={routing.get('agent')}, action={routing.get('action')}")
                    return data
            except json.JSONDecodeError as e:
                logger.debug(f"Repair strategy '{strategy_name}' failed: {e}")
                continue
            except Exception as e:
                logger.warning(f"Repair strategy '{strategy_name}' error: {e}")
                continue
        
        # Log the failed JSON for debugging
        logger.error(f"❌ All JSON repair strategies exhausted for message: {user_message[:50]}")
        logger.error(f"❌ Raw JSON that failed to parse (first 1000 chars):\n{json_str[:1000]}")
        
        # Try to identify the specific issue location
        try:
            json.loads(json_str)
        except json.JSONDecodeError as final_error:
            logger.error(f"❌ Final JSON error details: {final_error}")
            logger.error(f"❌ Error at line {final_error.lineno}, column {final_error.colno}")
            # Try to show context around the error
            lines = json_str.split('\n')
            if final_error.lineno <= len(lines):
                start_line = max(0, final_error.lineno - 3)
                end_line = min(len(lines), final_error.lineno + 2)
                context_lines = lines[start_line:end_line]
                logger.error(f"❌ Context around error (lines {start_line+1}-{end_line}):")
                for i, line in enumerate(context_lines, start=start_line+1):
                    marker = ">>>" if i == final_error.lineno else "   "
                    logger.error(f"   {marker} {i}: {line}")
        
        return None
    
    def _repair_basic_cleanup(self, json_str: str) -> str:
        """Basic cleanup: comments, Python literals, trailing commas."""
        repaired = json_str
        
        # Remove // comments (but not inside strings)
        repaired = re.sub(r'(?<!["\'])//[^\n]*', '', repaired)
        
        # Remove # comments
        repaired = re.sub(r'(?<!["\'])#[^\n]*', '', repaired)
        
        # Fix Python-style booleans/None (word boundaries)
        repaired = re.sub(r'\bTrue\b', 'true', repaired)
        repaired = re.sub(r'\bFalse\b', 'false', repaired)
        repaired = re.sub(r'\bNone\b', 'null', repaired)
        
        # Fix trailing commas before } or ]
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        
        return repaired
    
    def _repair_fix_structure(self, json_str: str) -> str:
        """Fix structural issues: missing commas, brackets."""
        repaired = self._repair_basic_cleanup(json_str)
        
        # Fix missing commas between key-value pairs
        # Pattern: "value"\n"key" or "value" "key"
        repaired = re.sub(r'(")\s*\n\s*(")', r'\1,\n\2', repaired)
        
        # Fix missing comma after } before "
        repaired = re.sub(r'(\})\s*\n?\s*(")', r'\1,\n\2', repaired)
        
        # Fix missing comma after ] before "
        repaired = re.sub(r'(\])\s*\n?\s*(")', r'\1,\n\2', repaired)
        
        # Fix missing comma after true/false/null before "
        repaired = re.sub(r'(true|false|null)\s*\n?\s*(")', r'\1,\n\2', repaired)
        
        # Fix missing comma after numbers before "
        repaired = re.sub(r'(\d)\s*\n?\s*(")', r'\1,\n\2', repaired)
        
        # Remove any multiple consecutive commas
        repaired = re.sub(r',\s*,+', ',', repaired)
        
        return repaired
    
    def _repair_aggressive(self, json_str: str) -> str:
        """Aggressive repair: handle edge cases and malformed structures."""
        repaired = self._repair_fix_structure(json_str)
        
        # Remove any text before the first {
        first_brace = repaired.find('{')
        if first_brace > 0:
            repaired = repaired[first_brace:]
        
        # Remove any text after the last }
        last_brace = repaired.rfind('}')
        if last_brace > 0 and last_brace < len(repaired) - 1:
            repaired = repaired[:last_brace + 1]
        
        # Fix single quotes to double quotes (for string values)
        # Only outside of already double-quoted strings
        repaired = re.sub(r"(?<![\"\\])'([^']*)'(?![\"\\])", r'"\1"', repaired)
        
        # Fix unquoted string values after colons
        # Pattern: : unquoted_value (not number, bool, null, object, array)
        repaired = re.sub(
            r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}\]])',
            lambda m: f': "{m.group(1)}"{m.group(2)}' if m.group(1) not in ['true', 'false', 'null'] else m.group(0),
            repaired
        )
        
        # Balance brackets - count and fix
        open_braces = repaired.count('{')
        close_braces = repaired.count('}')
        open_brackets = repaired.count('[')
        close_brackets = repaired.count(']')
        
        # Add missing closing braces/brackets
        if open_braces > close_braces:
            repaired += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            repaired += ']' * (open_brackets - close_brackets)
        
        return repaired
    
    def _repair_extract_partial(self, json_str: str) -> str:
        """Extract and repair partial JSON - find valid subsections."""
        # Try to find the main required sections
        sections = {
            "understanding": None,
            "routing": None,
            "memory_updates": None,
            "response_guidance": None,
            "reasoning_chain": None
        }
        
        # Extract each section using regex
        for section_name in sections:
            pattern = rf'"{section_name}"\s*:\s*(\{{[^}}]*\}}|\[[^\]]*\])'
            match = re.search(pattern, json_str, re.DOTALL)
            if match:
                try:
                    # Try to parse the section
                    section_json = match.group(1)
                    # Clean it up
                    section_json = self._repair_basic_cleanup(section_json)
                    json.loads(section_json)  # Validate
                    sections[section_name] = section_json
                except:
                    pass
        
        # Build a minimal valid JSON from extracted sections
        # At minimum we need understanding and routing
        if sections["understanding"] or sections["routing"]:
            built_json = "{"
            parts = []
            
            # Add understanding with defaults if missing
            if sections["understanding"]:
                parts.append(f'"understanding": {sections["understanding"]}')
            else:
                parts.append('"understanding": {"what_user_means": "Unable to parse", "is_continuation": false, "sentiment": "neutral"}')
            
            # Add routing with defaults if missing
            if sections["routing"]:
                parts.append(f'"routing": {sections["routing"]}')
            else:
                parts.append('"routing": {"agent": "general_assistant", "action": "understand_and_respond", "urgency": "routine"}')
            
            # Add optional sections
            if sections["memory_updates"]:
                parts.append(f'"memory_updates": {sections["memory_updates"]}')
            else:
                parts.append('"memory_updates": {"new_facts": {}, "system_action": "", "awaiting": ""}')
            
            if sections["response_guidance"]:
                parts.append(f'"response_guidance": {sections["response_guidance"]}')
            else:
                parts.append('"response_guidance": {"tone": "helpful", "minimal_context": {}}')
            
            if sections["reasoning_chain"]:
                parts.append(f'"reasoning_chain": {sections["reasoning_chain"]}')
            else:
                parts.append('"reasoning_chain": ["Partial JSON recovery"]')
            
            built_json += ", ".join(parts) + "}"
            return built_json
        
        # Can't extract anything useful
        return None
    
    def _smart_fallback_reasoning(
        self,
        user_message: str,
        memory: Any,
        patient_info: Dict[str, Any],
        continuation_context: Optional[Dict[str, Any]] = None,
        continuation_detection: Optional[Dict[str, Any]] = None
    ) -> ReasoningOutput:
        """
        LAYER 2: Smart keyword-based fallback with context awareness.
        
        Analyzes user message and context to determine appropriate routing
        when JSON parsing completely fails.
        """
        message_lower = user_message.lower().strip()
        
        logger.info(f"🔄 Smart fallback analyzing: '{user_message[:100]}'")
        
        # === EMERGENCY CHECK (highest priority) ===
        emergency_keywords = [
            "emergency", "urgent", "severe bleeding", "can't breathe",
            "knocked out", "broken jaw", "severe pain", "911", "ambulance",
            "heart attack", "dying", "help me"
        ]
        if any(kw in message_lower for kw in emergency_keywords):
            logger.info("🚨 Smart fallback: Emergency detected")
            return self._create_emergency_response(user_message)
        
        # === APPOINTMENT KEYWORDS (common case) ===
        appointment_keywords = [
            "appointment", "book", "schedule", "reschedule", "cancel",
            "my appointments", "upcoming", "when is my", "change my",
            "move my", "postpone", "available", "availability",
            "slot", "time slot", "doctor", "dentist", "visit"
        ]
        appointment_action_keywords = {
            "book": ["book", "schedule", "make", "create", "new appointment", "want to see"],
            "cancel": ["cancel", "remove", "delete", "stop", "don't want", "cancel all", "cancel them"],
            "reschedule": ["reschedule", "move", "change", "postpone", "different time", "another day"],
            "view": ["my appointments", "upcoming", "when is", "list", "show", "what appointments", "check my"]
        }
        
        is_appointment_related = any(kw in message_lower for kw in appointment_keywords)
        
        # Check continuation context for appointment flow
        if continuation_context:
            awaiting = continuation_context.get("awaiting", "").lower()
            if any(apt_term in awaiting for apt_term in ["appointment", "time", "doctor", "date", "slot", "cancel"]):
                is_appointment_related = True
                logger.info(f"🔄 Continuation context suggests appointment flow: awaiting={awaiting}")
        
        if is_appointment_related:
            # Determine specific action
            action = "handle_request"
            for act, keywords in appointment_action_keywords.items():
                if any(kw in message_lower for kw in keywords):
                    action = f"{act}_appointment"
                    break
            
            logger.info(f"📅 Smart fallback: Appointment request detected, action={action}")
            return ReasoningOutput(
                understanding=UnderstandingResult(
                    what_user_means=f"User wants to {action.replace('_', ' ')}",
                    is_continuation=bool(continuation_context),
                    continuation_type=continuation_detection.get("continuation_type") if continuation_detection else None,
                    selected_option=continuation_detection.get("selected_option") if continuation_detection else None,
                    sentiment="neutral",
                    is_conversation_restart=False
                ),
                routing=RoutingResult(
                    agent="appointment_manager",
                    action=action,
                    urgency="routine"
                ),
                memory_updates=MemoryUpdate(
                    system_action="routing_to_appointment_manager",
                    awaiting="action_completion"
                ),
                response_guidance=ResponseGuidance(
                    tone="helpful",
                    task_context=TaskContext(
                        user_intent=f"User wants to manage appointments: {user_message}",
                        is_continuation=bool(continuation_context),
                        continuation_context=continuation_context or {}
                    ),
                    minimal_context={
                        "what_user_means": f"User wants to {action.replace('_', ' ')}",
                        "action": action,
                        "original_message": user_message
                    }
                ),
                reasoning_chain=[
                    "JSON parsing failed, using smart fallback",
                    f"Appointment keywords detected: {action}",
                    "Routing to appointment_manager agent"
                ]
            )
        
        # === REGISTRATION KEYWORDS ===
        registration_keywords = [
            "register", "sign up", "new patient", "first time",
            "create account", "my name is", "i'm new"
        ]
        if any(kw in message_lower for kw in registration_keywords):
            logger.info("📝 Smart fallback: Registration request detected")
            return ReasoningOutput(
                understanding=UnderstandingResult(
                    what_user_means="User wants to register",
                    is_continuation=False,
                    sentiment="neutral",
                    is_conversation_restart=False
                ),
                routing=RoutingResult(
                    agent="registration",
                    action="start_registration",
                    urgency="routine"
                ),
                memory_updates=MemoryUpdate(
                    system_action="starting_registration",
                    awaiting="user_info"
                ),
                response_guidance=ResponseGuidance(
                    tone="helpful",
                    minimal_context={"what_user_means": "User wants to register", "action": "collect_info"}
                ),
                reasoning_chain=[
                    "JSON parsing failed, using smart fallback",
                    "Registration keywords detected",
                    "Routing to registration agent"
                ]
            )
        
        # === MEDICAL INQUIRY KEYWORDS ===
        medical_keywords = [
            "pain", "hurt", "ache", "tooth", "gum", "bleeding",
            "swollen", "sensitive", "cavity", "filling", "crown",
            "root canal", "extraction", "wisdom", "braces", "cleaning"
        ]
        if any(kw in message_lower for kw in medical_keywords) and not is_appointment_related:
            logger.info("🏥 Smart fallback: Medical inquiry detected")
            return ReasoningOutput(
                understanding=UnderstandingResult(
                    what_user_means="User has a dental/medical question",
                    is_continuation=False,
                    sentiment="neutral",
                    is_conversation_restart=False
                ),
                routing=RoutingResult(
                    agent="medical_inquiry",
                    action="answer_question",
                    urgency="routine"
                ),
                memory_updates=MemoryUpdate(
                    system_action="answering_medical_inquiry",
                    awaiting=""
                ),
                response_guidance=ResponseGuidance(
                    tone="empathetic",
                    minimal_context={"what_user_means": "User has a dental/medical question", "action": "provide_info"}
                ),
                reasoning_chain=[
                    "JSON parsing failed, using smart fallback",
                    "Medical keywords detected",
                    "Routing to medical_inquiry agent"
                ]
            )
        
        # === AFFIRMATIVE RESPONSE CHECK ===
        affirmatives = ["yes", "yeah", "yep", "sure", "okay", "ok", "alright", "please", "go ahead", "do it", "confirm"]
        is_affirmative = any(aff in message_lower.split() or message_lower == aff for aff in affirmatives)
        
        if is_affirmative and memory and memory.last_action:
            last_action = memory.last_action.lower()
            # Route based on what was last proposed
            if "registration" in last_action or "register" in last_action:
                logger.info("✅ Smart fallback: Affirmative to registration proposal")
                return ReasoningOutput(
                    understanding=UnderstandingResult(
                        what_user_means="User confirms registration",
                        is_continuation=True,
                        continuation_type="confirmation",
                        sentiment="affirmative",
                        is_conversation_restart=False
                    ),
                    routing=RoutingResult(
                        agent="registration",
                        action="start_registration",
                        urgency="routine"
                    ),
                    memory_updates=MemoryUpdate(
                        system_action="starting_registration",
                        awaiting="user_info"
                    ),
                    response_guidance=ResponseGuidance(
                        tone="helpful",
                        minimal_context={"what_user_means": "User wants to register", "action": "collect_info"}
                    ),
                    reasoning_chain=[
                        "JSON parsing failed, using smart fallback",
                        "Affirmative response to registration proposal",
                        "Routing to registration agent"
                    ]
                )
            elif "appointment" in last_action or "book" in last_action or "cancel" in last_action:
                logger.info("✅ Smart fallback: Affirmative to appointment action")
                return ReasoningOutput(
                    understanding=UnderstandingResult(
                        what_user_means="User confirms appointment action",
                        is_continuation=True,
                        continuation_type="confirmation",
                        sentiment="affirmative",
                        is_conversation_restart=False
                    ),
                    routing=RoutingResult(
                        agent="appointment_manager",
                        action="continue_appointment_flow",
                        urgency="routine"
                    ),
                    memory_updates=MemoryUpdate(
                        system_action="continuing_appointment_flow",
                        awaiting="action_completion"
                    ),
                    response_guidance=ResponseGuidance(
                        tone="helpful",
                        task_context=TaskContext(
                            user_intent="User confirms previous appointment action",
                            is_continuation=True,
                            continuation_type="confirmation",
                            continuation_context=continuation_context or {}
                        ),
                        minimal_context={
                            "what_user_means": "User confirms appointment action",
                            "action": "continue_flow",
                            "prior_context": last_action
                        }
                    ),
                    reasoning_chain=[
                        "JSON parsing failed, using smart fallback",
                        f"Affirmative response to: {last_action}",
                        "Routing to appointment_manager to continue"
                    ]
                )
        
        # === CONTINUATION CONTEXT ROUTING ===
        if continuation_context:
            awaiting = continuation_context.get("awaiting", "").lower()
            original_agent = continuation_context.get("agent", "general_assistant")
            
            logger.info(f"🔄 Smart fallback: Using continuation context, awaiting={awaiting}, agent={original_agent}")
            
            return ReasoningOutput(
                understanding=UnderstandingResult(
                    what_user_means=f"User responding to: {awaiting}",
                    is_continuation=True,
                    continuation_type=continuation_detection.get("continuation_type") if continuation_detection else "clarification",
                    selected_option=continuation_detection.get("selected_option") if continuation_detection else None,
                    sentiment="neutral",
                    is_conversation_restart=False
                ),
                routing=RoutingResult(
                    agent=original_agent if original_agent else "general_assistant",
                    action="handle_continuation",
                    urgency="routine"
                ),
                memory_updates=MemoryUpdate(
                    system_action="handling_continuation",
                    awaiting=""
                ),
                response_guidance=ResponseGuidance(
                    tone="helpful",
                    task_context=TaskContext(
                        user_intent=f"Continuation of previous flow: {user_message}",
                        is_continuation=True,
                        continuation_context=continuation_context
                    ),
                    minimal_context={
                        "what_user_means": "User responding to previous request",
                        "action": "handle_continuation",
                        "prior_context": awaiting
                    }
                ),
                reasoning_chain=[
                    "JSON parsing failed, using smart fallback",
                    f"Continuation context available: awaiting {awaiting}",
                    f"Routing to {original_agent} to handle continuation"
                ]
            )
        
        # === LAYER 3: EMERGENCY FINAL FALLBACK ===
        logger.warning(f"⚠️ Smart fallback exhausted, using emergency fallback for: {user_message[:50]}")
        return self._emergency_fallback_reasoning(user_message, memory)
    
    def _create_emergency_response(self, user_message: str) -> ReasoningOutput:
        """Create standardized emergency response."""
        return ReasoningOutput(
            understanding=UnderstandingResult(
                what_user_means="User has a dental emergency",
                is_continuation=False,
                sentiment="urgent",
                is_conversation_restart=False
            ),
            routing=RoutingResult(
                agent="emergency_response",
                action="assess_and_respond",
                urgency="emergency"
            ),
            memory_updates=MemoryUpdate(
                system_action="responding_to_emergency",
                awaiting=""
            ),
            response_guidance=ResponseGuidance(
                tone="urgent",
                minimal_context={"what_user_means": "User has a dental emergency", "action": "assess_emergency"}
            ),
            reasoning_chain=["Emergency keywords detected", "Route to emergency response"]
        )
    
    def _emergency_fallback_reasoning(
        self,
        user_message: str,
        memory: Any
    ) -> ReasoningOutput:
        """
        LAYER 3: Emergency final fallback - NEVER fails.
        
        This is the absolute last resort. Returns a safe, generic response
        that routes to general_assistant with full context for debugging.
        """
        logger.error(f"🆘 EMERGENCY FALLBACK ACTIVATED for message: {user_message}")
        
        # Log diagnostic information
        logger.error(f"  Memory state: last_action={getattr(memory, 'last_action', 'N/A')}, "
                    f"awaiting={getattr(memory, 'awaiting', 'N/A')}")
        
        return ReasoningOutput(
            understanding=UnderstandingResult(
                what_user_means=user_message,  # Pass through raw message
                is_continuation=False,
                sentiment="neutral",
                is_conversation_restart=False
            ),
            routing=RoutingResult(
                agent="general_assistant",
                action="understand_and_respond",
                urgency="routine"
            ),
            memory_updates=MemoryUpdate(
                system_action="emergency_fallback_used",
                awaiting=""
            ),
            response_guidance=ResponseGuidance(
                tone="helpful",
                task_context=TaskContext(
                    user_intent=user_message,
                    prior_context="Reasoning fallback was used - treat message with care"
                ),
                minimal_context={
                    "what_user_means": "User needs assistance",
                    "action": "help_user",
                    "fallback_reason": "reasoning_parse_failure",
                    "original_message": user_message
                }
            ),
            reasoning_chain=[
                "All parsing and smart fallback strategies exhausted",
                "Emergency fallback activated",
                "Routing to general_assistant for safe handling"
            ]
        )

    def _fallback_reasoning(
        self,
        user_message: str,
        memory: Any,
        patient_info: Dict[str, Any]
    ) -> ReasoningOutput:
        """Legacy fallback reasoning - now delegates to smart fallback."""
        # This method is kept for backward compatibility
        # It now delegates to the smarter fallback system
        return self._smart_fallback_reasoning(user_message, memory, patient_info, None, None)

    def _legacy_fallback_reasoning(
        self,
        user_message: str,
        memory: Any,
        patient_info: Dict[str, Any]
    ) -> ReasoningOutput:
        """Original fallback reasoning (preserved for reference)."""

        message_lower = user_message.lower().strip()

        # Check for emergency keywords
        emergency_keywords = [
            "emergency", "urgent", "severe bleeding", "can't breathe",
            "knocked out", "broken jaw", "severe pain", "911", "ambulance"
        ]
        is_emergency = any(kw in message_lower for kw in emergency_keywords)

        if is_emergency:
            return ReasoningOutput(
                understanding=UnderstandingResult(
                    what_user_means="User has a dental emergency",
                    is_continuation=False,
                    sentiment="urgent",
                    is_conversation_restart=False
                ),
                routing=RoutingResult(
                    agent="emergency_response",
                    action="assess_and_respond",
                    urgency="emergency"
                ),
                memory_updates=MemoryUpdate(
                    system_action="responding_to_emergency",
                    awaiting=""
                ),
                response_guidance=ResponseGuidance(
                    tone="urgent",
                    minimal_context={"what_user_means": "User has a dental emergency", "action": "assess_emergency"}
                ),
                reasoning_chain=["Emergency keywords detected", "Route to emergency response"]
            )

        # Check for affirmative responses to proposals
        affirmatives = ["yes", "yeah", "yep", "sure", "okay", "ok", "alright", "please", "go ahead"]
        is_affirmative = any(aff in message_lower for aff in affirmatives)

        if is_affirmative and "registration" in memory.last_action.lower():
            return ReasoningOutput(
                understanding=UnderstandingResult(
                    what_user_means="Agrees to register",
                    is_continuation=True,
                    sentiment="affirmative",
                    is_conversation_restart=False
                ),
                routing=RoutingResult(
                    agent="registration",
                    action="start_registration",
                    urgency="routine"
                ),
                memory_updates=MemoryUpdate(
                    system_action="starting_registration",
                    awaiting="user_info"
                ),
                response_guidance=ResponseGuidance(
                    tone="helpful",
                    minimal_context={"what_user_means": "User wants to register", "action": "collect_info"}
                ),
                reasoning_chain=["User affirmed", "Last action was registration proposal", "Start registration"]
            )

        # Default fallback
        logger.warning(f"Using default fallback reasoning for: {user_message}")
        return ReasoningOutput(
            understanding=UnderstandingResult(
                what_user_means=user_message,
                is_continuation=False,
                sentiment="neutral",
                is_conversation_restart=False
            ),
            routing=RoutingResult(
                agent="general_assistant",
                action="understand_and_respond",
                urgency="routine"
            ),
            memory_updates=MemoryUpdate(),
            response_guidance=ResponseGuidance(
                tone="helpful",
                minimal_context={}
            ),
            reasoning_chain=["Fallback reasoning used"]
        )

    async def validate_response(
        self,
        session_id: str,
        original_reasoning: ReasoningOutput,
        agent_response: str,
        execution_log: ExecutionLog
    ) -> ValidationResult:
        """
        Validate agent response against original intent and execution reality.

        This closes the loop by checking:
        1. Did agent complete the expected task?
        2. Does response match tool results?
        3. Is agent being honest about what happened?
        4. Any safety/policy violations?

        Args:
            session_id: Session identifier
            original_reasoning: The original reasoning output that routed to this agent
            agent_response: The response the agent generated
            execution_log: Log of tools executed and their results

        Returns:
            ValidationResult with decision on whether to send, retry, or fallback
        """
        try:
            # Build validation prompt
            prompt = self._build_validation_prompt(
                original_reasoning,
                agent_response,
                execution_log
            )

            # Call LLM for validation (low temperature for consistency)
            # Get hierarchical LLM config for validation
            llm_config = self.llm_config_manager.get_config(
                agent_name="reasoning",
                function_name="validate_response"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="reasoning",
                function_name="validate_response"
            )

            obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
            llm_start_time = time.time()

            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=self._get_validation_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=self._get_validation_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call with hierarchical config
            llm_call = None
            if obs_logger:
                llm_call = obs_logger.record_llm_call(
                    component="validation",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(self._get_validation_system_prompt()),
                    messages_count=1,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="validate_response"
                )

            # Parse validation response
            validation = self._parse_validation_response(response)
            
            # Add LLM call to validation details
            if obs_logger:
                from patient_ai_service.models.observability import ValidationDetails
                validation_details = ValidationDetails(
                    is_valid=validation.is_valid,
                    confidence=validation.confidence,
                    decision=validation.decision,
                    issues=validation.issues,
                    reasoning=validation.reasoning,
                    feedback_to_agent=validation.feedback_to_agent,
                    llm_call=llm_call,
                    retry_count=0
                )
                obs_logger.set_validation_details(validation_details)

            logger.info(f"Validation complete for session {session_id}: "
                       f"valid={validation.is_valid}, "
                       f"decision={validation.decision}, "
                       f"confidence={validation.confidence}")

            return validation

        except Exception as e:
            logger.error(f"Error in validation: {e}", exc_info=True)
            # On validation error, assume response is valid (fail open)
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                decision="send",
                issues=[f"Validation error: {str(e)}"],
                reasoning=["Validation failed, defaulting to send"]
            )

    async def finalize_response(
        self,
        session_id: str,
        original_reasoning: ReasoningOutput,
        agent_response: str,
        execution_log: ExecutionLog,
        validation_result: ValidationResult
    ) -> ValidationResult:
        """
        Finalize agent response by approving or editing before sending to user.

        This is the SECOND layer of quality control that runs AFTER validation (and retry if needed).
        Even if validation passed, finalization ensures response is grounded in tool results.

        The reasoning engine examines:
        1. Original user intent (from reasoning output)
        2. Actual tool execution results (from execution log)
        3. Agent's response accuracy
        4. Previous validation feedback (if retry occurred)

        Three outcomes:
        - APPROVE: Response is accurate and complete, send as-is
        - EDIT: Minor issues or improvements needed, provide edited version
        - FALLBACK: Cannot confidently approve, escalate to human

        Args:
            session_id: Session identifier
            original_reasoning: Initial reasoning before agent execution
            agent_response: The response from agent (possibly after retry)
            execution_log: Record of all tool calls and results
            validation_result: Result from validation layer (for context)

        Returns:
            ValidationResult with optional rewritten_response
        """
        try:
            # Build finalization prompt
            prompt = self._build_finalization_prompt(
                original_reasoning,
                agent_response,
                execution_log,
                validation_result
            )

            # Call LLM for finalization (slightly higher temp for natural edits)
            # Get hierarchical LLM config for finalization
            llm_config = self.llm_config_manager.get_config(
                agent_name="reasoning",
                function_name="finalize_response"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="reasoning",
                function_name="finalize_response"
            )

            obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
            llm_start_time = time.time()

            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=self._get_finalization_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=self._get_finalization_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call with hierarchical config
            llm_call = None
            if obs_logger:
                llm_call = obs_logger.record_llm_call(
                    component="finalization",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(self._get_finalization_system_prompt()),
                    messages_count=1,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="finalize_response"
                )

            # Parse finalization response
            finalization = self._parse_finalization_response(response)
            
            # Add LLM call to finalization details
            if obs_logger:
                from patient_ai_service.models.observability import FinalizationDetails
                finalization_details = FinalizationDetails(
                    decision=finalization.decision,
                    confidence=finalization.confidence,
                    was_rewritten=finalization.was_rewritten,
                    rewritten_response_preview=finalization.rewritten_response[:200] if finalization.rewritten_response else "",
                    issues=finalization.issues,
                    reasoning=finalization.reasoning,
                    llm_call=llm_call
                )
                obs_logger.set_finalization_details(finalization_details)

            logger.info(f"Finalization complete for session {session_id}: "
                       f"decision={finalization.decision}, "
                       f"edited={finalization.was_rewritten}, "
                       f"confidence={finalization.confidence}")

            return finalization

        except Exception as e:
            logger.error(f"Error in finalization: {e}", exc_info=True)
            # On finalization error, approve agent response (fail open)
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                decision="send",
                issues=[f"Finalization error: {str(e)}"],
                reasoning=["Finalization failed, defaulting to agent response"]
            )

    def _get_finalization_system_prompt(self) -> str:
        """System prompt for response finalization (second-layer quality check)."""
        return """You are a response finalizer for a dental clinic AI system.

CONTEXT:
This is the FINAL quality check before sending response to user. The response has already been:
1. Validated for major issues
2. Potentially retried if validation failed
Now you perform a final check to approve or make minor edits.

YOUR TASK:
Review the agent's response and ensure it:
1. Accurately reflects tool execution results
2. Answers the user's actual request
3. Is grounded in facts (no hallucinations)
4. Is complete and helpful

CRITICAL CHECKS:
- If agent says "appointment confirmed" → verify book_appointment was called with success=true
- If agent provides data (dates, times, doctor names, confirmation numbers) → verify exact match with tool outputs
- If agent says "no availability" → verify check_availability returned available=false
- Agent should NEVER invent information not present in tool results

DECISION OUTCOMES:
1. "send" - Response is accurate and complete, send as-is (rewritten_response = null, is_valid = true)
2. "edit" - Minor issues detected, provide edited version (rewritten_response = edited text, is_valid = false, decision = "edit")
3. "fallback" - Cannot confidently approve, escalate to human (decision = "fallback")

RESPONSE FORMAT (JSON):
{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "decision": "send|edit|fallback",
    "issues": ["issue 1", "issue 2"],
    "reasoning": ["reasoning step 1", "step 2"],
    "rewritten_response": "edited response text" or null,
    "was_rewritten": true/false
}

GROUNDING PRINCIPLE:
All responses must be grounded in actual tool outputs. If tools say available=true,
response cannot say "no availability". If book_appointment returns "APT-123",
response MUST include "APT-123".

WHEN EDITING:
- Maintain the agent's conversational tone
- Fix only inaccuracies or missing information
- Include all relevant information from tool outputs
- Keep it natural and helpful
- Make minimal changes (don't rewrite unnecessarily)

FAIL OPEN:
If you're uncertain or detect a complex issue, use decision="fallback" to escalate to human."""

    def _build_finalization_prompt(
        self,
        original_reasoning: ReasoningOutput,
        agent_response: str,
        execution_log: ExecutionLog,
        validation_result: ValidationResult
    ) -> str:
        """Build prompt for response finalization (second-layer quality check)."""
        import json

        # Format tool executions for context
        tools_section = ""
        if execution_log.tools_used:
            tools_section = "\n\nTOOL EXECUTIONS:\n"
            for i, tool in enumerate(execution_log.tools_used, 1):
                tools_section += f"\n{i}. {tool.tool_name}"
                tools_section += f"\n   Inputs: {json.dumps(tool.inputs, indent=2)}"
                tools_section += f"\n   Outputs: {json.dumps(tool.outputs, indent=2)}"
        else:
            tools_section = "\n\nTOOL EXECUTIONS: None (no tools were called)"

        # Include validation context if retry occurred
        validation_context = ""
        if not validation_result.is_valid:
            validation_context = f"""

VALIDATION HISTORY:
The response was validated and retried. Previous validation found these issues:
{chr(10).join(f"- {issue}" for issue in validation_result.issues)}

The agent was given this feedback and retried:
{validation_result.feedback_to_agent}

The current response is the result AFTER retry."""

        prompt = f"""ORIGINAL USER INTENT:
{original_reasoning.understanding.what_user_means}

ROUTING DECISION:
Agent: {original_reasoning.routing.agent}
Action: {original_reasoning.routing.action}
{tools_section}
{validation_context}

AGENT'S RESPONSE (final version after validation/retry):
{agent_response}

TASK:
Perform final quality check on the agent's response.
- If response is accurate and complete → approve (decision: "send")
- If minor issues detected → provide edited version (decision: "edit")
- If you cannot confidently approve → escalate (decision: "fallback")

Respond in JSON format as specified in the system prompt."""

        return prompt

    def _parse_finalization_response(self, response: str) -> ValidationResult:
        """Parse LLM finalization response into ValidationResult."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in finalization response")

            import json
            data = json.loads(json_match.group())

            # Create ValidationResult with finalization fields
            return ValidationResult(
                is_valid=data.get("is_valid", True),
                confidence=data.get("confidence", 1.0),
                decision=data.get("decision", "send"),
                issues=data.get("issues", []),
                reasoning=data.get("reasoning", []),
                feedback_to_agent=data.get("feedback_to_agent", ""),
                rewritten_response=data.get("rewritten_response"),
                was_rewritten=data.get("was_rewritten", False)
            )

        except Exception as e:
            logger.error(f"Error parsing finalization response: {e}")
            logger.debug(f"Response was: {response}")
            # On parse error, approve agent response (fail open)
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                decision="send",
                issues=[f"Parse error: {str(e)}"],
                reasoning=["Failed to parse finalization, approving agent response"]
            )

    def _get_validation_system_prompt(self) -> str:
        """Get system prompt for validation LLM call."""
        return """You are a validation system for a dental clinic AI assistant.

Your job is to verify that agent responses are:
1. COMPLETE - Did the agent finish the task?
2. ACCURATE - Does the response match what actually happened?
3. HONEST - Is the agent claiming something that didn't occur?
4. SAFE - No policy violations or dangerous advice?

CRITICAL VALIDATION CHECKS:

Tool Usage:
- If task requires booking: Did agent call book_appointment tool?
- If agent says "booked": Is there a book_appointment result?
- If tool returned data: Did agent use it correctly?

Completeness:
- If user asked for confirmation number: Is it in the response?
- If user requested action: Was action completed?

Accuracy:
- Does response match tool results?
- Are dates/times valid?
- Do entity references exist?

Safety:
- No medical diagnosis
- No unauthorized actions
- No false confirmations

Always respond with structured JSON for automated processing."""

    def _build_validation_prompt(
        self,
        original_reasoning: ReasoningOutput,
        agent_response: str,
        execution_log: ExecutionLog
    ) -> str:
        """Build validation prompt with context."""
        tools_summary = "\n".join([
            f"- {exec.tool_name}: inputs={exec.inputs}, outputs={exec.outputs}"
            for exec in execution_log.tools_used
        ])

        return f"""VALIDATION REQUEST

═══ ORIGINAL ANALYSIS ═══
User wanted: {original_reasoning.understanding.what_user_means}
Expected action: {original_reasoning.routing.action}
Expected tools: (inferred from action type)

═══ WHAT ACTUALLY HAPPENED ═══
Agent response: "{agent_response}"

Tools executed:
{tools_summary if tools_summary else "No tools used"}

═══ VALIDATION TASK ═══
Check if response is valid and safe to send to user.

Respond with JSON:
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "issues": ["list of specific problems found"],
  "decision": "send" | "retry" | "redirect" | "fallback",
  "feedback_to_agent": "specific guidance if retry needed",
  "reasoning": ["step 1: ...", "step 2: ..."]
}}

DECISION GUIDE:
- "send": Response is valid, send to user
- "retry": Fixable issue, give agent specific feedback
- "redirect": Wrong agent, need different approach
- "fallback": Unfixable, use safe fallback response

IMPORTANT: Only mark is_valid=true if you're confident response is complete, accurate, and safe."""

    def _parse_validation_response(self, response: str) -> ValidationResult:
        """Parse LLM validation response into ValidationResult."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in validation response")

            data = json.loads(json_match.group())

            # Create ValidationResult
            return ValidationResult(
                is_valid=data.get("is_valid", False),
                confidence=data.get("confidence", 1.0),
                issues=data.get("issues", []),
                decision=data.get("decision", "send"),
                feedback_to_agent=data.get("feedback_to_agent", ""),
                reasoning=data.get("reasoning", [])
            )

        except Exception as e:
            logger.error(f"Error parsing validation response: {e}")
            logger.debug(f"Response was: {response}")
            # On parse error, assume valid (fail open)
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                decision="send",
                issues=[f"Parse error: {str(e)}"],
                reasoning=["Failed to parse validation, defaulting to send"]
            )


# Global instance
_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine() -> ReasoningEngine:
    """Get or create the global reasoning engine instance."""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine()
    return _reasoning_engine


def reset_reasoning_engine():
    """Reset the global reasoning engine (for testing)."""
    global _reasoning_engine
    _reasoning_engine = None
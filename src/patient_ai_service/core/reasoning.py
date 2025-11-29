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
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from patient_ai_service.core.llm import LLMClient, get_llm_client
from patient_ai_service.core.conversation_memory import (
    ConversationMemoryManager,
    get_conversation_memory_manager
)
from patient_ai_service.models.validation import (
    ValidationResult,
    ExecutionLog,
    ToolExecution
)

logger = logging.getLogger(__name__)


class UnderstandingResult(BaseModel):
    """What the reasoning engine understood about the user's message."""
    what_user_means: str
    is_continuation: bool = False
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


class ResponseGuidance(BaseModel):
    """Guidance for the selected agent's response."""
    tone: str = "helpful"  # "helpful", "empathetic", "urgent", "professional"
    minimal_context: Dict[str, Any] = Field(default_factory=dict)


class ReasoningOutput(BaseModel):
    """Complete output from unified reasoning engine."""
    understanding: UnderstandingResult
    routing: RoutingResult
    memory_updates: MemoryUpdate
    response_guidance: ResponseGuidance
    reasoning_chain: List[str] = Field(default_factory=list)


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
        self.test_mode = test_mode

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

        # Build unified reasoning prompt
        prompt = self._build_reasoning_prompt(user_message, memory, patient_info or {})

        try:
            # Single LLM call does everything
            response = self.llm_client.create_message(
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Parse and validate response
            output = self._parse_reasoning_response(response, user_message, memory)

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

            return output

        except Exception as e:
            logger.error(f"Error in reasoning engine: {e}", exc_info=True)
            return self._fallback_reasoning(user_message, memory, patient_info or {})

    def _get_system_prompt(self) -> str:
        """Get system prompt for reasoning engine."""
        return """You are the reasoning engine for a dental clinic AI assistant.

Your job is to THINK through each user message and understand:
1. What the user is really saying (in context)
2. What they actually need
3. How best to help them

You analyze conversations holistically, reading between the lines to provide
natural, context-aware assistance.

IMPORTANT PRINCIPLES:
- Short responses like "yeah", "ok", "sure" are usually responses to what the system said
- If system proposed something and user agrees, honor that
- Don't force users to repeat themselves or use specific keywords
- Consider emotional state - frustrated users need empathy
- Emergency situations take priority (severe pain, bleeding, knocked out teeth)
- The goal is to HELP, not to categorize

Always respond with your reasoning in the specified JSON format."""

    def _build_reasoning_prompt(
        self,
        user_message: str,
        memory: Any,
        patient_info: Dict[str, Any]
    ) -> str:
        """Build the unified reasoning prompt."""

        # Format recent turns
        recent_turns_formatted = "\n".join([
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
            for turn in memory.recent_turns
        ])

        return f"""Analyze this conversation situation and respond with ONE complete JSON.

═══════════════════════════════════════════════════════════════
CONVERSATION STATE
═══════════════════════════════════════════════════════════════

**User Facts (persistent, never lost):**
{json.dumps(memory.user_facts, indent=2) if memory.user_facts else "None collected yet"}

**Conversation Summary:**
{memory.summary if memory.summary else "No previous conversation"}

**Recent Messages (last {len(memory.recent_turns)} turns):**
{recent_turns_formatted if recent_turns_formatted else "No previous messages"}

**System State:**
- Last Action: {memory.last_action or "None"}
- Awaiting Response: {memory.awaiting or "Nothing specific"}

═══════════════════════════════════════════════════════════════
PATIENT INFORMATION
═══════════════════════════════════════════════════════════════

- Name: {patient_info.get('first_name', 'Unknown')}
- Registered: {'Yes' if patient_info.get('patient_id') else 'No - NOT registered yet'}
- Patient ID: {patient_info.get('patient_id', 'None')}

═══════════════════════════════════════════════════════════════
NEW USER MESSAGE
═══════════════════════════════════════════════════════════════

"{user_message}"

═══════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════

Analyze and respond with ONE JSON object:

{{
    "understanding": {{
        "what_user_means": "Plain English explanation of what user actually wants",
        "is_continuation": true/false,  # Is this continuing previous topic?
        "sentiment": "affirmative/negative/neutral/unclear",
        "is_conversation_restart": false
    }},
    "routing": {{
        "agent": "appointment_manager/registration/general_assistant/medical_inquiry/emergency_response",
        "action": "Specific action for agent to take",
        "urgency": "routine/urgent/emergency"
    }},
    "memory_updates": {{
        "new_facts": {{}},  # NEW facts only (don't repeat existing user_facts)
        "system_action": "What system is about to do (proposed_registration, asked_question, provided_info, etc.)",
        "awaiting": "What system is waiting for (confirmation, date_selection, etc.)"
    }},
    "response_guidance": {{
        "tone": "helpful/empathetic/urgent/professional",
        "minimal_context": {{
            "user_wants": "Brief what user wants",
            "action": "Suggested action",
            "prior_context": "Any relevant prior context"
        }}
    }},
    "reasoning_chain": [
        "Step 1: What I observe...",
        "Step 2: What I conclude...",
        "Step 3: Therefore..."
    ]
}}

KEY RULES:
1. If system proposed action (check "Last Action") and user said "yeah/ok/sure/yes" → is_continuation=true, sentiment=affirmative
2. Emergency keywords (severe bleeding, can't breathe, knocked out, severe pain) → urgency=emergency
3. Extract ONLY NEW facts to new_facts (don't repeat what's in user_facts already)
4. For minimal_context, keep it BRIEF - just essential info
5. If user not registered and wants to book appointment → recommend registration first
6. Short ambiguous responses like "tomorrow" → likely continuation of previous topic

RESPOND WITH VALID JSON ONLY - NO OTHER TEXT."""

    def _parse_reasoning_response(
        self,
        response: str,
        user_message: str,
        memory: Any
    ) -> ReasoningOutput:
        """Parse LLM response into structured reasoning output."""

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            # Parse nested structures
            understanding = UnderstandingResult(**data.get("understanding", {}))
            routing = RoutingResult(**data.get("routing", {}))
            memory_updates = MemoryUpdate(**data.get("memory_updates", {}))
            response_guidance = ResponseGuidance(**data.get("response_guidance", {}))
            reasoning_chain = data.get("reasoning_chain", [])

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

    def _fallback_reasoning(
        self,
        user_message: str,
        memory: Any,
        patient_info: Dict[str, Any]
    ) -> ReasoningOutput:
        """Fallback reasoning when LLM fails."""

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
                    minimal_context={"user_wants": "emergency help", "action": "assess_emergency"}
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
                    minimal_context={"user_wants": "register", "action": "collect_info"}
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
            response = self.llm_client.create_message(
                system=self._get_validation_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2  # Low temp for deterministic validation
            )

            # Parse validation response
            validation = self._parse_validation_response(response)

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

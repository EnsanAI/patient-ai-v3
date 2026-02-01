"""
Orchestrator - Main coordinator for the multi-agent system.

Coordinates:
- Message routing via unified reasoning
- Agent execution
- State management
- Translation
- Pub/sub messaging
"""

import json
import logging
import re
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from patient_ai_service.core import (
    get_llm_client,
    get_state_manager,
    get_message_broker,
)
from patient_ai_service.core.config import settings
from patient_ai_service.core.feature_flags import FeatureFlags
from patient_ai_service.core.llm_config import get_llm_config_manager, LLMConfig
from patient_ai_service.core.reasoning import get_reasoning_engine, ReasoningEngine, ReasoningOutput
from patient_ai_service.core.conversation_memory import get_conversation_memory_manager
from patient_ai_service.core.native_language_memory import get_native_language_memory_manager
from patient_ai_service.core.observability import get_observability_logger, clear_observability_logger
from patient_ai_service.core.unified_reasoning import (
    UnifiedReasoning,
    get_unified_reasoning,
)
from patient_ai_service.models.unified_reasoning import (
    UnifiedReasoningOutput,
    RouteType,
    PlanDecision,
)
from patient_ai_service.models.situation_assessment import (
    SituationType,
)
from patient_ai_service.models.observability import TokenUsage

# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL FAST PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# These situation types bypass the full agent reasoning loop and go directly
# to a single LLM call for response generation.
#
# IMPORTANT: These are ONLY for messages that require NO information retrieval:
# - GREETING: "hi", "hello", "مرحبا"
# - FAREWELL: "bye", "goodbye", "مع السلامة"
# - THANKS: "thank you", "شكراً"
# - PLEASANTRY: "how are you?", "I'm fine" (NO info request)
#
# NOT for: "what are your hours?", "where are you?", "how do I book?"
# Those are NEW_INTENT and need the agent.

CONVERSATIONAL_FAST_PATH = frozenset({
    SituationType.GREETING,
    SituationType.FAREWELL,
    SituationType.THANKS,
    SituationType.PLEASANTRY,
})

# NEW: Phase 2 imports
from patient_ai_service.models.patient_entities import EntitySource
from patient_ai_service.models.entity_state import EntityState
from patient_ai_service.models.messages import Topics, ChatResponse
from patient_ai_service.models.enums import UrgencyLevel
from patient_ai_service.models.validation import ExecutionLog, ValidationResult
# NEW: Phase 3 imports - Agent Plan Management
from patient_ai_service.models.agent_plan import AgentPlan, PlanAction
from patient_ai_service.core.state_manager import ContinuationContext
from patient_ai_service.agents import (
    AppointmentManagerAgent,
    MedicalInquiryAgent,
    EmergencyResponseAgent,
    RegistrationAgent,
    TranslationAgent,
    GeneralAssistantAgent,
)
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator for the dental clinic AI system.

    Responsibilities:
    - Route messages to appropriate agents
    - Manage agent lifecycle
    - Coordinate translation
    - Handle state transitions
    - Publish/subscribe to message broker
    """

    def __init__(self, db_client: Optional[DbOpsClient] = None):
        self.llm_client = get_llm_client()
        self.state_manager = get_state_manager()
        self.message_broker = get_message_broker()
        self.memory_manager = get_conversation_memory_manager()
        self.native_memory_manager = get_native_language_memory_manager()
        self.db_client = db_client or DbOpsClient()
        self.llm_config_manager = get_llm_config_manager()
        
        # Unified Reasoning (replaces situation_assessor + reasoning_engine for routing)
        self.unified_reasoning = get_unified_reasoning()
        
        # Keep reasoning_engine only for validation (if enabled)
        self.reasoning_engine = get_reasoning_engine()

        # Initialize agents
        self._init_agents()
        
        # Metrics
        self._request_count = 0
        self._focused_count = 0
        self._comprehensive_count = 0

        logger.info("Orchestrator initialized with integrated architecture")

    def _init_agents(self):
        """Initialize all specialized agents."""
        # Create appointment manager first so we can pass it to general assistant
        appointment_manager = AppointmentManagerAgent(
            db_client=self.db_client
        )

        # Import language selection agent
        from patient_ai_service.agents.language_selection import LanguageSelectionAgent

        self.agents: Dict[str, Any] = {
            "language_selection": LanguageSelectionAgent(
                llm_config_manager=self.llm_config_manager
            ),
            "appointment_manager": appointment_manager,
            "medical_inquiry": MedicalInquiryAgent(
                db_client=self.db_client
            ),
            "emergency_response": EmergencyResponseAgent(
                db_client=self.db_client
            ),
            "registration": RegistrationAgent(
                db_client=self.db_client
            ),
            "translation": TranslationAgent(),
            "general_assistant": GeneralAssistantAgent(
                db_client=self.db_client,
                appointment_manager=appointment_manager  # Pass appointment manager for doctor search and availability
            ),
        }

        logger.info(f"Initialized {len(self.agents)} agents")

    async def start(self):
        """Start the orchestrator and message broker."""
        await self.message_broker.start()
        logger.info("Orchestrator started")

    async def stop(self):
        """Stop the orchestrator and message broker."""
        await self.message_broker.stop()
        logger.info("Orchestrator stopped")

    async def _conversational_response(
        self,
        message: str,
        situation: SituationType,
        patient_name: Optional[str],
        is_registered: bool,
        language: str,
        dialect: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        why_unclear: Optional[str] = None
    ) -> Tuple[str, TokenUsage]:
        """
        Handle all conversational fast-path responses in a single LLM call.
        
        This handles:
        - GREETING: "hi", "hello", "good morning", "مرحبا"
        - FAREWELL: "bye", "goodbye", "see you", "مع السلامة"
        - THANKS: "thank you", "thanks", "شكراً"
        - PLEASANTRY: "how are you?", "I'm fine" (pure social, NO info request)
        
        NOT for information requests like "what are your hours?" - those go to agent.
        
        Args:
            message: User's original message
            situation: The classified situation type
            patient_name: Patient's first name if known
            is_registered: Whether patient is registered
            language: Target response language code (e.g., "en", "ar")
            dialect: Target dialect code (e.g., "ae", "eg") or None
            conversation_history: Optional list of last 4 conversation turns (each with "role" and "content")
            
        Returns:
            Natural response in the user's language/dialect
        """
        name = patient_name or "Unknown"

        # Build language instruction
        lang_instruction = ""
        if language == "ar":
            dialect_map = {
                "ae": "Emirati Arabic (UAE dialect)",
                "sa": "Saudi Arabic (Gulf dialect)", 
                "eg": "Egyptian Arabic",
                "lv": "Levantine Arabic",
                None: "Emirati Arabic (UAE dialect)"
            }
            dialect_name = dialect_map.get(dialect, dialect_map[None])
            lang_instruction = f"\n\nIMPORTANT: Respond in {dialect_name}. Use natural, conversational Arabic appropriate for the UAE."
        elif language == "en":
            lang_instruction = f"\n\nIMPORTANT: Respond in {language}."
        
        # Situation-specific context
        situation_context = {
            SituationType.GREETING: "They are greeting you. Welcome them warmly.",
            SituationType.FAREWELL: "They are saying goodbye. Wish them well.",
            SituationType.THANKS: "They are thanking you. Acknowledge graciously.",
            SituationType.PLEASANTRY: "They are making a social/polite exchange. Respond naturally and warmly.",
            SituationType.UNCLEAR_REQUEST: "They are making a request that is unclear. Ask them to clarify.",
        }
        
        context = situation_context.get(situation, "Respond naturally and helpfully.")
        
        # For unclear_request, use the why_unclear explanation if provided
        if situation == SituationType.UNCLEAR_REQUEST and why_unclear:
            context = f"They are making a request that is unclear. {why_unclear} Ask them to clarify based on this understanding."
        
        # Registration hint for greetings only
        registration_hint = ""
        if situation == SituationType.GREETING and not is_registered:
            registration_hint = " Since they're new, you may briefly mention registration is available (but don't push it)."
        
        # Build conversation context if available
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nRecent conversation:\n"
            for turn in conversation_history:
                role_label = "User" if turn.get("role") == "user" else "Assistant"
                timestamp_str = ""
                timestamp = turn.get('timestamp')
                if timestamp:
                    try:
                        # Parse ISO format timestamp and format it nicely
                        ts_str = str(timestamp).replace('Z', '+00:00')
                        # Parse the timestamp
                        ts = datetime.fromisoformat(ts_str)
                        # Convert to naive datetime if timezone-aware (for formatting)
                        if ts.tzinfo:
                            ts = ts.replace(tzinfo=None)
                        timestamp_str = f" [{ts.strftime('%Y-%m-%d %H:%M:%S')}]"
                    except (ValueError, AttributeError, TypeError):
                        # Fallback to original if parsing fails
                        timestamp_str = f" [{timestamp}]"
                conversation_context += f"{role_label}{timestamp_str}: {turn.get('content', '')}\n"
            conversation_context += "\nUse this context to make your response more natural and relevant."
        
        prompt = f"""You are a friendly clinic receptionist.

Patient: {name} ({'returning patient' if is_registered else 'new visitor'})
User: "{message}"

{context}{registration_hint}{conversation_context}

1- RESPOND WARMTH AND NATURALLY 
2- KEEP IT CONCISE: 1-2 sentences maximum

MANDATORY: {lang_instruction}

Your response:"""

        # Log all inputs and built prompt
        import json
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("=" * 80)
        logger.info(f"💬 [Fast Path] [{timestamp}] CONVERSATIONAL RESPONSE GENERATOR - INPUTS:")
        logger.info("=" * 80)
        inputs = {
            "message": message,
            "situation": situation.value,
            "patient_name": patient_name,
            "is_registered": is_registered,
            "language": language,
            "dialect": dialect,
            "conversation_history": conversation_history,
            "session_id": session_id
        }
        logger.info(f"💬 [Fast Path] [{timestamp}] Inputs JSON:\n{json.dumps(inputs, indent=2, default=str)}")
        logger.info("=" * 80)
        
        system_prompt = "You are a warm, friendly dental clinic receptionist in the UAE. Be natural, concise, and culturally appropriate."
        
        logger.info(f"💬 [Fast Path] [{timestamp}] BUILT PROMPTS:")
        logger.info("=" * 80)
        logger.info(f"💬 [Fast Path] [{timestamp}] System Prompt:\n{system_prompt}")
        logger.info("-" * 80)
        logger.info(f"💬 [Fast Path] [{timestamp}] User Prompt:\n{prompt}")
        logger.info("=" * 80)

        try:
            # Get hierarchical LLM config for conversational_fast_path
            llm_config = self.llm_config_manager.get_config(agent_name="conversational_fast_path")
            llm_client = self.llm_config_manager.get_client(agent_name="conversational_fast_path")
            
            # Get observability logger
            obs_logger = get_observability_logger(session_id) if session_id and settings.enable_observability else None
            
            # Build messages list with conversation history if available
            messages = []
            if conversation_history:
                # Add conversation history as context messages
                for turn in conversation_history:
                    messages.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", "")
                    })
            
            # Add current prompt as final user message
            messages.append({"role": "user", "content": prompt})
            
            # Make LLM call with token tracking
            llm_start_time = time.time()
            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=messages,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=system_prompt,
                    messages=messages,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time
            
            # Log raw LLM response
            response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("=" * 80)
            logger.info(f"💬 [Fast Path] [{response_timestamp}] RAW LLM RESPONSE:")
            logger.info("=" * 80)
            logger.info(f"💬 [Fast Path] [{response_timestamp}] Response Content:\n{response}")
            logger.info("=" * 80)
            
            # Log LLM configuration used
            logger.info(f"💬 [Fast Path] [{response_timestamp}] LLM Configuration:")
            logger.info(f"💬 [Fast Path] [{response_timestamp}]   Provider: {llm_config.provider}")
            logger.info(f"💬 [Fast Path] [{response_timestamp}]   Model: {llm_config.model}")
            logger.info(f"💬 [Fast Path] [{response_timestamp}]   Temperature: {llm_config.temperature}")
            logger.info(f"💬 [Fast Path] [{response_timestamp}]   Max Tokens: {llm_config.max_tokens}")
            logger.info(f"💬 [Fast Path] [{response_timestamp}]   Tokens Used: {tokens.input_tokens}/{tokens.output_tokens} (total: {tokens.total_tokens})")
            logger.info(f"💬 [Fast Path] [{response_timestamp}]   Duration: {llm_duration_seconds:.3f}s")
            
            # Record LLM call for observability
            if obs_logger:
                llm_call = obs_logger.record_llm_call(
                    component="conversational_fast_path",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=len(messages),
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="conversational_response"
                )
                if llm_call and llm_call.cost and settings.cost_tracking_enabled:
                    cost_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"💬 [Fast Path] [{cost_timestamp}]   Cost: ${llm_call.cost.total_cost_usd:.6f} (Input: ${llm_call.cost.input_cost_usd:.6f}, Output: ${llm_call.cost.output_cost_usd:.6f})")
                # Also record tokens in token tracker for component-level tracking
                obs_logger.token_tracker.record_tokens(
                    component="conversational_fast_path",
                    input_tokens=tokens.input_tokens,
                    output_tokens=tokens.output_tokens
                )
            
            logger.info(f"[Conversational] Generated response ({len(response)} chars) for {situation.value}, tokens: {tokens.input_tokens}/{tokens.output_tokens} (total: {tokens.total_tokens})")
            return response.strip(), tokens
            
        except Exception as e:
            logger.error(f"[Conversational] Error generating response: {e}")
            # Safe fallbacks by language
            fallbacks = {
                "ar": {
                    SituationType.GREETING: "أهلاً! أقدر أساعدك ازاي ؟",
                    SituationType.FAREWELL: "مع السلامة! اتمنى لك يوماً سعيداً.",
                    SituationType.THANKS: "عفواً! هل هناك شيء آخر يمكنني مساعدتك به؟",
                    SituationType.PLEASANTRY: "الحمد لله بخير! كيف يمكنني مساعدتك اليوم؟",
                },
                "en": {
                    SituationType.GREETING: "Hello! How can I help you today?",
                    SituationType.FAREWELL: "Goodbye! Have a wonderful day.",
                    SituationType.THANKS: "You're welcome! Is there anything else I can help with?",
                    SituationType.PLEASANTRY: "I'm doing well, thank you! How can I help you today?",
                }
            }
            lang_fallbacks = fallbacks.get(language, fallbacks["en"])
            fallback_response = lang_fallbacks.get(situation, lang_fallbacks[SituationType.GREETING])
            return fallback_response, TokenUsage()

    def _detect_language_change_request(self, message: str, english_message: str) -> Optional[str]:
        """Detect if user is requesting to change language."""
        msg_lower = english_message.lower()

        # English change requests
        if any(phrase in msg_lower for phrase in [
            "switch to english", "change to english", "speak english",
            "use english", "english please", "change language", "switch language"
        ]):
            logger.info("[Language Change] English change request detected")
            return "en"

        if any(phrase in msg_lower for phrase in [
            "switch to arabic", "change to arabic", "speak arabic",
            "use arabic", "arabic please"
        ]):
            logger.info("[Language Change] Arabic change request detected (from English)")
            return "ar"

        # Arabic change requests (check original message)
        msg_original = message.lower()
        if any(phrase in msg_original for phrase in [
            "تكلم انجليزي", "غير للانجليزي", "استخدم انجليزي", "انجليزي"
        ]):
            logger.info("[Language Change] English change request detected (from Arabic)")
            return "en"

        if any(phrase in msg_original for phrase in [
            "تكلم عربي", "غير للعربي", "استخدم عربي", "عربي"
        ]):
            logger.info("[Language Change] Arabic change request detected (from Arabic)")
            return "ar"

        return None

    async def _information_collection_response(
        self,
        session_id: str,
        message: str,
        information_collection: Dict[str, Any],
        recent_messages: List[Dict[str, str]],
        language: str,
        dialect: Optional[str],
        patient_name: Optional[str]
    ) -> Tuple[str, TokenUsage, Optional[str], bool]:
        """
        Lightweight response for information collection non-brainer flow.

        Uses GPT-4o mini with 8 recent messages for context. Extracts information from user's response
        and generates a natural follow-up or acknowledgment. Also determines if collection is complete.

        Args:
            session_id: Session identifier
            message: User's latest message
            information_collection: Dict with 'information_needed', 'information_question', 'collected_information', etc.
            recent_messages: Last 8 conversation turns for context
            language: Target response language code
            dialect: Target dialect code or None
            patient_name: Patient's first name if known

        Returns:
            Tuple of (response_text, token_usage, extracted_information, collection_complete)
            - response_text: Natural language response to user
            - token_usage: Token usage for this call
            - extracted_information: String describing what info was provided (None if nothing useful)
            - collection_complete: Boolean indicating if all required information has been collected
        """
        # Log all inputs
        logger.info("=" * 80)
        logger.info(f"⚡ [Info Collection] INFORMATION COLLECTION RESPONSE GENERATOR - INPUTS:")
        logger.info("=" * 80)
        inputs = {
            "session_id": session_id,
            "message": message,
            "information_needed": information_collection.get('information_needed', 'N/A'),
            "information_question": information_collection.get('information_question', 'N/A'),
            "context": information_collection.get('context', 'N/A'),
            "patient_name": patient_name,
            "language": language,
            "dialect": dialect,
            "recent_messages_count": len(recent_messages)
        }
        logger.info(f"⚡ [Info Collection] Inputs JSON:\n{json.dumps(inputs, indent=2, default=str)}")
        logger.info("=" * 80)

        if patient_name:
            address_instruction = f"You may address the patient as '{patient_name}'"
        else:
            address_instruction = "DO NOT use any name - patient name is not available"

        # Build language instruction (reuse pattern from _conversational_response)
        lang_instruction = ""
        if language == "ar":
            lang_instruction = "Respond in Emirati Arabic (UAE). Use natural, colloquial Emirati Arabic. authentic but not slangy - professionally warm."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in EXACT user Language from RECENT CONVERSATION."

        # Build conversation context from last 8 messages
        conversation_context = ""
        if recent_messages:
            conversation_context = "RECENT CONVERSATION:\n"
            for turn in recent_messages[-6:]:
                role_label = "User" if turn.get("role") == "user" else "You"
                conversation_context += f"{role_label}: {turn.get('content', '')}\n"

        # Extract information collection details
        information_needed = information_collection.get('information_needed', 'user information')
        awaiting_info = information_collection.get('awaiting_info', 'user information')
        information_question = information_collection.get('information_question', '')
        collected_so_far = information_collection.get('collected_information', [])
        context = information_collection.get('context', 'general inquiry')

        # Format collected information for display
        collected_display = "\n".join([f"  - {item}" for item in collected_so_far]) if collected_so_far else ""

        # Build JSON example separately to avoid f-string brace escaping issues
        json_example = '''{
  "extracted_info": "concise description of what the user provided ALWAYS IN ENGLISH" OR null,
  "collection_complete": true OR false,
  "response": "(ONLY if collection_complete is false)" OR null
}'''

        prompt = f"""You are a friendly clinic receptionist in the UAE helping collect information. 

{address_instruction}

INFORMATION NEEDED (overall):
{information_needed}
{awaiting_info}

INFORMATION COLLECTED SO FAR:
{collected_display}

USER'S LATEST RESPONSE:
"{message}"

{conversation_context}


YOUR TASK:
1. Extract what information the user provided (if any)
2. Determine if all required information is now collected
3. If NOT complete, generate a follow-up question continuing "RECENT CONVERSATION".

BEHAVIOUR RULES:
1. If the user ask any questions, politely inform them that you will get back to them after collecting ALL the needed information.
2. Be FLEXIBLE with what counts as information: Vague answers like "I don't know", "I don't care", "anything" -> count as answers → Extract what you can

"response" RULES:
1. Speak warmly professional
2. Be conversational and natural, but avoid redundancy - don't repeat what was already said
3. AVOID repeating yourself. If user doesnt understand your question, ask again in a different way.
4. {lang_instruction}

COMPLETION CHECK:
Assess if ALL required information from "{information_needed}" is now collected (INFORMATION COLLECTED SO FAR + USER'S LATEST RESPONSE).

If ALL INFO COLLECTED:
  → collection_complete: true
  → extracted_info: Description of what user just provided
  → response: null (DO NOT generate a response - the system will handle it)

If STILL MISSING INFO:
  → collection_complete: false
  → extracted_info: Description of what user just provided (or null if nothing useful)
  → response: Warm, natural follow-up question for missing pieces
    - Example: "Got it! And what time of day works best for you?"
    - Example: "I understand, but to help you I need to know what type of visit you need."

Respond in JSON format:
{json_example}
"""

        # Log built prompt
        logger.info("=" * 80)
        logger.info(f"⚡ [Info Collection] BUILT PROMPT:")
        logger.info("=" * 80)
        logger.info(f"⚡ [Info Collection] Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        logger.info(f"⚡ [Info Collection] Prompt:\n{prompt}")
        logger.info("=" * 80)

        try:
            # Get hierarchical LLM config for _information_collection_response function
            llm_config = self.llm_config_manager.get_config(
                agent_name="orchestrator",
                function_name="_information_collection_response"
            )

            # Get LLM client using config manager (handles API keys automatically)
            llm_client = self.llm_config_manager.get_client(
                agent_name="orchestrator",
                function_name="_information_collection_response"
            )

            # Log LLM configuration
            logger.info("=" * 80)
            logger.info(f"⚡ [Info Collection] LLM CONFIGURATION:")
            logger.info("=" * 80)
            logger.info(f"⚡ [Info Collection]   Provider: {llm_config.provider}")
            logger.info(f"⚡ [Info Collection]   Model: {llm_config.model}")
            logger.info(f"⚡ [Info Collection]   Temperature: {llm_config.temperature}")
            logger.info(f"⚡ [Info Collection]   Max Tokens: {llm_config.max_tokens}")
            logger.info(f"⚡ [Info Collection]   Timeout: {llm_config.timeout}")
            logger.info("=" * 80)

            # Make LLM call
            llm_start_time = time.time()
            if hasattr(llm_client, 'chat_completion_json'):
                response_text, tokens = await llm_client.chat_completion_json(
                    messages=[{"role": "user", "content": prompt}],
                    config=llm_config,
                    session_id=session_id
                )
            else:
                # Fallback to regular message creation
                response_text, tokens = llm_client.create_message_with_usage(
                    system="You are a helpful assistant. Return JSON only.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )

            llm_duration_seconds = time.time() - llm_start_time
            llm_duration_ms = llm_duration_seconds * 1000

            # Log raw LLM response
            logger.info("=" * 80)
            logger.info(f"⚡ [Info Collection] RAW LLM RESPONSE:")
            logger.info("=" * 80)
            logger.info(f"⚡ [Info Collection] Response Content:\n{response_text}")
            logger.info("=" * 80)
            logger.info(f"⚡ [Info Collection] LLM call completed in {llm_duration_ms:.0f}ms")
            logger.info(f"⚡ [Info Collection] Tokens: {tokens.input_tokens} in / {tokens.output_tokens} out (total: {tokens.total_tokens})")
            logger.info("=" * 80)

            # Get observability logger
            obs_logger = get_observability_logger(session_id) if settings.enable_observability else None

            # Record LLM call in observability
            if obs_logger:
                llm_call = obs_logger.record_llm_call(
                    component="orchestrator.information_collection",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=0,  # No system prompt for JSON mode
                    messages_count=1,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="_information_collection_response"
                )
                if llm_call and llm_call.cost and settings.cost_tracking_enabled:
                    logger.info(f"⚡ [Info Collection]   Cost: ${llm_call.cost.total_cost_usd:.6f} (Input: ${llm_call.cost.input_cost_usd:.6f}, Output: ${llm_call.cost.output_cost_usd:.6f})")
                # Also record tokens in token tracker for component-level tracking
                obs_logger.token_tracker.record_tokens(
                    component="orchestrator.information_collection",
                    input_tokens=tokens.input_tokens,
                    output_tokens=tokens.output_tokens
                )

            # Parse response
            try:
                # Strip markdown code blocks if present
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```json"):
                    # Remove opening ```json
                    cleaned_response = cleaned_response[7:]
                elif cleaned_response.startswith("```"):
                    # Remove opening ```
                    cleaned_response = cleaned_response[3:]

                # Remove closing ```
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]

                cleaned_response = cleaned_response.strip()

                logger.info(f"⚡ [Info Collection] Cleaned response for parsing (removed markdown if present)")

                from patient_ai_service.core.json_utils import safe_json_loads
                result = safe_json_loads(cleaned_response, "Info Collection")
                extracted_info = result.get("extracted_info")  # Can be string or None
                collection_complete = result.get("collection_complete", False)  # Boolean flag
                response = result.get("response")  # Can be string or None
                # Handle None response (when JSON has "response": null)
                if response is None:
                    response = "I'm not sure I understood. Could you please provide the information I requested?"

                logger.info("=" * 80)
                logger.info(f"⚡ [Info Collection] PARSED RESPONSE:")
                logger.info("=" * 80)
                logger.info(f"⚡ [Info Collection] Extracted info: {extracted_info if extracted_info else '(none)'}")
                logger.info(f"⚡ [Info Collection] Collection complete: {collection_complete}")
                logger.info(f"⚡ [Info Collection] Response length: {len(response)} chars")
                logger.info(f"⚡ [Info Collection] Response:\n{response}")
                logger.info("=" * 80)

            except json.JSONDecodeError as e:
                logger.error("=" * 80)
                logger.error(f"⚡ [Info Collection] JSON PARSING ERROR:")
                logger.error("=" * 80)
                logger.error(f"⚡ [Info Collection] Error: {e}")
                logger.error(f"⚡ [Info Collection] Raw response: {response_text}")
                logger.error("=" * 80)
                extracted_info = None
                collection_complete = False
                response = "I'm not sure I understood. Could you please provide the information I requested?"

            logger.info(f"⚡ [Info Collection] ✅ Information collection response completed successfully")
            logger.info(f"⚡ [Info Collection] Final result: {'Info extracted: ' + str(extracted_info) if extracted_info else 'No info extracted'}, Complete: {collection_complete}")

            return response, tokens, extracted_info, collection_complete

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"⚡ [Info Collection] EXCEPTION:")
            logger.error("=" * 80)
            logger.error(f"⚡ [Info Collection] Error type: {type(e).__name__}")
            logger.error(f"⚡ [Info Collection] Error message: {e}")
            logger.error(f"⚡ [Info Collection] Traceback:", exc_info=True)
            logger.error("=" * 80)

            # Fallback response
            fallback_response = "I'm not sure I understood. Could you please provide that information again?"
            if language == "ar":
                fallback_response = "لم أفهم جيداً. هل يمكنك تقديم هذه المعلومات مرة أخرى؟"

            logger.info(f"⚡ [Info Collection] Returning fallback response: {fallback_response}")
            return fallback_response, TokenUsage(), None, False

    async def process_message(
        self,
        session_id: str,
        message: str,
        language: Optional[str] = None,
        clinic_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Process a user message through the complete pipeline.

        Args:
            session_id: Unique session identifier (typically phone number)
            message: User's message
            language: Optional language hint
            clinic_id: Clinic identifier for multi-tenant support

        Returns:
            ChatResponse with agent's reply
        """
        # Create clinic context and composite session key for multi-tenant isolation
        from patient_ai_service.models.clinic_context import ClinicContext

        if clinic_id:
            clinic_context = ClinicContext(clinic_id=clinic_id, session_id=session_id)
            session_id = clinic_context.composite_key
            # Set clinic context on db_client for RLS header propagation
            self.db_client.set_clinic_context(clinic_context)
        else:
            # Fallback for backward compatibility (should not happen in production)
            session_id = session_id
            clinic_context = None
            logger.warning(f"No clinic_id provided for session {session_id} - using non-namespaced key")

        pipeline_start_time = time.time()
        # Import settings at function level to avoid UnboundLocalError
        from patient_ai_service.core.config import settings as config_settings
        obs_logger = get_observability_logger(session_id) if config_settings.enable_observability else None

        logger.info("=" * 100)
        logger.info("ORCHESTRATOR: process_message() CALLED")
        logger.info("=" * 100)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Clinic ID: {clinic_id}")
        logger.info(f"Composite Session Key: {session_id}")
        logger.info(f"Input Message: {message}")
        logger.info(f"Language Hint: {language}")
        logger.info(f"Pipeline Start Time: {pipeline_start_time}")
        
        try:
            logger.info(f"Processing message for session: {session_id}")

            # Step 1: Load or initialize patient
            step1_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 1: Load Patient")
            logger.info("-" * 100)
            with obs_logger.pipeline_step(1, "load_patient", "orchestrator", {"session_id": session_id}) if obs_logger else nullcontext():
                await self._ensure_patient_loaded(session_id)
            step1_duration = (time.time() - step1_start) * 1000
            logger.info(f"Step 1 completed in {step1_duration:.2f}ms")

            # Step 2: Translation (input) - OPTIMIZED: Single LLM call for detect + translate
            step2_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 2: Translation (Input) [OPTIMIZED]")
            logger.info("-" * 100)
            logger.info(f"Original message: {message[:200]}")

            # Get current language context
            global_state = self.state_manager.get_global_state(session_id)
            language_context = global_state.language_context

            # DEBUG: Log language context state
            logger.info(f"[DEBUG] Language Context State:")
            logger.info(f"  - language_selected: {language_context.language_selected}")
            logger.info(f"  - current_language: {language_context.current_language}")
            logger.info(f"  - current_dialect: {language_context.current_dialect}")
            logger.info(f"  - preferred_language: {language_context.preferred_language}")
            logger.info(f"  - preferred_dialect: {language_context.preferred_dialect}")

            with obs_logger.pipeline_step(2, "translation_input", "translation", {"message": message[:100]}) if obs_logger else nullcontext():
                from patient_ai_service.core.feature_flags import is_translation_enabled
                from patient_ai_service.core.script_detector import ScriptDetector

                # Skip translation if language not yet selected (language_selection agent will handle it)
                if not language_context.language_selected:
                    logger.info("[Translation] Skipping - language not yet selected")
                    english_message = message
                    detected_lang = "en"
                    detected_dialect = None
                    translation_succeeded = True
                elif is_translation_enabled(session_id):
                    # ============================================================================
                    # NEW SIMPLIFIED TRANSLATION LOGIC (Post Language Selection)
                    # ============================================================================
                    # Use the selected language preference instead of automatic detection
                    # Only keep Arabic script detection as a safety barrier
                    # ============================================================================

                    translation_agent = self.agents["translation"]
                    user_preference = language_context.preferred_language  # ISO code: "en" or "ar"

                    # Determine if translation is needed
                    needs_translation = False
                    detected_lang = user_preference  # Default to user's preference
                    detected_dialect = language_context.preferred_dialect

                    # ONLY check for Arabic - don't "detect" other languages
                    if user_preference == "ar":
                        # User prefers Arabic - always translate Arabic input to English for processing
                        needs_translation = True
                        detected_lang = "ar"
                        detected_dialect = "ae"
                        logger.info("[Translation] User preference: Arabic - translating to English for processing")
                    elif ScriptDetector._has_arabic_letters(message):
                        # SAFETY BARRIER: Arabic script detected regardless of preference
                        needs_translation = True
                        detected_lang = "ar"
                        detected_dialect = "ae"
                        logger.info("[Translation] SAFETY BARRIER: Arabic script detected - translating to prevent system break")
                    else:
                        # Use user's preference as-is, no translation needed
                        needs_translation = False
                        logger.info(f"[Translation] Using user preference: {user_preference}, no Arabic detected - no translation needed")

                    # Perform translation if needed
                    if needs_translation:
                        translate_start = time.time()

                        # Use simple translation without detection
                        if detected_lang == "ar":
                            english_message = await translation_agent._translate_arabic_to_english(message, session_id)
                            translation_succeeded = True
                        else:
                            # Fallback for other languages
                            english_message = await translation_agent.translate_to_english_with_dialect(
                                message, detected_lang, detected_dialect
                            )
                            translation_succeeded = True

                        translate_duration = (time.time() - translate_start) * 1000
                        logger.info(
                            f"[Translation] {detected_lang} → en completed in {translate_duration:.2f}ms"
                        )
                        logger.info(f"[Translation] Result: {message[:100]}... → {english_message[:100]}...")
                    else:
                        english_message = message
                        translation_succeeded = True

                    # ============================================================================
                    # DEPRECATED: Old automatic language detection layer
                    # ============================================================================
                    # This has been replaced with preference-based translation + Arabic safety check
                    # The old detect_and_translate() method is no longer used here
                    #
                    # OLD CODE (DEPRECATED):
                    # english_message, detected_lang, detected_dialect, translation_succeeded =
                    #     await translation_agent.detect_and_translate(message, session_id)
                    # ============================================================================

                else:
                    logger.info("[SKIP] Translation disabled, using original message")
                    english_message = message
                    detected_lang = language_context.current_language or "en"
                    detected_dialect = language_context.current_dialect
                    translation_succeeded = True

                # Update language context
                language_context.turn_count += 1
                language_context.last_detected_at = datetime.utcnow()

                # Update current language based on user preference (already set by language_selection agent)
                # detected_lang and detected_dialect are now derived from preference, not detection
                if not detected_lang:
                    detected_lang = language_context.current_language or "en"
                if not detected_dialect and detected_lang == "ar":
                    detected_dialect = "ae"

                # CRITICAL FIX: Update current_language to match detected language
                # This ensures that if Arabic script was detected (safety barrier),
                # the preference is updated to "ar" for this turn and all future turns
                if detected_lang != language_context.current_language:
                    logger.info(f"[Language Context] Updating current_language from {language_context.current_language} to {detected_lang}")
                    language_context.current_language = detected_lang
                    if detected_dialect:
                        language_context.current_dialect = detected_dialect

                # Update global state with new language context (EVERY TURN)
                self.state_manager.update_global_state(
                    session_id,
                    language_context=language_context
                )

                logger.info(f"[Language Context] Saved to global state - current_language: {language_context.current_language}, preferred_language: {language_context.preferred_language}")

                # Check for language change request
                if language_context.language_selected:
                    new_lang = self._detect_language_change_request(message, english_message)
                    if new_lang and new_lang != language_context.current_language:
                        logger.info(f"[Language Change] Requested change to: {new_lang}")

                        new_dialect = "ae" if new_lang == "ar" else None
                        language_context.record_language_switch(new_lang, new_dialect, language_context.turn_count)
                        language_context.mark_language_selected(new_lang, new_dialect)

                        self.state_manager.update_global_state(
                            session_id,
                            language_context=language_context
                        )

                        # Generate acknowledgment
                        if new_lang == "en":
                            ack = "Switched to English. How can I help you?"
                        else:
                            ack = "تمام، غيرنا للعربي. كيف أقدر أساعدك؟"

                        response = ChatResponse(
                            response=ack,
                            session_id=session_id,
                            detected_language=language_context.get_full_language_code(),
                            intent="language_changed",
                            urgency=UrgencyLevel.LOW,
                            metadata={
                                "agent": "orchestrator",
                                "flow": "language_change",
                                "previous_language": language_context.language_history[-1]["from_language"] if language_context.language_history else "unknown",
                                "new_language": new_lang
                            }
                        )

                        # Store in memory
                        self.memory_manager.add_user_turn(session_id, message)
                        self.memory_manager.add_assistant_turn(session_id, ack)

                        logger.info(f"[Language Change] Changed language preference to {new_lang}")
                        return response

                logger.info(
                    f"Language: {language_context.get_full_language_code()} | "
                    f"Message: '{message[:50]}...' → '{english_message[:50]}...'"
                )

                if obs_logger:
                    obs_logger.record_pipeline_step(
                        2, "translation_input", "translation",
                        inputs={"message": message[:100]},
                        outputs={
                            "english_message": english_message[:100],
                            "detected_lang": detected_lang,
                            "detected_dialect": detected_dialect,
                            "translation_succeeded": translation_succeeded
                        }
                    )
            step2_duration = (time.time() - step2_start) * 1000
            logger.info(f"Step 2 completed in {step2_duration:.2f}ms")

            logger.info(f"Detected language: {language_context.get_full_language_code()}")

            # Step 3: Add user message to conversation memory
            step3_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 3: Add to Memory")
            logger.info("-" * 100)
            with obs_logger.pipeline_step(3, "add_to_memory", "memory_manager", {"message": english_message[:100]}) if obs_logger else nullcontext():
                # Store in ConversationMemory (English - for Unified Reasoning & _think())
                self.memory_manager.add_user_turn(session_id, english_message)
                logger.info(f"📝 [ConversationMemory] Stored user turn (English): {english_message[:80]}...")

                # Also store in NativeLanguageMemory (Original language - for response generators)
                self.native_memory_manager.add_user_turn(
                    session_id=session_id,
                    content=message,  # Original message before translation
                    language=detected_lang,
                    dialect=detected_dialect
                )
                logger.info(f"📝 [NativeLanguageMemory] Stored user turn ({detected_lang}): {message[:80]}...")
            step3_duration = (time.time() - step3_start) * 1000
            logger.info(f"Step 3 completed in {step3_duration:.2f}ms")

            # =========================================================================
            # Step 4: UNIFIED REASONING (replaces assessment + plan lifecycle + routing)
            # =========================================================================
            step4_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 4: Unified Reasoning")
            logger.info("-" * 100)

            global_state = self.state_manager.get_global_state(session_id)

            # Store clinic_id in global state if not already set
            if clinic_id and not global_state.clinic_id:
                global_state.clinic_id = clinic_id
                self.state_manager.update_global_state(session_id, clinic_id=clinic_id)

            # Fetch and cache clinic metadata
            if clinic_id and not global_state.clinic_metadata:
                clinic_metadata = self.state_manager.get_clinic_metadata(clinic_id)
                if clinic_metadata:
                    global_state.clinic_metadata = clinic_metadata
                    self.state_manager.update_global_state(
                        session_id,
                        clinic_metadata=clinic_metadata
                    )
                    # Log detailed clinic metadata
                    logger.info("=" * 80)
                    logger.info("🏥 CLINIC METADATA LOADED")
                    logger.info("=" * 80)
                    logger.info(f"Clinic Name: {clinic_metadata.name}")
                    logger.info(f"Clinic ID: {clinic_metadata.clinic_id}")
                    if clinic_metadata.address:
                        logger.info(f"Location: {clinic_metadata.address}")
                    logger.info(f"Timezone: {clinic_metadata.timezone}")
                    logger.info(f"Current Date & Time: {clinic_metadata.get_current_datetime_str()}")
                    if clinic_metadata.phone_number:
                        logger.info(f"Phone: {clinic_metadata.phone_number}")
                    if clinic_metadata.email:
                        logger.info(f"Email: {clinic_metadata.email}")
                    logger.info("=" * 80)
                else:
                    logger.warning(f"Could not load clinic metadata for {clinic_id} - proceeding without clinic context")

            patient_info = {
                "patient_id": global_state.patient_profile.patient_id,
                "first_name": global_state.patient_profile.first_name,
                "last_name": global_state.patient_profile.last_name,
                "phone": global_state.patient_profile.phone,
            }

            # Get continuation context
            continuation_context = self.state_manager.get_continuation_context(session_id)
            awaiting = continuation_context.get("awaiting") if continuation_context else None
            awaiting_context = continuation_context.get("awaiting_context") if continuation_context else None  # NEW
            pending_action = continuation_context.get("pending_action") if continuation_context else None      # NEW
            information_collection = continuation_context.get("information_collection") if continuation_context else None  # NEW

            # DEBUG: Log information_collection state
            if information_collection:
                logger.info("=" * 80)
                logger.info(f"📥 [DEBUG] LOADED information_collection from state:")
                logger.info(f"📥 Has information_needed: {'information_needed' in information_collection}")
                logger.info(f"📥 information_needed: {information_collection.get('information_needed', 'MISSING')}")
                logger.info(f"📥 collected_information: {information_collection.get('collected_information', [])}")
                logger.info("=" * 80)

            # Get existing plan
            existing_plan = self.state_manager.get_any_active_plan(session_id)

            # 🔍 ADD LOGGING HERE
            logger.info(f"🔍🔍🔍 [ORCHESTRATOR] PLAN RETRIEVAL ══════════════════")
            logger.info(f"🔍 Session ID: {session_id}")
            if existing_plan:
                logger.info(f"🔍 ✅ PLAN FOUND!")
                logger.info(f"🔍   Agent name: {existing_plan.agent_name}")
                logger.info(f"🔍   Status: {existing_plan.status.value}")
                logger.info(f"🔍   Objective: {existing_plan.objective}")
                logger.info(f"🔍   Tasks: {len(existing_plan.tasks)}")
                logger.info(f"🔍   Awaiting: {existing_plan.awaiting_info}")
                logger.info(f"🔍   Entities: {list(existing_plan.entities.keys())}")
            else:
                logger.info(f"🔍 ❌ NO PLAN FOUND")
                logger.info(f"🔍   Continuation context exists: {bool(continuation_context)}")
                if continuation_context:
                    logger.info(f"🔍   Continuation awaiting: {continuation_context.get('awaiting')}")
                    logger.info(f"🔍   Continuation entities: {list(continuation_context.get('entities', {}).keys())}")
            logger.info(f"🔍🔍🔍 ══════════════════════════════════════════════════")

            # Get recent conversation turns (last 8 messages)
            recent_turns = self._get_recent_turns(session_id, limit=6)

            # Check if planning is enabled for this session (A/B testing)
            planning_enabled = FeatureFlags.is_enabled("planning", session_id)
            if not planning_enabled:
                logger.info(f"📋 [Orchestrator] Planning DISABLED for session {session_id}")

            # Single unified reasoning call
            with obs_logger.pipeline_step(4, "unified_reasoning", "unified_reasoning", {"message": english_message[:100]}) if obs_logger else nullcontext():
                unified_output = await self.unified_reasoning.reason(
                    session_id=session_id,
                    message=english_message,
                    patient_info=patient_info,
                    active_agent=global_state.active_agent,
                    awaiting=awaiting,
                    awaiting_context=awaiting_context,    # NEW
                    pending_action=pending_action,         # NEW
                    information_collection=information_collection,  # NEW
                    recent_turns=recent_turns,
                    existing_plan=existing_plan,
                    planning_enabled=planning_enabled,  # A/B testing flag
                    clinic_metadata=global_state.clinic_metadata  # NEW
                )

            step4_duration = (time.time() - step4_start) * 1000
            logger.info(f"Step 4 completed in {step4_duration:.2f}ms")

            if obs_logger:
                obs_logger.record_pipeline_step(
                    4, "unified_reasoning", "unified_reasoning",
                    inputs={"message": english_message[:100]},
                    outputs={
                        "route_type": unified_output.route_type.value,
                        "situation_type": unified_output.situation_type.value,
                        "agent": unified_output.agent,
                        "plan_decision": unified_output.plan_decision.value if unified_output.plan_decision else None,
                        "duration_ms": step4_duration
                    }
                )

            # =========================================================================
            # Step 5: HANDLE ROUTING
            # =========================================================================
            step5_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 5: Handle Routing")
            logger.info("-" * 100)

            # === FAST PATH ===
            if unified_output.is_fast_path():
                fast_path_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_msg = f"⚡ [Fast Path] [{fast_path_timestamp}] Situation: {unified_output.situation_type.value}"
                if unified_output.situation_type == SituationType.UNCLEAR_REQUEST and unified_output.why_unclear:
                    log_msg += f" | Why unclear: {unified_output.why_unclear}"
                logger.info(log_msg)

                with obs_logger.pipeline_step(5, "fast_path_routing", "orchestrator", {"situation": unified_output.situation_type.value}) if obs_logger else nullcontext():
                    # Get conversation history from NativeLanguageMemory for context-aware response
                    # This provides original language context for better tone matching
                    native_turns = self.native_memory_manager.get_recent_turns(session_id, limit=4)
                    logger.info(f"🔍 [_conversational_response] Retrieved {len(native_turns)} turns from NativeLanguageMemory")
                    if native_turns:
                        logger.info(f"🔍 [_conversational_response] Language: {native_turns[0].get('language', 'unknown')}")
                    conversation_history = [
                        {
                            "role": t["role"],
                            "content": t["content"],
                            "timestamp": t.get("timestamp")
                        }
                        for t in native_turns
                    ] if native_turns else None

                    conv_start = time.time()
                    english_response, conv_tokens = await self._conversational_response(
                    message=english_message,
                    situation=unified_output.situation_type,
                    patient_name=patient_info.get('first_name'),
                    is_registered=bool(patient_info.get('patient_id')),
                    language=language_context.current_language,
                    dialect=language_context.current_dialect,
                    conversation_history=conversation_history,
                    session_id=session_id,
                    why_unclear=unified_output.why_unclear
                )
                    conv_duration = (time.time() - conv_start) * 1000
                    conv_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"⚡ [Fast Path] [{conv_timestamp}] Response generated in {conv_duration:.0f}ms")

                    # Build execution log for fast-path
                    execution_log = ExecutionLog(
                        session_id=session_id,
                        agent_name="fast_path",
                        tools_used=[]
                    )

                    step5_duration = (time.time() - step5_start) * 1000
                    logger.info(f"Step 5 completed in {step5_duration:.2f}ms")

                # TRANSLATION OUTPUT DISABLED
                # Fast path already generates response in target language via language instructions in prompt
                # No translation needed - response is already in correct language
                if obs_logger:
                    obs_logger.record_pipeline_step(
                        13, "translation_output", "translation",
                        metadata={"status": "disabled", "reason": "fast_path_generates_in_target_language", "language": language_context.current_language}
                    )

                # Add assistant message to memory
                with obs_logger.pipeline_step(12, "add_assistant_to_memory", "memory_manager", {"response_preview": english_response[:100]}) if obs_logger else nullcontext():
                    # Store in ConversationMemory (English - for Unified Reasoning & _think())
                    self.memory_manager.add_assistant_turn(
                        session_id=session_id,
                        content=english_response
                    )
                    logger.info(f"📝 [ConversationMemory] Stored assistant turn: {english_response[:80]}...")

                    # Also store in NativeLanguageMemory (for response generators)
                    self.native_memory_manager.add_assistant_turn(
                        session_id=session_id,
                        content=english_response  # Already in target language
                    )
                    logger.info(f"📝 [NativeLanguageMemory] Stored assistant turn ({language_context.current_language}): {english_response[:80]}...")

                total_duration = (time.time() - pipeline_start_time) * 1000
                
                # Log final response summary
                pipeline_duration_seconds = time.time() - pipeline_start_time
                logger.info("=" * 100)
                logger.info("ORCHESTRATOR: process_message() COMPLETED (Fast Path)")
                logger.info("=" * 100)
                logger.info(f"Session ID: {session_id}")
                logger.info(f"Final Response: {english_response[:200]}...")
                logger.info(f"Route: fast_path")
                logger.info(f"Situation: {unified_output.situation_type.value}")
                logger.info(f"Total Pipeline Duration: {pipeline_duration_seconds:.3f}s")
                logger.info(f"Tools Used: 0")
                logger.info("=" * 100)
                
                # Log observability summary for fast path
                if obs_logger:
                    obs_logger.record_pipeline_step(
                        15, "pipeline_complete", "orchestrator",
                        metadata={"total_duration_seconds": pipeline_duration_seconds, "route": "fast_path"}
                    )
                    obs_logger.log_summary()
                    
                    # Reset observability logger for next request (preserves accumulative cost)
                    clear_observability_logger(session_id)

                return ChatResponse(
                    session_id=session_id,
                    response=english_response,
                    intent=unified_output.situation_type.value,
                    metadata={
                        "route_type": "fast_path",
                        "situation": unified_output.situation_type.value,
                        "language": language_context.current_language,
                        "duration_ms": total_duration
                    }
                )

            # === INFORMATION COLLECTION NON-BRAINER ===
            if unified_output.routing_action == "collect_information":
                logger.info(f"⚡ [Information Collection] Non-brainer response for info collection")

                # Validate context exists (reuse from line 820-824)
                if not continuation_context:
                    logger.error("⚡ [Info Collection] CRITICAL: continuation_context is None!")
                    logger.error("⚡ [Info Collection] routing_action was 'collect_information' but no context found")
                    # Fall through to normal agent routing
                elif not information_collection:
                    logger.error("⚡ [Info Collection] CRITICAL: information_collection is None/empty!")
                    logger.error("⚡ [Info Collection] routing_action was 'collect_information' but information_collection is missing")
                    # Fall through to normal agent routing
                elif 'information_needed' not in information_collection:
                    logger.error("=" * 80)
                    logger.error("⚡ [Info Collection] CRITICAL: information_needed is missing from context!")
                    logger.error(f"⚡ [Info Collection] information_collection: {information_collection}")
                    logger.error(f"⚡ [Info Collection] This should have been set by agent._think() and persisted")
                    logger.error("=" * 80)
                    # Fall through to normal agent routing
                else:
                    with obs_logger.pipeline_step(5, "information_collection_response", "orchestrator",
                                                  {"information_needed": information_collection.get('information_needed', 'N/A')}) if obs_logger else nullcontext():

                        # Get recent messages from NativeLanguageMemory (last 8 for context)
                        # This provides original language context for better tone matching
                        recent_messages = self.native_memory_manager.get_recent_turns(session_id, limit=8)
                        logger.info(f"🔍 [_information_collection_response] Retrieved {len(recent_messages)} turns from NativeLanguageMemory")
                        if recent_messages:
                            logger.info(f"🔍 [_information_collection_response] Language: {recent_messages[0].get('language', 'unknown')}")

                        # Get language context
                        global_state = self.state_manager.get_global_state(session_id)
                        language_context = global_state.language_context

                        # Generate lightweight response
                        info_start = time.time()
                        english_response, info_tokens, extracted_info, collection_complete = await self._information_collection_response(
                            session_id=session_id,
                            message=english_message,
                            information_collection=information_collection,
                            recent_messages=recent_messages,
                            language=language_context.current_language,
                            dialect=language_context.current_dialect,
                            patient_name=patient_info.get('first_name')
                        )
                        info_duration = (time.time() - info_start) * 1000
                        logger.info(f"⚡ [Information Collection] Response generated in {info_duration:.0f}ms")
                        logger.info(f"⚡ [Information Collection] Extracted info: {extracted_info if extracted_info else '(none)'}")
                        logger.info(f"⚡ [Information Collection] Collection complete: {collection_complete}")

                        # Check for completion signal
                        if collection_complete:
                            logger.info("=" * 80)
                            logger.info("⚡ [Info Collection] COMPLETION DETECTED!")
                            logger.info("⚡ [Info Collection] Lightweight agent determined all info collected")
                            logger.info("⚡ [Info Collection] Proceeding to normal agentic workflow")
                            logger.info(f"⚡ [Info Collection] Will use agent: {unified_output.agent}")
                            logger.info(f"⚡ [Info Collection] Will use plan_decision: {unified_output.plan_decision}")
                            logger.info("=" * 80)

                            # Clear awaiting state and save final information_collection
                            information_collection['collection_status'] = 'complete'
                            # Add the final extracted info to collected list
                            if extracted_info:
                                collected_info_list = information_collection.get('collected_information', [])
                                collected_info_list.append(extracted_info)
                                information_collection['collected_information'] = collected_info_list

                            # Save with cleared awaiting
                            updated_context_dict = {
                                'awaiting': '',  # Clear awaiting
                                'awaiting_context': None,
                                'waiting_turns': continuation_context.get('waiting_turns', 0) + 1,
                                'information_collection': information_collection,
                                'pending_action': continuation_context.get('pending_action', {}),
                                'presented_options': continuation_context.get('presented_options', []),
                                'original_request': continuation_context.get('original_request'),
                                'entities': continuation_context.get('entities', {}),
                                'llm_entities': continuation_context.get('llm_entities', {}),
                                'blocked_criteria': continuation_context.get('blocked_criteria', []),
                                'created_at': continuation_context.get('created_at'),
                            }

                            updated_context = ContinuationContext(**updated_context_dict)
                            state = self.state_manager.get_agentic_state(session_id)
                            state.continuation_context = updated_context
                            state.last_updated_at = datetime.utcnow()
                            self.state_manager._save_state(session_id, "agentic_state", state)

                            logger.info("⚡ [Info Collection] State saved, falling through to agent routing")
                            logger.info("⚡ [Info Collection] Agent's _think() will generate the response")
                            logger.info("=" * 80)

                            # DON'T return - fall through to agent routing
                            # Agent's _think() will generate the actual response with collected info

                        # === CONTINUE COLLECTING (not complete yet) ===
                        if not collection_complete:
                            # Build execution log
                            execution_log = ExecutionLog(
                                session_id=session_id,
                                agent_name="information_collection",
                                tools_used=[]
                            )

                            # Update collected_information if user provided new info
                            if extracted_info:
                                logger.info(f"⚡ [Information Collection] User provided new information - updating collected_information")

                                # Get current collected_information list
                                collected_info_list = information_collection.get('collected_information', [])

                                # Append new info
                                collected_info_list.append(extracted_info)

                                # Update information_collection dict
                                # CRITICAL: Preserve information_needed (set by agent._think(), must never be lost)
                                information_collection['collected_information'] = collected_info_list

                                # Update the question asked for context on next turn
                                information_collection['information_question'] = english_response

                                logger.info(f"⚡ [Information Collection] Total pieces collected: {len(collected_info_list)}")
                                logger.info(f"⚡ [Information Collection] Collected so far: {collected_info_list}")
                            else:
                                logger.info(f"⚡ [Information Collection] User didn't provide new information - keeping same state")

                            # Verify information_needed is still present before saving
                            if 'information_needed' not in information_collection:
                                logger.error(f"⚡ [Info Collection] CRITICAL: information_needed was lost during update!")
                                logger.error(f"⚡ [Info Collection] This is a bug - information_needed should always persist")
                                # This should never happen, but if it does, we can't continue safely
                                raise ValueError("information_needed was lost from information_collection context")

                            # Update continuation context with updated collected_information
                            # NEVER clear it - let unified reasoning decide on next turn if collection is complete
                            if continuation_context:
                                # CRITICAL: Validate required fields BEFORE updating
                                if 'awaiting' not in continuation_context:
                                    logger.error("⚡ [Info Collection] CRITICAL: 'awaiting' missing from continuation_context!")
                                    logger.error(f"⚡ [Info Collection] Context keys: {list(continuation_context.keys())}")
                                    raise ValueError("Cannot update context without 'awaiting' field")

                                if continuation_context.get('awaiting') != 'information':
                                    logger.error("⚡ [Info Collection] CRITICAL: awaiting != 'information'!")
                                    logger.error(f"⚡ [Info Collection] awaiting = {continuation_context.get('awaiting')}")
                                    raise ValueError("Cannot collect information when awaiting != 'information'")

                                # Create new context dict with EXPLICIT field preservation
                                updated_context_dict = {
                                    # CRITICAL: Preserve awaiting - never change it (only agent._think() can)
                                    'awaiting': continuation_context['awaiting'],
                                    'awaiting_context': continuation_context.get('awaiting_context'),

                                    # Update only these two fields
                                    'waiting_turns': continuation_context.get('waiting_turns', 0) + 1,
                                    'information_collection': information_collection,

                                    # Preserve all other fields explicitly
                                    'pending_action': continuation_context.get('pending_action', {}),
                                    'presented_options': continuation_context.get('presented_options', []),
                                    'original_request': continuation_context.get('original_request'),
                                    'entities': continuation_context.get('entities', {}),
                                    'llm_entities': continuation_context.get('llm_entities', {}),
                                    'blocked_criteria': continuation_context.get('blocked_criteria', []),
                                    'created_at': continuation_context.get('created_at'),
                                }

                                # Validate critical fields before reconstruction
                                if 'information_needed' not in information_collection:
                                    logger.error("⚡ [Info Collection] CRITICAL: information_needed missing from information_collection!")
                                    raise ValueError("Cannot save context without information_needed")

                                # Log what we're saving
                                logger.info("⚡ [Information Collection] Saving updated context:")
                                logger.info(f"  awaiting: {updated_context_dict['awaiting']}")
                                logger.info(f"  information_needed: {information_collection.get('information_needed')}")
                                logger.info(f"  collected_information: {information_collection.get('collected_information', [])}")

                                # Reconstruct and save
                                updated_context = ContinuationContext(**updated_context_dict)
                                state = self.state_manager.get_agentic_state(session_id)
                                state.continuation_context = updated_context
                                state.last_updated_at = datetime.utcnow()
                                self.state_manager._save_state(session_id, "agentic_state", state)

                                # VERIFICATION: Read back state to ensure persistence
                                logger.info("⚡ [Information Collection] Verifying state was saved correctly...")
                                verification = self.state_manager.get_continuation_context(session_id)

                                if not verification:
                                    logger.error("⚡ [Info Collection] VERIFICATION FAILED: get_continuation_context returned None!")
                                    logger.error(f"⚡ [Info Collection] This means awaiting field might be empty or state wasn't saved")
                                    raise RuntimeError("State persistence verification failed - context is None")

                                if verification.get('awaiting') != 'information':
                                    logger.error("⚡ [Info Collection] VERIFICATION FAILED: awaiting field lost!")
                                    logger.error(f"⚡ [Info Collection] Expected: 'information', Got: {verification.get('awaiting')}")
                                    raise RuntimeError("State persistence verification failed - awaiting field lost")

                                if 'information_needed' not in verification.get('information_collection', {}):
                                    logger.error("⚡ [Info Collection] VERIFICATION FAILED: information_needed lost!")
                                    logger.error(f"⚡ [Info Collection] Verification context: {verification}")
                                    raise RuntimeError("State persistence verification failed - information_needed lost")

                                logger.info(f"⚡ [Information Collection] ✅ State verified - awaiting='information', information_needed preserved")

                                logger.info(f"⚡ [Information Collection] Updated continuation context with collected_information")
                                logger.info(f"⚡ [Information Collection] ✅ information_needed persisted: {information_collection.get('information_needed', 'ERROR')}")

                            # Add response to memory
                            # Store in ConversationMemory (English - for Unified Reasoning & _think())
                            self.memory_manager.add_assistant_turn(session_id, english_response)
                            logger.info(f"📝 [ConversationMemory] Stored assistant turn: {english_response[:80]}...")

                            # Also store in NativeLanguageMemory (for response generators)
                            self.native_memory_manager.add_assistant_turn(
                                session_id=session_id,
                                content=english_response  # Already in target language
                            )
                            logger.info(f"📝 [NativeLanguageMemory] Stored assistant turn ({language_context.current_language}): {english_response[:80]}...")

                            step5_duration = (time.time() - step5_start) * 1000
                            logger.info(f"Step 5 completed in {step5_duration:.2f}ms")

                            total_duration = (time.time() - pipeline_start_time) * 1000
                            logger.info("=" * 100)
                            logger.info("ORCHESTRATOR: process_message() COMPLETED (Information Collection)")
                            logger.info("=" * 100)
                            logger.info(f"Session ID: {session_id}")
                            logger.info(f"Final Response: {english_response[:200]}...")
                            logger.info(f"Route: information_collection (multi-turn)")
                            logger.info(f"Collected pieces: {len(information_collection.get('collected_information', []))}")
                            logger.info(f"Total Duration: {total_duration:.0f}ms")
                            logger.info("=" * 100)

                            # Clear observability for next request
                            if obs_logger:
                                obs_logger.log_summary()
                                clear_observability_logger(session_id)

                            return ChatResponse(
                                session_id=session_id,
                                response=english_response,
                                intent="information_collection_ongoing",
                                metadata={
                                    "route": "information_collection",
                                    "awaiting": "information",
                                    "collected_pieces": len(information_collection.get('collected_information', [])),
                                    "duration_ms": total_duration
                                }
                            )
                        # When collection_complete=True, we fall through to AGENT ROUTING below

            # === AGENT ROUTING ===
            with obs_logger.pipeline_step(6, "agent_routing", "orchestrator", {"agent": unified_output.agent, "plan_decision": unified_output.plan_decision.value if unified_output.plan_decision else None}) if obs_logger else nullcontext():
                logger.info(f"🎯 [Agent Routing] Agent: {unified_output.agent}")
                logger.info(f"🎯 [Agent Routing] Plan decision: {unified_output.plan_decision.value if unified_output.plan_decision else 'N/A'}")

                # Execute plan decision
                if unified_output.plan_decision == PlanDecision.ABANDON_CREATE:
                    if existing_plan:
                        self.state_manager.abandon_all_plans(session_id, "unified_reasoning_abandon")
                        logger.info(f"📋 [Plan] Abandoned existing plan for {existing_plan.agent_name}")

                if unified_output.plan_decision == PlanDecision.COMPLETE:
                    if existing_plan:
                        self.state_manager.abandon_all_plans(session_id, "unified_reasoning_complete")
                        logger.info(f"📋 [Plan] Cleared completed plan for {existing_plan.agent_name}")

                # Build reasoning-compatible output for agent activation
                reasoning = self._build_reasoning_from_unified(session_id, unified_output, existing_plan, continuation_context)

                # Determine plan_action for agent context (map from unified plan_decision)
                plan_action_map = {
                    PlanDecision.NO_PLAN: PlanAction.NO_PLAN,
                    PlanDecision.CREATE_NEW: PlanAction.CREATE_NEW,
                    PlanDecision.RESUME: PlanAction.RESUME,
                    PlanDecision.ABANDON_CREATE: PlanAction.ABANDON_AND_CREATE,
                    PlanDecision.COMPLETE: PlanAction.NO_PLAN,  # Complete clears plan, no new plan needed
                }
                plan_action = plan_action_map.get(unified_output.plan_decision, PlanAction.CREATE_NEW)

                step5_duration = (time.time() - step5_start) * 1000
                logger.info(f"Step 5 completed in {step5_duration:.2f}ms")

            # Continue with existing Step 6+ (agent execution)
            # The variable 'reasoning' now contains a ReasoningOutput-compatible object
            # The variable 'agent_name' should be set from unified_output
            agent_name = unified_output.agent

            # ============================================================================
            # SUCCESS CRITERIA - Use objective or derive from intent
            # ============================================================================
            # Success criteria now come from reasoning engine as outcomes, or derive from objective
            success_criteria = reasoning.response_guidance.task_context.success_criteria

            if not success_criteria:
                # Fallback: create outcome-based criterion from objective/intent
                objective = reasoning.response_guidance.task_context.objective
                user_intent = reasoning.response_guidance.task_context.user_intent
                
                if objective:
                    success_criteria = [objective]
                elif user_intent:
                    success_criteria = [f"Complete: {user_intent}"]
                else:
                    success_criteria = [f"Complete action: {reasoning.routing.action}"]
                
                reasoning.response_guidance.task_context.success_criteria = success_criteria

            logger.info(f"Success criteria: {success_criteria}")
            # ============================================================================

            logger.info(
                f"Reasoning: agent={reasoning.routing.agent}, "
                f"urgency={reasoning.routing.urgency}, "
                f"sentiment={reasoning.understanding.sentiment}"
            )
            
            # Print reasoning output summary in orchestrator
            import json
            logger.info("=" * 80)
            logger.info("ORCHESTRATOR: Reasoning Output Received")
            logger.info("=" * 80)
            logger.info(f"Session: {session_id}")
            logger.info(f"User Message: {english_message[:200]}...")
            logger.info(f"Routing Decision: {reasoning.routing.agent} -> {reasoning.routing.action} (urgency: {reasoning.routing.urgency})")
            logger.info(f"Understanding: {reasoning.understanding.what_user_means}")
            logger.info(f"Sentiment: {reasoning.understanding.sentiment}, Continuation: {reasoning.understanding.is_continuation}")
            logger.info(f"Memory Updates:")
            logger.info(f"  - system_action: {reasoning.memory_updates.system_action or '(empty)'}")
            logger.info(f"  - awaiting: {reasoning.memory_updates.awaiting or '(empty)'}")
            if reasoning.memory_updates.new_facts:
                logger.info(f"  - new_facts: {json.dumps(reasoning.memory_updates.new_facts, indent=2)}")
            if reasoning.response_guidance.minimal_context:
                logger.info(f"Response Guidance: {json.dumps(reasoning.response_guidance.minimal_context, indent=2)}")
            if reasoning.response_guidance.task_context.objective:
                logger.info(f"Agent Objective: {reasoning.response_guidance.task_context.objective}")
            logger.info("=" * 80)

            # Step 5: Handle conversation restart if detected
            if reasoning.understanding.is_conversation_restart:
                with obs_logger.pipeline_step(5, "conversation_restart", "memory_manager", {}) if obs_logger else nullcontext():
                    self.memory_manager.archive_and_reset(session_id)
                    logger.info(f"Conversation restarted for session {session_id}")

            # Initialize execution log at START of pipeline (before agent selection)
            # This ensures execution_log always exists, even in error paths
            execution_log = ExecutionLog(
                tools_used=[],
                conversation_turns=len(self.memory_manager.get_memory(session_id).recent_turns)
            )
            logger.debug(f"Initialized execution_log for session {session_id}")

            # Step 6: Select agent from reasoning
            agent_name = reasoning.routing.agent

            # Emergency routing via urgency field
            if reasoning.routing.urgency == "emergency":
                agent_name = "emergency_response"
                logger.info(f"Emergency routing for session {session_id}")

            # Update active agent in both global_state and reasoning_state
            self.state_manager.update_global_state(
                session_id,
                active_agent=agent_name
            )
            
            if obs_logger:
                obs_logger.agent_flow_tracker.record_transition(
                    from_agent=None,
                    to_agent=agent_name,
                    reason=f"Reasoning: {reasoning.routing.action}",
                    context=reasoning.response_guidance.minimal_context
                )

            # Step 7: Agent transition hook
            step7_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 7-8: Agent Activation & Execution")
            logger.info("-" * 100)
            logger.info(f"Selected Agent: {agent_name}")
            agent = self.agents.get(agent_name)
            if not agent:
                logger.error(f"Agent not found: {agent_name}")
                english_response = "I'm sorry, I encountered an error. Please try again."
                # execution_log already initialized above, no need to create again
            elif agent_name == "language_selection":
                # Special handling for language_selection agent
                logger.info("[Language Selection] Handling language_selection agent")

                # Build context for language selection agent
                global_state = self.state_manager.get_global_state(session_id)
                patient_info = {
                    "patient_id": global_state.patient_profile.patient_id,
                    "first_name": global_state.patient_profile.first_name,
                    "last_name": global_state.patient_profile.last_name,
                    "phone": global_state.patient_profile.phone,
                }
                # Get recent conversation turns
                memory = self.memory_manager.get_memory(session_id)
                recent_turns = memory.recent_turns if memory else []

                context = {
                    "patient_info": patient_info,
                    "recent_turns": recent_turns,
                    "session_id": session_id
                }

                # Call language selection agent
                try:
                    agent_result = await agent.process_message(
                        session_id=session_id,
                        message=message,  # Use original message (not translated)
                        context=context
                    )

                    logger.info(f"[Language Selection] Agent result: {agent_result}")

                    # Extract response
                    english_response = agent_result.get("response", "")

                    # Update language context if language was selected
                    if agent_result.get("language_selected"):
                        detected_lang = agent_result.get("detected_language")
                        detected_dialect = agent_result.get("detected_dialect")

                        logger.info(f"[Language Selection] Updating context: {detected_lang}-{detected_dialect}")

                        # Update language context
                        global_state = self.state_manager.get_global_state(session_id)
                        language_context = global_state.language_context
                        language_context.mark_language_selected(detected_lang, detected_dialect)

                        self.state_manager.update_global_state(
                            session_id,
                            language_context=language_context
                        )

                        logger.info(f"[Language Selection] Preference stored: {detected_lang}-{detected_dialect}")

                except Exception as e:
                    logger.error(f"[Language Selection] Error: {e}")
                    english_response = "Welcome to our dental clinic! How can I help you today?"
            else:

                # Call agent activation hook (for state setup)
                activation_start = time.time()
                logger.info(f"Activating agent: {agent_name}")
                with obs_logger.pipeline_step(7, "agent_activation", "agent", {"agent_name": agent_name}) if obs_logger else nullcontext():
                    if hasattr(agent, 'on_activated'):
                        await agent.on_activated(session_id, reasoning)

                    # Build comprehensive context for agent
                    if hasattr(agent, 'set_context'):
                        # Get continuation context from state manager
                        continuation_context = self.state_manager.get_continuation_context(session_id)
                        
                        # Get language context from global state
                        global_state = self.state_manager.get_global_state(session_id)
                        language_context = global_state.language_context
                        
                        # Extract task context using helper method
                        task_context = self._extract_task_context(reasoning, continuation_context)
                        
                        # Build agent context using helper method
                        agent_context = self._build_agent_context(
                            session_id,
                            task_context,
                            reasoning,
                            language_context,
                            continuation_context,
                            plan_action,  # NEW: Phase 3
                            existing_plan  # NEW: Phase 3
                        )
                        
                        logger.info(f"Context for agent: success_criteria={len(agent_context.get('success_criteria', []))}, entities={list(agent_context.get('entities', {}).keys())}")
                        agent.set_context(session_id, agent_context)
                activation_duration = (time.time() - activation_start) * 1000
                logger.info(f"Agent activation completed in {activation_duration:.2f}ms")

                # Check if we're transitioning from information collection to agent execution
                # Get continuation context to check for state transition
                transition_context = self.state_manager.get_continuation_context(session_id)
                if transition_context and transition_context.get('awaiting') == 'information':
                    logger.info("⚡ [State Transition] Moving from information collection to agent execution")
                    logger.info("⚡ [State Transition] Agent._think() will decide if collection is complete or needs continuation")
                    # Don't clear here - let agent._think() decide
                    # Agent will either:
                    # 1. Return COLLECT_INFORMATION with updated information_needed (continue)
                    # 2. Return EXECUTE_TOOL with collected information (complete)
                    # 3. Return something else (abort collection)

                # Execute agent with logging - PASS execution_log
                execution_start = time.time()
                logger.info(f"Executing agent: {agent_name} with message: {english_message[:200]}...")
                with obs_logger.pipeline_step(8, "agent_execution", "agent", {"agent_name": agent_name, "message": english_message[:100]}) if obs_logger else nullcontext():
                    english_response, execution_log = await agent.process_message_with_log(
                        session_id,
                        english_message,
                        execution_log  # Pass log to agent (agent will append tools)
                    )
                execution_duration = (time.time() - execution_start) * 1000
                logger.info(f"Agent execution completed in {execution_duration:.2f}ms")
                logger.info(f"Agent response preview: {english_response[:200]}...")
                logger.info(f"Tools used: {len(execution_log.tools_used)}")
                
                # Log token usage if available from observability
                if obs_logger and hasattr(obs_logger, 'agent_execution') and obs_logger.agent_execution:
                    agent_exec = obs_logger.agent_execution
                    if agent_exec.total_tokens:
                        logger.info(f"Token usage - Input: {agent_exec.total_tokens.input_tokens}, Output: {agent_exec.total_tokens.output_tokens}, Total: {agent_exec.total_tokens.total_tokens}")
                    if agent_exec.total_cost:
                        logger.info(f"Cost: ${agent_exec.total_cost.total_cost:.4f}")
                
                # Monitor execution log size after agent execution
                tool_count = len(execution_log.tools_used)
                if tool_count > 100:
                    logger.warning(
                        f"Execution log for session {session_id} has {tool_count} tools. "
                        f"Consider investigating potential loops or excessive tool calls."
                    )
                elif tool_count > 50:
                    logger.info(
                        f"Execution log for session {session_id} has {tool_count} tools "
                        f"(monitoring for potential issues)"
                    )
            
            step7_duration = (time.time() - step7_start) * 1000
            logger.info(f"Steps 7-8 completed in {step7_duration:.2f}ms")

            # Step 8: Validate response (CLOSED-LOOP) - ONLY IF ENABLED
            step9_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 9: Validation")
            logger.info("-" * 100)

            if config_settings.enable_validation:
                logger.info(f"Validation enabled - validating response from {agent_name}")
                logger.info(f"Response preview: {english_response[:200]}...")
                logger.info(f"Tools used: {len(execution_log.tools_used)}")
                with obs_logger.pipeline_step(9, "validation", "reasoning_engine", {"response_preview": english_response[:100]}) if obs_logger else nullcontext():
                    validation = await self.reasoning_engine.validate_response(
                        session_id=session_id,
                        original_reasoning=reasoning,
                        agent_response=english_response,
                        execution_log=execution_log
                    )
                    if obs_logger:
                        obs_logger.record_pipeline_step(
                            9, "validation", "reasoning_engine",
                            inputs={"response_preview": english_response[:100]},
                            outputs={
                                "is_valid": validation.is_valid,
                                "decision": validation.decision,
                                "confidence": validation.confidence
                            }
                        )

                logger.info(f"Validation result: valid={validation.is_valid}, "
                           f"decision={validation.decision}, "
                           f"confidence={validation.confidence}")
                if not validation.is_valid:
                    logger.info(f"Validation issues: {validation.issues}")
                    logger.info(f"Validation feedback: {validation.feedback_to_agent}")
            else:
                # Validation disabled - create a pass-through validation result
                logger.info("Validation layer DISABLED - skipping validation")
                validation = ValidationResult(
                    is_valid=True,
                    confidence=1.0,
                    decision="send",
                    issues=[],
                    reasoning=["Validation layer disabled in config"]
                )
            step9_duration = (time.time() - step9_start) * 1000
            logger.info(f"Step 9 completed in {step9_duration:.2f}ms")

            # Step 9: Handle validation result (retry loop) - ONLY IF VALIDATION ENABLED
            step10_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 10: Validation Retry Loop")
            logger.info("-" * 100)

            max_retries = config_settings.validation_max_retries if config_settings.enable_validation else 0
            retry_count = 0
            logger.info(f"Max retries allowed: {max_retries}")

            while not validation.is_valid and retry_count < max_retries and config_settings.enable_validation:
                if validation.decision == "retry":
                    # Store tool count before retry for validation
                    tools_before_retry = len(execution_log.tools_used)
                    logger.info(
                        f"RETRY {retry_count + 1}/{max_retries}: Retrying with feedback (current tools: {tools_before_retry}): "
                        f"{validation.feedback_to_agent}"
                    )
                    logger.debug(f"Execution log before retry: {len(execution_log.tools_used)} tools")
                    
                    with obs_logger.pipeline_step(10, "agent_retry", "agent", {"retry_count": retry_count + 1}) if obs_logger else nullcontext():
                        # Retry with specific feedback - PASS SAME execution_log
                        # Tools from first attempt are preserved, new tools will be appended
                        english_response, execution_log = await agent.process_message_with_log(
                            session_id,
                            f"[VALIDATION FEEDBACK]: {validation.feedback_to_agent}\n\n"
                            f"Original user request: {english_message}",
                            execution_log  # Same log - tools accumulate across retries
                        )
                        
                        # Validate tools accumulated (not replaced)
                        tools_after_retry = len(execution_log.tools_used)
                        logger.info(f"Retry complete (tools now: {tools_after_retry}, was: {tools_before_retry})")
                        logger.debug(f"Execution log after retry: {len(execution_log.tools_used)} tools")
                        
                        # Assertion: tools should only increase, never decrease
                        # This catches regressions where log is replaced instead of appended
                        assert tools_after_retry >= tools_before_retry, (
                            f"Execution log tools decreased during retry: "
                            f"{tools_before_retry} -> {tools_after_retry}. "
                            f"This indicates execution_log was replaced instead of appended!"
                        )

                        # Re-validate with accumulated execution_log
                        retry_validation_start = time.time()
                        validation = await self.reasoning_engine.validate_response(
                            session_id,
                            reasoning,
                            english_response,
                            execution_log  # Contains tools from ALL attempts
                        )
                        retry_validation_duration = (time.time() - retry_validation_start) * 1000
                        retry_count += 1
                        logger.info(f"Retry {retry_count} validation completed in {retry_validation_duration:.2f}ms")
                        logger.info(f"Validation result after retry: valid={validation.is_valid}, decision={validation.decision}")
                        
                        if obs_logger:
                            obs_logger._validation_details.retry_count = retry_count

                elif validation.decision == "redirect":
                    # Could try different agent, but for MVP just break
                    logger.warning(f"Validation suggests redirect, but using current response")
                    break
                else:
                    break
            
            # Log retry step completion
            if obs_logger and (not config_settings.enable_validation or retry_count == 0):
                obs_logger.record_pipeline_step(
                    10, "validation_retry", "reasoning_engine",
                    metadata={"status": "disabled" if not config_settings.enable_validation else "no_retry_needed", "retry_count": retry_count}
                )
            
            step10_duration = (time.time() - step10_start) * 1000
            logger.info(f"Step 10 completed in {step10_duration:.2f}ms (retries: {retry_count})")

            # LAYER 2: Finalization (final quality check) - ONLY IF ENABLED
            step11_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 11: Finalization")
            logger.info("-" * 100)
            
            if config_settings.enable_finalization:
                logger.info(f"Finalization enabled - processing response from {agent_name}")
                logger.info(f"Response preview: {english_response[:200]}...")
                with obs_logger.pipeline_step(11, "finalization", "reasoning_engine", {"response_preview": english_response[:100]}) if obs_logger else nullcontext():
                    finalization = await self.reasoning_engine.finalize_response(
                        session_id=session_id,
                        original_reasoning=reasoning,
                        agent_response=english_response,
                        execution_log=execution_log,
                        validation_result=validation
                    )
                    if obs_logger:
                        obs_logger.record_pipeline_step(
                            11, "finalization", "reasoning_engine",
                            inputs={"response_preview": english_response[:100]},
                            outputs={
                                "decision": finalization.decision,
                                "was_rewritten": finalization.was_rewritten,
                                "confidence": finalization.confidence
                            }
                        )

                logger.info(f"Finalization result: decision={finalization.decision}, "
                           f"edited={finalization.was_rewritten}, "
                           f"confidence={finalization.confidence}")

                # Use finalized response
                if finalization.should_use_rewritten():
                    logger.info(f"Using edited response from finalization layer")
                    logger.info(f"Original: {english_response[:200]}...")
                    english_response = finalization.rewritten_response
                    logger.info(f"Rewritten: {english_response[:200]}...")
                elif finalization.should_fallback():
                    english_response = self._get_validation_fallback(finalization.issues)
                    logger.warning(f"Finalization triggered fallback: {finalization.issues}")
                else:
                    logger.info("Finalization approved agent's response")
            else:
                # Finalization disabled - skip
                logger.info("Finalization layer DISABLED - using agent response as-is")
                finalization = None
            
            step11_duration = (time.time() - step11_start) * 1000
            logger.info(f"Step 11 completed in {step11_duration:.2f}ms")

            # Step 12: Add assistant response to conversation memory (English version)
            step12_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 12: Add Assistant Response to ConversationMemory (English)")
            logger.info("-" * 100)
            with obs_logger.pipeline_step(12, "add_assistant_to_memory", "memory_manager", {"response_preview": english_response[:100]}) if obs_logger else nullcontext():
                # Store in ConversationMemory (English - for Unified Reasoning & _think())
                # CRITICAL: Always use suggested_response from execution_log (pre-humanization English from _think())
                # This ensures ConversationMemory stores English, not translated/humanized responses
                if execution_log.suggested_response:
                    response_for_conv_memory = execution_log.suggested_response
                    logger.info(f"📝 [ConversationMemory] Using suggested_response from _think() (English): {response_for_conv_memory[:80]}...")
                else:
                    # Fallback: If suggested_response is not available, use english_response
                    # NOTE: This should rarely happen - suggested_response should always be set by _think()
                    # If it's missing, english_response might be in user's language (humanized/generated)
                    # Log a warning to help debug this edge case
                    response_for_conv_memory = english_response
                    logger.warning(f"⚠️ [ConversationMemory] suggested_response not found in execution_log! Using english_response as fallback: {response_for_conv_memory[:80]}...")
                    logger.warning(f"⚠️ [ConversationMemory] This may store a non-English response if humanizer/generator was used!")
                
                self.memory_manager.add_assistant_turn(session_id, response_for_conv_memory)
                logger.info(f"📝 [ConversationMemory] Stored response (English): {response_for_conv_memory[:80]}...")
            step12_duration = (time.time() - step12_start) * 1000
            logger.info(f"Step 12 completed in {step12_duration:.2f}ms")

            # Step 13: Translation (output) - Configurable via feature flag
            step13_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 13: Translation (Output)")
            logger.info("-" * 100)
            from patient_ai_service.core.feature_flags import is_output_translation_enabled
            if is_output_translation_enabled(session_id):
                with obs_logger.pipeline_step(13, "translation_output", "translation", {"status": "enabled", "language": language_context.current_language}) if obs_logger else nullcontext():
                    logger.info(f"TRANSLATION OUTPUT ENABLED - Translating response to {language_context.current_language}")
                    translated_response = await self._translate_output(
                        session_id,
                        english_response,
                        language_context.current_language
                    )
                    logger.info(f"Translated response ({language_context.current_language}): {translated_response[:200]}...")
            else:
                with obs_logger.pipeline_step(13, "translation_output", "translation", {"status": "disabled", "language": language_context.current_language}) if obs_logger else nullcontext():
                    logger.info("TRANSLATION OUTPUT DISABLED - Using agent response as-is")
                    # Use agent response as-is (already in target language)
                    translated_response = english_response
                    logger.info(f"Response used as-is ({language_context.current_language}): {translated_response[:200]}...")
            
            # Store translated response in NativeLanguageMemory (AFTER translation)
            with obs_logger.pipeline_step(13, "add_translated_to_native_memory", "memory_manager", {"response_preview": translated_response[:100]}) if obs_logger else nullcontext():
                self.native_memory_manager.add_assistant_turn(
                    session_id=session_id,
                    content=translated_response  # Translated response in user's language
                )
                logger.info(f"📝 [NativeLanguageMemory] Stored translated response ({language_context.current_language}): {translated_response[:80]}...")
            
            step13_duration = (time.time() - step13_start) * 1000
            logger.info(f"Step 13 completed in {step13_duration:.2f}ms")

            # Step 14: Build response
            with obs_logger.pipeline_step(14, "build_response", "orchestrator", {}) if obs_logger else nullcontext():
                response = ChatResponse(
                    response=translated_response,
                    session_id=session_id,
                    detected_language=language_context.get_full_language_code(),
                    intent=reasoning.understanding.what_user_means,
                    urgency=reasoning.routing.urgency,
                    metadata={
                        "agent": agent_name,
                        "sentiment": reasoning.understanding.sentiment,
                        "reasoning_summary": reasoning.reasoning_chain[0] if reasoning.reasoning_chain else "",
                        "language_context": {
                            "language": language_context.current_language,
                            "dialect": language_context.current_dialect,
                            "full_code": language_context.get_full_language_code(),
                            "switched": len(language_context.language_history) > 0
                        },
                        "validation": {
                            "passed": validation.is_valid,
                            "retries": retry_count,
                            "confidence": validation.confidence,
                            "issues": validation.issues if not validation.is_valid else []
                        },
                        "finalization": {
                            "decision": finalization.decision if finalization else "disabled",
                            "was_edited": finalization.was_rewritten if finalization else False,
                            "confidence": finalization.confidence if finalization else None,
                            "issues": finalization.issues if finalization else []
                        } if finalization is not None else {"enabled": False}
                    }
                )

            logger.info(f"Response generated by {agent_name}")
            
            # Log final response summary
            pipeline_duration_seconds = time.time() - pipeline_start_time
            logger.info("=" * 100)
            logger.info("ORCHESTRATOR: process_message() COMPLETED")
            logger.info("=" * 100)
            logger.info(f"Session ID: {session_id}")
            logger.info(f"Final Response: {translated_response[:200]}...")
            logger.info(f"Agent Used: {agent_name}")
            logger.info(f"Total Pipeline Duration: {pipeline_duration_seconds:.3f}s")
            logger.info(f"Tools Used: {len(execution_log.tools_used)}")
            logger.info("=" * 100)
            
            # Optional: Persist execution_log for debugging/auditing (if enabled)
            # Note: This is optional and should not fail the pipeline if it fails
            if getattr(config_settings, 'persist_execution_logs', False):
                try:
                    self.state_manager.save_execution_log(session_id, execution_log)
                    logger.debug(
                        f"Persisted execution_log for session {session_id} "
                        f"({len(execution_log.tools_used)} tools)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist execution_log for session {session_id}: {e}")
            
            # Log observability summary
            if obs_logger:
                pipeline_duration_seconds = time.time() - pipeline_start_time
                obs_logger.record_pipeline_step(
                    15, "pipeline_complete", "orchestrator",
                    metadata={"total_duration_seconds": pipeline_duration_seconds}
                )
                obs_logger.log_summary()
                
                # Reset observability logger for next request (preserves accumulative cost)
                clear_observability_logger(session_id)

            return response

        except Exception as e:
            pipeline_duration_seconds = time.time() - pipeline_start_time
            logger.error(f"Error in orchestrator: {e}", exc_info=True)
            
            # Reset observability logger even on error
            if obs_logger:
                clear_observability_logger(session_id)
            logger.info("=" * 100)
            logger.info("ORCHESTRATOR: process_message() FAILED")
            logger.info("=" * 100)
            logger.info(f"Session ID: {session_id}")
            logger.info(f"Error: {str(e)}")
            logger.info(f"Pipeline Duration Before Error: {pipeline_duration_seconds:.3f}s")
            logger.info("=" * 100)
            
            if obs_logger:
                obs_logger.record_pipeline_step(
                    999, "error", "orchestrator",
                    error=str(e)
                )

            # Return error response
            return ChatResponse(
                response="I apologize, but I encountered an error. Please try again or contact support.",
                session_id=session_id,
                detected_language=language or "en",
                metadata={"error": str(e)}
            )

    def _get_validation_fallback(self, issues: List[str]) -> str:
        """
        Generate safe fallback response when validation fails.

        Args:
            issues: List of validation issues detected

        Returns:
            Safe fallback response string
        """
        return (
            "I want to make sure I give you accurate information. "
            "Let me connect you with a team member who can help you with this request."
        )

    async def _ensure_patient_loaded(self, session_id: str):
        """Ensure patient data is loaded in state."""
        try:
            logger.info(f"🔍 _ensure_patient_loaded called for session: {session_id}")
            global_state = self.state_manager.get_global_state(session_id)

            # If patient already loaded, return
            if global_state.patient_profile.patient_id:
                logger.info(f"✅ Patient already loaded in state: {global_state.patient_profile.patient_id}")
                return
            
            logger.info(f"⚠️ Patient not in state, attempting to load from DB...")

            # Extract phone from session_id
            # Format: "clinic:{clinic_id}:session:{phone_number}" OR legacy plain phone number
            phone_number = None
            if session_id.startswith("clinic:") and ":session:" in session_id:
                # Extract phone number from composite key: clinic:xxx:session:{phone}
                phone_number = session_id.split(":session:")[-1]
            elif session_id.startswith("+") or session_id.isdigit():
                # Legacy format: plain phone number
                phone_number = session_id

            if phone_number:
                patient = self.db_client.get_patient_by_phone_number(phone_number)

                if patient:
                    # Load patient data into state
                    self.state_manager.update_patient_profile(
                        session_id,
                        patient_id=patient.get("id"),
                        user_id=patient.get("userId"),
                        first_name=patient.get("first_name"),
                        last_name=patient.get("last_name"),
                        phone=patient.get("phone_number") or phone_number,
                        email=patient.get("email"),
                        date_of_birth=patient.get("date_of_birth"),
                        preferred_language=patient.get("user", {}).get("languagePreference", "en")
                        if patient.get("user") else "en",
                        allergies=patient.get("allergies", []),
                        medications=patient.get("medications", []),
                    )

                    logger.info(f"Loaded patient: {patient.get('id')} for session: {session_id}")
                else:
                    logger.info(f"New user detected for session: {session_id}")
                    # Patient not found - they might need to register
                    # Store phone for later use
                    self.state_manager.update_patient_profile(
                        session_id,
                        phone=phone_number
                    )

        except Exception as e:
            logger.error(f"Error loading patient: {e}")

    # =========================================================================
    # STEP IMPLEMENTATIONS
    # =========================================================================
    
    async def _translate_input(
        self,
        session_id: str,
        message: str,
        language_hint: Optional[str]
    ) -> Tuple[str, str]:
        """Translate input to English and detect language."""
        translation_agent = self.agents["translation"]
        
        # Use existing translation logic
        english_message, detected_lang, detected_dialect, translation_succeeded = await translation_agent.detect_and_translate(
            message, session_id
        )
        
        # Update global state with detected language
        global_state = self.state_manager.get_global_state(session_id)
        language_context = global_state.language_context
        
        if language_context.current_language != detected_lang:
            language_context.record_language_switch(
                detected_lang,
                detected_dialect,
                language_context.turn_count
            )
        else:
            language_context.current_language = detected_lang
            language_context.current_dialect = detected_dialect
            language_context.last_detected_at = datetime.utcnow()
        
        language_context.turn_count += 1
        self.state_manager.update_global_state(
            session_id,
            language_context=language_context
        )
        
        return english_message, detected_lang
    
    async def _translate_output(
        self,
        session_id: str,
        response: str,
        target_language: str
    ) -> str:
        """Translate output to user's language."""
        if target_language == "en":
            return response
        
        translation_agent = self.agents["translation"]
        global_state = self.state_manager.get_global_state(session_id)
        language_context = global_state.language_context
        
        return await translation_agent.translate_from_english_with_dialect(
            response,
            target_language,
            language_context.current_dialect,
            session_id
        )
    
    
    # =========================================================================
    # METRICS & MONITORING
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        total = self._request_count or 1
        return {
            "total_requests": self._request_count,
            "focused_requests": self._focused_count,
            "comprehensive_requests": self._comprehensive_count,
            "focused_rate": self._focused_count / total,
            "comprehensive_rate": self._comprehensive_count / total
        }

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get complete session state including conversation memory."""
        state = self.state_manager.export_session(session_id)

        # Add conversation memory context
        memory = self.memory_manager.get_memory(session_id)
        state["conversation_memory"] = {
            "user_facts": memory.user_facts,
            "summary": memory.summary,
            "recent_turns_count": len(memory.recent_turns),
            "last_action": memory.last_action,
            "awaiting": memory.awaiting,
            "turn_count": memory.turn_count,
        }

        return state

    def clear_session(self, session_id: str):
        """Clear session state and conversation memory."""
        self.state_manager.clear_session(session_id)
        self.memory_manager.clear_session(session_id)
        logger.info(f"Session cleared: {session_id}")

    # =============================================================================
    # HELPER METHODS FOR AGENTIC ARCHITECTURE
    # =============================================================================

    def _extract_task_context(
        self,
        reasoning: 'ReasoningOutput',
        continuation_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract task context from reasoning output.

        Merges with continuation context if resuming.
        """
        from patient_ai_service.core.reasoning import ReasoningOutput

        # Get from reasoning
        if hasattr(reasoning.response_guidance, 'task_context'):
            tc = reasoning.response_guidance.task_context
            # Extract situation_type from minimal_context
            situation_type = None
            if hasattr(reasoning.response_guidance, 'minimal_context'):
                situation_type = reasoning.response_guidance.minimal_context.get("situation_type")
            
            task_context = {
                "user_intent": tc.user_intent if hasattr(tc, 'user_intent') else reasoning.understanding.what_user_means,
                "objective": tc.objective if hasattr(tc, 'objective') else "",  # NEW: Extract objective
                "entities": tc.entities if hasattr(tc, 'entities') else {},
                "success_criteria": tc.success_criteria if hasattr(tc, 'success_criteria') else [],
                "constraints": tc.constraints if hasattr(tc, 'constraints') else [],
                "prior_context": tc.prior_context if hasattr(tc, 'prior_context') else None,
                "is_continuation": tc.is_continuation if hasattr(tc, 'is_continuation') else False,
                "continuation_type": tc.continuation_type if hasattr(tc, 'continuation_type') else None,
                "selected_option": tc.selected_option if hasattr(tc, 'selected_option') else None,
                "situation_type": situation_type,  # NEW: Add situation_type from unified reasoning
            }
            logger.info(f"{task_context}")

            # Log entities extracted from reasoning
            entities_from_reasoning = task_context["entities"]
            if entities_from_reasoning:
                logger.info(f"📊 Entities extracted from reasoning engine: {json.dumps(entities_from_reasoning, default=str)}")
            else:
                logger.info("📊 No entities extracted from reasoning engine")
        else:
            # Fallback to minimal_context
            mc = reasoning.response_guidance.minimal_context
            task_context = {
                "user_intent": mc.get("what_user_means", reasoning.understanding.what_user_means),
                "objective": "",  # NEW: Empty objective if no task_context available
                "entities": {},
                "success_criteria": [],
                "constraints": [],
                "prior_context": mc.get("prior_context"),
                "is_continuation": mc.get("is_continuation", False),
                "situation_type": mc.get("situation_type"),  # NEW: Add situation_type from minimal_context
            }
            logger.info("📊 Using minimal_context (no task_context available)")

        # Merge with continuation context if resuming
        if continuation_context and task_context.get("is_continuation"):
            resolved = continuation_context.get("entities", {})
            if resolved:
                logger.info(f"🔄 Merging resolved entities from continuation: {json.dumps(resolved, default=str)}")
                merged_count = 0
                for key, value in resolved.items():
                    if key not in task_context["entities"]:
                        task_context["entities"][key] = value
                        merged_count += 1
                logger.info(f"🔄 Merged {merged_count} entities from continuation context")

            # Use same success criteria if resuming
            if not task_context["success_criteria"]:
                blocked = continuation_context.get("blocked_criteria", [])
                if blocked:
                    task_context["success_criteria"] = blocked

            # Store continuation context for agent
            task_context["continuation_context"] = continuation_context

        # Log final entities after merging
        final_entities = task_context["entities"]
        logger.info(f"📊 Final task context entities (after merge): {json.dumps(final_entities, default=str)}")

        return task_context

    def _get_recent_turns(self, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
        """Get recent conversation turns for context."""
        memory = self.memory_manager.get_memory(session_id)
        if not memory.recent_turns:
            return []
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat() if hasattr(turn, 'timestamp') and turn.timestamp else None
            }
            for turn in memory.recent_turns[-limit:]
        ]

    def _build_reasoning_from_unified(
        self,
        session_id: str,
        unified_output: UnifiedReasoningOutput,
        existing_plan: Optional[AgentPlan],
        continuation_context: Optional[Dict[str, Any]]
    ) -> 'ReasoningOutput':
        """
        Build ReasoningOutput-compatible object from UnifiedReasoningOutput.

        This allows agents to continue using their existing on_activated() signature
        while we transition to unified reasoning.
        """
        from patient_ai_service.core.reasoning import (
            ReasoningOutput,
            UnderstandingResult,
            RoutingResult,
            MemoryUpdate,
            ResponseGuidance,
            TaskContext
        )

        # Build understanding
        understanding = UnderstandingResult(
            what_user_means=unified_output.what_user_means or "",
            is_continuation=unified_output.is_continuation,
            continuation_type=unified_output.continuation_type,
            sentiment="neutral"
        )

        # Build routing - USE routing_action IF PROVIDED (takes precedence)
        routing = RoutingResult(
            agent=unified_output.agent or "general_assistant",
            action=unified_output.routing_action or unified_output.continuation_type or "new_request",  # CHANGED
            urgency="emergency" if unified_output.situation_type == SituationType.EMERGENCY else "routine"
        )

        # Build task context
        # For resume, use existing plan's objective
        if unified_output.plan_decision == PlanDecision.RESUME and existing_plan:
            objective = existing_plan.objective
            entities = existing_plan.entities.copy() if existing_plan.entities else {}
        else:
            objective = unified_output.objective or unified_output.what_user_means or ""
            entities = {}

        # Merge entities from continuation context if available
        # Restore multi-source entities from continuation context
        if continuation_context:
            resolved = continuation_context.get("entities", {})
            for k, v in resolved.items():
                if k not in entities:
                    entities[k] = v

        # NEW: Load LLM entities from GlobalState (persistent session memory)
        llm_entities = self.state_manager.get_llm_entities(session_id)
        logger.info(f"🔒 Loaded LLM entities from GlobalState: {list(llm_entities.keys())}")

        task_context = TaskContext(
            user_intent=unified_output.what_user_means or "",
            objective=objective,
            entities=entities,  # Internal only
            llm_entities=llm_entities,  # Persistent session memory
            is_continuation=unified_output.is_continuation,
            continuation_type=unified_output.continuation_type
        )

        response_guidance = ResponseGuidance(
            tone="helpful",
            task_context=task_context,
            minimal_context={
                "situation_type": unified_output.situation_type.value if unified_output.situation_type else None
            }
        )

        return ReasoningOutput(
            understanding=understanding,
            routing=routing,
            memory_updates=MemoryUpdate(),
            response_guidance=response_guidance,
            reasoning_chain=[f"Unified reasoning: {unified_output.situation_type.value}"]
        )

    def _build_agent_context(
        self,
        session_id: str,
        task_context: Dict[str, Any],
        reasoning: 'ReasoningOutput',
        language_context: Any,
        continuation_context: Optional[Dict[str, Any]],
        plan_action: Optional[PlanAction] = None,  # NEW: Phase 3
        existing_plan: Optional[AgentPlan] = None  # NEW: Phase 3
    ) -> Dict[str, Any]:
        """
        Build the context dict to pass to the agent.
        Injects critical parameters like patient_id to prevent hallucinations.
        
        NEW (Phase 3): Includes plan_action and existing_plan for plan-based execution.
        """
        # Get entities from task context (conversation entities - global)
        entities = task_context.get("entities", {}).copy()
        logger.info(f"📊 Conversation entities: {json.dumps(entities, default=str)}")

        # Get agent_name from reasoning routing
        agent_name = reasoning.routing.agent if hasattr(reasoning.routing, 'agent') else None

        # ✅ NEW: Add AGENT-SCOPED derived entities
        entity_state = self.state_manager.get_entity_state(session_id)
        if entity_state and agent_name:
            derived_added = 0
            derived_skipped_other_agent = 0
            derived_skipped_invalid = 0

            for key, derived_entity in entity_state.derived.entities.items():
                # ✅ CRITICAL: Only include derived entities from THIS agent
                if derived_entity.agent_name != agent_name:
                    derived_skipped_other_agent += 1
                    logger.debug(
                        f"⏭️ Skipped derived entity '{key}' from other agent '{derived_entity.agent_name}'"
                    )
                    continue

                # Check if valid
                if not derived_entity.is_valid():
                    derived_skipped_invalid += 1
                    logger.debug(
                        f"⚠️ Skipped invalid derived entity '{key}': "
                        f"{derived_entity.invalidation_reason or 'expired'}"
                    )
                    continue

                # Add to entities (don't overwrite conversation entities)
                if key not in entities:
                    entities[key] = derived_entity.value
                    derived_added += 1
                    logger.info(
                        f"✅ Added derived entity '{key}' from {derived_entity.source_tool} "
                        f"(age: {derived_entity.age_display()})"
                    )

            # Summary log
            logger.info(
                f"📊 Agent {agent_name} context: "
                f"{len(entities)} total entities "
                f"(derived: +{derived_added}, "
                f"skipped_other_agents: {derived_skipped_other_agent}, "
                f"skipped_invalid: {derived_skipped_invalid})"
            )

        # CRITICAL: Inject patient_id from global state to prevent hallucinations
        # The LLM should not have to extract this from the system prompt
        entities_added = []
        try:
            global_state = self.state_manager.get_global_state(session_id)
            if global_state and global_state.patient_profile:
                patient_id = global_state.patient_profile.patient_id
                if patient_id and patient_id.strip():
                    if "patient_id" not in entities:
                        entities["patient_id"] = patient_id
                        entities_added.append(f"patient_id={patient_id}")
                        logger.info(f"✅ Injected patient_id into agent context: {patient_id}")
                    else:
                        logger.info(f"📌 patient_id already in entities: {patient_id} (from reasoning)")
                else:
                    logger.warning("⚠️ Patient ID not available in global state - agent may need to prompt for registration")
        except Exception as e:
            logger.error(f"Error injecting patient_id into context: {e}")

        # Log final enhanced entities
        if entities_added:
            logger.info(f"📊 Enhanced entities with {len(entities_added)} injection(s): {', '.join(entities_added)}")
        logger.info(f"📊 Final agent context entities: {json.dumps(entities, default=str)}")

        # Build ISOLATED LLM entities from GlobalState
        llm_entities = self.state_manager.get_llm_entities(session_id)

        # Inject patient_id if not present
        try:
            global_state = self.state_manager.get_global_state(session_id)
            if global_state and global_state.patient_profile:
                patient_id = global_state.patient_profile.patient_id
                if patient_id and patient_id.strip():
                    if "patient_id" not in llm_entities:
                        # Add to session and save
                        llm_entities["patient_id"] = patient_id
                        self.state_manager.update_llm_entities(session_id, {"patient_id": patient_id})
                        logger.info(f"🔒 Injected patient_id into global llm_entities")
        except Exception as e:
            logger.error(f"Error injecting patient_id: {e}")

        logger.info(
            f"🔒 LLM entities (from GlobalState): {list(llm_entities.keys())}"
        )
        logger.info(
            f"📊 Internal entities (multi-source): {len(entities)} items"
        )

        # Calculate entity statistics
        total_entities = len(entities)
        # Estimate conversation entities (non-derived keys)
        conversation_entities = sum(
            1 for k in entities
            if not k.endswith('_uuid') and not k.endswith('_id') and k != 'available_slots'
        )
        derived_entities_count = total_entities - conversation_entities

        # Log entity statistics
        logger.info(
            f"📊 Entity stats: total={total_entities}, "
            f"conversation={conversation_entities}, derived={derived_entities_count}"
        )
        
        # Track with metrics
        try:
            from patient_ai_service.core.observability import (
                entity_count,
                derived_entity_count
            )
            entity_count.observe(total_entities)
            derived_entity_count.observe(derived_entities_count)
        except Exception:
            pass  # Don't fail on metrics errors

        context = {
            # Task context
            "user_intent": task_context.get("user_intent", ""),
            "objective": task_context.get("objective", ""),  # NEW: Include objective
            "entities": entities,  # Multi-source (internal only, NEVER shown to LLM)
            "llm_entities": llm_entities,  # NEW: ISOLATED (shown in prompts)
            "success_criteria": task_context.get("success_criteria", []),
            "constraints": task_context.get("constraints", []),
            "prior_context": task_context.get("prior_context"),

            # Continuation info
            "is_continuation": task_context.get("is_continuation", False),
            "continuation_type": task_context.get("continuation_type"),
            "selected_option": task_context.get("selected_option"),
            "continuation_context": continuation_context or {},
            
            # NEW: Unified reasoning output fields
            "situation_type": task_context.get("situation_type"),  # From unified reasoning

            # Routing info
            "routing_action": reasoning.routing.action,
            "routing_urgency": reasoning.routing.urgency,

            # Language
            "current_language": language_context.current_language,
            "current_dialect": language_context.current_dialect,

            # Response guidance
            "tone": reasoning.response_guidance.tone,

            # Backward compatibility - use what_user_means from understanding
            "what_user_means": reasoning.understanding.what_user_means,
            "action": reasoning.routing.action,
            
            # NEW: Phase 3 - Agent Plan Management
            "plan_action": plan_action.value if plan_action else "create_new",
            "existing_plan": existing_plan.model_dump() if existing_plan else None,
        }

        return context

    async def _handle_agentic_completion(
        self,
        session_id: str,
        agentic_state: 'AgenticExecutionState',
        reasoning: 'ReasoningOutput',
        response: str,
        execution_log: 'ExecutionLog',
        config_settings: Any
    ) -> 'ValidationResult':
        """
        Handle different agentic completion states.
        
        Returns appropriate ValidationResult based on state.
        """
        from patient_ai_service.models.agentic import AgenticExecutionState
        from patient_ai_service.models.validation import ValidationResult
        
        status = agentic_state.status
        
        if status == "complete":
            # Task completed successfully
            logger.info("✅ Agentic task completed successfully")
            
            # Clear any continuation context
            self.state_manager.clear_continuation_context(session_id)
            
            return ValidationResult(
                is_valid=True,
                confidence=0.95,
                decision="send",
                issues=[],
                reasoning=[f"Task completed in {agentic_state.iteration} iterations"]
            )
        
        elif status == "blocked":
            # Task blocked - waiting for user input
            logger.info("⏸️ Agentic task blocked - awaiting user input")
            
            # Continuation context should already be set by agent
            # Just verify it exists
            if not self.state_manager.has_continuation(session_id):
                logger.warning("Blocked status but no continuation context!")
            
            return ValidationResult(
                is_valid=True,  # Response is valid (presenting options)
                confidence=0.9,
                decision="send",
                issues=[],
                reasoning=["Task blocked awaiting user input"]
            )
        
        elif status == "failed":
            # Task failed
            logger.warning(f"❌ Agentic task failed: {agentic_state.failure_reason}")
            
            return ValidationResult(
                is_valid=True,  # Failure response is valid
                confidence=0.7,
                decision="send",
                issues=[agentic_state.failure_reason or "Task failed"],
                reasoning=["Task could not be completed"]
            )
        
        elif status == "max_iterations":
            # Hit max iterations
            logger.warning(f"⚠️ Max iterations reached ({agentic_state.max_iterations})")
            
            # Run validation to check response quality
            if config_settings.enable_validation:
                validation = await self.reasoning_engine.validate_response(
                    session_id=session_id,
                    original_reasoning=reasoning,
                    agent_response=response,
                    execution_log=execution_log
                )
                return validation
            
            return ValidationResult(
                is_valid=True,
                confidence=0.6,
                decision="send",
                issues=["Max iterations reached"],
                reasoning=["Task incomplete due to iteration limit"]
            )
        
        else:
            # Unknown or in_progress status - run validation
            if config_settings.enable_validation:
                validation = await self.reasoning_engine.validate_response(
                    session_id=session_id,
                    original_reasoning=reasoning,
                    agent_response=response,
                    execution_log=execution_log
                )
                return validation
            
            return ValidationResult(
                is_valid=True,
                confidence=0.8,
                decision="send"
            )

    def _build_response_metadata(
        self,
        agent_name: str,
        reasoning: 'ReasoningOutput',
        validation: 'ValidationResult',
        finalization: Optional['ValidationResult'],
        agentic_summary: Dict[str, Any],
        language_context: Any
    ) -> Dict[str, Any]:
        """
        Build metadata dict for ChatResponse including agentic info.
        """
        return {
            "agent": agent_name,
            "sentiment": reasoning.understanding.sentiment,
            "is_continuation": reasoning.understanding.is_continuation,
            "reasoning_summary": reasoning.reasoning_chain[0] if reasoning.reasoning_chain else "",
            
            "language_context": {
                "language": language_context.current_language,
                "dialect": language_context.current_dialect,
                "full_code": language_context.get_full_language_code(),
            },
            
            "validation": {
                "passed": validation.is_valid,
                "confidence": validation.confidence,
                "decision": validation.decision,
            },
            
            "finalization": {
                "enabled": finalization is not None,
                "decision": finalization.decision if finalization else None,
                "was_edited": finalization.was_rewritten if finalization else False,
            } if finalization else {"enabled": False},
            
            "agentic": {
                "status": agentic_summary["status"],
                "iterations": agentic_summary["iterations"],
                "max_iterations": agentic_summary["max_iterations"],
                "criteria": agentic_summary["criteria"],
                "has_continuation": agentic_summary["has_continuation"],
                "awaiting": agentic_summary.get("awaiting"),
                "tool_calls": agentic_summary["tool_calls"],
                "llm_calls": agentic_summary["llm_calls"],
            }
        }
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
from patient_ai_service.core.reasoning import get_reasoning_engine, ReasoningEngine, ReasoningOutput
from patient_ai_service.core.conversation_memory import get_conversation_memory_manager
from patient_ai_service.core.observability import get_observability_logger, clear_observability_logger
from patient_ai_service.core.situation_assessor import (
    SituationAssessor,
    get_situation_assessor
)
from patient_ai_service.models.situation_assessment import (
    SituationAssessment,
    SituationType,
    ReasoningNeeds
)
from patient_ai_service.models.observability import TokenUsage
from patient_ai_service.core.focused_prompts import FocusedResponseGenerator

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
        self.reasoning_engine = get_reasoning_engine()
        self.memory_manager = get_conversation_memory_manager()
        self.db_client = db_client or DbOpsClient()
        
        # NEW: Situation Assessor
        self.situation_assessor = get_situation_assessor()
        self.focused_generator = FocusedResponseGenerator()

        # Initialize agents
        self._init_agents()
        
        # Metrics
        self._request_count = 0
        self._focused_count = 0
        self._comprehensive_count = 0

        logger.info("Orchestrator initialized with integrated architecture")

    def _init_agents(self):
        """Initialize all specialized agents."""
        self.agents: Dict[str, Any] = {
            "appointment_manager": AppointmentManagerAgent(
                db_client=self.db_client
            ),
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
                db_client=self.db_client
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
        session_id: Optional[str] = None
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
        name = patient_name or "there"
        
        # Build language instruction
        lang_instruction = ""
        if language == "ar":
            dialect_map = {
                "ae": "Emirati Arabic (UAE dialect)",
                "sa": "Saudi Arabic (Gulf dialect)", 
                "eg": "Egyptian Arabic",
                "lv": "Levantine Arabic",
                None: "Modern Standard Arabic"
            }
            dialect_name = dialect_map.get(dialect, dialect_map[None])
            lang_instruction = f"\n\nIMPORTANT: Respond in {dialect_name}. Use natural, conversational Arabic appropriate for the UAE."
        elif language != "en":
            lang_instruction = f"\n\nIMPORTANT: Respond in {language}."
        
        # Situation-specific context
        situation_context = {
            SituationType.GREETING: "They are greeting you. Welcome them warmly.",
            SituationType.FAREWELL: "They are saying goodbye. Wish them well.",
            SituationType.THANKS: "They are thanking you. Acknowledge graciously.",
            SituationType.PLEASANTRY: "They are making a social/polite exchange. Respond naturally and warmly.",
        }
        
        context = situation_context.get(situation, "Respond naturally and helpfully.")
        
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
                conversation_context += f"{role_label}: {turn.get('content', '')}\n"
            conversation_context += "\nUse this context to make your response more natural and relevant."
        
        prompt = f"""You are a friendly dental clinic receptionist in the UAE.

Patient: {name} ({'returning patient' if is_registered else 'new visitor'})
They said: "{message}"

{context}{registration_hint}{conversation_context}

Keep your response to 1-2 sentences maximum. Be warm and natural.{lang_instruction}

Your response:"""

        try:
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
            
            system_prompt = "You are a warm, friendly dental clinic receptionist in the UAE. Be natural, concise, and culturally appropriate."
            
            # Make LLM call with token tracking
            llm_start_time = time.time()
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150
                )
            else:
                response = self.llm_client.create_message(
                    system=system_prompt,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150
                )
                tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time
            
            # Record LLM call for observability
            if obs_logger:
                obs_logger.record_llm_call(
                    component="conversational_fast_path",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=len(messages),
                    temperature=0.7,
                    max_tokens=150
                )
            
            logger.info(f"[Conversational] Generated response ({len(response)} chars) for {situation.value}, tokens: {tokens.total_tokens}")
            return response.strip(), tokens
            
        except Exception as e:
            logger.error(f"[Conversational] Error generating response: {e}")
            # Safe fallbacks by language
            fallbacks = {
                "ar": {
                    SituationType.GREETING: "أهلاً وسهلاً! كيف يمكنني مساعدتك؟",
                    SituationType.FAREWELL: "مع السلامة! نتمنى لك يوماً سعيداً.",
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

    async def process_message(
        self,
        session_id: str,
        message: str,
        language: Optional[str] = None
    ) -> ChatResponse:
        """
        Process a user message through the complete pipeline.

        Args:
            session_id: Unique session identifier
            message: User's message
            language: Optional language hint

        Returns:
            ChatResponse with agent's reply
        """
        pipeline_start_time = time.time()
        # Import settings at function level to avoid UnboundLocalError
        from patient_ai_service.core.config import settings as config_settings
        obs_logger = get_observability_logger(session_id) if config_settings.enable_observability else None
        
        logger.info("=" * 100)
        logger.info("ORCHESTRATOR: process_message() CALLED")
        logger.info("=" * 100)
        logger.info(f"Session ID: {session_id}")
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
            with obs_logger.pipeline_step(2, "translation_input", "translation", {"message": message[:100]}) if obs_logger else nullcontext():
                translation_agent = self.agents["translation"]

                # OPTIMIZED: Single LLM call for detection AND translation
                translate_start = time.time()
                english_message, detected_lang, detected_dialect, translation_succeeded = await translation_agent.detect_and_translate(message, session_id)
                translate_duration = (time.time() - translate_start) * 1000

                logger.info(
                    f"[FAST] Language: {detected_lang}-{detected_dialect or 'unknown'}, "
                    f"translation_succeeded: {translation_succeeded} "
                    f"(took {translate_duration:.2f}ms - single LLM call)"
                )

                # Get current language context
                global_state = self.state_manager.get_global_state(session_id)
                language_context = global_state.language_context

                # Check if language switched
                if language_context.current_language != detected_lang:
                    logger.info(
                        f"Language switch detected: {language_context.get_full_language_code()} "
                        f"→ {detected_lang}-{detected_dialect or 'unknown'}"
                    )
                    language_context.record_language_switch(
                        detected_lang,
                        detected_dialect,
                        language_context.turn_count
                    )
                else:
                    # Update both language and dialect from detect_and_translate results
                    # This ensures they stay in sync even if dialect changes within same language
                    language_context.current_language = detected_lang
                    language_context.current_dialect = detected_dialect
                    language_context.last_detected_at = datetime.utcnow()

                language_context.turn_count += 1

                # Track translation status explicitly
                if not translation_succeeded:
                    logger.warning(
                        f"⚠️ TRANSLATION FALLBACK for session {session_id}: "
                        f"passing original message to reasoning engine"
                    )
                    language_context.translation_failures += 1
                    language_context.last_translation_error = "Translation failed - using original"
                elif detected_lang != "en":
                    logger.info(f"Translation to English: {message[:100]}... -> {english_message[:100]}...")
                else:
                    logger.info("No translation needed (already in English)")

                # Update global state with new language context
                self.state_manager.update_global_state(
                    session_id,
                    language_context=language_context
                )

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
                self.memory_manager.add_user_turn(session_id, english_message)
            step3_duration = (time.time() - step3_start) * 1000
            logger.info(f"Step 3 completed in {step3_duration:.2f}ms")

            # Step 4: Situation Assessment (NEW)
            global_state = self.state_manager.get_global_state(session_id)

            patient_info = {
                "patient_id": global_state.patient_profile.patient_id,
                "first_name": global_state.patient_profile.first_name,
                "last_name": global_state.patient_profile.last_name,
                "phone": global_state.patient_profile.phone,
            }

            step4_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 4: Situation Assessment")
            logger.info("-" * 100)
            
            # Get continuation context BEFORE assessment so assessor knows what we're awaiting
            continuation_context = self.state_manager.get_continuation_context(session_id)
            active_agent = global_state.active_agent
            
            with obs_logger.pipeline_step(4, "situation_assessment", "situation_assessor", {"message": english_message[:100]}) if obs_logger else nullcontext():
                assessment = await self.situation_assessor.assess(
                    session_id=session_id,
                    message=english_message,
                    patient_info=patient_info,
                    continuation_context=continuation_context,  # NEW: What we're waiting for
                    active_agent=active_agent  # NEW: Current agent handling the flow
                )
            
            step4_duration = (time.time() - step4_start) * 1000
            logger.info(
                f"Assessment complete in {step4_duration:.0f}ms: "
                f"type={assessment.situation_type.value}, "
                f"confidence={assessment.confidence:.2f}, "
                f"reasoning_needs={assessment.reasoning_needs.value}"
            )
            
            if obs_logger:
                obs_logger.record_pipeline_step(
                    4, "situation_assessment", "situation_assessor",
                    inputs={"message": english_message[:100]},
                    outputs={
                        "situation_type": assessment.situation_type.value,
                        "confidence": assessment.confidence,
                        "reasoning_needs": assessment.reasoning_needs.value,
                        "duration_ms": step4_duration
                    }
                )
            
            # =========================================================================
            # Step 4.5: PLAN LIFECYCLE MANAGEMENT (NEW - Phase 3)
            # =========================================================================
            step45_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 4.5: Plan Lifecycle Management")
            logger.info("-" * 100)
            
            # Load any existing active plan
            existing_plan = self.state_manager.get_any_active_plan(session_id)
            
            # Determine what to do with plans based on assessment
            plan_action = self._determine_plan_action(assessment, existing_plan)
            
            logger.info(f"📋 [Orchestrator] Plan action: {plan_action.value}")
            if existing_plan:
                logger.info(f"📋 [Orchestrator] Existing plan: {existing_plan.get_summary()}")
            
            # Handle plan lifecycle based on assessment
            if plan_action == PlanAction.ABANDON_AND_CREATE:
                if existing_plan:
                    self.state_manager.abandon_all_plans(session_id, "topic_change")
                    logger.info(f"📋 [Orchestrator] Abandoned existing plan due to topic change")
            
            step45_duration = (time.time() - step45_start) * 1000
            logger.info(f"Step 4.5 completed in {step45_duration:.2f}ms")
            
            if obs_logger:
                obs_logger.record_pipeline_step(
                    4.5, "plan_lifecycle", "orchestrator",  # Using float 4.5 to maintain step numbering
                    inputs={
                        "situation_type": assessment.situation_type.value,
                        "has_existing_plan": existing_plan is not None
                    },
                    outputs={
                        "plan_action": plan_action.value,
                        "plan_agent": existing_plan.agent_name if existing_plan else None,
                        "duration_ms": step45_duration
                    }
                )
            
            # =========================================================================
            # Step 5: ROUTING DECISION
            # =========================================================================
            step5_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 5: Routing Decision")
            logger.info("-" * 100)
            
            # ═══════════════════════════════════════════════════════════════════════
            # CONVERSATIONAL FAST PATH - Check FIRST before any other routing
            # ═══════════════════════════════════════════════════════════════════════
            if assessment.situation_type in CONVERSATIONAL_FAST_PATH:
                logger.info(f"⚡ [Conversational Fast Path] Situation: {assessment.situation_type.value}")
                logger.info(f"⚡ [Conversational Fast Path] Bypassing agent reasoning loop")
                
                # Get last 4 messages from conversation memory for context-aware response
                memory = self.memory_manager.get_memory(session_id)
                conversation_history = []
                if memory.recent_turns:
                    # Get last 4 turns (excluding current message which was just added)
                    recent_turns = memory.recent_turns[:-1] if memory.recent_turns else []
                    last_4_turns = recent_turns[-4:] if len(recent_turns) > 4 else recent_turns
                    conversation_history = [
                        {"role": turn.role, "content": turn.content}
                        for turn in last_4_turns
                    ]
                    logger.info(f"⚡ [Conversational Fast Path] Using {len(conversation_history)} previous messages for context")
                
                # Generate response directly with pipeline step tracking
                conv_start = time.time()
                with obs_logger.pipeline_step(5, "conversational_fast_path", "orchestrator", {"situation": assessment.situation_type.value, "message": english_message[:100]}) if obs_logger else nullcontext():
                    english_response, conv_tokens = await self._conversational_response(
                        message=english_message,
                        situation=assessment.situation_type,
                        patient_name=patient_info.get('first_name'),
                        is_registered=bool(patient_info.get('patient_id')),
                        language=language_context.current_language,
                        dialect=language_context.current_dialect,
                        conversation_history=conversation_history if conversation_history else None,
                        session_id=session_id
                    )
                conv_duration = (time.time() - conv_start) * 1000
                
                logger.info(f"⚡ [Conversational Fast Path] Response generated in {conv_duration:.0f}ms, tokens: {conv_tokens.total_tokens} (in: {conv_tokens.input_tokens}, out: {conv_tokens.output_tokens})")
                
                # Record pipeline step with full details
                if obs_logger:
                    obs_logger.record_pipeline_step(
                        5, "conversational_fast_path", "orchestrator",
                        inputs={
                            "situation": assessment.situation_type.value,
                            "message": english_message[:100]
                        },
                        outputs={
                            "response": english_response[:100],
                            "tokens": {
                                "total": conv_tokens.total_tokens,
                                "input": conv_tokens.input_tokens,
                                "output": conv_tokens.output_tokens
                            },
                            "duration_ms": conv_duration
                        }
                    )
                
                # Add to conversation memory
                self.memory_manager.add_assistant_turn(session_id, english_response)
                
                # Build minimal metadata
                step5_duration = (time.time() - step5_start) * 1000
                pipeline_duration = (time.time() - pipeline_start_time) * 1000
                
                logger.info(f"⚡ [Conversational Fast Path] Total pipeline: {pipeline_duration:.0f}ms")
                logger.info("=" * 100)
                
                # Return early - skip all agent execution, validation, finalization
                return ChatResponse(
                    response=english_response,
                    session_id=session_id,
                    detected_language=language_context.get_full_language_code(),
                    intent=assessment.situation_type.value,
                    urgency="routine",
                    metadata={
                        "path": "conversational_fast",
                        "situation": assessment.situation_type.value,
                        "llm_calls": 1,
                        "pipeline_duration_ms": pipeline_duration,
                        "response_duration_ms": conv_duration,
                        "tokens": {
                            "total": conv_tokens.total_tokens,
                            "input": conv_tokens.input_tokens,
                            "output": conv_tokens.output_tokens
                        },
                        "agent": None,  # No agent used
                        "validation": {"passed": True, "skipped": True},
                        "finalization": {"enabled": False}
                    }
                )
            
            # ═══════════════════════════════════════════════════════════════════════
            # NORMAL ROUTING - Conditional based on plan_action (Phase 3)
            # ═══════════════════════════════════════════════════════════════════════
            
            # Determine if we need reasoning based on plan action
            if plan_action in [PlanAction.CREATE_NEW, PlanAction.ABANDON_AND_CREATE]:
                # Need full reasoning to get objective for new plan
                if assessment.needs_comprehensive_reasoning():
                    # COMPREHENSIVE: Use full reasoning engine
                    self._comprehensive_count += 1
                    logger.info("⚡ [Routing] ════════════════════════════════════════════════════════")
                    logger.info("⚡ [Routing] → Using COMPREHENSIVE reasoning (new plan)")
                    logger.info(f"⚡ [Routing]   Reason: confidence={assessment.confidence:.2f}, needs={assessment.reasoning_needs.value}")
                    reasoning = await self._comprehensive_reasoning(
                        session_id, english_message, patient_info, assessment
                    )
                else:
                    # FOCUSED: Use lightweight handling
                    self._focused_count += 1
                    logger.info("⚡ [Routing] ════════════════════════════════════════════════════════")
                    logger.info("⚡ [Routing] → Using FOCUSED handling (new plan)")
                    logger.info(f"⚡ [Routing]   Reason: confidence={assessment.confidence:.2f}, type={assessment.situation_type.value}")
                    reasoning = await self._focused_handling(
                        session_id, english_message, patient_info, assessment
                    )
            
            elif plan_action in [PlanAction.RESUME, PlanAction.UPDATE_AND_RESUME]:
                # Continuation - use minimal reasoning to extract new entities
                self._focused_count += 1  # Count as focused since it's lightweight
                logger.info("⚡ [Routing] ════════════════════════════════════════════════════════")
                logger.info("⚡ [Routing] → Using CONTINUATION reasoning (resume plan)")
                logger.info(f"⚡ [Routing]   Reason: plan_action={plan_action.value}, existing_plan={existing_plan.agent_name}")
                reasoning = await self._continuation_reasoning(
                    session_id, english_message, assessment, existing_plan
                )
            
            else:
                # Fallback: use existing logic
                if assessment.needs_comprehensive_reasoning():
                    self._comprehensive_count += 1
                    logger.info("⚡ [Routing] → Using COMPREHENSIVE reasoning (fallback)")
                    reasoning = await self._comprehensive_reasoning(
                        session_id, english_message, patient_info, assessment
                    )
                else:
                    self._focused_count += 1
                    logger.info("⚡ [Routing] → Using FOCUSED handling (fallback)")
                    reasoning = await self._focused_handling(
                        session_id, english_message, patient_info, assessment
                    )
            
            step5_duration = (time.time() - step5_start) * 1000
            logger.info(f"Routing decision complete in {step5_duration:.0f}ms")
            
            if obs_logger:
                obs_logger.record_pipeline_step(
                    5, "routing_decision", "orchestrator",
                    inputs={"assessment": assessment.situation_type.value},
                    outputs={
                        "agent": reasoning.routing.agent,
                        "action": reasoning.routing.action,
                        "urgency": reasoning.routing.urgency,
                        "sentiment": reasoning.understanding.sentiment,
                        "duration_ms": step5_duration
                    }
                )

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
            logger.info(f"Step 4 completed in {step4_duration:.2f}ms")

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
            
            # Sync reasoning_state.active_agent to match global_state
            reasoning_state = self.situation_assessor._get_reasoning_state(session_id)
            reasoning_state.active_agent = agent_name
            self.situation_assessor._save_reasoning_state(session_id, reasoning_state)
            logger.info(f"🎯 [ReasoningState] Active agent synced to: {agent_name}")
            
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

            # Step 12: Add assistant response to conversation memory
            step12_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 12: Add Assistant Response to Memory")
            logger.info("-" * 100)
            with obs_logger.pipeline_step(12, "add_assistant_to_memory", "memory_manager", {"response_preview": english_response[:100]}) if obs_logger else nullcontext():
                self.memory_manager.add_assistant_turn(session_id, english_response)
            step12_duration = (time.time() - step12_start) * 1000
            logger.info(f"Step 12 completed in {step12_duration:.2f}ms")

            # Step 13: Translation (output) - DISABLED
            # The agent's _generate_focused_response already generates responses in the target language
            # based on context.get("current_language"). No additional translation needed.
            step13_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 13: Translation (Output) - DISABLED")
            logger.info("-" * 100)
            logger.info("Translation handled by agent's _generate_focused_response method")
            logger.info("Agent generates response directly in target language based on context.current_language")
            
            # Use agent response as-is (already in target language)
            translated_response = english_response
            logger.info(f"Using agent response as-is (already in target language): {translated_response[:200]}...")
            
            step13_duration = (time.time() - step13_start) * 1000
            logger.info(f"Step 13 completed in {step13_duration:.2f}ms (skipped translation)")

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

            # Try to extract phone from session_id (common pattern: phone number as session)
            # This is a simple heuristic - adjust based on your session ID strategy
            if session_id.startswith("+") or session_id.isdigit():
                phone_number = session_id
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

    async def _comprehensive_reasoning(
        self,
        session_id: str,
        message: str,
        patient_info: Dict[str, Any],
        assessment: SituationAssessment
    ) -> 'ReasoningOutput':
        """
        Use full reasoning engine for complex situations.
        
        This is the existing behavior, now only triggered when needed.
        """
        logger.info(
            f"[Comprehensive] Reason: confidence={assessment.confidence:.2f}, "
            f"type={assessment.situation_type.value}, "
            f"needs={assessment.reasoning_needs.value}"
        )
        
        # Use existing reasoning engine
        reasoning = await self.reasoning_engine.reason(
            session_id,
            message,
            patient_info
        )
        
        # Update patient entities from assessment if any extracted
        self._apply_assessment_entities(session_id, assessment)
        
        return reasoning

    async def _focused_handling(
        self,
        session_id: str,
        message: str,
        patient_info: Dict[str, Any],
        assessment: SituationAssessment
    ) -> 'ReasoningOutput':
        """
        Use focused prompts for simple situations.
        
        Skips full reasoning, uses targeted mini-prompts.
        """
        from patient_ai_service.core.reasoning import (
            ReasoningOutput,
            UnderstandingResult,
            RoutingResult,
            MemoryUpdate,
            ResponseGuidance,
            TaskContext
        )
        
        logger.info(
            f"[Focused] Type={assessment.situation_type.value}, "
            f"confidence={assessment.confidence:.2f}"
        )
        
        # Get reasoning state for context
        reasoning_state = self.situation_assessor._get_reasoning_state(session_id)
        
        # Get active_agent from global_state (which is correctly maintained by orchestrator)
        # rather than reasoning_state (which isn't synced when agent is activated)
        global_state = self.state_manager.get_global_state(session_id)
        active_agent = global_state.active_agent
        
        # Apply extracted entities from assessment
        self._apply_assessment_entities(session_id, assessment)
        
        # Build minimal reasoning output from assessment
        understanding = UnderstandingResult(
            what_user_means=assessment.key_understanding,
            is_continuation=assessment.continue_with_active_agent,
            continuation_type=assessment.situation_type.value if assessment.is_response_to_options else None,
            selected_option=assessment.extracted_entities.selected_option,
            sentiment=assessment.user_sentiment
        )
        
        # Determine routing based on situation type and awaiting context
        logger.info(
            f"⚡ [Routing] Determining agent: "
            f"active_agent={active_agent}, "
            f"is_response_to_awaiting={assessment.is_response_to_awaiting}, "
            f"continue_with_active_agent={assessment.continue_with_active_agent}, "
            f"situation_type={assessment.situation_type.value}"
        )
        
        if assessment.is_response_to_awaiting:
            # User is responding to what we asked - continue with active agent
            agent = active_agent or assessment.suggested_agent or self._infer_agent_from_assessment(assessment)
            
            if assessment.situation_type == SituationType.DIRECT_ANSWER:
                action = "process_answer"
            elif assessment.situation_type == SituationType.SELECTION:
                action = "process_selection"
            elif assessment.situation_type == SituationType.CONFIRMATION:
                action = "execute_confirmed_action"
            elif assessment.situation_type == SituationType.REJECTION:
                action = "handle_rejection"
            elif assessment.situation_type == SituationType.MODIFICATION:
                action = "handle_modification"
            else:
                action = "continue_flow"
                
        elif assessment.situation_type == SituationType.PIVOT_SAME_FLOW:
            # Different question but same overall goal - keep agent, new action
            agent = active_agent or assessment.suggested_agent
            action = "handle_pivot"
            
        elif assessment.situation_type in [
            SituationType.DIRECT_CONTINUATION, 
            SituationType.DIRECT_ANSWER,
            SituationType.CLARIFICATION_RESPONSE
        ] and active_agent:
            # These continuation types with active agent should always continue with that agent
            agent = active_agent
            
            # Determine action based on type
            if assessment.situation_type == SituationType.DIRECT_ANSWER:
                action = "process_answer"
            elif assessment.situation_type == SituationType.CLARIFICATION_RESPONSE:
                action = "process_clarification"
            else:
                action = "continue_flow"
                
            logger.info(
                f"⚡ [Routing] {assessment.situation_type.value} detected with "
                f"active agent={active_agent}, continuing with action={action}"
            )
            
        elif assessment.continue_with_active_agent and active_agent:
            agent = active_agent
            action = "continue_flow"
            
        else:
            agent = assessment.suggested_agent or self._infer_agent_from_assessment(assessment)
            action = self._infer_action_from_assessment(assessment)
        
        routing = RoutingResult(
            agent=agent,
            action=action,
            urgency="emergency" if assessment.situation_type == SituationType.EMERGENCY else "routine"
        )
        
        # Build task context from assessment
        entities = {}
        if assessment.extracted_entities.doctor_name:
            entities["doctor_preference"] = assessment.extracted_entities.doctor_name
        if assessment.extracted_entities.date:
            entities["date_preference"] = assessment.extracted_entities.date
        if assessment.extracted_entities.time:
            entities["time_preference"] = assessment.extracted_entities.time
        if assessment.extracted_entities.procedure:
            entities["procedure_preference"] = assessment.extracted_entities.procedure
        if assessment.extracted_entities.selected_option:
            entities["selected_option"] = assessment.extracted_entities.selected_option
        
        task_context = TaskContext(
            user_intent=assessment.key_understanding,
            objective=assessment.key_understanding,
            entities=entities,
            is_continuation=assessment.continue_with_active_agent,
            continuation_type=assessment.situation_type.value,
            selected_option=assessment.extracted_entities.selected_option
        )
        
        response_guidance = ResponseGuidance(
            tone="helpful",
            task_context=task_context,
            minimal_context={
                "what_user_means": assessment.key_understanding,
                "action": action,
                "situation_type": assessment.situation_type.value
            }
        )
        
        return ReasoningOutput(
            understanding=understanding,
            routing=routing,
            memory_updates=MemoryUpdate(),
            response_guidance=response_guidance,
            reasoning_chain=[f"Focused handling: {assessment.situation_type.value}"]
        )

    async def _continuation_reasoning(
        self,
        session_id: str,
        message: str,
        assessment: SituationAssessment,
        existing_plan: AgentPlan
    ) -> 'ReasoningOutput':
        """
        Use minimal reasoning for continuations (Phase 3).
        
        When resuming an existing plan, we don't need full reasoning.
        Just extract new entities and build context from the existing plan.
        """
        from patient_ai_service.core.reasoning import (
            ReasoningOutput,
            UnderstandingResult,
            RoutingResult,
            MemoryUpdate,
            ResponseGuidance,
            TaskContext
        )
        
        logger.info(
            f"[Continuation] Resuming plan for {existing_plan.agent_name}, "
            f"situation={assessment.situation_type.value}"
        )
        
        # Apply any new entities from assessment
        self._apply_assessment_entities(session_id, assessment)
        
        # Build understanding from assessment
        understanding = UnderstandingResult(
            what_user_means=assessment.key_understanding,
            is_continuation=True,
            continuation_type=assessment.situation_type.value,
            selected_option=assessment.extracted_entities.selected_option,
            sentiment=assessment.user_sentiment
        )
        
        # Routing: continue with the plan's agent
        routing = RoutingResult(
            agent=existing_plan.agent_name,
            action="continue_plan",  # Signal to agent to resume plan
            urgency="normal",
            reasoning=f"Continuing existing plan: {existing_plan.objective}"
        )
        
        # Response guidance from assessment
        response_guidance = ResponseGuidance(
            tone=assessment.suggested_tone or "professional_friendly",
            should_ask_question=False,
            formality_level="conversational"
        )
        
        # Task context from existing plan
        task_context = TaskContext(
            objective=existing_plan.objective,
            user_intent=existing_plan.objective,
            entities=existing_plan.resolved_entities.copy(),
            constraints=existing_plan.constraints.copy(),
            success_criteria=[],  # Phase 6: criteria removed, using plan-based execution
            action=routing.action,
            prior_context=f"Resuming: {existing_plan.get_summary()}",
            is_continuation=True,
            continuation_type=assessment.situation_type.value
        )
        
        # Return minimal reasoning output
        return ReasoningOutput(
            task_context=task_context,
            understanding=understanding,
            routing=routing,
            memory_updates=MemoryUpdate(),
            response_guidance=response_guidance,
            reasoning_chain=[
                f"Continuation reasoning: {assessment.situation_type.value}",
                f"Resuming plan: {existing_plan.plan_id}",
                f"Agent: {existing_plan.agent_name}"
            ]
        )

    def _apply_assessment_entities(
        self,
        session_id: str,
        assessment: SituationAssessment,
        entity_state: 'EntityState' = None
    ) -> List[str]:
        """
        Apply extracted entities from assessment to entity state.
        
        Returns list of fields that were updated.
        """
        from patient_ai_service.models.patient_entities import EntitySource
        from patient_ai_service.models.entity_state import EntityState
        
        updated = []
        ext = assessment.extracted_entities
        
        # Get entity state if not provided
        if entity_state is None:
            entity_state = self.state_manager.get_entity_state(session_id)
        
        if ext.doctor_name:
            result = entity_state.update_patient_preference(
                "appointment.doctor_preference",
                ext.doctor_name,
                EntitySource.EXTRACTED
            )
            if result["changed"]:
                updated.append("doctor_preference")
                if result["invalidated"]:
                    logger.info(f"Invalidated derived: {result['invalidated']}")
        
        if ext.time:
            result = entity_state.update_patient_preference(
                "appointment.time_preference",
                ext.time,
                EntitySource.EXTRACTED
            )
            if result["changed"]:
                updated.append("time_preference")
        
        if ext.date:
            result = entity_state.update_patient_preference(
                "appointment.date_preference",
                ext.date,
                EntitySource.EXTRACTED
            )
            if result["changed"]:
                updated.append("date_preference")
        
        if ext.procedure:
            result = entity_state.update_patient_preference(
                "appointment.procedure_preference",
                ext.procedure,
                EntitySource.EXTRACTED
            )
            if result["changed"]:
                updated.append("procedure_preference")
        
        # Save updated entity state
        if updated:
            self.state_manager.save_entity_state(session_id, entity_state)
        
        return updated

    def _infer_agent_from_assessment(self, assessment: SituationAssessment) -> str:
        """Infer agent from assessment when not continuing active agent."""
        situation = assessment.situation_type
        
        if situation in [SituationType.GREETING, SituationType.FAREWELL, SituationType.PLEASANTRY]:
            return "general_assistant"
        
        if situation == SituationType.EMERGENCY:
            return "emergency_response"
        
        # Default based on entities
        entities = assessment.extracted_entities
        if entities.doctor_name or entities.time or entities.date:
            return "appointment_manager"
        
        return "general_assistant"

    def _infer_action_from_assessment(self, assessment: SituationAssessment) -> str:
        """Infer action from assessment."""
        situation = assessment.situation_type
        
        action_map = {
            SituationType.DIRECT_CONTINUATION: "continue",
            SituationType.SELECTION: "process_selection",
            SituationType.CONFIRMATION: "confirm",
            SituationType.REJECTION: "handle_rejection",
            SituationType.MODIFICATION: "modify",
            SituationType.NEW_INTENT: "new_request",
            SituationType.GREETING: "greet",
            SituationType.FAREWELL: "farewell",
            SituationType.EMERGENCY: "emergency_response",
            SituationType.CLARIFICATION_RESPONSE: "process_clarification"
        }
        
        return action_map.get(situation, "process")
    
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
    
    def _check_continuation(
        self,
        session_id: str,
        assessment: SituationAssessment,
        entity_state: 'EntityState'
    ) -> Optional[Dict[str, Any]]:
        """
        Check if this is a continuation of a blocked flow.
        
        Returns continuation context if resuming, None otherwise.
        """
        logger.info("🔄 [Continuation] ════════════════════════════════════════════════════════")
        logger.info(f"🔄 [Continuation] Checking for continuation: session={session_id}")
        
        if not assessment.continue_with_active_agent:
            logger.info("🔄 [Continuation] ⏭️  Not a continuation (new intent)")
            return None
        
        logger.info("🔄 [Continuation] ✅ This is a continuation")
        
        # Check for blocked task plan
        task_plan = self.state_manager.get_task_plan(session_id)
        if task_plan and task_plan.has_blocked_tasks():
            blocked_tasks = [t for t in task_plan.tasks if t.status.value == "blocked"]
            logger.info(f"🔄 [Continuation] 📋 Found blocked task plan with {len(blocked_tasks)} blocked tasks")
            context = task_plan.get_continuation_context()
            logger.info(f"🔄 [Continuation]   Context: {context.get('awaiting', 'N/A')}")
            return context
        
        # Check reasoning state for awaiting
        reasoning_state = self.situation_assessor._get_reasoning_state(session_id)
        if reasoning_state.awaiting:
            logger.info(f"🔄 [Continuation] 🎯 Found awaiting in reasoning state: {reasoning_state.awaiting}")
            context = {
                "awaiting": reasoning_state.awaiting,
                "presented_options": reasoning_state.presented_options,
                "established_intent": reasoning_state.established_intent
            }
            logger.info(f"🔄 [Continuation]   Options: {len(reasoning_state.presented_options)}")
            return context
        
        logger.info("🔄 [Continuation] ⚠️  Continuation detected but no context found")
        return None
    
    def _build_execution_context(
        self,
        reasoning: 'ReasoningOutput',
        assessment: SituationAssessment,
        continuation_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context dict for agent execution."""
        context = {
            # From reasoning
            "objective": reasoning.response_guidance.task_context.objective,
            "user_intent": reasoning.response_guidance.task_context.user_intent,
            "entities": reasoning.response_guidance.task_context.entities,
            "constraints": reasoning.response_guidance.task_context.constraints,
            "tone": reasoning.response_guidance.tone,
            "action": reasoning.routing.action,
            
            # From assessment
            "situation_type": assessment.situation_type.value,
            "confidence": assessment.confidence,
            "is_continuation": assessment.continue_with_active_agent,
            "selected_option": assessment.extracted_entities.selected_option,
            
            # Continuation
            "continuation_context": continuation_context,
        }
        
        return context
    
    def _infer_agent(
        self,
        assessment: SituationAssessment,
        patient_info: Dict[str, Any]
    ) -> str:
        """Infer agent from assessment."""
        situation = assessment.situation_type
        
        # Special situations
        if situation == SituationType.EMERGENCY:
            return "emergency_response"
        
        if situation in [SituationType.GREETING, SituationType.FAREWELL, SituationType.PLEASANTRY]:
            return "general_assistant"
        
        # Check if patient needs registration
        is_registered = bool(patient_info.get("patient_id"))
        if not is_registered:
            # Check if trying to do something that needs registration
            entities = assessment.extracted_entities
            if entities.doctor_name or entities.time or entities.date:
                return "registration"
        
        # Infer from entities
        entities = assessment.extracted_entities
        if entities.doctor_name or entities.time or entities.date or entities.procedure:
            return "appointment_manager"
        
        return "general_assistant"
    
    def _infer_action(self, assessment: SituationAssessment) -> str:
        """Infer action from assessment."""
        return self._infer_action_from_assessment(assessment)
    
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

    def _determine_plan_action(
        self,
        assessment: SituationAssessment,
        existing_plan: Optional[AgentPlan]
    ) -> PlanAction:
        """
        Determine what to do with plans based on situation assessment.
        
        Maps situation types to plan actions:
        - NEW_INTENT → Create new (abandon old if exists)
        - TOPIC_SHIFT → Abandon old, create new
        - CONTINUATION/SELECTION/CONFIRMATION → Resume or update existing
        - No plan exists → Create new
        
        Args:
            assessment: SituationAssessment from situation_assessor
            existing_plan: Current active plan if any
            
        Returns:
            PlanAction indicating what agent should do
        """
        situation_type = assessment.situation_type
        
        logger.info(f"📋 [PlanLifecycle] Determining plan action:")
        logger.info(f"📋 [PlanLifecycle]   Situation: {situation_type.value}")
        logger.info(f"📋 [PlanLifecycle]   Existing plan: {existing_plan.agent_name if existing_plan else 'None'}")
        
        # NEW INTENT - always create new plan
        if situation_type == SituationType.NEW_INTENT:
            if existing_plan:
                logger.info(f"📋 [PlanLifecycle]   → ABANDON_AND_CREATE (new intent detected)")
                return PlanAction.ABANDON_AND_CREATE
            logger.info(f"📋 [PlanLifecycle]   → CREATE_NEW (new intent, no existing plan)")
            return PlanAction.CREATE_NEW
        
        # TOPIC SHIFT - abandon old, create new
        if situation_type == SituationType.TOPIC_SHIFT:
            logger.info(f"📋 [PlanLifecycle]   → ABANDON_AND_CREATE (topic shift)")
            return PlanAction.ABANDON_AND_CREATE
        
        # No existing plan and not a new intent - create new
        if not existing_plan:
            logger.info(f"📋 [PlanLifecycle]   → CREATE_NEW (no existing plan)")
            return PlanAction.CREATE_NEW
        
        # CONTINUATIONS - resume or update existing plan
        continuation_types = {
            SituationType.DIRECT_CONTINUATION,
            SituationType.SELECTION,
            SituationType.CONFIRMATION,
            SituationType.DIRECT_ANSWER
        }
        
        if situation_type in continuation_types:
            # Check if assessment extracted new entities
            has_new_entities = bool(
                assessment.extracted_entities.doctor_name or
                assessment.extracted_entities.date or
                assessment.extracted_entities.time or
                assessment.extracted_entities.procedure or
                assessment.extracted_entities.selected_option
            )
            
            if has_new_entities:
                logger.info(f"📋 [PlanLifecycle]   → UPDATE_AND_RESUME (new entities extracted)")
                return PlanAction.UPDATE_AND_RESUME
            
            logger.info(f"📋 [PlanLifecycle]   → RESUME (continue existing plan)")
            return PlanAction.RESUME
        
        # MODIFICATION - update and resume
        if situation_type == SituationType.MODIFICATION:
            logger.info(f"📋 [PlanLifecycle]   → UPDATE_AND_RESUME (modification)")
            return PlanAction.UPDATE_AND_RESUME
        
        # REJECTION - might abandon or modify depending on context
        if situation_type == SituationType.REJECTION:
            # Check if rejecting the whole task or just an option
            if existing_plan and existing_plan.is_blocked():
                # Rejecting an option - let agent handle
                logger.info(f"📋 [PlanLifecycle]   → RESUME (rejection of option)")
                return PlanAction.RESUME
            # Rejecting the whole thing
            logger.info(f"📋 [PlanLifecycle]   → ABANDON_AND_CREATE (rejection of task)")
            return PlanAction.ABANDON_AND_CREATE
        
        # Default: create new
        logger.info(f"📋 [PlanLifecycle]   → CREATE_NEW (default)")
        return PlanAction.CREATE_NEW

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
            }
            logger.info("📊 Using minimal_context (no task_context available)")

        # Merge with continuation context if resuming
        if continuation_context and task_context.get("is_continuation"):
            resolved = continuation_context.get("resolved_entities", {})
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
        # Get entities from task context
        entities = task_context.get("entities", {}).copy()
        logger.info(f"📊 Entities before enhancement: {json.dumps(entities, default=str)}")

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

        context = {
            # Task context
            "user_intent": task_context.get("user_intent", ""),
            "objective": task_context.get("objective", ""),  # NEW: Include objective
            "entities": entities,  # Use enhanced entities with patient_id
            "success_criteria": task_context.get("success_criteria", []),
            "constraints": task_context.get("constraints", []),
            "prior_context": task_context.get("prior_context"),

            # Continuation info
            "is_continuation": task_context.get("is_continuation", False),
            "continuation_type": task_context.get("continuation_type"),
            "selected_option": task_context.get("selected_option"),
            "continuation_context": continuation_context or {},

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
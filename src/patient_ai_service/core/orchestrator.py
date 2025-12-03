"""
Orchestrator - Main coordinator for the multi-agent system.

Coordinates:
- Message routing via unified reasoning
- Agent execution
- State management
- Translation
- Pub/sub messaging
"""

import logging
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Optional, Dict, Any, List

from patient_ai_service.core import (
    get_llm_client,
    get_state_manager,
    get_message_broker,
)
from patient_ai_service.core.config import settings
from patient_ai_service.core.reasoning import get_reasoning_engine
from patient_ai_service.core.conversation_memory import get_conversation_memory_manager
from patient_ai_service.core.observability import get_observability_logger, clear_observability_logger
from patient_ai_service.models.messages import Topics, ChatResponse
from patient_ai_service.models.enums import UrgencyLevel
from patient_ai_service.models.validation import ExecutionLog, ValidationResult
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

        # Initialize agents
        self._init_agents()

        logger.info("Orchestrator initialized with reasoning engine")

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
        
        try:
            logger.info(f"Processing message for session: {session_id}")

            # Step 1: Load or initialize patient
            with obs_logger.pipeline_step(1, "load_patient", "orchestrator", {"session_id": session_id}) if obs_logger else nullcontext():
                await self._ensure_patient_loaded(session_id)

            # Step 2: Translation (input)
            with obs_logger.pipeline_step(2, "translation_input", "translation", {"message": message[:100]}) if obs_logger else nullcontext():
                translation_agent = self.agents["translation"]

                # Detect language and dialect
                detected_lang, detected_dialect = await translation_agent.detect_language_and_dialect(message)

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
                    # Update current dialect if detected
                    language_context.current_language = detected_lang
                    language_context.current_dialect = detected_dialect
                    language_context.last_detected_at = datetime.utcnow()

                language_context.turn_count += 1

                # Translate to English if needed
                if detected_lang != "en":
                    english_message = await translation_agent.translate_to_english_with_dialect(
                        message,
                        detected_lang,
                        detected_dialect
                    )
                else:
                    english_message = message

                # Validate translation succeeded
                if not english_message or len(english_message.strip()) == 0:
                    logger.error(f"Translation failed for session {session_id}")
                    language_context.translation_failures += 1
                    language_context.last_translation_error = "Empty translation result"
                    english_message = message  # Fallback to original

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
                            "detected_dialect": detected_dialect
                        }
                    )

            logger.info(f"Detected language: {language_context.get_full_language_code()}")

            # Step 3: Add user message to conversation memory
            with obs_logger.pipeline_step(3, "add_to_memory", "memory_manager", {"message": english_message[:100]}) if obs_logger else nullcontext():
                self.memory_manager.add_user_turn(session_id, english_message)

            # Step 4: Unified reasoning (replaces intent classification)
            global_state = self.state_manager.get_global_state(session_id)

            patient_info = {
                "patient_id": global_state.patient_profile.patient_id,
                "first_name": global_state.patient_profile.first_name,
                "last_name": global_state.patient_profile.last_name,
                "phone": global_state.patient_profile.phone,
            }

            # Step 4: Unified reasoning
            with obs_logger.pipeline_step(4, "reasoning", "reasoning_engine", {"message": english_message[:100]}) if obs_logger else nullcontext():
                reasoning = await self.reasoning_engine.reason(
                    session_id,
                    english_message,
                    patient_info
                )
                if obs_logger:
                    obs_logger.record_pipeline_step(
                        4, "reasoning", "reasoning_engine",
                        inputs={"message": english_message[:100]},
                        outputs={
                            "agent": reasoning.routing.agent,
                            "action": reasoning.routing.action,
                            "urgency": reasoning.routing.urgency,
                            "sentiment": reasoning.understanding.sentiment
                        }
                    )

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
            if reasoning.response_guidance.plan:
                logger.info(f"Agent Plan: {reasoning.response_guidance.plan}")
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

            # Update active agent
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
            agent = self.agents.get(agent_name)
            if not agent:
                logger.error(f"Agent not found: {agent_name}")
                english_response = "I'm sorry, I encountered an error. Please try again."
                # execution_log already initialized above, no need to create again
            else:
                
                # Call agent activation hook (for state setup)
                with obs_logger.pipeline_step(7, "agent_activation", "agent", {"agent_name": agent_name}) if obs_logger else nullcontext():
                    if hasattr(agent, 'on_activated'):
                        await agent.on_activated(session_id, reasoning)

                    # Pass minimal context to agent
                    if hasattr(agent, 'set_context'):
                        agent.set_context(session_id, reasoning.response_guidance.minimal_context)

                # Execute agent with logging - PASS execution_log
                with obs_logger.pipeline_step(8, "agent_execution", "agent", {"agent_name": agent_name, "message": english_message[:100]}) if obs_logger else nullcontext():
                    english_response, execution_log = await agent.process_message_with_log(
                        session_id,
                        english_message,
                        execution_log  # Pass log to agent (agent will append tools)
                    )
                    
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

            # Step 8: Validate response (CLOSED-LOOP) - ONLY IF ENABLED

            if config_settings.enable_validation:
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

            # Step 9: Handle validation result (retry loop) - ONLY IF VALIDATION ENABLED

            max_retries = config_settings.validation_max_retries if config_settings.enable_validation else 0
            retry_count = 0

            while not validation.is_valid and retry_count < max_retries and config_settings.enable_validation:
                if validation.decision == "retry":
                    # Store tool count before retry for validation
                    tools_before_retry = len(execution_log.tools_used)
                    logger.info(
                        f"Retrying with feedback (current tools: {tools_before_retry}): "
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
                        validation = await self.reasoning_engine.validate_response(
                            session_id,
                            reasoning,
                            english_response,
                            execution_log  # Contains tools from ALL attempts
                        )
                        retry_count += 1
                        
                        if obs_logger:
                            obs_logger._validation_details.retry_count = retry_count

                elif validation.decision == "redirect":
                    # Could try different agent, but for MVP just break
                    logger.warning(f"Validation suggests redirect, but using current response")
                    break
                else:
                    break

            # LAYER 2: Finalization (final quality check) - ONLY IF ENABLED
            if config_settings.enable_finalization:
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
                    english_response = finalization.rewritten_response
                    logger.info("Using edited response from finalization layer")
                elif finalization.should_fallback():
                    english_response = self._get_validation_fallback(finalization.issues)
                    logger.warning(f"Finalization triggered fallback: {finalization.issues}")
                else:
                    logger.info("Finalization approved agent's response")
            else:
                # Finalization disabled - skip
                logger.info("Finalization layer DISABLED - using agent response as-is")
                finalization = None

            # Step 12: Add assistant response to conversation memory
            with obs_logger.pipeline_step(12, "add_assistant_to_memory", "memory_manager", {"response_preview": english_response[:100]}) if obs_logger else nullcontext():
                self.memory_manager.add_assistant_turn(session_id, english_response)

            # Step 13: Translation (output)
            # Get current language context for translation
            language_context = global_state.language_context

            if language_context.current_language != "en":
                with obs_logger.pipeline_step(13, "translation_output", "translation", {"response_preview": english_response[:100]}) if obs_logger else nullcontext():
                    # process_output now uses dialect-aware translation internally
                    translated_response = await translation_agent.process_output(
                        session_id,
                        english_response
                    )
                    logger.info(f"Translated response to {language_context.get_full_language_code()}")
            else:
                translated_response = english_response

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
                pipeline_duration_ms = (time.time() - pipeline_start_time) * 1000
                obs_logger.record_pipeline_step(
                    15, "pipeline_complete", "orchestrator",
                    metadata={"total_duration_ms": pipeline_duration_ms}
                )
                obs_logger.log_summary()

            return response

        except Exception as e:
            logger.error(f"Error in orchestrator: {e}", exc_info=True)
            
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

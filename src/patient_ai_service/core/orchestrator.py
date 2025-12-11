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

            # Step 4: Unified reasoning (replaces intent classification)
            global_state = self.state_manager.get_global_state(session_id)

            patient_info = {
                "patient_id": global_state.patient_profile.patient_id,
                "first_name": global_state.patient_profile.first_name,
                "last_name": global_state.patient_profile.last_name,
                "phone": global_state.patient_profile.phone,
            }

            # Step 4: Unified reasoning
            step4_start = time.time()
            logger.info("-" * 100)
            logger.info("ORCHESTRATOR STEP 4: Unified Reasoning")
            logger.info("-" * 100)
            logger.info(f"Reasoning input: message={english_message[:200]}..., patient_info={patient_info}")
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

            step4_duration = (time.time() - step4_start) * 1000

            # ============================================================================
            # FIX #1: EXTRACT SUCCESS CRITERIA FROM PLAN
            # ============================================================================
            # The reasoning engine provides a 'plan' string (e.g., "1. Verify doctor, 2. Check availability...")
            # BaseAgent expects 'success_criteria' as a list
            # Transform plan into success criteria
            success_criteria = []
            if reasoning.response_guidance.plan:
                # Split plan by numbered steps
                plan_steps = re.split(r'\d+\.\s+', reasoning.response_guidance.plan)
                # Filter empty strings and clean up
                success_criteria = [
                    step.strip().rstrip(',') 
                    for step in plan_steps 
                    if step.strip()
                ]
                logger.info(f"Extracted {len(success_criteria)} success criteria from plan")
                for i, criterion in enumerate(success_criteria, 1):
                    logger.info(f"  Criterion {i}: {criterion}")
            else:
                # Fallback: create generic criterion from action
                success_criteria = [f"Complete action: {reasoning.routing.action}"]
                logger.warning(f"No plan provided by reasoning engine, using fallback criterion")

            # Store success_criteria for agent context
            reasoning.response_guidance.task_context.success_criteria = success_criteria
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
            if reasoning.response_guidance.plan:
                logger.info(f"Agent Plan: {reasoning.response_guidance.plan}")
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
                            continuation_context
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
            task_context = {
                "user_intent": tc.user_intent if hasattr(tc, 'user_intent') else reasoning.understanding.what_user_means,
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
                "user_intent": mc.get("user_wants", reasoning.understanding.what_user_means),
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
        continuation_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build the context dict to pass to the agent.
        Injects critical parameters like patient_id to prevent hallucinations.
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

            # Backward compatibility
            "user_wants": task_context.get("user_intent", ""),
            "action": reasoning.routing.action,
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
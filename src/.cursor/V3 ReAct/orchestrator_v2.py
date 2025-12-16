"""
Enhanced Orchestrator - Blocked Handling and Continuation Support

This file contains updates to orchestrator.py to support:
1. Detecting and handling blocked flows
2. Passing continuation context to agents
3. Managing state transitions for multi-turn flows
4. Enhanced logging for agentic execution

Merge these changes with your existing orchestrator.py
"""

import logging
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
from patient_ai_service.core.reasoning import get_reasoning_engine
from patient_ai_service.core.conversation_memory import get_conversation_memory_manager
from patient_ai_service.core.observability import get_observability_logger
from patient_ai_service.models.messages import ChatResponse
from patient_ai_service.models.validation import ExecutionLog, ValidationResult

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED process_message METHOD
# =============================================================================

async def process_message_enhanced(
    self,
    session_id: str,
    message: str,
    language: Optional[str] = None
) -> ChatResponse:
    """
    Enhanced process_message with continuation and blocked state handling.
    
    Key additions:
    1. Check for continuation context before reasoning
    2. Initialize agentic state from reasoning
    3. Pass structured context to agents
    4. Handle blocked/continuation states
    5. Rich metadata in response
    """
    pipeline_start_time = time.time()
    from patient_ai_service.core.config import settings as config_settings
    obs_logger = get_observability_logger(session_id) if config_settings.enable_observability else None
    
    logger.info("=" * 100)
    logger.info("ORCHESTRATOR: process_message() START")
    logger.info(f"Session: {session_id}")
    logger.info(f"Message: {message[:200]}...")
    logger.info("=" * 100)
    
    try:
        # =====================================================================
        # STEP 0: Check for existing continuation context
        # =====================================================================
        
        continuation_context = self.state_manager.get_continuation_context(session_id)
        is_resuming = bool(continuation_context)
        
        if is_resuming:
            logger.info("-" * 60)
            logger.info("CONTINUATION DETECTED - Resuming blocked flow")
            logger.info(f"Awaiting: {continuation_context.get('awaiting')}")
            logger.info(f"Options: {continuation_context.get('presented_options', [])[:5]}")
            logger.info("-" * 60)
            
            # Increment waiting turns
            self.state_manager.increment_waiting_turns(session_id)
        else:
            # New request - reset agentic state
            self.state_manager.reset_agentic_state(session_id)
            logger.debug("Reset agentic state for new request")
        
        # =====================================================================
        # STEP 1: Load Patient
        # =====================================================================
        
        step1_start = time.time()
        with obs_logger.pipeline_step(1, "load_patient", "orchestrator") if obs_logger else nullcontext():
            await self._ensure_patient_loaded(session_id)
        logger.info(f"Step 1 (load_patient): {(time.time() - step1_start)*1000:.2f}ms")
        
        # =====================================================================
        # STEP 2: Translation (Input)
        # =====================================================================
        
        step2_start = time.time()
        with obs_logger.pipeline_step(2, "translation_input", "translation") if obs_logger else nullcontext():
            translation_agent = self.agents["translation"]
            detected_lang, detected_dialect = await translation_agent.detect_language_and_dialect(message)
            
            global_state = self.state_manager.get_global_state(session_id)
            language_context = global_state.language_context
            language_context.current_language = detected_lang
            language_context.current_dialect = detected_dialect
            language_context.turn_count += 1
            
            if detected_lang != "en":
                english_message = await translation_agent.translate_to_english_with_dialect(
                    message, detected_lang, detected_dialect
                )
            else:
                english_message = message
            
            self.state_manager.update_global_state(session_id, language_context=language_context)
        
        logger.info(f"Step 2 (translation): {(time.time() - step2_start)*1000:.2f}ms")
        logger.info(f"Language: {language_context.get_full_language_code()}")
        
        # =====================================================================
        # STEP 3: Add to Memory
        # =====================================================================
        
        step3_start = time.time()
        with obs_logger.pipeline_step(3, "add_to_memory", "memory") if obs_logger else nullcontext():
            self.memory_manager.add_user_turn(session_id, english_message)
        logger.info(f"Step 3 (memory): {(time.time() - step3_start)*1000:.2f}ms")
        
        # =====================================================================
        # STEP 4: Unified Reasoning (with continuation awareness)
        # =====================================================================
        
        step4_start = time.time()
        global_state = self.state_manager.get_global_state(session_id)
        patient_info = {
            "patient_id": global_state.patient_profile.patient_id,
            "first_name": global_state.patient_profile.first_name,
            "last_name": global_state.patient_profile.last_name,
            "phone": global_state.patient_profile.phone,
        }
        
        with obs_logger.pipeline_step(4, "reasoning", "reasoning_engine") if obs_logger else nullcontext():
            # Pass continuation context to reasoning
            reasoning = await self.reasoning_engine.reason(
                session_id,
                english_message,
                patient_info
            )
        
        logger.info(f"Step 4 (reasoning): {(time.time() - step4_start)*1000:.2f}ms")
        logger.info(f"Routing: {reasoning.routing.agent} -> {reasoning.routing.action}")
        logger.info(f"Intent: {reasoning.understanding.what_user_means}")
        logger.info(f"Is Continuation: {reasoning.understanding.is_continuation}")
        
        # =====================================================================
        # STEP 5: Initialize/Update Agentic State
        # =====================================================================
        
        step5_start = time.time()
        
        # Extract task context
        task_context = self._extract_task_context(reasoning, continuation_context)
        
        if is_resuming and reasoning.understanding.is_continuation:
            # Resuming - update existing state with new info
            logger.info("Updating agentic state for continuation")
            
            # Unblock criteria if user provided selection
            if hasattr(reasoning.response_guidance, 'task_context'):
                tc = reasoning.response_guidance.task_context
                if tc.continuation_type == "selection" and tc.selected_option:
                    # User made a selection - unblock criteria
                    blocked = self.state_manager.get_blocked_criteria(session_id)
                    for bc in blocked:
                        self.state_manager.unblock_criterion(session_id, bc.get("description", ""))
                    
                    # Update resolved entities
                    task_context["entities"]["selected_time"] = tc.selected_option
                    
                    # Clear continuation context - we're proceeding
                    # Don't clear yet - let agent handle
        else:
            # New request - initialize fresh
            max_iterations = getattr(config_settings, 'max_agentic_iterations', 15)
            self.state_manager.initialize_agentic_state(
                session_id,
                task_context=task_context,
                max_iterations=max_iterations
            )
            logger.info(f"Initialized agentic state: {len(task_context.get('success_criteria', []))} criteria")
        
        logger.info(f"Step 5 (agentic_init): {(time.time() - step5_start)*1000:.2f}ms")
        
        # =====================================================================
        # STEP 6: Handle Conversation Restart
        # =====================================================================
        
        if reasoning.understanding.is_conversation_restart:
            with obs_logger.pipeline_step(6, "restart", "memory") if obs_logger else nullcontext():
                self.memory_manager.archive_and_reset(session_id)
                self.state_manager.clear_continuation_context(session_id)
                logger.info("Conversation restarted")
        
        # =====================================================================
        # STEP 7: Initialize Execution Log
        # =====================================================================
        
        execution_log = ExecutionLog(
            tools_used=[],
            conversation_turns=len(self.memory_manager.get_memory(session_id).recent_turns)
        )
        
        # =====================================================================
        # STEP 8: Select and Activate Agent
        # =====================================================================
        
        step8_start = time.time()
        agent_name = reasoning.routing.agent
        
        # Emergency override
        if reasoning.routing.urgency == "emergency":
            agent_name = "emergency_response"
            logger.info("Emergency routing activated")
        
        self.state_manager.update_global_state(session_id, active_agent=agent_name)
        
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"Agent not found: {agent_name}")
            return self._error_response(session_id, "Agent not found", language_context)
        
        # Build context for agent
        context_for_agent = self._build_agent_context(
            task_context,
            reasoning,
            language_context,
            continuation_context
        )
        
        # Activate agent
        with obs_logger.pipeline_step(8, "agent_activation", "agent") if obs_logger else nullcontext():
            if hasattr(agent, 'on_activated'):
                await agent.on_activated(session_id, reasoning)
            if hasattr(agent, 'set_context'):
                agent.set_context(session_id, context_for_agent)
        
        logger.info(f"Step 8 (agent_activation): {(time.time() - step8_start)*1000:.2f}ms")
        logger.info(f"Agent: {agent_name}")
        
        # =====================================================================
        # STEP 9: Execute Agent
        # =====================================================================
        
        step9_start = time.time()
        with obs_logger.pipeline_step(9, "agent_execution", "agent") if obs_logger else nullcontext():
            english_response, execution_log = await agent.process_message_with_log(
                session_id,
                english_message,
                execution_log
            )
        
        agent_duration = (time.time() - step9_start) * 1000
        logger.info(f"Step 9 (agent_execution): {agent_duration:.2f}ms")
        logger.info(f"Response preview: {english_response[:200]}...")
        logger.info(f"Tools used: {len(execution_log.tools_used)}")
        
        # =====================================================================
        # STEP 10: Check Agentic State After Execution
        # =====================================================================
        
        step10_start = time.time()
        agentic_state = self.state_manager.get_agentic_state(session_id)
        agentic_summary = self.state_manager.get_agentic_summary(session_id)
        
        logger.info("-" * 60)
        logger.info("AGENTIC EXECUTION SUMMARY:")
        logger.info(f"  Status: {agentic_summary['status']}")
        logger.info(f"  Iterations: {agentic_summary['iterations']}/{agentic_summary['max_iterations']}")
        logger.info(f"  Criteria: {agentic_summary['criteria']}")
        logger.info(f"  Has Continuation: {agentic_summary['has_continuation']}")
        logger.info(f"  Tool Calls: {agentic_summary['tool_calls']}")
        logger.info("-" * 60)
        
        # Handle different completion states
        validation = await self._handle_agentic_completion(
            session_id,
            agentic_state,
            reasoning,
            english_response,
            execution_log,
            config_settings
        )
        
        logger.info(f"Step 10 (completion_check): {(time.time() - step10_start)*1000:.2f}ms")
        
        # =====================================================================
        # STEP 11: Finalization (if enabled)
        # =====================================================================
        
        step11_start = time.time()
        finalization = None
        
        if config_settings.enable_finalization and agentic_state.status == "complete":
            with obs_logger.pipeline_step(11, "finalization", "reasoning") if obs_logger else nullcontext():
                finalization = await self.reasoning_engine.finalize_response(
                    session_id=session_id,
                    original_reasoning=reasoning,
                    agent_response=english_response,
                    execution_log=execution_log,
                    validation_result=validation
                )
                
                if finalization.should_use_rewritten():
                    english_response = finalization.rewritten_response
                    logger.info("Using finalized response")
        
        logger.info(f"Step 11 (finalization): {(time.time() - step11_start)*1000:.2f}ms")
        
        # =====================================================================
        # STEP 12: Add to Memory
        # =====================================================================
        
        step12_start = time.time()
        with obs_logger.pipeline_step(12, "add_response_to_memory", "memory") if obs_logger else nullcontext():
            self.memory_manager.add_assistant_turn(session_id, english_response)
        logger.info(f"Step 12 (memory): {(time.time() - step12_start)*1000:.2f}ms")
        
        # =====================================================================
        # STEP 13: Translation (Output)
        # =====================================================================
        
        step13_start = time.time()
        if language_context.current_language != "en":
            with obs_logger.pipeline_step(13, "translation_output", "translation") if obs_logger else nullcontext():
                translated_response = await translation_agent.process_output(session_id, english_response)
        else:
            translated_response = english_response
        
        logger.info(f"Step 13 (translation): {(time.time() - step13_start)*1000:.2f}ms")
        
        # =====================================================================
        # STEP 14: Build Response
        # =====================================================================
        
        response = ChatResponse(
            response=translated_response,
            session_id=session_id,
            detected_language=language_context.get_full_language_code(),
            intent=reasoning.understanding.what_user_means,
            urgency=reasoning.routing.urgency,
            metadata=self._build_response_metadata(
                agent_name,
                reasoning,
                validation,
                finalization,
                agentic_summary,
                language_context
            )
        )
        
        # =====================================================================
        # FINAL: Log Summary
        # =====================================================================
        
        pipeline_duration = (time.time() - pipeline_start_time) * 1000
        logger.info("=" * 100)
        logger.info("ORCHESTRATOR: process_message() COMPLETE")
        logger.info(f"Total Duration: {pipeline_duration:.2f}ms")
        logger.info(f"Agent: {agent_name}")
        logger.info(f"Status: {agentic_summary['status']}")
        logger.info(f"Tools: {agentic_summary['tool_calls']}")
        logger.info("=" * 100)
        
        return response
    
    except Exception as e:
        logger.error(f"Error in orchestrator: {e}", exc_info=True)
        return self._error_response(session_id, str(e), None)


# =============================================================================
# HELPER METHODS
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
    
    # Merge with continuation context if resuming
    if continuation_context and task_context.get("is_continuation"):
        resolved = continuation_context.get("resolved_entities", {})
        for key, value in resolved.items():
            if key not in task_context["entities"]:
                task_context["entities"][key] = value
        
        # Use same success criteria if resuming
        if not task_context["success_criteria"]:
            blocked = continuation_context.get("blocked_criteria", [])
            if blocked:
                task_context["success_criteria"] = blocked
        
        # Store continuation context for agent
        task_context["continuation_context"] = continuation_context
    
    return task_context


def _build_agent_context(
    self,
    task_context: Dict[str, Any],
    reasoning: 'ReasoningOutput',
    language_context: Any,
    continuation_context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build the context dict to pass to the agent.
    """
    context = {
        # Task context
        "user_intent": task_context.get("user_intent", ""),
        "entities": task_context.get("entities", {}),
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
    Build metadata dict for ChatResponse.
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


def _error_response(
    self,
    session_id: str,
    error: str,
    language_context: Optional[Any]
) -> 'ChatResponse':
    """
    Generate error response.
    """
    return ChatResponse(
        response="I apologize, but I encountered an error. Please try again or contact support.",
        session_id=session_id,
        detected_language=language_context.get_full_language_code() if language_context else "en",
        metadata={"error": error}
    )


# =============================================================================
# CONFIG ADDITIONS
# =============================================================================

"""
Add these settings to your config.py:

class Settings:
    # ... existing settings ...
    
    # Agentic execution
    max_agentic_iterations: int = 15
    agentic_thinking_temperature: float = 0.2
    agentic_response_temperature: float = 0.5
    
    # Trust settings
    trust_agentic_completion: bool = True  # Skip validation for complete tasks
    
    # Continuation settings
    max_continuation_turns: int = 5  # Max turns to wait for user input
"""

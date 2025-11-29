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
from typing import Optional, Dict, Any, List

from patient_ai_service.core import (
    get_llm_client,
    get_state_manager,
    get_message_broker,
)
from patient_ai_service.core.reasoning import get_reasoning_engine
from patient_ai_service.core.conversation_memory import get_conversation_memory_manager
from patient_ai_service.models.messages import Topics, ChatResponse
from patient_ai_service.models.enums import UrgencyLevel
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
        try:
            logger.info(f"Processing message for session: {session_id}")

            # Step 1: Load or initialize patient
            await self._ensure_patient_loaded(session_id)

            # Step 2: Translation (input)
            translation_agent = self.agents["translation"]
            english_message, detected_lang = await translation_agent.process_input(
                session_id,
                message
            )

            logger.info(f"Detected language: {detected_lang}")

            # Step 3: Add user message to conversation memory
            self.memory_manager.add_user_turn(session_id, english_message)

            # Step 4: Unified reasoning (replaces intent classification)
            global_state = self.state_manager.get_global_state(session_id)

            patient_info = {
                "patient_id": global_state.patient_profile.patient_id,
                "first_name": global_state.patient_profile.first_name,
                "last_name": global_state.patient_profile.last_name,
                "phone": global_state.patient_profile.phone,
            }

            reasoning = await self.reasoning_engine.reason(
                session_id,
                english_message,
                patient_info
            )

            logger.info(
                f"Reasoning: agent={reasoning.routing.agent}, "
                f"urgency={reasoning.routing.urgency}, "
                f"sentiment={reasoning.understanding.sentiment}"
            )

            # Step 5: Handle conversation restart if detected
            if reasoning.understanding.is_conversation_restart:
                self.memory_manager.archive_and_reset(session_id)
                logger.info(f"Conversation restarted for session {session_id}")

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

            # Step 7: Agent transition hook
            agent = self.agents.get(agent_name)
            if not agent:
                logger.error(f"Agent not found: {agent_name}")
                english_response = "I'm sorry, I encountered an error. Please try again."
            else:
                # Call agent activation hook (for state setup)
                if hasattr(agent, 'on_activated'):
                    await agent.on_activated(session_id, reasoning)

                # Pass minimal context to agent
                if hasattr(agent, 'set_context'):
                    agent.set_context(session_id, reasoning.response_guidance.minimal_context)

                # Execute agent with logging
                english_response, execution_log = await agent.process_message_with_log(
                    session_id,
                    english_message
                )

            # Step 8: Validate response (CLOSED-LOOP)
            validation = await self.reasoning_engine.validate_response(
                session_id=session_id,
                original_reasoning=reasoning,
                agent_response=english_response,
                execution_log=execution_log
            )

            logger.info(f"Validation result: valid={validation.is_valid}, "
                       f"decision={validation.decision}, "
                       f"confidence={validation.confidence}")

            # Step 9: Handle validation result (retry loop)
            max_retries = 1  # Single retry to minimize latency
            retry_count = 0

            while not validation.is_valid and retry_count < max_retries:
                if validation.decision == "retry":
                    logger.info(f"Retrying with feedback: {validation.feedback_to_agent}")

                    # Retry with specific feedback
                    english_response, execution_log = await agent.process_message_with_log(
                        session_id,
                        f"[VALIDATION FEEDBACK]: {validation.feedback_to_agent}\n\n"
                        f"Original user request: {english_message}"
                    )

                    # Re-validate
                    validation = await self.reasoning_engine.validate_response(
                        session_id,
                        reasoning,
                        english_response,
                        execution_log
                    )
                    retry_count += 1

                elif validation.decision == "redirect":
                    # Could try different agent, but for MVP just break
                    logger.warning(f"Validation suggests redirect, but using current response")
                    break
                else:
                    break

            # Step 10: Use fallback if still invalid
            if not validation.is_valid and validation.confidence < 0.5:
                logger.error(f"Validation failed after retries: {validation.issues}")
                english_response = self._get_validation_fallback(validation.issues)

            # Step 11: Add assistant response to conversation memory
            self.memory_manager.add_assistant_turn(session_id, english_response)

            # Step 9: Translation (output)
            if detected_lang != "en":
                translated_response = await translation_agent.process_output(
                    session_id,
                    english_response
                )
            else:
                translated_response = english_response

            # Step 10: Build response
            response = ChatResponse(
                response=translated_response,
                session_id=session_id,
                detected_language=detected_lang,
                intent=reasoning.understanding.what_user_means,
                urgency=reasoning.routing.urgency,
                metadata={
                    "agent": agent_name,
                    "sentiment": reasoning.understanding.sentiment,
                    "reasoning_summary": reasoning.reasoning_chain[0] if reasoning.reasoning_chain else "",
                    "validation": {
                        "passed": validation.is_valid,
                        "retries": retry_count,
                        "confidence": validation.confidence,
                        "issues": validation.issues if not validation.is_valid else []
                    }
                }
            )

            logger.info(f"Response generated by {agent_name}")

            return response

        except Exception as e:
            logger.error(f"Error in orchestrator: {e}", exc_info=True)

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

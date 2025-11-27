"""
Orchestrator - Main coordinator for the multi-agent system.

Coordinates:
- Message routing via intent classification
- Agent execution
- State management
- Translation
- Pub/sub messaging
"""

import logging
from typing import Optional, Dict, Any

from patient_ai_service.core import (
    get_llm_client,
    get_state_manager,
    get_message_broker,
)
from patient_ai_service.core.intent_router import get_intent_router
from patient_ai_service.models.messages import Topics, ChatResponse
from patient_ai_service.models.enums import IntentType, UrgencyLevel
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
        self.intent_router = get_intent_router()
        self.db_client = db_client or DbOpsClient()

        # Initialize agents
        self._init_agents()

        logger.info("Orchestrator initialized")

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

            # Step 3: Intent classification
            global_state = self.state_manager.get_global_state(session_id)
            
            # Get last few intents for better context
            recent_intents = global_state.intent_history[-3:] if global_state.intent_history else []
            
            context = {
                "patient_id": global_state.patient_profile.patient_id,
                "active_agent": global_state.active_agent,
                "conversation_stage": global_state.conversation_stage,
                "recent_intents": [str(i) for i in recent_intents],  # Convert to strings for JSON
            }

            classification = self.intent_router.route(english_message, context)

            logger.info(
                f"Intent: {classification.intent}, "
                f"Urgency: {classification.urgency}"
            )

            # Update state with intent
            self.state_manager.update_global_state(
                session_id,
                intent_history=global_state.intent_history + [classification.intent],
                entities_collected={
                    **global_state.entities_collected,
                    **classification.entities
                }
            )

            # Step 4: Route to agent
            agent_name = self._get_agent_for_intent(classification.intent, global_state, session_id)

            # Check for emergency override
            if classification.urgency == UrgencyLevel.CRITICAL:
                agent_name = "emergency_response"

            # Update active agent
            self.state_manager.update_global_state(
                session_id,
                active_agent=agent_name
            )

            # Step 5: Execute agent
            agent = self.agents.get(agent_name)
            if not agent:
                logger.error(f"Agent not found: {agent_name}")
                english_response = "I'm sorry, I encountered an error. Please try again."
            else:
                english_response = await agent.process_message(
                    session_id,
                    english_message
                )

            # Step 6: Translation (output)
            if detected_lang != "en":
                translated_response = await translation_agent.process_output(
                    session_id,
                    english_response
                )
            else:
                translated_response = english_response

            # Step 7: Build response
            response = ChatResponse(
                response=translated_response,
                session_id=session_id,
                detected_language=detected_lang,
                intent=classification.intent,
                urgency=classification.urgency,
                metadata={
                    "agent": agent_name,
                    "confidence": classification.confidence,
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

    def _get_agent_for_intent(self, intent: IntentType, global_state, session_id: str) -> str:
        """
        Determine which agent should handle the intent.

        Considers:
        - Intent type
        - Current active agent (for continuity)
        - Conversation state
        - Active operations/workflows
        """
        # If emergency agent is active AND intent is emergency, maintain control
        # But allow other intents to override if user explicitly changes topic
        if global_state.active_agent == "emergency_response" and intent == IntentType.EMERGENCY:
            return "emergency_response"
        
        # If registration agent is active and intent is registration, maintain continuity
        if global_state.active_agent == "registration" and intent == IntentType.REGISTRATION:
            return "registration"
        
        # CRITICAL: If appointment_manager is active AND the last intent was appointment-related,
        # maintain continuity for multi-turn workflows (e.g., selecting which appointment to reschedule)
        # BUT: Medical inquiries and emergencies should always switch to their specialized agents
        if global_state.active_agent == "appointment_manager":
            # Check if last intent was appointment-related
            if global_state.intent_history:
                last_intent = global_state.intent_history[-1]
                appointment_intents = [
                    IntentType.APPOINTMENT_BOOKING,
                    IntentType.APPOINTMENT_RESCHEDULE,
                    IntentType.APPOINTMENT_CANCEL,
                    IntentType.APPOINTMENT_CHECK,
                ]
                
                # ALWAYS switch to medical_inquiry or emergency agents regardless of context
                if intent in [IntentType.MEDICAL_INQUIRY, IntentType.EMERGENCY]:
                    logger.info(
                        f"Switching from appointment_manager to specialized agent - "
                        f"Intent: {intent} requires specialized handling"
                    )
                    # Don't maintain appointment_manager, fall through to normal routing
                # If last intent was appointment-related AND current intent is also appointment-related,
                # likely a follow-up response - maintain appointment_manager
                elif last_intent in appointment_intents and intent in appointment_intents + [IntentType.FOLLOW_UP]:
                    logger.info(
                        f"Maintaining appointment_manager context - "
                        f"Last intent: {last_intent}, Current intent: {intent}"
                    )
                    return "appointment_manager"

        # Map intent to agent
        intent_agent_map = {
            IntentType.APPOINTMENT_BOOKING: "appointment_manager",
            IntentType.APPOINTMENT_RESCHEDULE: "appointment_manager",
            IntentType.APPOINTMENT_CANCEL: "appointment_manager",
            IntentType.APPOINTMENT_CHECK: "appointment_manager",
            IntentType.FOLLOW_UP: "appointment_manager",
            IntentType.MEDICAL_INQUIRY: "medical_inquiry",
            IntentType.EMERGENCY: "emergency_response",
            IntentType.REGISTRATION: "registration",
            IntentType.GENERAL_INQUIRY: "general_assistant",  # Route to general assistant for info queries
            IntentType.GREETING: "general_assistant",  # Route greetings to general assistant
        }

        agent_name = intent_agent_map.get(intent, "appointment_manager")

        # Check if patient is actually registered (exists in DB)
        # Also check registration state to see if registration was completed
        reg_state = self.state_manager.get_registration_state(session_id)
        patient_registered = (
            global_state.patient_profile.patient_id is not None and
            global_state.patient_profile.patient_id != ""
        )
        registration_complete = reg_state.registration_complete if reg_state else False

        # If patient doesn't exist in DB AND registration isn't marked complete, route to registration
        # BUT: Allow these intents to go through without registration:
        # - GENERAL_INQUIRY: Users should be able to ask about doctors, procedures, clinic info
        # - GREETING: Users should be able to greet and get general info
        # - APPOINTMENT_BOOKING: Users might be trying to register through appointment flow
        # - APPOINTMENT_CHECK: Users might want to check existing appointments
        # - EMERGENCY: Always allow emergency queries
        if (not patient_registered and 
            not registration_complete and 
            intent != IntentType.REGISTRATION and
            intent not in [
                IntentType.GENERAL_INQUIRY,  # Allow info queries without registration
                IntentType.GREETING,  # Allow greetings without registration
                IntentType.APPOINTMENT_BOOKING,  # Allow appointment requests
                IntentType.APPOINTMENT_CHECK,  # Allow appointment checks
                IntentType.EMERGENCY  # Always allow emergencies
            ]):
            logger.info("Patient not registered, routing to registration")
            return "registration"

        return agent_name

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get complete session state."""
        return self.state_manager.export_session(session_id)

    def clear_session(self, session_id: str):
        """Clear session state."""
        self.state_manager.clear_session(session_id)
        logger.info(f"Session cleared: {session_id}")

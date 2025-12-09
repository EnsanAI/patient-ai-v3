"""
Intent Router for classifying user messages and routing to appropriate agents.
"""

import json
import logging
import re
from typing import Tuple, Dict, Any

from patient_ai_service.models.enums import IntentType, UrgencyLevel
from patient_ai_service.models.messages import IntentClassification
from patient_ai_service.core.config import settings
from .llm import get_llm_client, LLMClient

logger = logging.getLogger(__name__)


class IntentRouter:
    """
    Routes user messages to appropriate agents based on intent classification.

    Uses LLM for intelligent classification with keyword-based fallback.
    """

    # Emergency keywords for fast-path detection
    EMERGENCY_KEYWORDS = [
        "emergency", "urgent", "severe bleeding", "can't breathe",
        "knocked out", "broken jaw", "severe pain", "911",
        "ambulance", "immediate", "critical"
    ]

    # Appointment keywords
    APPOINTMENT_KEYWORDS = [
        "appointment", "book", "schedule", "reschedule", "cancel",
        "check appointment", "when is my", "confirm", "need to schedule",
        "want to book", "make an appointment", "set up appointment"
    ]

    # Medical keywords - questions requiring doctor expertise
    MEDICAL_KEYWORDS = [
        "pain", "hurt", "swelling", "bleeding", "infection",
        "toothache", "sensitivity", "symptom",
        "toothpaste", "mouthwash", "medication", "medicine",
        "which", "what should", "recommend", "advise",
        "treatment option", "side effect", "is it normal",
        "post-procedure", "after procedure", "diagnosis"
    ]

    # Registration keywords
    REGISTRATION_KEYWORDS = [
        "new patient", "register", "sign up", "first time",
        "create account", "join"
    ]

    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or get_llm_client()
        logger.info("IntentRouter initialized")

    def route(self, message: str, context: Dict[str, Any] = None) -> IntentClassification:
        """
        Classify user message intent and determine urgency.

        Args:
            message: User's message
            context: Optional context (session state, history, etc.)

        Returns:
            IntentClassification with intent, urgency, entities, and routing info
        """
        try:
            # Fast-path: Check for emergencies first
            if self._is_emergency(message):
                return IntentClassification(
                    intent=IntentType.EMERGENCY,
                    urgency=UrgencyLevel.CRITICAL,
                    entities={"detected_emergency": True},
                    confidence=1.0,
                    reasoning="Emergency keywords detected"
                )

            # Use LLM for classification
            classification = self._classify_with_llm(message, context)
            return classification

        except Exception as e:
            logger.error(f"Error in intent routing: {e}", exc_info=True)
            # Fallback to keyword-based classification
            return self._classify_with_keywords(message)

    def _is_emergency(self, message: str) -> bool:
        """Fast emergency detection using keywords."""
        message_lower = message.lower()
        
        # Exclude false positives - "emergency contact" is not an emergency
        if "emergency contact" in message_lower or "contact name" in message_lower or "contact phone" in message_lower:
            return False
        
        return any(keyword in message_lower for keyword in self.EMERGENCY_KEYWORDS)

    def _classify_with_llm(
        self,
        message: str,
        context: Dict[str, Any] = None
    ) -> IntentClassification:
        """Use LLM for intent classification."""
        system_prompt = self._get_classification_prompt()

        # Build context string
        context_str = ""
        if context:
            context_str = f"\n\nContext: {json.dumps(context, indent=2)}"

        user_prompt = f"""Classify this user message:

Message: "{message}"{context_str}

**CONTEXT ANALYSIS REQUIRED:**
If context shows active_agent="appointment_manager", the user is likely in an ongoing appointment workflow.
Short responses (dates, numbers, "tomorrow", "the first one") are typically CONTINUATIONS, not new requests.

Examples:
- Context shows active_agent="appointment_manager", last conversation about rescheduling
- User says: "Tomorrow (Nov 25)" 
- Classification: appointment_reschedule (continuing reschedule workflow, NOT a new booking)

Provide your classification in JSON format:
{{
    "intent": "<one of: appointment_booking, appointment_reschedule, appointment_cancel, appointment_check, follow_up, medical_inquiry, emergency, registration, general_inquiry, greeting>",
    "urgency": "<one of: low, medium, high, critical>",
    "entities": {{<extracted entities like dates, times, names, symptoms>}},
    "reasoning": "<brief explanation>"
}}"""

        # Call LLM
        response = self.llm_client.create_message(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=settings.intent_router_temperature
        )

        # Parse response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                return IntentClassification(
                    intent=IntentType(result.get("intent")),
                    urgency=UrgencyLevel(result.get("urgency")),
                    entities=result.get("entities", {}),
                    reasoning=result.get("reasoning"),
                    confidence=0.9  # High confidence for LLM classification
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM classification: {e}")

        # Fallback
        return self._classify_with_keywords(message)

    def _classify_with_keywords(self, message: str) -> IntentClassification:
        """Fallback keyword-based classification."""
        message_lower = message.lower()

        # Check appointment intent (check this FIRST before other intents)
        if any(kw in message_lower for kw in self.APPOINTMENT_KEYWORDS):
            if "cancel" in message_lower:
                intent = IntentType.APPOINTMENT_CANCEL
            elif "reschedule" in message_lower or "change" in message_lower:
                intent = IntentType.APPOINTMENT_RESCHEDULE
            elif "check" in message_lower or "when" in message_lower or "status" in message_lower:
                intent = IntentType.APPOINTMENT_CHECK
            else:
                intent = IntentType.APPOINTMENT_BOOKING

            return IntentClassification(
                intent=intent,
                urgency=UrgencyLevel.MEDIUM,
                entities={},
                confidence=0.8,  # Higher confidence for appointment keywords
                reasoning="Appointment keyword detected"
            )

        # Check medical intent
        if any(kw in message_lower for kw in self.MEDICAL_KEYWORDS):
            # Determine urgency based on pain descriptors
            if any(word in message_lower for word in ["severe", "unbearable", "extreme"]):
                urgency = UrgencyLevel.HIGH
            elif any(word in message_lower for word in ["pain", "hurt", "bleeding"]):
                urgency = UrgencyLevel.MEDIUM
            else:
                urgency = UrgencyLevel.LOW

            return IntentClassification(
                intent=IntentType.MEDICAL_INQUIRY,
                urgency=urgency,
                entities={},
                confidence=0.7,
                reasoning="Keyword-based classification"
            )

        # Check registration intent
        if any(kw in message_lower for kw in self.REGISTRATION_KEYWORDS):
            return IntentClassification(
                intent=IntentType.REGISTRATION,
                urgency=UrgencyLevel.LOW,
                entities={},
                confidence=0.7,
                reasoning="Keyword-based classification"
            )

        # Check greetings
        if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
            return IntentClassification(
                intent=IntentType.GREETING,
                urgency=UrgencyLevel.LOW,
                entities={},
                confidence=0.8,
                reasoning="Greeting detected"
            )

        # Default: general inquiry
        return IntentClassification(
            intent=IntentType.GENERAL_INQUIRY,
            urgency=UrgencyLevel.LOW,
            entities={},
            confidence=0.5,
            reasoning="No specific intent detected"
        )

    def _get_classification_prompt(self) -> str:
        """Get system prompt for classification."""
        return """You are an intent classification system for a dental clinic.

Your task is to analyze user messages and classify them into one of these intents:

1. **appointment_booking**: User wants to schedule a new appointment
2. **appointment_reschedule**: User wants to change an existing appointment
3. **appointment_cancel**: User wants to cancel an appointment
4. **appointment_check**: User wants to check appointment status/details
5. **follow_up**: User asking about follow-up care or post-treatment INSTRUCTIONS (e.g., "when can I eat after extraction?")
6. **medical_inquiry**: User has medical/dental QUESTIONS that need doctor answers (e.g., symptoms, medication advice, product recommendations like toothpaste/mouthwash, treatment options, side effects, diagnosis questions)
7. **emergency**: Dental emergency requiring immediate attention
8. **registration**: New patient wants to register
9. **general_inquiry**: General questions about services, hours, location, prices, insurance
10. **greeting**: Simple greeting or small talk

IMPORTANT DISTINCTION:
- **follow_up**: Post-procedure INSTRUCTIONS (e.g., "how long until I can eat?", "when should I remove the gauze?")
- **medical_inquiry**: Medical QUESTIONS requiring doctor expertise (e.g., "what toothpaste should I use?", "which medication is better?", "is this symptom normal?", "what are my treatment options?")

Also determine urgency level:
- **low**: Routine, non-urgent matter
- **medium**: Should be addressed soon
- **high**: Urgent, needs same-day attention
- **critical**: Emergency, immediate attention required

Extract any relevant entities like:
- Dates/times
- Names
- Symptoms
- Appointment IDs
- Doctor preferences
- Procedures mentioned

Be precise and consider context when available.

CRITICAL CONTEXT AWARENESS:
- **If context shows active_agent = "appointment_manager"**, the user is likely mid-workflow
- **If the user provides simple selections (like "1", "2", "Tomorrow", "the second one")**, 
  they are RESPONDING to a previous question, NOT starting a new request
- **Maintain the current workflow intent** unless the user clearly changes topic
- Example: If user is rescheduling and says "Tomorrow (Nov 25)", they're selecting an appointment,
  NOT booking a new appointment - keep the reschedule intent

IMPORTANT: 
- "Emergency contact" or "emergency contact name/phone" is NOT an emergency - it's part of registration
- Only classify as emergency if there's an actual dental emergency happening right now
- If user is providing information during registration, classify as registration intent
- Short responses like dates, numbers, or simple selections usually mean continuing current workflow"""

    def get_agent_for_intent(self, intent: IntentType) -> str:
        """Map intent to agent name."""
        intent_to_agent = {
            IntentType.APPOINTMENT_BOOKING: "appointment_manager",
            IntentType.APPOINTMENT_RESCHEDULE: "appointment_manager",
            IntentType.APPOINTMENT_CANCEL: "appointment_manager",
            IntentType.APPOINTMENT_CHECK: "appointment_manager",
            IntentType.FOLLOW_UP: "appointment_manager",
            IntentType.MEDICAL_INQUIRY: "medical_inquiry",
            IntentType.EMERGENCY: "emergency_response",
            IntentType.REGISTRATION: "registration",
            IntentType.GENERAL_INQUIRY: "general_assistant",
            IntentType.GREETING: "general_assistant",
        }

        return intent_to_agent.get(intent, "general_assistant")


# Global intent router instance
_intent_router: IntentRouter = None


def get_intent_router() -> IntentRouter:
    """Get or create the global intent router instance."""
    global _intent_router
    if _intent_router is None:
        _intent_router = IntentRouter()
    return _intent_router


def reset_intent_router():
    """Reset the global intent router (useful for testing)."""
    global _intent_router
    _intent_router = None

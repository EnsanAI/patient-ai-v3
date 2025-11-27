"""
Medical Inquiry Agent.

IMPORTANT: This agent does NOT provide medical advice.
It only:
- Classifies inquiries (preventive tips, side effects, general info)
- Assesses urgency/triage
- Recommends appropriate next steps (appointments, emergency services)
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from patient_ai_service.models.enums import TriageLevel
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

logger = logging.getLogger(__name__)


class MedicalInquiryAgent(BaseAgent):
    """
    Agent for handling medical inquiries WITHOUT providing medical advice.

    Responsibilities:
    - Classify inquiry type (preventive, side effect, general)
    - Assess urgency/triage level
    - Recommend seeing a dentist
    - Escalate emergencies
    - Log inquiries for staff review
    """

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        super().__init__(agent_name="MedicalInquiry", **kwargs)
        self.db_client = db_client or DbOpsClient()

    def _register_tools(self):
        """Register medical inquiry tools."""

        # Assess triage level
        self.register_tool(
            name="assess_triage",
            function=self.tool_assess_triage,
            description="Assess the urgency level of a medical inquiry",
            parameters={
                "symptoms": {
                    "type": "array",
                    "description": "List of symptoms mentioned",
                    "items": {"type": "string"}
                },
                "pain_level": {
                    "type": "integer",
                    "description": "Pain level 0-10 if mentioned"
                },
                "duration": {
                    "type": "string",
                    "description": "How long symptoms have persisted"
                }
            }
        )

        # Recommend appointment
        self.register_tool(
            name="recommend_appointment",
            function=self.tool_recommend_appointment,
            description="Recommend booking an appointment based on urgency",
            parameters={
                "urgency": {
                    "type": "string",
                    "description": "Urgency level: routine, soon, urgent, emergency"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for appointment"
                }
            }
        )

        # Log inquiry
        self.register_tool(
            name="log_inquiry",
            function=self.tool_log_inquiry,
            description="Log the medical inquiry for staff review",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID"
                },
                "inquiry_type": {
                    "type": "string",
                    "description": "Type: preventive_tip, side_effect, or general"
                },
                "content": {
                    "type": "string",
                    "description": "Inquiry details"
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level"
                }
            }
        )

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate system prompt with strict medical guidelines."""
        global_state = self.state_manager.get_global_state(session_id)
        medical_state = self.state_manager.get_medical_state(session_id)
        patient = global_state.patient_profile

        return f"""You are a medical inquiry assistant for Bright Smile Dental Clinic.

CRITICAL: YOU CANNOT AND MUST NOT PROVIDE MEDICAL ADVICE OR ANSWER MEDICAL QUESTIONS DIRECTLY.

PATIENT INFO:
- Name: {patient.first_name} {patient.last_name}
- Patient ID: {patient.patient_id}

CURRENT ASSESSMENT:
- Symptoms: {', '.join(medical_state.symptoms) if medical_state.symptoms else 'None recorded'}
- Pain Level: {medical_state.pain_level if medical_state.pain_level else 'Not assessed'}
- Triage Level: {medical_state.triage_level}

YOUR PRIMARY ROLE:
When a patient asks a medical question (e.g., "What toothpaste should I use?", "Which medication is better?", "Is this symptom normal?"):
1. **Listen and acknowledge** - Show empathy for their concern
2. **DO NOT ANSWER the medical question** - You are NOT qualified to provide medical advice
3. **Log the inquiry** - Use the log_inquiry tool to save it for doctor review
4. **Inform the patient** - Tell them their question has been forwarded to a doctor who will respond soon
5. **Assess urgency** - If urgent, recommend booking an appointment

MANDATORY WORKFLOW FOR ALL MEDICAL QUESTIONS:
Step 1: Acknowledge the question empathetically
Step 2: Call log_inquiry tool with the patient's question
Step 3: Tell patient: "I've forwarded your question to our dentist who will provide you with a professional answer soon."
Step 4: If urgent, offer to help book an appointment

WHAT YOU CAN DO:
✓ Log medical questions for doctor review (ALWAYS use log_inquiry tool)
✓ Assess urgency level
✓ Recommend booking an appointment
✓ Escalate emergencies
✓ Listen empathetically

WHAT YOU CANNOT DO:
✗ Answer medical questions directly (even if you "know" the answer)
✗ Provide product recommendations (toothpaste, mouthwash, medications)
✗ Give treatment advice
✗ Diagnose conditions
✗ Suggest specific medications
✗ Provide "general tips" as an answer to a specific medical question

TRIAGE GUIDELINES:
- **EMERGENCY**: Severe bleeding, difficulty breathing, severe facial swelling, knocked out tooth
  → Recommend calling 911 or going to ER immediately
- **URGENT**: Severe pain (8-10/10), swelling with fever, signs of infection
  → Recommend same-day appointment
- **SOON**: Moderate pain, persistent discomfort, broken tooth
  → Recommend next-day or this-week appointment
- **ROUTINE**: General questions, mild sensitivity, preventive care
  → Regular scheduling

EXAMPLE CORRECT RESPONSES:
Patient: "What toothpaste should I use after my root canal?"
You: "That's a great question! Let me forward this to our dentist who can provide you with the best recommendation based on your specific procedure. [CALL log_inquiry TOOL] I've logged your question and one of our dentists will get back to you with a professional answer soon. In the meantime, would you like me to help schedule a follow-up appointment?"

Patient: "Which medication is better for pain?"
You: "I understand you want to know about pain management options. I've forwarded your question to our dentist who can properly advise you on the most appropriate medication for your situation. [CALL log_inquiry TOOL] You should receive a professional response soon. If your pain is severe, I'd recommend booking an urgent appointment today."

INCORRECT RESPONSE (DON'T DO THIS):
Patient: "What toothpaste should I use?"
You: "Generally, fluoride toothpaste is recommended..." ✗ WRONG! You answered instead of logging for doctor review.

Remember: Your job is to LOG medical questions, NOT to ANSWER them."""

    # Tool implementations

    def tool_assess_triage(
        self,
        session_id: str,
        symptoms: list,
        pain_level: Optional[int] = None,
        duration: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assess medical urgency."""
        try:
            # Emergency symptoms
            emergency_symptoms = [
                "severe bleeding", "can't breathe", "difficulty breathing",
                "knocked out tooth", "broken jaw", "severe facial swelling"
            ]

            # Urgent symptoms
            urgent_symptoms = [
                "severe pain", "swelling with fever", "infection",
                "pus", "abscess", "broken tooth"
            ]

            symptoms_lower = [s.lower() for s in symptoms]

            # Determine triage level
            if any(emergency in ' '.join(symptoms_lower) for emergency in emergency_symptoms):
                triage = TriageLevel.EMERGENCY
                recommendation = "Seek immediate emergency care - call 911 or go to ER"
            elif pain_level and pain_level >= 8:
                triage = TriageLevel.URGENT
                recommendation = "Schedule same-day appointment"
            elif any(urgent in ' '.join(symptoms_lower) for urgent in urgent_symptoms):
                triage = TriageLevel.URGENT
                recommendation = "Schedule same-day appointment"
            elif pain_level and pain_level >= 5:
                triage = TriageLevel.SOON
                recommendation = "Schedule appointment within 1-2 days"
            else:
                triage = TriageLevel.ROUTINE
                recommendation = "Schedule routine appointment"

            # Update state
            self.state_manager.update_medical_state(
                session_id,
                symptoms=symptoms,
                pain_level=pain_level,
                symptom_duration=duration,
                triage_level=triage,
                recommended_action=recommendation,
                assessment_complete=True
            )

            return {
                "success": True,
                "triage_level": triage,
                "recommendation": recommendation,
                "symptoms": symptoms,
                "pain_level": pain_level
            }

        except Exception as e:
            logger.error(f"Error in triage assessment: {e}")
            return {"error": str(e)}

    def tool_recommend_appointment(
        self,
        session_id: str,
        urgency: str,
        reason: str
    ) -> Dict[str, Any]:
        """Recommend booking an appointment."""
        try:
            urgency_messages = {
                "emergency": "I strongly recommend seeking immediate emergency care. Should I provide emergency contact information?",
                "urgent": "Based on what you've described, I recommend booking a same-day appointment. Would you like me to check availability?",
                "soon": "I recommend scheduling an appointment within the next 1-2 days. Shall I help you find a suitable time?",
                "routine": "I recommend scheduling a routine appointment. Would you like to see our available times?"
            }

            message = urgency_messages.get(urgency.lower(), urgency_messages["routine"])

            return {
                "success": True,
                "urgency": urgency,
                "message": message,
                "next_step": "offer_booking"
            }

        except Exception as e:
            logger.error(f"Error recommending appointment: {e}")
            return {"error": str(e)}

    def tool_log_inquiry(
        self,
        session_id: str,
        patient_id: str,
        inquiry_type: str,
        content: str,
        priority: str
    ) -> Dict[str, Any]:
        """Log medical inquiry to database."""
        try:
            # Map inquiry type to database enum values
            type_mapping = {
                "preventive_tip": "HEALTH_TIP_REQUEST",
                "side_effect": "SIDE_EFFECTS",
                "general": "GENERAL_MEDICAL_QUESTION",
                "medication": "GENERAL_MEDICAL_QUESTION",
                "appointment": "APPOINTMENT_QUERY",
                "billing": "BILLING_QUERY"
            }
            
            # Map priority to database enum values
            priority_mapping = {
                "routine": "LOW",
                "low": "LOW",
                "medium": "MEDIUM",
                "high": "HIGH",
                "urgent": "HIGH",
                "critical": "HIGH"
            }
            
            # Get mapped values with defaults
            mapped_type = type_mapping.get(inquiry_type.lower(), "GENERAL_MEDICAL_QUESTION")
            mapped_priority = priority_mapping.get(priority.lower(), "LOW")
            
            # Use the inquiries API with correct field names
            inquiry_data = {
                "patientId": patient_id,
                "inquiryType": mapped_type,  # Must be one of the enum values
                "inquiryText": content,      # Changed from "content" to "inquiryText"
                "priority": mapped_priority  # Must be LOW, MEDIUM, or HIGH
            }

            result = self.db_client.create_inquiry(inquiry_data)

            if result:
                # Update state
                self.state_manager.update_medical_state(
                    session_id,
                    inquiry_type=inquiry_type
                )

                return {
                    "success": True,
                    "inquiry_id": result.get("id"),
                    "message": "Inquiry logged for dentist review"
                }
            else:
                return {"error": "Failed to log inquiry"}

        except Exception as e:
            logger.error(f"Error logging inquiry: {e}")
            return {"error": str(e)}

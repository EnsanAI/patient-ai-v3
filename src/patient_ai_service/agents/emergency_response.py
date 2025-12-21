"""
Emergency Response Agent.

Handles critical dental emergencies with highest priority.
Provides immediate first aid guidance and directs patients to appropriate care.
"""

import logging
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from patient_ai_service.models.agentic import ToolResultType
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

logger = logging.getLogger(__name__)


# Constants
EMERGENCY_CONTACTS = {
    "911": "Emergency Services (911)",
    "clinic_emergency": "+971-XXX-XXXX (24/7 Emergency Line)",
    "poison_control": "1-800-222-1222"
}

EMERGENCY_TYPES_REQUIRING_911 = {
    "severe_bleeding",
    "difficulty_breathing",
    "unconsciousness",
    "facial_trauma",
    "broken_jaw"
}

EMERGENCY_TYPES_FOR_CLINIC = {
    "knocked_out_tooth",
    "severe_pain",
    "broken_tooth",
    "lost_filling",
    "abscess"
}

FIRST_AID_INSTRUCTIONS: Dict[str, List[str]] = {
    "bleeding": [
        "Apply clean gauze or cloth to the area",
        "Apply firm but gentle pressure for 10-15 minutes",
        "Do not repeatedly check if bleeding has stopped",
        "If bleeding doesn't stop, seek immediate medical attention"
    ],
    "knocked_out_tooth": [
        "Find the tooth and pick it up by the crown (not the root)",
        "Rinse gently with water if dirty (do not scrub)",
        "Try to place back in socket if possible",
        "If not possible, place in milk or saliva",
        "See dentist within 1 hour - time is critical"
    ],
    "broken_tooth": [
        "Rinse mouth with warm water",
        "Save any broken pieces",
        "Apply cold compress to reduce swelling",
        "Avoid using the tooth",
        "Contact dentist immediately"
    ],
    "severe_pain": [
        "Rinse mouth with warm salt water",
        "Apply cold compress to outside of cheek",
        "Take over-the-counter pain reliever",
        "Do not apply heat or aspirin directly to gum",
        "Seek dental care as soon as possible"
    ]
}

DEFAULT_FIRST_AID_INSTRUCTIONS = [
    "Remain calm",
    "Contact emergency services or dentist immediately"
]


class EmergencyResponseAgent(BaseAgent):
    """
    Agent for handling dental emergencies.

    Features:
    - Immediate response to emergency situations
    - Provide first aid instructions
    - Direct to appropriate emergency services
    - Maintain control until resolved
    - Log all emergency interactions
    """

    EMERGENCY_CONTACTS = EMERGENCY_CONTACTS

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        """
        Initialize Emergency Response Agent.

        Args:
            db_client: Database operations client. If None, creates a new instance.
            **kwargs: Additional arguments passed to BaseAgent.
        """
        super().__init__(agent_name="EmergencyResponse", **kwargs)
        self.db_client = db_client or DbOpsClient()

    def _get_agent_instructions(self) -> str:
        """Emergency response-specific behavioral instructions."""
        return """CRITICAL: Patient safety is the absolute priority. Provide first aid instructions immediately, stay calm, and guide patient through emergency steps. Log emergency after providing immediate assistance."""

    def _register_tools(self) -> None:
        """Register emergency response tools."""
        self._register_report_emergency_tool()
        self._register_first_aid_tool()
        self._register_emergency_contacts_tool()

    def _register_report_emergency_tool(self) -> None:
        """Register the report_emergency tool."""
        self.register_tool(
            name="report_emergency",
            function=self.tool_report_emergency,
            description="""Report and log a dental emergency to the clinic's emergency response system.

This tool creates an official emergency record in the database that:
- Notifies clinic staff immediately
- Creates emergency ID for tracking
- Logs patient information and emergency details
- Updates emergency state for the session

WHEN TO USE:
- After providing first aid instructions
- After assessing emergency severity
- For ANY dental emergency situation (bleeding, knocked out tooth, severe pain, trauma)

The tool returns:
- SUCCESS: Emergency logged with emergency_id, staff notified
- SYSTEM_ERROR: Failed to log, should retry (but continue with first aid guidance)

IMPORTANT: Patient safety comes first. Even if logging fails, continue providing emergency guidance.""",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "REQUIRED: Patient's ID from PATIENT INFORMATION section. If patient not registered, use session_id or temporary ID.",
                    "required": True
                },
                "emergency_type": {
                    "type": "string",
                    "description": "REQUIRED: Type of emergency. Valid values: 'bleeding', 'knocked_out_tooth', 'broken_tooth', 'severe_pain', 'facial_trauma', 'broken_jaw', 'lost_filling', 'abscess', 'difficulty_breathing', 'severe_facial_swelling'. Use the most specific type available.",
                    "required": True
                },
                "description": {
                    "type": "string",
                    "description": "REQUIRED: Detailed description of the emergency in patient's own words. Include: what happened, when it happened, current symptoms, pain level if mentioned. Be comprehensive.",
                    "required": True
                },
                "severity": {
                    "type": "string",
                    "description": "REQUIRED: Severity level. Valid values: 'critical' (life-threatening, needs 911), 'high' (urgent, needs immediate clinic care), 'moderate' (serious but stable). Match to triage assessment.",
                    "required": True
                }
            }
        )

    def _register_first_aid_tool(self) -> None:
        """Register the provide_first_aid_instructions tool."""
        self.register_tool(
            name="provide_first_aid_instructions",
            function=self.tool_first_aid,
            description="""Provide immediate, step-by-step first aid instructions for a specific dental emergency.

This tool provides emergency-specific guidance for:
- bleeding: Pressure, gauze application, when to seek help
- knocked_out_tooth: How to handle tooth, storage method (milk), time-critical care (1 hour)
- broken_tooth: Rinsing, saving pieces, cold compress, avoid using tooth
- severe_pain: Cold compress, OTC pain relief, what NOT to do

CRITICAL WORKFLOW:
1. Call this tool FIRST when emergency is identified
2. Present instructions clearly to patient
3. Then call report_emergency to log the incident
4. Then call get_emergency_contacts if needed

The tool returns:
- SUCCESS: Step-by-step instructions formatted and ready to share
- SYSTEM_ERROR: Failed to retrieve instructions (provide general emergency guidance)

PRIORITY: Patient safety is paramount. Instructions must be clear, calm, and actionable.""",
            parameters={
                "emergency_type": {
                    "type": "string",
                    "description": "REQUIRED: Type of emergency needing first aid. Valid values: 'bleeding', 'knocked_out_tooth', 'broken_tooth', 'severe_pain', 'abscess', 'lost_filling', 'facial_trauma'. Use exact match for specific instructions.",
                    "required": True
                }
            }
        )

    def _register_emergency_contacts_tool(self) -> None:
        """Register the get_emergency_contacts tool."""
        self.register_tool(
            name="get_emergency_contacts",
            function=self.tool_emergency_contacts,
            description="""Retrieve emergency contact information including 911, clinic emergency line, and poison control.

This tool provides formatted emergency contact numbers:
- 911: For life-threatening emergencies (severe bleeding, difficulty breathing, unconsciousness)
- Clinic Emergency Line: 24/7 dental emergency hotline for knocked out teeth, severe pain, trauma
- Poison Control: For medication or chemical exposure emergencies

WHEN TO USE:
- After providing first aid instructions
- When patient needs immediate care contact
- For critical/high severity emergencies
- When directing patient to appropriate emergency service

The tool returns:
- SUCCESS: Formatted list of emergency contacts with context
- SYSTEM_ERROR: Failed to retrieve (provide 911 as fallback)

The tool requires no parameters and always returns current emergency contact information.""",
            parameters={}
        )

    def _get_system_prompt(self, session_id: str) -> str:
        """
        Generate emergency response system prompt.

        Args:
            session_id: Current session identifier.

        Returns:
            Formatted system prompt string with patient and emergency context.
        """
        global_state = self.state_manager.get_global_state(session_id)
        emergency_state = self.state_manager.get_emergency_state(session_id)
        patient = global_state.patient_profile

        patient_info = self._format_patient_info(patient)
        emergency_status = self._format_emergency_status(emergency_state)
        emergency_guidance = self._format_emergency_guidance()
        emergency_contacts = self._format_emergency_contacts()

        return f"""You are an emergency response coordinator for Bright Smile Dental Clinic.

🚨 EMERGENCY MODE ACTIVE 🚨

{patient_info}

{emergency_status}

CRITICAL PRIORITIES:
1. **Patient Safety First** - Immediate response
2. **Assess Severity** - Determine if 911 is needed
3. **Provide Instructions** - Clear, calm first aid guidance
4. **Direct to Care** - Emergency services or clinic
5. **Document Everything** - Log all interactions

{emergency_guidance}

EMERGENCY CONTACTS:
{emergency_contacts}

COMMUNICATION STYLE:
- **Stay calm and reassuring**
- Use clear, simple instructions
- Speak in short sentences
- Repeat critical information
- Confirm understanding
- Provide immediate next steps

NEVER:
- Minimize patient's concern
- Delay emergency response
- Provide medical treatment advice beyond first aid
- Leave patient without clear next steps

Your goal is to ensure patient safety and direct them to appropriate care immediately."""

    def _format_patient_info(self, patient: Any) -> str:
        """
        Format patient information for system prompt.

        Args:
            patient: Patient profile object.

        Returns:
            Formatted patient information string.
        """
        emergency_contact = (
            f"{patient.emergency_contact_name} ({patient.emergency_contact_phone})"
            if patient.emergency_contact_name and patient.emergency_contact_phone
            else "Not provided"
        )

        return f"""PATIENT INFO:
- Name: {patient.first_name} {patient.last_name}
- Phone: {patient.phone}
- Emergency Contact: {emergency_contact}"""

    def _format_emergency_status(self, emergency_state: Any) -> str:
        """
        Format emergency status for system prompt.

        Args:
            emergency_state: Emergency state object.

        Returns:
            Formatted emergency status string.
        """
        return f"""EMERGENCY STATUS:
- Type: {emergency_state.emergency_type or 'Being assessed'}
- Severity: {emergency_state.severity}
- Location Confirmed: {emergency_state.location_confirmed}"""

    def _format_emergency_guidance(self) -> str:
        """
        Format emergency type guidance for system prompt.

        Returns:
            Formatted emergency guidance string.
        """
        return """EMERGENCY TYPES & RESPONSES:

**CALL 911 IMMEDIATELY:**
- Severe uncontrolled bleeding
- Difficulty breathing/swallowing
- Unconsciousness
- Severe facial trauma
- Suspected broken jaw

**CLINIC EMERGENCY LINE:**
- Knocked out tooth (within 1 hour)
- Severe tooth pain
- Broken tooth with sharp edges
- Lost filling/crown with pain
- Abscess or severe swelling

**FIRST AID GUIDANCE:**
- Bleeding: Apply clean gauze, pressure for 10-15 minutes
- Knocked out tooth: Rinse gently, place in milk, see dentist within 1 hour
- Broken tooth: Rinse mouth, save pieces, apply cold compress
- Severe pain: Cold compress (not heat), over-the-counter pain reliever"""

    def _format_emergency_contacts(self) -> str:
        """
        Format emergency contacts for display.

        Returns:
            Formatted emergency contacts string.
        """
        return "\n".join([
            f"- {name}: {number}"
            for name, number in self.EMERGENCY_CONTACTS.items()
        ])

    # ==================== TOOL IMPLEMENTATIONS ====================

    def tool_report_emergency(
        self,
        session_id: str,
        patient_id: str,
        emergency_type: str,
        description: str,
        severity: str
    ) -> Dict[str, Any]:
        """
        Report and log a dental emergency.

        Args:
            session_id: Current session identifier.
            patient_id: Patient identifier.
            emergency_type: Type of emergency (e.g., "bleeding", "knocked_out_tooth").
            description: Detailed description of the emergency.
            severity: Severity level (critical, high, moderate).

        Returns:
            Dictionary with success status, emergency_id (if successful), and message.
        """
        try:
            clinic_id = self._get_clinic_id()
            emergency_data = self._build_emergency_data(
                clinic_id, patient_id, emergency_type, description, severity
            )

            result = self.db_client.report_emergency(emergency_data)
            self._update_emergency_state(session_id, emergency_type, severity)

            if result:
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "emergency_id": result.get("id"),
                    "message": "Emergency reported and logged",
                    "suggested_response": f"I've reported your {emergency_type} emergency to our clinic. Emergency ID: {result.get('id')}. Please follow the first aid instructions I provided."
                }

            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": "Failed to log emergency",
                "should_retry": True,
                "suggested_response": "I'm having trouble logging the emergency. Let me try again while you follow the first aid instructions..."
            }

        except Exception as e:
            logger.error(f"Error reporting emergency for patient {patient_id}: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": f"Failed to report emergency: {str(e)}",
                "should_retry": True,
                "suggested_response": "I'm having trouble logging the emergency, but please follow the first aid instructions I provided. Your safety is the priority."
            }

    def _get_clinic_id(self) -> str:
        """
        Get clinic ID from database or return default.

        Returns:
            Clinic ID string.
        """
        clinic_info = self.db_client.get_clinic_info()
        return clinic_info.get("id") if clinic_info else "clinic_001"

    def _build_emergency_data(
        self,
        clinic_id: str,
        patient_id: str,
        emergency_type: str,
        description: str,
        severity: str
    ) -> Dict[str, Any]:
        """
        Build emergency data dictionary for database.

        Args:
            clinic_id: Clinic identifier.
            patient_id: Patient identifier.
            emergency_type: Type of emergency.
            description: Emergency description.
            severity: Severity level.

        Returns:
            Dictionary with emergency data.
        """
        return {
            "clinicId": clinic_id,
            "patientId": patient_id,
            "emergencyType": emergency_type,
            "description": description,
            "severity": severity,
            "status": "reported"
        }

    def _update_emergency_state(
        self,
        session_id: str,
        emergency_type: str,
        severity: str
    ) -> None:
        """
        Update emergency state in state manager.

        Args:
            session_id: Current session identifier.
            emergency_type: Type of emergency.
            severity: Severity level.
        """
        self.state_manager.update_emergency_state(
            session_id,
            emergency_type=emergency_type,
            severity=severity
        )

    def tool_first_aid(
        self,
        session_id: str,
        emergency_type: str
    ) -> Dict[str, Any]:
        """
        Provide first aid instructions for a specific emergency type.

        Args:
            session_id: Current session identifier.
            emergency_type: Type of emergency (e.g., "bleeding", "knocked_out_tooth").

        Returns:
            Dictionary with success status, emergency_type, and instructions list.
        """
        try:
            normalized_type = self._normalize_emergency_type(emergency_type)
            instructions = self._get_first_aid_instructions(normalized_type)

            self._update_first_aid_state(session_id, instructions)

            # Format instructions as numbered list for suggested response
            formatted_instructions = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(instructions)])

            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "emergency_type": emergency_type,
                "instructions": instructions,
                "suggested_response": f"Here's what to do right now:\n\n{formatted_instructions}\n\nPlease follow these steps carefully. Are you able to do this?"
            }

        except Exception as e:
            logger.error(
                f"Error providing first aid for {emergency_type}: {e}",
                exc_info=True
            )
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": f"Failed to provide first aid instructions: {str(e)}",
                "should_retry": True,
                "suggested_response": "I'm having trouble retrieving first aid instructions. The most important thing is to stay calm. Can you describe your emergency again?"
            }

    def _normalize_emergency_type(self, emergency_type: str) -> str:
        """
        Normalize emergency type string for lookup.

        Args:
            emergency_type: Raw emergency type string.

        Returns:
            Normalized emergency type (lowercase, underscores instead of spaces).
        """
        return emergency_type.lower().replace(" ", "_")

    def _get_first_aid_instructions(self, emergency_type: str) -> List[str]:
        """
        Get first aid instructions for emergency type.

        Args:
            emergency_type: Normalized emergency type.

        Returns:
            List of first aid instruction strings.
        """
        return FIRST_AID_INSTRUCTIONS.get(
            emergency_type,
            DEFAULT_FIRST_AID_INSTRUCTIONS
        )

    def _update_first_aid_state(
        self,
        session_id: str,
        instructions: List[str]
    ) -> None:
        """
        Update emergency state with first aid instructions.

        Args:
            session_id: Current session identifier.
            instructions: List of first aid instructions.
        """
        self.state_manager.update_emergency_state(
            session_id,
            immediate_actions=instructions,
            first_aid_provided=True
        )

    def tool_emergency_contacts(self, session_id: str) -> Dict[str, Any]:
        """
        Get emergency contact information.

        Args:
            session_id: Current session identifier (unused but required by tool signature).

        Returns:
            Dictionary with success status, contacts dictionary, and message.
        """
        try:
            # Format contacts for suggested response
            contacts_formatted = "\n".join([
                f"• {name}: {number}"
                for name, number in self.EMERGENCY_CONTACTS.items()
            ])

            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "contacts": self.EMERGENCY_CONTACTS,
                "message": "For life-threatening emergencies, call 911 immediately",
                "suggested_response": f"Here are the emergency contacts:\n\n{contacts_formatted}\n\n⚠️ For life-threatening emergencies, call 911 immediately."
            }

        except Exception as e:
            logger.error(f"Error getting emergency contacts: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": f"Failed to retrieve emergency contacts: {str(e)}",
                "should_retry": True,
                "suggested_response": "I'm having trouble retrieving emergency contacts. For life-threatening emergencies, please call 911 immediately."
            }

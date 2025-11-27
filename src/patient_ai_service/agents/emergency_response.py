"""
Emergency Response Agent.

Handles critical dental emergencies with highest priority.
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

logger = logging.getLogger(__name__)


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

    # Emergency contact information
    EMERGENCY_CONTACTS = {
        "911": "Emergency Services (911)",
        "clinic_emergency": "+971-XXX-XXXX (24/7 Emergency Line)",
        "poison_control": "1-800-222-1222"
    }

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        super().__init__(agent_name="EmergencyResponse", **kwargs)
        self.db_client = db_client or DbOpsClient()

    def _register_tools(self):
        """Register emergency response tools."""

        # Report emergency
        self.register_tool(
            name="report_emergency",
            function=self.tool_report_emergency,
            description="Report and log a dental emergency",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID"
                },
                "emergency_type": {
                    "type": "string",
                    "description": "Type of emergency"
                },
                "description": {
                    "type": "string",
                    "description": "Emergency description"
                },
                "severity": {
                    "type": "string",
                    "description": "Severity: critical, high, moderate"
                }
            }
        )

        # Provide first aid
        self.register_tool(
            name="provide_first_aid_instructions",
            function=self.tool_first_aid,
            description="Provide immediate first aid instructions",
            parameters={
                "emergency_type": {
                    "type": "string",
                    "description": "Type of emergency"
                }
            }
        )

        # Get emergency contacts
        self.register_tool(
            name="get_emergency_contacts",
            function=self.tool_emergency_contacts,
            description="Get emergency contact information",
            parameters={}
        )

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate emergency response prompt."""
        global_state = self.state_manager.get_global_state(session_id)
        emergency_state = self.state_manager.get_emergency_state(session_id)
        patient = global_state.patient_profile

        return f"""You are an emergency response coordinator for Bright Smile Dental Clinic.

🚨 EMERGENCY MODE ACTIVE 🚨

PATIENT INFO:
- Name: {patient.first_name} {patient.last_name}
- Phone: {patient.phone}
- Emergency Contact: {patient.emergency_contact_name} ({patient.emergency_contact_phone})

EMERGENCY STATUS:
- Type: {emergency_state.emergency_type or 'Being assessed'}
- Severity: {emergency_state.severity}
- Location Confirmed: {emergency_state.location_confirmed}

CRITICAL PRIORITIES:
1. **Patient Safety First** - Immediate response
2. **Assess Severity** - Determine if 911 is needed
3. **Provide Instructions** - Clear, calm first aid guidance
4. **Direct to Care** - Emergency services or clinic
5. **Document Everything** - Log all interactions

EMERGENCY TYPES & RESPONSES:

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
- Severe pain: Cold compress (not heat), over-the-counter pain reliever

EMERGENCY CONTACTS:
{self._format_emergency_contacts()}

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

    def _format_emergency_contacts(self) -> str:
        """Format emergency contacts for display."""
        return "\n".join([f"- {name}: {number}"
                         for name, number in self.EMERGENCY_CONTACTS.items()])

    # Tool implementations

    def tool_report_emergency(
        self,
        session_id: str,
        patient_id: str,
        emergency_type: str,
        description: str,
        severity: str
    ) -> Dict[str, Any]:
        """Report and log emergency."""
        try:
            # Get clinic ID
            clinic_info = self.db_client.get_clinic_info()
            clinic_id = clinic_info.get("id") if clinic_info else "clinic_001"

            # Report emergency to database
            emergency_data = {
                "clinicId": clinic_id,
                "patientId": patient_id,
                "emergencyType": emergency_type,
                "description": description,
                "severity": severity,
                "status": "reported"
            }

            result = self.db_client.report_emergency(emergency_data)

            # Update state
            self.state_manager.update_emergency_state(
                session_id,
                emergency_type=emergency_type,
                severity=severity
            )

            if result:
                return {
                    "success": True,
                    "emergency_id": result.get("id"),
                    "message": "Emergency reported and logged"
                }
            else:
                return {"error": "Failed to log emergency"}

        except Exception as e:
            logger.error(f"Error reporting emergency: {e}")
            return {"error": str(e)}

    def tool_first_aid(
        self,
        session_id: str,
        emergency_type: str
    ) -> Dict[str, Any]:
        """Provide first aid instructions."""
        try:
            first_aid_instructions = {
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

            instructions = first_aid_instructions.get(
                emergency_type.lower().replace(" ", "_"),
                ["Remain calm", "Contact emergency services or dentist immediately"]
            )

            # Update state
            self.state_manager.update_emergency_state(
                session_id,
                immediate_actions=instructions,
                first_aid_provided=True
            )

            return {
                "success": True,
                "emergency_type": emergency_type,
                "instructions": instructions
            }

        except Exception as e:
            logger.error(f"Error providing first aid: {e}")
            return {"error": str(e)}

    def tool_emergency_contacts(self, session_id: str) -> Dict[str, Any]:
        """Get emergency contact information."""
        try:
            return {
                "success": True,
                "contacts": self.EMERGENCY_CONTACTS,
                "message": "For life-threatening emergencies, call 911 immediately"
            }

        except Exception as e:
            logger.error(f"Error getting emergency contacts: {e}")
            return {"error": str(e)}

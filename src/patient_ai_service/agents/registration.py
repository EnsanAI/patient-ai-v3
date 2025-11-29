"""
Registration Agent.

Handles new patient registration flow.
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

logger = logging.getLogger(__name__)


class RegistrationAgent(BaseAgent):
    """
    Agent for new patient registration.

    Features:
    - Collect patient information
    - Validate required fields
    - Create user and patient records
    - Track completion progress
    """

    REQUIRED_FIELDS = [
        "first_name",
        "last_name",
        "phone",
        "date_of_birth",
        "gender"
    ]

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        super().__init__(agent_name="Registration", **kwargs)
        self.db_client = db_client or DbOpsClient()

    async def on_activated(self, session_id: str, reasoning: Any):
        """
        Set up registration workflow when agent is activated.

        Args:
            session_id: Session identifier
            reasoning: ReasoningOutput from reasoning engine
        """
        # Get current registration state
        reg_state = self.state_manager.get_registration_state(session_id)
        global_state = self.state_manager.get_global_state(session_id)

        # If registration not started, initialize it
        if not reg_state.form_completion:
            logger.info(f"Initializing registration workflow for session {session_id}")

            # Determine missing fields
            missing_fields = []
            for field in self.REQUIRED_FIELDS:
                # Check if field exists in patient profile
                profile_value = getattr(global_state.patient_profile, field, None)
                if not profile_value:
                    missing_fields.append(field)

            # Update registration state
            self.state_manager.update_registration_state(
                session_id,
                current_section="personal_info",
                workflow_step="collecting",
                missing_fields=missing_fields
            )

            logger.info(f"Registration initialized with {len(missing_fields)} missing fields")

    def _register_tools(self):
        """Register registration tools."""

        # Save field
        self.register_tool(
            name="save_field",
            function=self.tool_save_field,
            description="Save a single registration field",
            parameters={
                "field_name": {
                    "type": "string",
                    "description": "Field name (e.g., 'first_name', 'phone')"
                },
                "value": {
                    "type": "string",
                    "description": "Field value"
                }
            }
        )

        # Save list field
        self.register_tool(
            name="save_list",
            function=self.tool_save_list,
            description="Save a list field (allergies, medications, etc.)",
            parameters={
                "field_name": {
                    "type": "string",
                    "description": "Field name (e.g., 'allergies', 'medications')"
                },
                "values": {
                    "type": "array",
                    "description": "List of values",
                    "items": {"type": "string"}
                }
            }
        )

        # Check completion status
        self.register_tool(
            name="check_completion",
            function=self.tool_check_completion,
            description="Check registration completion status",
            parameters={}
        )

        # Complete registration
        self.register_tool(
            name="complete_registration",
            function=self.tool_complete_registration,
            description="Finalize and submit registration",
            parameters={}
        )

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate registration system prompt."""
        reg_state = self.state_manager.get_registration_state(session_id)
        global_state = self.state_manager.get_global_state(session_id)
        
        # Check if registration is actually complete (patient exists in DB)
        patient_registered = (
            global_state.patient_profile.patient_id is not None and
            global_state.patient_profile.patient_id != ""
        )
        registration_complete = reg_state.registration_complete if reg_state else False
        
        # Determine actual registration status
        if patient_registered or registration_complete:
            registration_status = "✅ REGISTRATION COMPLETE - Patient is registered in system"
        elif reg_state.form_completion >= 1.0:
            registration_status = f"⚠️ FORM COMPLETE ({reg_state.form_completion * 100:.0f}%) BUT NOT SUBMITTED - Use complete_registration tool to finalize"
        else:
            registration_status = f"🔄 IN PROGRESS: {reg_state.form_completion * 100:.0f}% complete"

        return f"""You are a patient registration assistant for Bright Smile Dental Clinic.

REGISTRATION STATUS:
{registration_status}
- Current Section: {reg_state.current_section}
- Missing Fields: {', '.join(reg_state.missing_fields) if reg_state.missing_fields else 'None'}

COLLECTED INFORMATION:
{self._format_collected_fields(reg_state.collected_fields)}

YOUR ROLE:
1. Welcome new patients warmly
2. Collect ONLY the 5 required fields (first name, last name, phone, DOB, gender)
3. DO NOT ask for any optional information (email, address, emergency contacts, insurance, allergies, medications, etc.)
4. Validate inputs (phone numbers, dates, etc.)
5. Track progress and show completion status
6. Complete registration as soon as all 5 fields are collected
7. **IMPORTANT**: If user wants to schedule an appointment, redirect them to appointment scheduling AFTER registration is complete

REQUIRED FIELDS:
✓ First Name
✓ Last Name
✓ Phone Number
✓ Date of Birth
✓ Gender

IMPORTANT INSTRUCTIONS:
- When ALL required fields are collected (form_completion = 100%), you MUST call the complete_registration tool
- Do NOT wait for the user to explicitly ask to complete - if form is 100% complete, complete it automatically
- After calling complete_registration successfully, you MUST provide a warm, congratulatory message to the user
- DO NOT just say "Using tool: complete_registration" - instead, provide a complete response like:
  "🎉 Congratulations! Your registration is now complete! You're all set to book appointments with us. How can I help you today?"
- Always follow up tool calls with a natural, conversational response

COLLECTION STRATEGY:
1. Collect first name
2. Collect last name
3. Collect phone number (international format)
4. Collect date of birth (YYYY-MM-DD)
5. Collect gender
6. Complete registration - DO NOT ask for any other information

GUIDELINES:
- One question at a time (don't overwhelm)
- Explain data privacy and security
- Validate formats (phone: international format, DOB: YYYY-MM-DD)
- **CRITICAL: You MUST use the save_field tool to save each piece of information the user provides**
- **When user provides their name, phone, DOB, gender, or emergency contact info, IMMEDIATELY call save_field tool**
- **DO NOT claim registration is complete unless registration_complete flag is True**
- **If form is 100% complete (form_completion >= 1.0), you MUST automatically call complete_registration tool - do not wait for user to ask**
- **If user requests appointment scheduling and registration is incomplete, complete registration first, then redirect to appointment scheduling**
- Be patient and understanding
- Confirm information before finalizing
- Congratulate on completion

PRIVACY STATEMENT:
"Your information is securely stored and used only for providing dental care. We comply with all data protection regulations and never share your information without consent."

Use tools to save information and track progress."""

        return prompt

    def _format_collected_fields(self, fields: Dict[str, Any]) -> str:
        """Format collected fields for display."""
        if not fields:
            return "None yet"

        lines = []
        for key, value in fields.items():
            if value:
                display_key = key.replace("_", " ").title()
                lines.append(f"- {display_key}: {value}")

        return "\n".join(lines) if lines else "None yet"

    # Tool implementations

    def tool_save_field(
        self,
        session_id: str,
        field_name: str,
        value: str
    ) -> Dict[str, Any]:
        """Save a registration field."""
        try:
            reg_state = self.state_manager.get_registration_state(session_id)

            # Update collected fields
            collected = reg_state.collected_fields.copy()
            collected[field_name] = value

            # Update patient profile in global state
            self.state_manager.update_patient_profile(
                session_id,
                **{field_name: value}
            )

            # Calculate completion
            required_collected = sum(
                1 for field in self.REQUIRED_FIELDS
                if field in collected and collected[field]
            )
            completion = required_collected / len(self.REQUIRED_FIELDS)

            # Update missing fields
            missing = [
                field for field in self.REQUIRED_FIELDS
                if field not in collected or not collected[field]
            ]

            # Update state
            self.state_manager.update_registration_state(
                session_id,
                collected_fields=collected,
                form_completion=completion,
                missing_fields=missing
            )
            
            # Auto-complete registration if all required fields are collected
            result = {
                "success": True,
                "field": field_name,
                "value": value,
                "completion": f"{completion * 100:.0f}%",
                "remaining": len(missing),
                "form_complete": completion >= 1.0
            }
            
            # If form is 100% complete, automatically complete registration
            if completion >= 1.0:
                logger.info(f"Form is 100% complete for session {session_id}, auto-completing registration...")
                completion_result = self.tool_complete_registration(session_id)
                if "error" not in completion_result:
                    result["registration_completed"] = True
                    result["patient_id"] = completion_result.get("patient_id")
                    result["user_id"] = completion_result.get("user_id")
                else:
                    result["registration_error"] = completion_result.get("error")
                    logger.error(f"Failed to auto-complete registration: {completion_result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Error saving field: {e}")
            return {"error": str(e)}

    def tool_save_list(
        self,
        session_id: str,
        field_name: str,
        values: list
    ) -> Dict[str, Any]:
        """Save a list field."""
        try:
            # Update patient profile
            self.state_manager.update_patient_profile(
                session_id,
                **{field_name: values}
            )

            # Update collected fields
            reg_state = self.state_manager.get_registration_state(session_id)
            collected = reg_state.collected_fields.copy()
            collected[field_name] = values

            self.state_manager.update_registration_state(
                session_id,
                collected_fields=collected
            )

            return {
                "success": True,
                "field": field_name,
                "values": values,
                "count": len(values)
            }

        except Exception as e:
            logger.error(f"Error saving list: {e}")
            return {"error": str(e)}

    def tool_check_completion(self, session_id: str) -> Dict[str, Any]:
        """Check registration completion status."""
        try:
            reg_state = self.state_manager.get_registration_state(session_id)

            return {
                "success": True,
                "completion": f"{reg_state.form_completion * 100:.0f}%",
                "complete": reg_state.form_completion >= 1.0,
                "missing_fields": reg_state.missing_fields,
                "collected_count": len(reg_state.collected_fields)
            }

        except Exception as e:
            logger.error(f"Error checking completion: {e}")
            return {"error": str(e)}

    def tool_complete_registration(self, session_id: str) -> Dict[str, Any]:
        """Complete registration and create user/patient records."""
        try:
            global_state = self.state_manager.get_global_state(session_id)
            reg_state = self.state_manager.get_registration_state(session_id)
            patient = global_state.patient_profile

            # Validate completion
            if reg_state.form_completion < 1.0:
                return {
                    "error": "Registration incomplete",
                    "missing": reg_state.missing_fields
                }

            # Check if user already exists
            existing_user = self.db_client.get_user_by_phone_number(patient.phone)
            
            if existing_user:
                # User already exists, use existing user_id
                user_id = existing_user.get("id")
                logger.info(f"User already exists with ID: {user_id}, skipping user creation")
            else:
                # Create user account
                logger.info(f"Creating new user for phone: {patient.phone}")
                user_data = self.db_client.register_user(
                    email=patient.email or f"{patient.phone}@temp.clinic",
                    full_name=f"{patient.first_name} {patient.last_name}",
                    phone_number=patient.phone,
                    role_id="patient_role_id",  # Default patient role
                    language_preference=patient.preferred_language
                )

                if not user_data:
                    # If registration failed, try to get user again (might have been created)
                    logger.warning("User registration returned None, checking if user exists...")
                    existing_user = self.db_client.get_user_by_phone_number(patient.phone)
                    if existing_user:
                        user_id = existing_user.get("id")
                        logger.info(f"User found after failed registration attempt: {user_id}")
                    else:
                        return {"error": "Failed to create user account"}
                else:
                    user_id = user_data.get("userId") or user_data.get("id")
                    if not user_id:
                        # Try to get user by phone as fallback
                        existing_user = self.db_client.get_user_by_phone_number(patient.phone)
                        if existing_user:
                            user_id = existing_user.get("id")
                            logger.info(f"Got user_id from phone lookup: {user_id}")
                        else:
                            return {"error": "Failed to get user ID from registration response"}

            # Create patient record
            patient_data = self.db_client.create_patient(
                user_id=user_id,
                first_name=patient.first_name,
                last_name=patient.last_name,
                date_of_birth=patient.date_of_birth,
                gender=patient.gender,
                emergency_contact_name=patient.emergency_contact_name,
                emergency_contact_phone=patient.emergency_contact_phone,
                insurance_provider=patient.insurance_provider,
                insurance_policy_number=patient.insurance_id,
                allergies=patient.allergies,
                medications=patient.medications
            )

            if not patient_data:
                return {"error": "Failed to create patient record"}

            patient_id = patient_data.get("id")

            # Update state
            self.state_manager.update_patient_profile(
                session_id,
                patient_id=patient_id,
                user_id=user_id
            )

            self.state_manager.update_registration_state(
                session_id,
                registration_complete=True
            )

            logger.info(f"Registration completed for patient: {patient_id}")

            return {
                "success": True,
                "patient_id": patient_id,
                "user_id": user_id,
                "message": "Registration completed successfully!"
            }

        except Exception as e:
            logger.error(f"Error completing registration: {e}")
            return {"error": str(e)}

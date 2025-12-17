"""
Registration Agent.

Simplified agent for new patient registration using a single tool with state persistence.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient
from patient_ai_service.models.agentic import ToolResultType

logger = logging.getLogger(__name__)


class RegistrationAgent(BaseAgent):
    """
    Agent for new patient registration.

    Features:
    - Single register_patient tool that accepts partial info
    - State persistence across conversation turns
    - Dynamic result types (SUCCESS, USER_INPUT_NEEDED, SYSTEM_ERROR)
    - Natural conversation flow with entity extraction
    """

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        super().__init__(agent_name="registration", **kwargs)
        self.db_client = db_client or DbOpsClient()

    async def on_activated(self, session_id: str, reasoning: Any):
        """
        Set up registration context when agent is activated.

        Extracts any entities from reasoning and persists them to state.
        """
        logger.info(f"Registration agent activated for session {session_id}")

        # Extract entities from reasoning if available
        if reasoning and hasattr(reasoning, 'response_guidance'):
            task_context = reasoning.response_guidance.task_context
            if hasattr(task_context, 'entities') and task_context.entities:
                entities = task_context.entities
                # Filter to only registration-relevant fields
                valid_fields = ['first_name', 'last_name', 'phone', 'date_of_birth', 'gender']
                updates = {k: v for k, v in entities.items() if k in valid_fields and v}

                if updates:
                    self.state_manager.update_patient_profile(session_id, **updates)
                    logger.info(f"Extracted entities from reasoning: {list(updates.keys())}")

    def _register_tools(self):
        """Register the single registration tool."""
        self.register_tool(
            name="register_patient",
            function=self.tool_register_patient,
            description="""Register a new patient with the clinic.

This tool requires ALL 5 fields to be provided. The tool will:
1. Validate all required fields are present
2. Normalize date format automatically
3. Create user account and patient record in database
4. Return SUCCESS with patient_id when registration is complete

If any field is missing or invalid, the tool returns USER_INPUT_NEEDED with details about what's missing.

REQUIRED FIELDS (all must be provided):
- first_name: Patient's first name
- last_name: Patient's last name
- phone: Phone number (any format accepted)
- date_of_birth: Date of birth (YYYY-MM-DD preferred, but flexible formats accepted)
- gender: Gender (male, female, or other)

The tool will automatically normalize date formats and handle existing users.""",
            parameters={
                "first_name": {
                    "type": "string",
                    "description": "REQUIRED: Patient's first name",
                    "required": True
                },
                "last_name": {
                    "type": "string",
                    "description": "REQUIRED: Patient's last name",
                    "required": True
                },
                "phone": {
                    "type": "string",
                    "description": "REQUIRED: Phone number in any format (e.g., +1234567890, 123-456-7890)",
                    "required": True
                },
                "date_of_birth": {
                    "type": "string",
                    "description": "REQUIRED: Date of birth. Preferred format: YYYY-MM-DD (e.g., 1990-05-15). Also accepts: DD-MM-YYYY, DD/MM/YYYY, MM-DD-YYYY formats which will be auto-normalized",
                    "required": True
                },
                "gender": {
                    "type": "string",
                    "description": "REQUIRED: Gender. Valid values: male, female, other",
                    "required": True
                }
            }
        )

    def _get_agent_instructions(self) -> str:
        """Registration-specific behavioral instructions."""
        return """MANDATORY: Always confirm ALL details with user before calling register_patient.
Show: name, phone, date_of_birth, gender. ONCE confirmed, call register_patient with ALL 5 fields immediately."""

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate simplified registration system prompt."""
        global_state = self.state_manager.get_global_state(session_id)
        patient = global_state.patient_profile
        reg_state = self.state_manager.get_registration_state(session_id)

        # Check if already registered
        if patient.patient_id:
            return f"""You are a patient registration assistant for Bright Smile Dental Clinic.

✅ REGISTRATION COMPLETE
Patient {patient.first_name} {patient.last_name} is already registered.
Patient ID: {patient.patient_id}

If user needs to update information or has other requests, guide them appropriately."""

        # Build status display
        known_info = []
        missing_info = []

        fields = [
            ("first_name", "First name", patient.first_name),
            ("last_name", "Last name", patient.last_name),
            ("phone", "Phone", patient.phone),
            ("date_of_birth", "Date of birth", patient.date_of_birth),
            ("gender", "Gender", patient.gender),
        ]

        for field_name, display_name, value in fields:
            if value:
                known_info.append(f"✓ {display_name}: {value}")
            else:
                missing_info.append(display_name)

        status_section = "\n".join(known_info) if known_info else "No information collected yet"
        missing_section = ", ".join(missing_info) if missing_info else "All fields collected!"

        return f"""You are a patient registration assistant for Bright Smile Dental Clinic.

REGISTRATION STATUS:
{status_section}

STILL NEEDED:
{missing_section}

INSTRUCTIONS:
1. Greet new patients warmly
2. Call register_patient with any new info you learn (the tool saves progress automatically)
3. If tool returns USER_INPUT_NEEDED → ask user for missing fields naturally
4. If tool returns SUCCESS → congratulate and confirm registration
5. You can ask for multiple fields at once to speed up the process

TOOL: register_patient(first_name, last_name, phone, date_of_birth, gender)
- ALL parameters are REQUIRED - you must have all 5 fields before calling
- Date format: YYYY-MM-DD preferred (but tool accepts various formats and normalizes automatically)
- If you don't have all fields yet, use COLLECT_INFORMATION to ask the user for missing fields

PRIVACY: Your information is securely stored and used only for providing dental care."""

    def tool_register_patient(
        self,
        session_id: str,
        first_name: str,
        last_name: str,
        phone: str,
        date_of_birth: str,
        gender: str
    ) -> Dict[str, Any]:
        """
        Register a new patient with the clinic.

        All parameters are REQUIRED. The tool validates all fields, normalizes the date,
        and creates the user account and patient record.

        Returns:
            - SUCCESS: All fields valid, patient registered
            - USER_INPUT_NEEDED: Missing or invalid fields, agent should collect correct info
            - SYSTEM_ERROR: DB failure, should retry
        """
        try:
            # === STEP 1: Check if already registered ===
            global_state = self.state_manager.get_global_state(session_id)
            patient = global_state.patient_profile

            if patient.patient_id:
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "patient_id": patient.patient_id,
                    "user_id": patient.user_id,
                    "already_registered": True,
                    "suggested_response": f"You're already registered! Your patient ID is {patient.patient_id}."
                }

            # === STEP 2: Validate all required fields are provided ===
            fields = {
                "first_name": first_name.strip() if first_name else None,
                "last_name": last_name.strip() if last_name else None,
                "phone": phone.strip() if phone else None,
                "date_of_birth": date_of_birth.strip() if date_of_birth else None,
                "gender": gender.strip().lower() if gender else None
            }

            # Check for missing fields
            missing = [k for k, v in fields.items() if not v]
            if missing:
                friendly_missing = [f.replace('_', ' ') for f in missing]
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "missing_fields": missing,
                    "blocks_criteria": "registration_complete",
                    "suggested_response": f"I need your {', '.join(friendly_missing)} to complete registration.",
                    "next_action": "collect_information"
                }

            # === STEP 3: Normalize date ===
            normalized_dob = self._normalize_date_of_birth(fields["date_of_birth"])
            if not normalized_dob:
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": "invalid_date_format",
                    "invalid_field": "date_of_birth",
                    "invalid_value": fields["date_of_birth"],
                    "blocks_criteria": "registration_complete",
                    "suggested_response": f"I couldn't parse '{fields['date_of_birth']}' as a date. Could you provide your date of birth as YYYY-MM-DD (e.g., 1990-05-15)?"
                }

            # Update fields with normalized date
            fields["date_of_birth"] = normalized_dob

            # === STEP 4: Save to state (for persistence) ===
            self.state_manager.update_patient_profile(session_id, **fields)
            logger.info(f"Saving registration info for patient: {fields['first_name']} {fields['last_name']}")

            # === STEP 4: Normalize date ===
            normalized_dob = self._normalize_date_of_birth(fields["date_of_birth"])
            if not normalized_dob:
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": "invalid_date_format",
                    "invalid_field": "date_of_birth",
                    "invalid_value": fields["date_of_birth"],
                    "blocks_criteria": "registration_complete",
                    "suggested_response": f"I couldn't parse '{fields['date_of_birth']}' as a date. Could you provide your date of birth as YYYY-MM-DD (e.g., 1990-05-15)?"
                }

            # === STEP 5: Create user & patient in DB ===
            existing_user = self.db_client.get_user_by_phone_number(fields["phone"])

            if existing_user:
                user_id = existing_user.get("id")
                logger.info(f"Found existing user: {user_id}")
            else:
                # Create user
                user_data = self.db_client.register_user(
                    email=f"{fields['phone']}@temp.clinic",
                    full_name=f"{fields['first_name']} {fields['last_name']}",
                    phone_number=fields["phone"],
                    role_id="patient_role_id"
                )
                if not user_data:
                    # Try to fetch user again (might have been created by race condition)
                    existing_user = self.db_client.get_user_by_phone_number(fields["phone"])
                    if existing_user:
                        user_id = existing_user.get("id")
                    else:
                        return {
                            "success": False,
                            "result_type": ToolResultType.SYSTEM_ERROR.value,
                            "error": "user_creation_failed",
                            "should_retry": True,
                            "suggested_response": "I'm having trouble creating your account. Let me try again..."
                        }
                else:
                    user_id = user_data.get("userId") or user_data.get("id")
                    if not user_id:
                        # Fallback: try to get user by phone
                        existing_user = self.db_client.get_user_by_phone_number(fields["phone"])
                        if existing_user:
                            user_id = existing_user.get("id")
                        else:
                            return {
                                "success": False,
                                "result_type": ToolResultType.SYSTEM_ERROR.value,
                                "error": "user_id_not_returned",
                                "should_retry": True,
                                "suggested_response": "I'm having trouble setting up your account. Let me try again..."
                            }

            # Create patient record
            patient_data = self.db_client.create_patient(
                user_id=user_id,
                first_name=fields["first_name"],
                last_name=fields["last_name"],
                date_of_birth=normalized_dob,
                gender=fields["gender"]
            )

            if not patient_data:
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "patient_creation_failed",
                    "should_retry": True,
                    "suggested_response": "I'm having trouble creating your patient record. Let me try again..."
                }

            patient_id = patient_data.get("id")

            # === STEP 6: Update final state ===
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
                "result_type": ToolResultType.SUCCESS.value,
                "patient_id": patient_id,
                "user_id": user_id,
                "satisfies_criteria": ["registration_complete"],
                "suggested_response": f"Welcome {fields['first_name']}! You're now registered. Your patient ID is {patient_id}. You're all set to book appointments!"
            }

        except Exception as e:
            logger.error(f"Registration error: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True,
                "suggested_response": "Something went wrong. Let me try again..."
            }

    def _normalize_date_of_birth(self, date_str: str) -> Optional[str]:
        """
        Normalize date of birth to YYYY-MM-DD format.

        Accepts various formats:
        - YYYY-MM-DD (already correct)
        - DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        - MM-DD-YYYY, MM/DD/YYYY (US format)

        Returns normalized date in YYYY-MM-DD format or None if invalid.
        """
        if not date_str or not date_str.strip():
            return None

        date_str = date_str.strip()

        # If already in YYYY-MM-DD format, validate and return
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except ValueError:
                pass

        # Try various date formats
        date_formats = [
            ('%d-%m-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
            ('%d/%m/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%d.%m.%Y', r'^\d{1,2}\.\d{1,2}\.\d{4}$'),
            ('%m-%d-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
            ('%m/%d/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
        ]

        for fmt, pattern in date_formats:
            if re.match(pattern, date_str):
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    normalized = parsed_date.strftime('%Y-%m-%d')
                    logger.info(f"Normalized date '{date_str}' to '{normalized}'")
                    return normalized
                except ValueError:
                    continue

        # Flexible approach: extract numbers and try to construct a date
        numbers = re.findall(r'\d+', date_str)
        if len(numbers) >= 3:
            try:
                day, month, year = map(int, numbers[:3])

                # Validate ranges
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                    try:
                        parsed_date = datetime(year, month, day)
                        normalized = parsed_date.strftime('%Y-%m-%d')
                        logger.info(f"Normalized date '{date_str}' to '{normalized}'")
                        return normalized
                    except ValueError:
                        # Try swapping day/month
                        try:
                            parsed_date = datetime(year, day, month)
                            normalized = parsed_date.strftime('%Y-%m-%d')
                            logger.info(f"Normalized date '{date_str}' to '{normalized}' (swapped day/month)")
                            return normalized
                        except ValueError:
                            pass
            except (ValueError, IndexError):
                pass

        logger.warning(f"Could not normalize date format: {date_str}")
        return None

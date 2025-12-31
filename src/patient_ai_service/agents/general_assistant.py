"""
General Assistant Agent.

Handles greetings, general inquiries, and clinic information.
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient
from patient_ai_service.models.agentic import ToolResultType

logger = logging.getLogger(__name__)


class GeneralAssistantAgent(BaseAgent):
    """
    Agent for general inquiries and greetings.

    Features:
    - Welcome and greet patients
    - Answer questions about clinic operations
    - Provide clinic information (hours, location, services)
    - Guide patients to appropriate services
    """

    def __init__(self, db_client: Optional[DbOpsClient] = None, appointment_manager=None, **kwargs):
        super().__init__(agent_name="GeneralAssistant", **kwargs)
        self.db_client = db_client or DbOpsClient()
        self.appointment_manager = appointment_manager  # Will be set by orchestrator if needed

    def _register_tools(self):
        """Register general assistant tools."""

        # Get clinic information
        self.register_tool(
            name="get_clinic_info",
            function=self.tool_get_clinic_info,
            description="Get general clinic information (location, hours, contact)",
            parameters={}
        )

        # Get all clinic branches
        self.register_tool(
            name="get_all_clinics",
            function=self.tool_get_all_clinics,
            description="Get information about all clinic branches",
            parameters={}
        )

        # Get available services/procedures
        self.register_tool(
            name="get_services",
            function=self.tool_get_services,
            description="Get list of dental services and procedures offered",
            parameters={}
        )

        # Get insurance providers
        self.register_tool(
            name="get_insurance_info",
            function=self.tool_get_insurance_info,
            description="Get information about accepted insurance providers",
            parameters={}
        )

        # Get payment methods
        self.register_tool(
            name="get_payment_methods",
            function=self.tool_get_payment_methods,
            description="Get available payment methods",
            parameters={}
        )

        # Get doctors list
        self.register_tool(
            name="get_doctors",
            function=self.tool_get_doctors,
            description="Get list of available doctors with their specialties, languages, and information",
            parameters={}
        )

        # Get specific doctor information
        self.register_tool(
            name="get_doctor_info",
            function=self.tool_get_doctor_info,
            description="Get detailed information about a specific doctor by name or ID",
            parameters={
                "doctor_name": {
                    "type": "string",
                    "description": "Full name or partial name of the doctor (e.g., 'Mohammed Atef' or 'Atef')"
                }
            }
        )

        # Get specific procedure information
        self.register_tool(
            name="get_procedure_info",
            function=self.tool_get_procedure_info,
            description="Get detailed information about a specific dental procedure including price, duration, and description",
            parameters={
                "procedure_name": {
                    "type": "string",
                    "description": "Name of the procedure (e.g., 'Root Canal Treatment', 'Implant', 'Veneers')"
                }
            }
        )

        # Find doctor by name (with fuzzy matching)
        self.register_tool(
            name="find_doctor_by_name",
            function=self.tool_find_doctor_by_name,
            description="Find a doctor by name using fuzzy matching. Handles variations in spelling, transliterations, and multilingual names (e.g., 'Mohamed' vs 'Mohammed', 'mo7amed')",
            parameters={
                "doctor_name": {
                    "type": "string",
                    "description": "Doctor's name to search for (e.g., 'Dr. Ahmed', 'Mohammed Atef', 'mo7amed')"
                }
            }
        )

        # Check doctor availability
        self.register_tool(
            name="check_doctor_availability",
            function=self.tool_check_doctor_availability,
            description="Check if a doctor is available on a specific date and optionally at a specific time. Requires doctor_id (use find_doctor_by_name first to get the ID)",
            parameters={
                "doctor_id": {
                    "type": "string",
                    "description": "Doctor's UUID (obtained from find_doctor_by_name or get_doctors)"
                },
                "date": {
                    "type": "string",
                    "description": "Date to check availability in YYYY-MM-DD format (e.g., '2024-03-15')"
                },
                "requested_time": {
                    "type": "string",
                    "description": "Optional specific time to check. Supports formats like '14:00', '2pm', '14:30' (24-hour or 12-hour with AM/PM)"
                }
            }
        )

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate system prompt for general assistant."""
        global_state = self.state_manager.get_global_state(session_id)
        patient = global_state.patient_profile
        
        # Check if patient is registered
        patient_registered = (
            patient.patient_id is not None and
            patient.patient_id != ""
        )
        
        # Check if this is a new conversation (first message)
        conversation_history = self.conversation_history.get(session_id, [])
        is_first_message = len(conversation_history) == 0

        return f"""You are a friendly, helpful assistant for Bright Smile Dental Clinic.

PATIENT INFO:
- Name: {patient.first_name or 'Guest'} {patient.last_name or ''}
- Phone: {patient.phone or 'Not provided'}
- Language: {global_state.detected_language}
- Registration Status: {'✅ Registered' if patient_registered else '❌ Not Registered'}

CONVERSATION CONTEXT:
- This is {'the FIRST message' if is_first_message else 'a CONTINUATION'} in this conversation
- Patient is {'registered' if patient_registered else 'NOT registered'}

YOUR ROLE:
You're the first point of contact for patients. You help them with:
1. **Greetings & Welcome** - Make them feel comfortable
2. **General Questions** - Clinic hours, location, services, prices, doctors, procedures
3. **Navigation** - Guide them to the right service or agent
4. **Information** - Insurance, payments, procedures

WHAT YOU CAN HELP WITH:
✓ Clinic information (location, hours, contact)
✓ Services and procedures offered (use get_services tool)
✓ Doctor information and specialties (use get_doctors and get_doctor_info tools)
✓ Finding doctors by name (use find_doctor_by_name tool for fuzzy matching)
✓ Checking doctor availability (use check_doctor_availability tool)
✓ Specific procedure details, prices, and descriptions (use get_procedure_info tool)
✓ Insurance and payment options
✓ General dental health information
✓ Directing to appointments, emergencies, or registration
✓ Answering common questions

IMPORTANT: When users ask about:
- **Doctors**: ALWAYS use get_doctors or get_doctor_info tool to provide accurate information
- **Finding specific doctors by name**: Use find_doctor_by_name tool - it handles variations in spelling and transliterations (e.g., "Mohamed" vs "Mohammed", "mo7amed")
- **Doctor availability**: Use check_doctor_availability tool with doctor_id and date. MUST get doctor_id first using find_doctor_by_name
- **Procedures**: ALWAYS use get_services or get_procedure_info tool to provide accurate details including prices
- **Services**: ALWAYS use get_services tool to list available procedures

REGISTRATION HANDLING:
- **If this is the FIRST message and patient is NOT registered**: 
  - Welcome them warmly
  - Briefly mention that you can help with questions about the clinic
  - PROACTIVELY but GENTLY offer registration: "I notice you're not registered yet. Would you like to register with us? It's quick and will allow you to book appointments. Or feel free to ask me any questions about our clinic first!"
  - DO NOT force registration - let them ask questions if they want
  
- **If patient is NOT registered but asks questions**:
  - Answer their questions fully and helpfully
  - DO NOT push registration while they're asking questions
  - Let them explore and get information
  
- **If patient is NOT registered and wants to book an appointment**:
  - Politely explain: "To book an appointment, we'll need to complete a quick registration first. It only takes a few minutes. Would you like to start the registration process?"
  - Then redirect them to registration

- **If patient IS registered**:
  - Treat them as a returning patient
  - No need to mention registration

COMMUNICATION STYLE:
- **Warm and friendly** - Use a conversational, welcoming tone
- **Professional** - Maintain clinic standards
- **Helpful** - Anticipate needs and offer assistance
- **Clear** - Use simple language, avoid jargon
- **Proactive but not pushy** - Offer registration naturally, don't force it
- **Respectful** - If they want to ask questions first, let them!

EXAMPLE INTERACTIONS:

**First message from unregistered user:**
User: "Hello"
You: "Hello! Welcome to Bright Smile Dental Clinic! 😊 I'm here to help you with any questions about our services, doctors, procedures, or appointments. I notice you're not registered yet - would you like to register with us? It's quick and will allow you to book appointments. Or feel free to ask me any questions about our clinic first!"

**Unregistered user asks questions:**
User: "What are your hours?"
You: "We're here to help you! Let me get our current hours for you."
[Use get_clinic_info tool]
[Answer their question - don't push registration]

User: "What procedures do you offer?"
You: "Great question! Let me show you all our available procedures."
[Use get_services tool]
[Provide full information - let them explore]

**Unregistered user wants to book:**
User: "I want to book an appointment"
You: "I'd be happy to help you book an appointment! To schedule with us, we'll need to complete a quick registration first - it only takes a few minutes and will allow us to create your patient profile. Would you like to start the registration process now?"

**Registered user:**
User: "Hello"
You: "Hello! Welcome back to Bright Smile Dental Clinic! How can I help you today?"

IMPORTANT:
- Answer questions fully and helpfully - don't interrupt with registration offers
- Only mention registration when: (1) First message from unregistered user, or (2) They want to book
- Never force registration - make it feel natural and helpful
- If someone needs an appointment → Check if registered, if not, offer registration first
- If medical emergency → Escalate immediately
- If medical question → Acknowledge and recommend seeing dentist

Always be ready to hand off to specialized agents when needed.

CLINIC DETAILS:
- Name: iSmile Dental Clinic
- Services: General dentistry, cosmetic, orthodontics, emergency care
- Languages: We serve patients in multiple languages

Use your tools to provide accurate, up-to-date information!"""

    # Tool implementations

    def tool_get_clinic_info(self, session_id: str) -> Dict[str, Any]:
        """Get general clinic information."""
        try:
            clinic = self.db_client.get_clinic_info()
            
            if not clinic:
                # Return default info - still a success
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "clinic_info": {
                        "name": "Bright Smile Dental Clinic",
                        "phone": "+1-234-567-8900",
                        "address": "123 Smile Street",
                        "hours": "9:00 AM - 6:00 PM"
                    },
                    "can_proceed": True,
                    "next_action": "present_info_to_user"
                }
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "clinic_info": {
                    "name": clinic.get("name"),
                    "phone": clinic.get("phone"),
                    "address": clinic.get("address"),
                    "city": clinic.get("city"),
                    "hours": clinic.get("working_hours")
                },
                "can_proceed": True,
                "next_action": "present_info_to_user"
            }
            
        except Exception as e:
            logger.error(f"Error getting clinic info: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "CLINIC_INFO_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_get_all_clinics(self, session_id: str) -> Dict[str, Any]:
        """Get all clinic branches."""
        try:
            clinics = self.db_client.get_all_clinics()
            
            if not clinics:
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "clinics": [],
                    "count": 0,
                    "message": "No clinic information available",
                    "can_proceed": True,
                    "next_action": "inform_user_no_clinics"
                }
            
            clinic_list = []
            for clinic in clinics:
                clinic_list.append({
                    "id": clinic.get("id"),
                    "name": clinic.get("name"),
                    "address": clinic.get("address"),
                    "city": clinic.get("city"),
                    "phone": clinic.get("phone"),
                    "email": clinic.get("email")
                })
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "clinics": clinic_list,
                "count": len(clinic_list),
                "can_proceed": True,
                "next_action": "present_clinics_to_user"
            }
            
        except Exception as e:
            logger.error(f"Error getting all clinics: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "CLINICS_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_get_services(self, session_id: str) -> Dict[str, Any]:
        """Get available dental services."""
        try:
            procedures = self.db_client.get_all_dental_procedures()
            
            if not procedures:
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "services": [],
                    "count": 0,
                    "message": "Service information currently unavailable",
                    "can_proceed": True,
                    "next_action": "inform_user_no_services"
                }
            
            services = []
            for proc in procedures:
                services.append({
                    "name": proc.get("name"),
                    "description": proc.get("description"),
                    "category": proc.get("category"),
                    "estimated_duration": proc.get("estimated_duration")
                })
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "services": services,
                "count": len(services),
                "can_proceed": True,
                "next_action": "present_services_to_user"
            }
            
        except Exception as e:
            logger.error(f"Error getting services: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "SERVICES_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_get_insurance_info(self, session_id: str) -> Dict[str, Any]:
        """Get insurance provider information."""
        try:
            insurance_providers = self.db_client.get_insurance_providers()
            
            if not insurance_providers:
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "providers": [],
                    "count": 0,
                    "message": "Please contact us for insurance information",
                    "can_proceed": True,
                    "next_action": "inform_user_contact_for_insurance"
                }
            
            providers = []
            for provider in insurance_providers:
                providers.append({
                    "name": provider.get("provider_name"),
                    "network": provider.get("network_type"),
                    "coverage": provider.get("coverage_percentage")
                })
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "providers": providers,
                "count": len(providers),
                "message": "We accept these insurance providers",
                "can_proceed": True,
                "next_action": "present_insurance_to_user"
            }
            
        except Exception as e:
            logger.error(f"Error getting insurance info: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "INSURANCE_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_get_payment_methods(self, session_id: str) -> Dict[str, Any]:
        """Get available payment methods."""
        try:
            payment_methods = self.db_client.get_payment_methods()
            
            if not payment_methods:
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "methods": [],
                    "count": 0,
                    "message": "Please contact us for payment options",
                    "can_proceed": True,
                    "next_action": "inform_user_contact_for_payment"
                }
            
            methods = []
            for method in payment_methods:
                methods.append({
                    "type": method.get("payment_type"),
                    "accepted": method.get("is_accepted"),
                    "notes": method.get("notes")
                })
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "methods": methods,
                "count": len(methods),
                "can_proceed": True,
                "next_action": "present_payment_methods_to_user"
            }
            
        except Exception as e:
            logger.error(f"Error getting payment methods: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "PAYMENT_METHODS_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_get_doctors(self, session_id: str) -> Dict[str, Any]:
        """Get list of available doctors."""
        try:
            doctors = self.db_client.get_doctors()
            
            if not doctors:
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "doctors": [],
                    "count": 0,
                    "message": "Doctor information currently unavailable",
                    "can_proceed": True,
                    "next_action": "inform_user_no_doctors"
                }
            
            doctor_list = []
            for doctor in doctors:
                doctor_list.append({
                    "id": doctor.get("id"),
                    "name": f"{doctor.get('first_name', '')} {doctor.get('last_name', '')}".strip(),
                    "specialties": doctor.get("specialties", []),
                    "languages": doctor.get("languages", []),
                    "bio": doctor.get("bio"),
                    "clinic": doctor.get("clinic_name")
                })
            
            return {
                "success": True,
                "result_type": ToolResultType.PARTIAL.value,  # PARTIAL: data retrieved, agent needs to use it
                "doctors": doctor_list,
                "count": len(doctor_list),
                "can_proceed": True,
                "next_action": "select_doctor_or_present_to_user"
            }
            
        except Exception as e:
            logger.error(f"Error getting doctors: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "DOCTORS_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_get_doctor_info(self, session_id: str, doctor_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific doctor."""
        try:
            doctors = self.db_client.get_doctors()
            
            if not doctors:
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "Could not retrieve doctor information",
                    "error_code": "DOCTORS_UNAVAILABLE",
                    "should_retry": True,
                    "can_proceed": False
                }
            
            # Search for doctor by name (case-insensitive, partial match)
            doctor_name_lower = doctor_name.lower()
            matching_doctors = []
            
            for doctor in doctors:
                first_name = doctor.get("first_name", "").lower()
                last_name = doctor.get("last_name", "").lower()
                full_name = f"{first_name} {last_name}".strip()
                
                if (doctor_name_lower in first_name or 
                    doctor_name_lower in last_name or 
                    doctor_name_lower in full_name):
                    matching_doctors.append(doctor)
            
            if not matching_doctors:
                # No match found - need user to clarify
                all_doctor_names = [
                    f"{d.get('first_name', '')} {d.get('last_name', '')}".strip()
                    for d in doctors
                ]
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": f"No doctor found matching '{doctor_name}'",
                    "error_code": "DOCTOR_NOT_FOUND",
                    "alternatives": all_doctor_names,
                    "can_proceed": False,
                    "suggested_response": f"I couldn't find a doctor named '{doctor_name}'. Our available doctors are: {', '.join(all_doctor_names)}. Which doctor would you like to know about?"
                }
            
            # Found matching doctor(s)
            doctor = matching_doctors[0]
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "doctor": {
                    "id": doctor.get("id"),
                    "name": f"{doctor.get('first_name', '')} {doctor.get('last_name', '')}".strip(),
                    "title": doctor.get("title", "Dr."),
                    "specialties": doctor.get("specialties", []),
                    "languages": doctor.get("languages", []),
                    "bio": doctor.get("bio"),
                    "clinic": doctor.get("clinic_name"),
                    "experience_years": doctor.get("experience_years")
                },
                "can_proceed": True,
                "next_action": "present_doctor_info_to_user"
            }
            
        except Exception as e:
            logger.error(f"Error getting doctor info: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "DOCTOR_INFO_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_get_procedure_info(self, session_id: str, procedure_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific dental procedure."""
        try:
            procedures = self.db_client.get_all_dental_procedures()

            if not procedures:
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "Could not retrieve procedure information",
                    "error_code": "PROCEDURES_UNAVAILABLE",
                    "should_retry": True,
                    "can_proceed": False
                }

            # Search for procedure by name (case-insensitive, partial match)
            procedure_name_lower = procedure_name.lower()
            matching_procedures = []

            for proc in procedures:
                proc_name = proc.get("name", "").lower()
                if procedure_name_lower in proc_name or proc_name in procedure_name_lower:
                    matching_procedures.append(proc)

            if not matching_procedures:
                all_procedure_names = [p.get("name", "") for p in procedures]
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": f"No procedure found matching '{procedure_name}'",
                    "error_code": "PROCEDURE_NOT_FOUND",
                    "alternatives": all_procedure_names,
                    "can_proceed": False,
                    "suggested_response": f"I couldn't find a procedure called '{procedure_name}'. Our available procedures include: {', '.join(all_procedure_names[:5])}. Which one would you like to know about?"
                }

            procedure = matching_procedures[0]

            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "procedure": {
                    "name": procedure.get("name"),
                    "description": procedure.get("description"),
                    "category": procedure.get("category"),
                    "estimated_duration": procedure.get("estimated_duration"),
                    "price_range": procedure.get("price_range"),
                    "recovery_time": procedure.get("recovery_time")
                },
                "can_proceed": True,
                "next_action": "present_procedure_info_to_user"
            }

        except Exception as e:
            logger.error(f"Error getting procedure info: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "PROCEDURE_INFO_FETCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_find_doctor_by_name(self, session_id: str, doctor_name: str) -> Dict[str, Any]:
        """
        Find a doctor by name using the appointment manager's fuzzy matching.

        Args:
            session_id: Session identifier
            doctor_name: Doctor's name to search for

        Returns:
            Dictionary with doctor information or error
        """
        try:
            # Check if appointment manager is available
            if not self.appointment_manager:
                logger.warning("Appointment manager not available, falling back to basic search")
                # Fallback to basic get_doctor_info if appointment manager not available
                return self.tool_get_doctor_info(session_id, doctor_name)

            # Use appointment manager's fuzzy matching
            result = self.appointment_manager.tool_find_doctor_by_name(session_id, doctor_name)

            # Return the result from appointment manager
            return result

        except Exception as e:
            logger.error(f"Error finding doctor by name: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "DOCTOR_SEARCH_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

    def tool_check_doctor_availability(
        self,
        session_id: str,
        doctor_id: str,
        date: str,
        requested_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check doctor availability on a specific date and optionally at a specific time.

        Args:
            session_id: Session identifier
            doctor_id: Doctor's UUID (must be obtained from find_doctor_by_name or get_doctors first)
            date: Date in YYYY-MM-DD format
            requested_time: Optional time to check (e.g., '14:00', '2pm', '14:30')

        Returns:
            Dictionary with availability information or error
        """
        try:
            # Check if appointment manager is available
            if not self.appointment_manager:
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "Appointment manager not available",
                    "error_code": "APPOINTMENT_MANAGER_UNAVAILABLE",
                    "should_retry": False,
                    "can_proceed": False,
                    "suggested_response": "I apologize, but I cannot check doctor availability at the moment. Please try booking an appointment directly."
                }

            # Use appointment manager's availability check
            result = self.appointment_manager.tool_check_availability(
                session_id=session_id,
                doctor_id=doctor_id,
                date=date,
                requested_time=requested_time
            )

            return result

        except Exception as e:
            logger.error(f"Error checking doctor availability: {e}")
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "error_code": "AVAILABILITY_CHECK_FAILED",
                "should_retry": True,
                "can_proceed": False
            }

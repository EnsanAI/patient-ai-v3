"""
General Assistant Agent.

Handles greetings, general inquiries, and clinic information.
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

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

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        super().__init__(agent_name="GeneralAssistant", **kwargs)
        self.db_client = db_client or DbOpsClient()

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
✓ Specific procedure details, prices, and descriptions (use get_procedure_info tool)
✓ Insurance and payment options
✓ General dental health information
✓ Directing to appointments, emergencies, or registration
✓ Answering common questions

IMPORTANT: When users ask about:
- **Doctors**: ALWAYS use get_doctors or get_doctor_info tool to provide accurate information
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
        """Get clinic information."""
        try:
            clinic_info = self.db_client.get_clinic_info()

            if not clinic_info:
                return {
                    "success": False,
                    "error": "Could not retrieve clinic information"
                }

            return {
                "success": True,
                "clinic": {
                    "name": clinic_info.get("name"),
                    "address": clinic_info.get("address"),
                    "city": clinic_info.get("city"),
                    "country": clinic_info.get("country"),
                    "phone": clinic_info.get("phone"),
                    "email": clinic_info.get("email"),
                    "website": clinic_info.get("website")
                }
            }

        except Exception as e:
            logger.error(f"Error getting clinic info: {e}")
            return {"error": str(e)}

    def tool_get_all_clinics(self, session_id: str) -> Dict[str, Any]:
        """Get all clinic branches."""
        try:
            clinics = self.db_client.get_all_clinics()

            if not clinics:
                return {
                    "success": True,
                    "clinics": [],
                    "message": "No clinic information available"
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
                "clinics": clinic_list,
                "count": len(clinic_list)
            }

        except Exception as e:
            logger.error(f"Error getting all clinics: {e}")
            return {"error": str(e)}

    def tool_get_services(self, session_id: str) -> Dict[str, Any]:
        """Get available dental services."""
        try:
            procedures = self.db_client.get_all_dental_procedures()

            if not procedures:
                return {
                    "success": True,
                    "services": [],
                    "message": "Service information currently unavailable"
                }

            # Group by specialty if available
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
                "services": services,
                "count": len(services)
            }

        except Exception as e:
            logger.error(f"Error getting services: {e}")
            return {"error": str(e)}

    def tool_get_insurance_info(self, session_id: str) -> Dict[str, Any]:
        """Get insurance provider information."""
        try:
            insurance_providers = self.db_client.get_insurance_providers()

            if not insurance_providers:
                return {
                    "success": True,
                    "providers": [],
                    "message": "Please contact us for insurance information"
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
                "providers": providers,
                "count": len(providers),
                "message": "We accept these insurance providers"
            }

        except Exception as e:
            logger.error(f"Error getting insurance info: {e}")
            return {"error": str(e)}

    def tool_get_payment_methods(self, session_id: str) -> Dict[str, Any]:
        """Get available payment methods."""
        try:
            payment_methods = self.db_client.get_payment_methods()

            if not payment_methods:
                return {
                    "success": True,
                    "methods": [],
                    "message": "Please contact us for payment options"
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
                "methods": methods,
                "count": len(methods)
            }

        except Exception as e:
            logger.error(f"Error getting payment methods: {e}")
            return {"error": str(e)}

    def tool_get_doctors(self, session_id: str) -> Dict[str, Any]:
        """Get list of available doctors."""
        try:
            doctors = self.db_client.get_doctors()

            if not doctors:
                return {
                    "success": True,
                    "doctors": [],
                    "message": "Doctor information currently unavailable"
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
                "doctors": doctor_list,
                "count": len(doctor_list)
            }

        except Exception as e:
            logger.error(f"Error getting doctors: {e}")
            return {"error": str(e)}

    def tool_get_doctor_info(self, session_id: str, doctor_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific doctor."""
        try:
            # First, get all doctors
            doctors = self.db_client.get_doctors()

            if not doctors:
                return {
                    "success": False,
                    "error": "Could not retrieve doctor information"
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
                return {
                    "success": False,
                    "error": f"No doctor found matching '{doctor_name}'",
                    "suggestion": "Use get_doctors tool to see all available doctors"
                }

            # If multiple matches, return the first one (or could return all)
            doctor = matching_doctors[0]
            
            # Get detailed info if we have an ID
            doctor_detail = None
            if doctor.get("id"):
                doctor_detail = self.db_client.get_doctor_by_id(doctor.get("id"))

            # Use detailed info if available, otherwise use basic info
            result_doctor = doctor_detail if doctor_detail else doctor

            return {
                "success": True,
                "doctor": {
                    "id": result_doctor.get("id"),
                    "name": f"{result_doctor.get('first_name', '')} {result_doctor.get('last_name', '')}".strip(),
                    "specialties": result_doctor.get("specialties", []),
                    "languages": result_doctor.get("languages", []),
                    "bio": result_doctor.get("bio"),
                    "clinic": result_doctor.get("clinic_name"),
                    "email": result_doctor.get("email"),
                    "phone": result_doctor.get("phone")
                }
            }

        except Exception as e:
            logger.error(f"Error getting doctor info: {e}")
            return {"error": str(e)}

    def tool_get_procedure_info(self, session_id: str, procedure_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific procedure."""
        try:
            # First, try to get all procedures and do fuzzy matching
            all_procedures = self.db_client.get_all_dental_procedures()
            
            if not all_procedures:
                # Fallback: try API endpoint
                procedures = self.db_client.get_procedure_by_name(procedure_name)
                if not procedures or len(procedures) == 0:
                    return {
                        "success": False,
                        "error": f"No procedure found matching '{procedure_name}'",
                        "suggestion": "Use get_services tool to see all available procedures"
                    }
            else:
                # Do fuzzy matching on all procedures
                procedure_name_lower = procedure_name.lower()
                # Split search terms
                search_terms = procedure_name_lower.split()
                
                matching = []
                for proc in all_procedures:
                    proc_name_lower = proc.get("name", "").lower()
                    # Check if any search term is in the procedure name
                    if any(term in proc_name_lower for term in search_terms if len(term) > 2):
                        matching.append(proc)
                
                # If no matches, try exact substring match
                if not matching:
                    matching = [p for p in all_procedures 
                              if procedure_name_lower in p.get("name", "").lower()]
                
                procedures = matching

            if not procedures or len(procedures) == 0:
                return {
                    "success": False,
                    "error": f"No procedure found matching '{procedure_name}'",
                    "suggestion": "Use get_services tool to see all available procedures"
                }

            # Return the first matching procedure (or could return all)
            procedure = procedures[0] if isinstance(procedures, list) else procedures

            return {
                "success": True,
                "procedure": {
                    "id": procedure.get("id"),
                    "name": procedure.get("name"),
                    "description": procedure.get("description"),
                    "price": procedure.get("price"),
                    "duration_minutes": procedure.get("duration_minutes"),
                    "specialty": procedure.get("specialty_name"),
                    "requires_preparation": procedure.get("requires_preparation", False),
                    "preparation_instructions": procedure.get("preparation_instructions")
                },
                "message": f"Found procedure: {procedure.get('name')}"
            }

        except Exception as e:
            logger.error(f"Error getting procedure info: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

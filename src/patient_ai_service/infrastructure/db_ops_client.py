import logging
import re
import json
from typing import Optional, Dict, Any, List
from patient_ai_service.infrastructure.mcp_connection import mcp

logger = logging.getLogger(__name__)

class DbOpsClient:
    """
    MCP-Native Client (Async Only).
    Routes all PatientAI data requests to the CareBot MCP Server.
    """
    def __init__(self, base_url: Optional[str] = None, **kwargs):
        # We accept kwargs to maintain backward compatibility with old init calls
        logger.info("🚀 DbOpsClient initialized in MCP-Native Mode (Async)")
        self._tool_mapping = {}

    async def initialize(self):
        """Check connection and list available tools on startup."""
        logger.info("🔌 Connecting to MCP Server to verify tools...")
        try:
            result = await mcp.list_tools()
            if result and hasattr(result, 'tools'):
                tool_names = [t.name for t in result.tools]
                logger.info(f"✅ MCP Orchestrator Connected. Discovered {len(tool_names)} capabilities.")
                self._build_tool_mapping(tool_names)
                
                # Log missing critical tools
                critical_intents = ["get_doctors", "get_clinic_info", "book_appointment"]
                missing = [i for i in critical_intents if i not in self._tool_mapping]
                if missing:
                    logger.warning(f"⚠️ Missing critical tools: {missing} (will try resource fallbacks)")
            else:
                logger.warning("⚠️ MCP Connected but no tools found or list failed.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP client: {e}")

    def _build_tool_mapping(self, available_tools: List[str]):
        """Map internal method intents to actual MCP tool names."""
        # Map: internal_intent -> [possible_mcp_tool_names]
        mapping_rules = {
            "get_doctors": ["get_doctors", "list_doctors", "fetch_doctors"],
            "get_clinic_info": ["get_clinic_info", "fetch_clinic_info", "clinic_details"],
            "get_all_dental_procedures": ["get_all_dental_procedures", "list_procedures", "get_procedures"],
            "get_all_clinics": ["get_all_clinics", "list_clinics"],
            "get_payment_methods": ["get_payment_methods", "list_payment_methods"],
            "get_appointment_types": ["get_appointment_types", "list_appointment_types"],
            "get_visit_type_fees": ["get_visit_type_fees", "list_visit_fees"],
            "book_appointment": ["book_appointment", "create_appointment"],
            "get_specialties": ["get_specialties", "list_specialties"],
            "get_insurance_providers": ["get_insurance_providers", "list_insurance_providers", "get_insurance_providers_resource"],
            "check_system_health": ["check_system_health", "health_check"],
            "search_staff_tools": ["search_staff_tools", "search_tools"],
            "resolve_patient_by_phone": ["resolve_patient_by_phone", "find_patient"],
            "get_available_slots": ["get_available_slots", "list_available_slots"],
            "cancel_appointment": ["cancel_appointment", "delete_appointment"],
            "get_patient_appointments": ["get_patient_appointments", "list_patient_appointments"],
            "report_emergency": ["report_emergency", "create_emergency"],
            "create_medical_inquiry": ["create_medical_inquiry", "log_inquiry"],
            "log_communication": ["log_communication", "add_communication_log"],
            "get_doctor_by_id": ["get_doctor_by_id", "get_doctor"],
            "get_appointment_by_id": ["get_appointment_by_id", "get_appointment"],
            "reschedule_appointment": ["reschedule_appointment", "move_appointment"],
            "update_appointment": ["update_appointment", "patch_appointment"],
        }
        
        for intent, candidates in mapping_rules.items():
            for candidate in candidates:
                if candidate in available_tools:
                    self._tool_mapping[intent] = candidate
                    logger.info(f"  Mapped '{intent}' -> '{candidate}'")
                    break

    async def _execute_mapped(self, intent: str, args: dict = None):
        """Execute a tool using the resolved name."""
        tool_name = self._tool_mapping.get(intent, intent)
        logger.info(f"🎯 [Orchestrator] Mapping internal intent '{intent}' -> MCP Tool '{tool_name}'")
        return await mcp.execute(tool_name, args)

    # --- 1. CORE LOOKUP LOGIC ---

    async def get_patient_by_phone_number(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Smart Lookup via MCP."""
        try:
            text = await self._execute_mapped("resolve_patient_by_phone", {"phone_number": phone_number})
            if text and "Found:" in text:
                match = re.search(r'ID: ([a-f0-9\-]+)', text)
                if match:
                    # Return object to satisfy Agent expectation
                    return {"id": match.group(1), "phone": phone_number, "source": "mcp"}
            return None
        except Exception:
            return None

    # --- 2. APPOINTMENTS (Booking, Searching, Modifying) ---

    async def get_available_doctors(self, date: str, start_time: str, end_time: str, **kwargs) -> List[Dict]:
        """Search availability via MCP."""
        query = f"Find available doctors on {date} between {start_time} and {end_time}"
        try:
            res = await self._execute_mapped("search_staff_tools", {"query": query})
            # Wrap in list to satisfy legacy agents iterating over results
            return [{"raw_availability": res}] 
        except Exception:
            return []

    async def create_appointment(self, **kwargs) -> Dict[str, Any]:
        """Books appointment via MCP."""
        try:
            # Map to the expected MCP tool arguments (names and separate date/time)
            params = {
                "patient_name": kwargs.get("patient_name") or kwargs.get("full_name"),
                "doctor_name": kwargs.get("doctor_name"),
                "date": kwargs.get("appointment_date"),
                "start_time": kwargs.get("start_time"),
                "end_time": kwargs.get("end_time")
            }
            
            # Fallback for legacy calls that only provide IDs and 'time'
            if not params["date"] and kwargs.get("time"):
                time_parts = kwargs.get("time").split()
                if len(time_parts) >= 2:
                    params["date"] = time_parts[0]
                    params["start_time"] = time_parts[1]

            # Remove None values to avoid validation errors
            params = {k: v for k, v in params.items() if v is not None}
            
            res = await self._execute_mapped("book_appointment", params)
            return {"id": "mcp_booked", "status": "scheduled", "details": res}
        except Exception as e:
            logger.error(f"Booking failed: {e}")
            return {"error": f"Booking failed: {str(e)}"}

    async def cancel_appointment(self, appointment_id: str, cancellation_reason: str) -> bool:
        try:
            await self._execute_mapped("cancel_appointment", {"appointment_id": appointment_id, "reason": cancellation_reason})
            return True
        except:
            return False

    async def get_patient_appointments(self, patient_id: str) -> List[Dict]:
        """Get patient appointments via MCP."""
        try:
            res = await self._execute_mapped("get_patient_appointments", {"patient_id": patient_id})
            if res:
                return json.loads(res)
            return []
        except Exception as e:
            logger.error(f"Failed to get appointments: {e}")
            return []

    async def get_available_time_slots(self, doctor_id: str, date: str, **kwargs) -> List[Dict]:
        """Get available slots via MCP."""
        try:
            res = await self._execute_mapped("get_available_slots", {
                "doctor_id": doctor_id,
                "date": date
            })
            if res:
                return json.loads(res)
            return []
        except Exception as e:
            logger.error(f"Failed to get slots: {e}")
            return []

    # --- 3. NEW FAMILIES (Emergency, Insurance, Inquiry) ---

    async def report_emergency(self, emergency_data: Dict) -> Dict:
        res = await self._execute_mapped("report_emergency", {
            "clinic_id": emergency_data.get('clinicId', 'default'),
            "patient_id": emergency_data.get('patientId'),
            "description": emergency_data.get('description'),
            "priority": emergency_data.get('priority', 'routine')
        })
        return {"status": "success", "response": res}

    async def get_insurance_providers(self) -> List[Dict]:
        text = await self._execute_mapped("get_insurance_providers", {})
        return [{"info": text}]

    async def create_inquiry(self, data: Dict) -> Dict:
        res = await self._execute_mapped("create_medical_inquiry", {
            "patient_id": data.get('patientId'),
            "subject": data.get('subject'),
            "message": data.get('message')
        })
        return {"status": "success", "response": res}

    async def add_communication_log(self, patient_id: str, message: str, **kwargs) -> bool:
        try:
            await self._execute_mapped("log_communication", {
                "patient_id": patient_id, 
                "message": message
            })
            return True
        except:
            return False

    # --- 4. CORE DATA METHODS (Connected to MCP) ---
    
    def _parse_doctors_text(self, res: str) -> List[Dict]:
        """Parse text format: 'Clinic Staff Registry:\n- Name (Title) | Languages: ...'"""
        logger.info("Parsing text response from doctors list")
        doctors = []
        for line in res.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                # Format: "- First Last (Title) | Languages: A, B"
                try:
                    # Remove dash and split by pipe
                    content = line[1:].strip()
                    parts = content.split('|')
                    main_part = parts[0].strip()
                    langs_part = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Parse Name and Title
                    name_match = re.match(r"^(.*?)\s*\((.*?)\)$", main_part)
                    if name_match:
                        name = name_match.group(1).strip()
                        title = name_match.group(2).strip()
                    else:
                        name = main_part
                        title = "Doctor"
                    
                    # Parse Languages
                    languages = []
                    if "Languages:" in langs_part:
                        lang_str = langs_part.replace("Languages:", "").strip()
                        languages = [l.strip() for l in lang_str.split(',')]
                    
                    # Split name for first/last
                    name_parts = name.split()
                    first_name = name_parts[0] if name_parts else ""
                    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
                    
                    doctors.append({
                        "id": name.lower().replace(" ", "_"), # Fallback ID
                        "first_name": first_name,
                        "last_name": last_name,
                        "title": title,
                        "specialties": [title],
                        "languages": languages,
                        "bio": line
                    })
                except Exception as parse_err:
                    logger.warning(f"Failed to parse doctor line '{line}': {parse_err}")
                    continue
        return doctors

    async def get_doctors(self, **kwargs) -> List[Dict]:
        """Get doctors list via MCP."""
        res = None
        # Try tool first if mapped
        if "get_doctors" in self._tool_mapping:
            try:
                res = await self._execute_mapped("get_doctors", {})
            except Exception as e:
                logger.error(f"Failed to get doctors via tool: {e}")
        
        # Fallback: Try resource if tool failed or wasn't mapped
        if not res:
            try:
                logger.info("⚠️ Tool 'get_doctors' not found or failed. Trying resource 'doctors://list'...")
                res = await mcp.read_resource("doctors://list")
            except Exception as e:
                logger.error(f"Failed to get doctors via resource: {e}")
        
        if res:
            try:
                return json.loads(res)
            except json.JSONDecodeError:
                return self._parse_doctors_text(res)
            
        return []

    async def get_clinic_info(self, **kwargs) -> Dict:
        """Get clinic info via MCP."""
        try:
            res = await self._execute_mapped("get_clinic_info", {})
            if res:
                try:
                    return json.loads(res)
                except json.JSONDecodeError:
                    return {"info": res}
            return {}
        except Exception as e:
            logger.error(f"Failed to get clinic info: {e}")
            return {}

    # --- 5. MISSING METHODS IMPLEMENTATION (Mapped to MCP Tools) ---

    async def get_all_dental_procedures(self) -> List[Dict]:
        """Get all dental procedures via MCP."""
        try:
            res = await self._execute_mapped("get_all_dental_procedures", {})
            if res:
                try:
                    return json.loads(res)
                except json.JSONDecodeError:
                    return [{"info": res}]
            return []
        except Exception as e:
            logger.error(f"Failed to get procedures: {e}")
            return []

    async def get_all_clinics(self) -> List[Dict]:
        """Get all clinics via MCP."""
        try:
            res = await self._execute_mapped("get_all_clinics", {})
            if res:
                return json.loads(res)
            return []
        except Exception as e:
            logger.error(f"Failed to get clinics: {e}")
            return []

    async def get_payment_methods(self) -> List[Dict]:
        """Get payment methods via MCP."""
        try:
            res = await self._execute_mapped("get_payment_methods", {})
            if res:
                return json.loads(res)
            return []
        except Exception as e:
            logger.error(f"Failed to get payment methods: {e}")
            return []

    async def get_doctor_by_id(self, doctor_id: str) -> Optional[Dict]:
        """Get doctor by ID via MCP."""
        try:
            res = await self._execute_mapped("get_doctor_by_id", {"doctor_id": doctor_id})
            if res:
                return json.loads(res)
            return None
        except Exception as e:
            logger.error(f"Failed to get doctor {doctor_id}: {e}")
            return None

    async def get_appointment_types(self) -> List[Dict]:
        """Get appointment types via MCP."""
        try:
            res = await self._execute_mapped("get_appointment_types", {})
            if res:
                return json.loads(res)
            return []
        except Exception as e:
            logger.error(f"Failed to get appointment types: {e}")
            return []

    async def get_visit_type_fees(self) -> List[Dict]:
        """Get visit type fees via MCP."""
        try:
            res = await self._execute_mapped("get_visit_type_fees", {})
            if res:
                try:
                    return json.loads(res)
                except json.JSONDecodeError:
                    return [{"info": res}]
            return []
        except Exception as e:
            logger.error(f"Failed to get visit fees: {e}")
            return []

    async def get_appointment_by_id(self, appointment_id: str) -> Optional[Dict]:
        """Get appointment by ID via MCP."""
        try:
            res = await self._execute_mapped("get_appointment_by_id", {"appointment_id": appointment_id})
            if res:
                return json.loads(res)
            return None
        except Exception as e:
            logger.error(f"Failed to get appointment {appointment_id}: {e}")
            return None

    async def reschedule_appointment(self, appointment_id: str, new_date: str, new_time: str, new_end_time: str = None) -> Dict:
        """Reschedule appointment via MCP."""
        try:
            res = await self._execute_mapped("reschedule_appointment", {
                "appointment_id": appointment_id,
                "new_date": new_date,
                "new_time": new_time,
                "new_end_time": new_end_time
            })
            if res:
                return json.loads(res)
            return {}
        except Exception as e:
            logger.error(f"Failed to reschedule appointment: {e}")
            return {}

    async def update_appointment(self, appointment_id: str, updates: Dict) -> Dict:
        """Update appointment via MCP."""
        try:
            # Flatten updates into params
            params = {"appointment_id": appointment_id}
            params.update(updates)
            res = await self._execute_mapped("update_appointment", params)
            if res:
                return json.loads(res)
            return {}
        except Exception as e:
            logger.error(f"Failed to update appointment: {e}")
            return {}
            
    async def get_specialties(self) -> List[str]:
        """Get specialties via MCP."""
        try:
            res = await self._execute_mapped("get_specialties", {})
            if res:
                return json.loads(res)
            return []
        except Exception as e:
            logger.error(f"Failed to get specialties: {e}")
            return []

    async def check_system_health(self) -> Dict:
        """Diagnostic check via MCP."""
        try:
            res = await self._execute_mapped("check_system_health", {})
            return {"status": "online", "mcp_response": res}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

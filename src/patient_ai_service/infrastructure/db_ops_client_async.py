import logging
import re
import json
from typing import Optional, Dict, Any, List
from src.patient_ai_service.infrastructure.mcp_connection import mcp

logger = logging.getLogger(__name__)

class DbOpsClient:
    """
    MCP-Native Client (Async Only).
    Routes all PatientAI data requests to the CareBot MCP Server.
    """
    def __init__(self, base_url: Optional[str] = None, **kwargs):
        # We accept kwargs to maintain backward compatibility with old init calls
        logger.info("🚀 DbOpsClient initialized in MCP-Native Mode (Async)")

    # --- 1. CORE LOOKUP LOGIC ---

    async def get_patient_by_phone_number(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Smart Lookup via MCP."""
        try:
            text = await mcp.execute("resolve_patient_by_phone", {"phone_number": phone_number})
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
            res = await mcp.execute("search_staff_tools", {"query": query})
            # Wrap in list to satisfy legacy agents iterating over results
            return [{"raw_availability": res}] 
        except Exception:
            return []

    async def create_appointment(self, **kwargs) -> Dict[str, Any]:
        """Books appointment via MCP."""
        # Note: MCP 'book_appointment' tool handles the logic
        try:
            # We construct a natural language request for the MCP tool
            # or call the tool directly if arguments match perfect
            # Here we map specific legacy args to the tool
            res = await mcp.execute("book_appointment", {
                "doctor_id": kwargs.get("doctor_id"),
                "patient_id": kwargs.get("patient_id"),
                "time": f"{kwargs.get('appointment_date')} {kwargs.get('start_time')}"
            })
            return {"id": "mcp_booked", "status": "scheduled", "details": res}
        except Exception:
            return {"error": "Booking failed"}

    async def cancel_appointment(self, appointment_id: str, cancellation_reason: str) -> bool:
        try:
            await mcp.execute("cancel_appointment", {"appointment_id": appointment_id, "reason": cancellation_reason})
            return True
        except:
            return False

    async def get_patient_appointments(self, patient_id: str) -> List[Dict]:
        # Implementation depends on specific MCP tool availability
        # For now, return empty or implement specific 'list_appointments' tool
        return []

    async def get_available_time_slots(self, doctor_id: str, date: str, **kwargs) -> List[Dict]:
        # Return empty list to prevent crash, implement specific tool later
        return []

    # --- 3. NEW FAMILIES (Emergency, Insurance, Inquiry) ---

    async def report_emergency(self, emergency_data: Dict) -> Dict:
        res = await mcp.execute("report_emergency", {
            "clinic_id": emergency_data.get('clinicId', 'default'),
            "patient_id": emergency_data.get('patientId'),
            "description": emergency_data.get('description'),
            "priority": emergency_data.get('priority', 'routine')
        })
        return {"status": "success", "response": res}

    async def get_insurance_providers(self) -> List[Dict]:
        text = await mcp.execute("get_insurance_providers_resource", {})
        return [{"info": text}]

    async def create_inquiry(self, data: Dict) -> Dict:
        res = await mcp.execute("create_medical_inquiry", {
            "patient_id": data.get('patientId'),
            "subject": data.get('subject'),
            "message": data.get('message')
        })
        return {"status": "success", "response": res}

    async def add_communication_log(self, patient_id: str, message: str, **kwargs) -> bool:
        try:
            await mcp.execute("log_communication", {
                "patient_id": patient_id, 
                "message": message
            })
            return True
        except:
            return False

    # --- 4. LEGACY STUBS (To prevent AttributeErrors) ---
    
    async def get_doctors(self, **kwargs): return []
    async def get_clinic_info(self, **kwargs): return {}
    async def get_specialties(self): return []

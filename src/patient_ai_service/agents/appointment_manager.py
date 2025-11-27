"""
Appointment Manager Agent.

Handles appointment booking, rescheduling, cancellation, and checking.
"""

import logging
import asyncio
import hashlib
import time as time_module
import concurrent.futures
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from datetime import datetime as dt

from .base_agent import BaseAgent
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

logger = logging.getLogger(__name__)


class AppointmentManagerAgent(BaseAgent):
    """
    Agent responsible for all appointment-related operations.

    Features:
    - Book new appointments
    - Reschedule existing appointments
    - Cancel appointments
    - Check appointment status
    - Find available time slots
    """

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        super().__init__(agent_name="AppointmentManager", **kwargs)
        self.db_client = db_client or DbOpsClient()

    def _register_tools(self):
        """Register appointment-related tools."""

        # Get doctors list
        self.register_tool(
            name="list_doctors",
            function=self.tool_list_doctors,
            description="Get list of available doctors with their specialties and languages",
            parameters={}
        )

        # Check doctor availability
        # Calculate dates dynamically for the description
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        today_str = today.strftime('%Y-%m-%d')
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        
        self.register_tool(
            name="check_availability",
            function=self.tool_check_availability,
            description=f"Check available time slots for a specific doctor on a date. IMPORTANT: doctor_id must be a UUID (use list_doctors first to get it), not a doctor name. When parsing 'tomorrow', calculate it dynamically: today is {today_str}, so tomorrow is {tomorrow_str}.",
            parameters={
                "doctor_id": {
                    "type": "string",
                    "description": "Doctor's UUID (NOT name - use list_doctors tool first to get the UUID)"
                },
                "date": {
                    "type": "string",
                    "description": f"Date in YYYY-MM-DD format (e.g., '{tomorrow_str}'). Parse 'tomorrow' to actual date based on today ({today_str}). Always calculate tomorrow dynamically from the current date."
                },
                "requested_time": {
                    "type": "string",
                    "description": "Optional: Specific time to check (e.g., '14:00', '2pm', '2:00 PM'). Convert to 24-hour format."
                }
            }
        )

        # Book appointment
        self.register_tool(
            name="book_appointment",
            function=self.tool_book_appointment,
            description="Book a new appointment. MANDATORY: When check_availability returns 'MANDATORY_ACTION': 'CALL book_appointment TOOL IMMEDIATELY', you MUST call this tool immediately. Do NOT generate text - make the tool call. Use 'general consultation' for reason if not provided.",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "Patient's ID (get from Patient ID in PATIENT INFORMATION section - REQUIRED)"
                },
                "doctor_id": {
                    "type": "string",
                    "description": "Doctor's UUID (from find_doctor_by_name tool result - REQUIRED)"
                },
                "date": {
                    "type": "string",
                    "description": "Appointment date in YYYY-MM-DD format (e.g., '2025-11-26' - REQUIRED)"
                },
                "time": {
                    "type": "string",
                    "description": "Appointment time in HH:MM format (e.g., '15:00' for 3:00 PM - REQUIRED)"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for appointment (e.g., 'cleaning', 'checkup', 'consultation'). Use 'general consultation' if not provided."
                }
            }
        )

        # Check patient appointments
        self.register_tool(
            name="check_patient_appointments",
            function=self.tool_check_patient_appointments,
            description="Get list of patient's appointments",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "Patient's ID"
                }
            }
        )

        # Cancel appointment
        self.register_tool(
            name="cancel_appointment",
            function=self.tool_cancel_appointment,
            description="Cancel an existing appointment",
            parameters={
                "appointment_id": {
                    "type": "string",
                    "description": "Appointment ID to cancel"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for cancellation"
                }
            }
        )

        # Reschedule appointment
        self.register_tool(
            name="reschedule_appointment",
            function=self.tool_reschedule_appointment,
            description="Reschedule an existing appointment",
            parameters={
                "appointment_id": {
                    "type": "string",
                    "description": "Appointment ID to reschedule"
                },
                "new_date": {
                    "type": "string",
                    "description": "New date (YYYY-MM-DD)"
                },
                "new_time": {
                    "type": "string",
                    "description": "New time (HH:MM)"
                }
            }
        )

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate system prompt with context."""
        # Get state
        global_state = self.state_manager.get_global_state(session_id)
        agent_state = self.state_manager.get_appointment_state(session_id)
        patient = global_state.patient_profile

        # Check if patient is registered
        patient_registered = (
            patient.patient_id is not None and
            patient.patient_id != ""
        )
        
        # Calculate actual dates for dynamic examples
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        today_str = today.strftime('%Y-%m-%d')
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        current_year = today.year
        current_month = today.month
        
        # Debug logging
        logger.info(f"🔍 AppointmentManager._get_system_prompt - session: {session_id}, patient_id: {patient.patient_id}, registered: {patient_registered}")
        
        registration_status = "✅ Registered" if patient_registered else "❌ Not Registered - Registration Required"
        
        context = f"""You are an appointment manager for Bright Smile Dental Clinic.

PATIENT INFORMATION:
- Name: {patient.first_name or 'Not provided'} {patient.last_name or ''}
- Patient ID: {patient.patient_id or 'None - Registration Required'}
- Phone: {patient.phone or 'Not provided'}
- Language: {patient.preferred_language or 'Not set'}
- Registration Status: {registration_status}

CURRENT STATE:
- Workflow Step: {agent_state.workflow_step}
- Operation Type: {agent_state.operation_type or 'Not set'}

YOUR RESPONSIBILITIES:
1. Help patients book new appointments
2. Reschedule existing appointments
3. Cancel appointments
4. Check appointment status
5. Provide available time slots
6. Recommend appropriate appointment types and durations

CRITICAL: APPOINTMENT BOOKING REQUIREMENTS:
- **Patient MUST be registered before booking an appointment**
- If Patient ID is None or empty, you MUST redirect to registration first
- You CANNOT use book_appointment tool without a valid patient_id
- You CAN help them explore options (doctors, availability) before registration
- But when they actually want to BOOK, redirect to registration first

HANDLING UNREGISTERED USERS:
- If user wants to book but is NOT registered:
  1. Check availability first (they can see options)
  2. If available: "Dr. [Name] is available on [date] at [time]. To book, I'll need to complete your registration first - it takes just 2 minutes. Should I proceed?"
  3. Mention registration requirement ONCE only - don't repeat
  4. DO NOT attempt to use book_appointment tool without patient_id
- You CAN help them:
  - See available doctors (use list_doctors)
  - Check availability for specific dates (use check_availability)
  - Show them what's available before requiring registration

GUIDELINES:
- Be friendly, professional, and efficient
- **Interpret user intent immediately** - if they provide complete info, treat it as a booking request
- **Book immediately when user confirms** - don't ask for confirmation again
- Offer alternative times if preferred slot is unavailable (suggest 2-3 options ONCE)
- **Only ask for missing information** - never ask for details already provided
- Consider urgency when suggesting appointments
- Keep responses concise and human-like
- **If patient is not registered and wants to book, mention registration requirement ONCE only**
- **CRITICAL: After using tools, always provide a natural, conversational response - never just say "Using tool: X"**

AVAILABLE TOOLS:
- list_doctors: Get available doctors with their IDs (UUIDs) - **USE THIS FIRST** to get the doctor's UUID
- check_availability: Check doctor's available slots (REQUIRES doctor UUID from list_doctors, NOT doctor name)
- book_appointment: Create a new appointment (REQUIRES patient_id - only use if patient is registered)
- check_patient_appointments: View patient's appointments (REQUIRES patient_id)
- cancel_appointment: Cancel an appointment (REQUIRES patient_id)
- reschedule_appointment: Change appointment date/time (REQUIRES patient_id)

**CRITICAL TOOL USAGE RULES:**
1. **NEVER use check_availability with a doctor name** - always use list_doctors first to get the UUID
2. **ALWAYS parse dates**: "tomorrow" → actual date (YYYY-MM-DD), "25th november" → "{current_year}-11-25"
3. **ALWAYS parse times**: "2pm" → "14:00", "2:00 PM" → "14:00", "11 am" → "11:00"
4. **Use the doctor_id (UUID) from list_doctors result** for check_availability

CRITICAL BOOKING WORKFLOW - SMOOTH, FAST, HUMAN-LIKE:

GOAL: Book appointments in ≤3 messages with zero contradictions or unnecessary loops.

RULES:
1. **Interpret Complete Requests Immediately**: If user provides doctor, date, time, and reason in one message, treat it as a complete booking request.
2. **No Unnecessary Questions**: Never ask for details the user already provided.
3. **One Availability Check**: Check availability ONCE. If available, confirm immediately. If unavailable, suggest nearest times ONCE.
4. **No Contradictions**: Never say a doctor is unavailable then later say they are available. Trust your first availability check.
5. **Immediate Booking on Confirmation**: When user says "yes", "schedule", "book it", "do it", or similar, book IMMEDIATELY without repeating previous messages.
6. **Concise Responses**: Keep messages short, warm, and human-like. No verbose explanations.

WORKFLOW:
**Scenario A: User provides complete info (doctor, date, time, reason) in one message:**
1. **MANDATORY FIRST STEP**: Get doctor list using list_doctors tool, then find the doctor by name from the list
2. **CRITICAL**: Extract the doctor "id" (UUID) from list_doctors result - this is what you'll use for check_availability
3. **Parse date IMMEDIATELY and CORRECTLY**: 
   - "tomorrow" → Calculate: Today is {today_str}, so tomorrow is {tomorrow_str} (YYYY-MM-DD format)
   - "26th nov" or "26th november" or "november 26" → "{current_year}-11-26" (use current year {current_year} if year not specified)
   - "25th november" or "november 25" → "{current_year}-11-25" (use current year {current_year} if year not specified)
   - "{current_year}-{current_month:02d}-25" → Use as-is
   - ALWAYS use YYYY-MM-DD format for dates
   - Handle ordinal numbers: "26th", "27th", "28th", etc.
   - ALWAYS calculate dates dynamically based on today's date
4. **Parse time IMMEDIATELY**: 
   - "3pm" → "15:00"
   - "2pm" → "14:00"
   - "2:00 PM" → "14:00"
   - "11 am" → "11:00"
   - "3:00 PM" → "15:00"
   - "14:00" → Use as-is
   - ALWAYS use HH:MM format (24-hour) for times
5. **IMMEDIATELY call check_availability** with:
   - doctor_id: The UUID from list_doctors result (e.g., "c1111111-c111-c111-c111-c11111111111")
   - date: Parsed date in YYYY-MM-DD format (e.g., "{tomorrow_str}")
   - requested_time: Parsed time in HH:MM format (e.g., "14:00")
6. **Analyze availability result CRITICALLY - THIS IS MANDATORY**: 
   - When check_availability tool returns, you MUST check the tool result JSON
   - **LOOK FOR**: "available_at_requested_time": True AND "MANDATORY_ACTION": "CALL book_appointment TOOL IMMEDIATELY"
   - **CRITICAL RULE**: If you see BOTH of these in the tool result AND patient is registered (Patient ID exists), you MUST IMMEDIATELY call book_appointment tool
   - **DO NOT generate any text response** - when you see "MANDATORY_ACTION": "CALL book_appointment TOOL IMMEDIATELY", you MUST make a tool call
   - **DO NOT say "booking" or "would you like me to book"** - you MUST call the book_appointment tool
   - **DO NOT generate text saying "Using tool: book_appointment"** - you MUST actually make a tool call using the tool
   - The tool result will include "required_parameters" - use those exact values for the book_appointment tool call
   - If False or is_conflicting is True: The time is NOT available - proceed to step 9
   - DO NOT ask for more information if availability check shows the slot is free
7. **If available AND patient registered (Patient ID exists) - MANDATORY ACTION**: 
   - **YOU MUST CALL book_appointment tool IMMEDIATELY** - this is not optional, it's mandatory
   - The tool result from check_availability will show "available_at_requested_time": True
   - When you see this, call book_appointment tool immediately with:
     * patient_id: From PATIENT INFORMATION section
     * doctor_id: The UUID from find_doctor_by_name result
     * date: The parsed date (YYYY-MM-DD)
     * time: The parsed time (HH:MM, 24-hour format)
     * reason: From user message OR "general consultation" if not provided
   - NO confirmation needed
   - NO asking "would you like to book?"
   - NO waiting for user to say "yes"
   - NO asking for reason if user didn't provide it - use "general consultation"
   - After booking succeeds, provide a natural confirmation message like: "Sure! Your appointment with Dr. [Name] on [date] at [time] is scheduled. ✅"
8. **If available BUT patient NOT registered (Patient ID is None or empty)**: 
   - Say: "Dr. [Name] is available on [date] at [time]. To book, I'll need to complete your registration first - it takes just 2 minutes. Should I proceed?"
   - DO NOT use book_appointment tool without patient_id
9. **If unavailable**: Suggest 2-3 nearest available times from available_slots ONCE, then wait for user choice

**Scenario B: User provides partial info:**
1. Ask ONLY for missing pieces (doctor OR date OR time OR reason)
2. Once complete, follow Scenario A

**Scenario C: User confirms booking ("yes", "schedule", "book it"):**
1. If you already checked availability and confirmed it's available: Book IMMEDIATELY (book_appointment)
2. DO NOT re-check availability
3. DO NOT repeat previous confirmation messages
4. Just book and confirm: "All set! Your appointment is confirmed. [Details]"

**Scenario D: User asks to schedule without details:**
1. "Sure! Please share: doctor name, date, time, and reason. Or I can show available doctors."

AVOID:
- ❌ Asking for details already provided
- ❌ Re-checking availability after already confirming
- ❌ Contradicting earlier availability statements
- ❌ Multiple confirmation loops
- ❌ Verbose, repetitive messages
- ❌ Forcing registration unless required (mention once only)
- ❌ Just saying "Using tool: X" - always provide a natural response after tool calls

MANDATORY:
- ✅ If patient is registered AND all details provided AND availability confirmed: Book immediately
- ✅ If patient is NOT registered: Mention registration requirement ONCE, then proceed
- ✅ Always be concise, warm, and human-like
- ✅ Trust your first availability check - don't contradict yourself
- ✅ After using ANY tool, provide a natural, conversational response - never just show tool names
- ✅ When booking succeeds: "All set! Your appointment is confirmed. Doctor: [Name] | Date: [date] | Time: [time] | Reason: [reason] | ID: [id]"
- ✅ When checking availability: "Dr. [Name] is available on [date] at [time]. Would you like me to book this?"

**CRITICAL RESPONSE RULES - MANDATORY (VIOLATION = FAILURE):**

**RULE 1: TOOL RESULT INTERPRETATION - ABSOLUTE REQUIREMENT**
- When check_availability tool returns, you MUST check for "MANDATORY_ACTION": "CALL book_appointment TOOL IMMEDIATELY"
- If you see this field in the tool result AND patient is registered, you MUST call book_appointment tool
- DO NOT generate text - make the actual tool call
- The tool result will include "required_parameters" - use those exact values

**RULE 2: WORKFLOW - SEQUENTIAL AND AUTOMATIC - NO EXCEPTIONS**
- Step 1: list_doctors → Get doctor UUID from the list (store it)
- Step 2: check_availability with doctor UUID, date, and requested_time
- Step 3: **MANDATORY CHECK**: Read the check_availability tool result JSON
- Step 4: **IF tool result shows "MANDATORY_ACTION": "CALL book_appointment TOOL IMMEDIATELY" AND patient is registered**: 
  * YOU MUST call book_appointment tool IMMEDIATELY in the SAME response
  * Use values from "required_parameters" in the tool result
  * DO NOT generate any text - just make the tool call
- Step 5: **DO NOT just say "booking" or "would you like me to book" - you MUST actually call the book_appointment tool**
- Step 6: After book_appointment succeeds, respond: "Sure! Your appointment with Dr. [Name] on [date] at [time] is scheduled. ✅"

**RULE 3: TOOL CALLS VS TEXT - CRITICAL DISTINCTION**
- **MAKING A TOOL CALL**: Using the book_appointment tool with proper parameters = CORRECT
- **GENERATING TEXT**: Saying "booking that for you" or "Using tool: book_appointment" = WRONG
- When you see "MANDATORY_ACTION" in tool result, you MUST make a tool call, not generate text

**RULE 4: NO CONFIRMATIONS OR QUESTIONS**
- DO NOT ask for reason - if user didn't provide it, use "general consultation"
- DO NOT ask for confirmation - if available and patient registered, book immediately
- DO NOT wait for user to say "yes" - book immediately when you see "MANDATORY_ACTION"

**RULE 5: RESPONSE FORMAT**
- After successful booking: "Sure! Your appointment with Dr. [Name] on [date] at [time] is scheduled. ✅"
- Always be conversational and human-like - never show tool names to the user
- Never show "Tool result:" or JSON in responses

**RULE 6: REMEMBER**
- Saying "booking that for you" is NOT the same as calling book_appointment tool - you MUST call the tool
- If check_availability returns {{"MANDATORY_ACTION": "CALL book_appointment TOOL IMMEDIATELY"}}, immediately call book_appointment
- The tool result will tell you exactly what to do - follow it precisely
"""

        return context

    # Tool implementations

    def tool_list_doctors(self, session_id: str) -> Dict[str, Any]:
        """Get list of available doctors."""
        try:
            doctors = self.db_client.get_doctors()

            if not doctors:
                return {"error": "No doctors found"}

            # Format doctor list
            doctor_list = []
            for doc in doctors:
                doctor_list.append({
                    "id": doc.get("id"),
                    "name": f"Dr. {doc.get('first_name')} {doc.get('last_name')}",
                    "specialty": doc.get("specialty"),
                    "languages": doc.get("languages", ["en"])
                })

            return {
                "success": True,
                "doctors": doctor_list,
                "count": len(doctor_list)
            }

        except Exception as e:
            logger.error(f"Error listing doctors: {e}")
            return {"error": str(e)}

    def tool_find_doctor_by_name(self, session_id: str, doctor_name: str) -> Dict[str, Any]:
        """Find a doctor by name."""
        try:
            # Get all doctors
            doctors = self.db_client.get_doctors()

            if not doctors:
                return {"error": "No doctors found"}

            # Clean doctor name (remove "Dr.", "Dr", etc.)
            doctor_name_clean = doctor_name.lower().replace("dr.", "").replace("dr", "").strip()
            # Split into words for better matching
            search_words = [w.strip() for w in doctor_name_clean.split() if len(w.strip()) > 2]
            
            matching_doctors = []
            
            for doctor in doctors:
                first_name = doctor.get("first_name", "").lower()
                last_name = doctor.get("last_name", "").lower()
                full_name = f"{first_name} {last_name}".strip()
                
                # Multi-word search: all words should match (e.g., "mohammed atef" requires both words)
                if len(search_words) > 1:
                    # Check if all search words are in the full name
                    if all(word in full_name for word in search_words):
                        matching_doctors.append(doctor)
                else:
                    # Single word search: check if it's in first name, last name, or full name
                    search_term = search_words[0] if search_words else doctor_name_clean
                    if (search_term in first_name or 
                        search_term in last_name or 
                        search_term in full_name):
                        matching_doctors.append(doctor)

            if not matching_doctors:
                return {
                    "success": False,
                    "error": f"No doctor found matching '{doctor_name}'",
                    "suggestion": "Use list_doctors tool to see all available doctors"
                }

            # Return the first matching doctor (best match)
            doctor = matching_doctors[0]
            
            return {
                "success": True,
                "doctor": {
                    "id": doctor.get("id"),
                    "name": f"Dr. {doctor.get('first_name')} {doctor.get('last_name')}",
                    "first_name": doctor.get("first_name"),
                    "last_name": doctor.get("last_name"),
                    "specialty": doctor.get("specialty"),
                    "languages": doctor.get("languages", ["en"])
                }
            }

        except Exception as e:
            logger.error(f"Error finding doctor by name: {e}")
            return {"error": str(e)}

    def tool_check_availability(
        self,
        session_id: str,
        doctor_id: str,
        date: str,
        requested_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check doctor availability. 
        
        Args:
            doctor_id: Must be a valid UUID (use find_doctor_by_name first to get the ID)
            date: Date in YYYY-MM-DD format
            requested_time: Optional time in HH:MM format to check specific slot availability
        """
        try:
            # Validate doctor_id is a UUID (not a name)
            import uuid
            try:
                uuid.UUID(doctor_id)
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid doctor_id format",
                    "message": f"doctor_id must be a UUID, not a name. Got: {doctor_id}. Please use find_doctor_by_name first to get the correct doctor ID.",
                    "suggestion": "Use find_doctor_by_name tool first to get the doctor's UUID"
                }

            availability = self.db_client.get_doctor_availability(doctor_id, date)

            if not availability:
                return {
                    "success": True,
                    "available_slots": [],
                    "message": "No available slots found",
                    "date": date
                }

            # Get existing appointments for this doctor on this date to check for conflicts
            existing_appointments = self.db_client.get_doctor_appointments(
                doctor_id=doctor_id,
                date=date,
                status=None  # Get all non-cancelled appointments
            )
            
            # Filter out cancelled appointments
            booked_slots = []
            if existing_appointments:
                for apt in existing_appointments:
                    if apt.get("status") not in ["cancelled", "completed", "no_show"]:
                        booked_slots.append({
                            "start_time": apt.get("start_time"),
                            "end_time": apt.get("end_time")
                        })
            
            logger.info(f"Found {len(booked_slots)} booked slots for doctor {doctor_id} on {date}")

            # If requested_time is provided, check if that specific time is available
            if requested_time:
                # Parse requested time to HH:MM format
                from datetime import datetime
                try:
                    # Try parsing various formats
                    time_str = requested_time.replace(" ", "").upper()
                    parsed_time = None
                    
                    # Handle formats like "2pm", "2:00pm", "14:00", etc.
                    if "PM" in time_str or "AM" in time_str:
                        # 12-hour format
                        time_str_clean = time_str.replace("PM", "").replace("AM", "").strip()
                        if ":" in time_str_clean:
                            hour, minute = time_str_clean.split(":")
                            hour = int(hour)
                            minute = int(minute)
                        else:
                            hour = int(time_str_clean)
                            minute = 0
                        
                        if "PM" in time_str and hour < 12:
                            hour += 12
                        elif "AM" in time_str and hour == 12:
                            hour = 0
                        
                        parsed_time = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
                    else:
                        # 24-hour format
                        if ":" in time_str:
                            parsed_time = datetime.strptime(time_str, "%H:%M").time()
                        else:
                            # Just hour
                            hour = int(time_str)
                            parsed_time = datetime.strptime(f"{hour:02d}:00", "%H:%M").time()
                except Exception as e:
                    logger.warning(f"Could not parse requested time '{requested_time}': {e}")
                    parsed_time = None

                # Check if requested time falls within any availability slot AND is not already booked
                if parsed_time:
                    available_at_requested_time = False
                    matching_slot = None
                    is_conflicting = False
                    
                    # Calculate requested time slot (30 min default)
                    from datetime import timedelta
                    requested_start = parsed_time
                    requested_end_dt = datetime.combine(datetime.today(), requested_start) + timedelta(minutes=30)
                    requested_end = requested_end_dt.time()
                    
                    # Check for conflicts with existing appointments
                    for booked in booked_slots:
                        booked_start_str = booked.get("start_time", "00:00:00")
                        booked_end_str = booked.get("end_time", "23:59:59")
                        booked_start = datetime.strptime(booked_start_str[:5], "%H:%M").time() if len(booked_start_str) >= 5 else datetime.strptime("00:00", "%H:%M").time()
                        booked_end = datetime.strptime(booked_end_str[:5], "%H:%M").time() if len(booked_end_str) >= 5 else datetime.strptime("23:59", "%H:%M").time()
                        
                        # Check if requested time overlaps with booked slot
                        if not (requested_end <= booked_start or requested_start >= booked_end):
                            is_conflicting = True
                            logger.warning(f"Requested time {requested_time} conflicts with existing appointment {booked_start}-{booked_end}")
                            break
                    
                    # Only check availability slots if no conflict
                    if not is_conflicting:
                        for slot in availability:
                            slot_start_str = slot.get("start_time", "00:00:00")
                            slot_end_str = slot.get("end_time", "23:59:59")
                            
                            # Parse slot times
                            slot_start = datetime.strptime(slot_start_str[:5], "%H:%M").time() if len(slot_start_str) >= 5 else datetime.strptime("00:00", "%H:%M").time()
                            slot_end = datetime.strptime(slot_end_str[:5], "%H:%M").time() if len(slot_end_str) >= 5 else datetime.strptime("23:59", "%H:%M").time()
                            
                            # Check if requested time is within slot range
                            if slot_start <= parsed_time < slot_end:
                                available_at_requested_time = True
                                matching_slot = slot
                                break
                    
                    # Update agent state
                    self.state_manager.update_appointment_state(
                        session_id,
                        available_slots=availability
                    )
                    
                    if is_conflicting:
                        return {
                            "success": True,
                            "date": date,
                            "requested_time": requested_time,
                            "available_at_requested_time": False,
                            "is_conflicting": True,
                            "available_slots": availability,
                            "booked_slots": booked_slots,
                            "message": f"Requested time {requested_time} is already booked. Please choose a different time."
                        }
                    elif available_at_requested_time:
                        logger.info(f"✅ Time {requested_time} on {date} is AVAILABLE for doctor {doctor_id}")
                        return {
                            "success": True,
                            "date": date,
                            "requested_time": requested_time,
                            "available_at_requested_time": True,
                            "available_slots": availability,
                            "matching_slot": matching_slot,
                            "message": f"Dr. is available on {date} at {requested_time}",
                            "MANDATORY_ACTION": "CALL book_appointment TOOL IMMEDIATELY",
                            "instructions": "You MUST call the book_appointment tool NOW with patient_id, doctor_id, date, time, and reason. Do NOT generate text - make the tool call.",
                            "tool_to_call": "book_appointment",
                            "required_parameters": {
                                "patient_id": "From PATIENT INFORMATION section",
                                "doctor_id": "The UUID from find_doctor_by_name result",
                                "date": f"{date}",
                                "time": f"{requested_time}",
                                "reason": "From user message OR 'general consultation'"
                            }
                        }
                    else:
                        return {
                            "success": True,
                            "date": date,
                            "requested_time": requested_time,
                            "available_at_requested_time": False,
                            "available_slots": availability,
                            "message": f"Requested time {requested_time} is not available, but other slots are available"
                        }

            # Filter out slots that conflict with existing appointments
            # This provides available time ranges, but individual slots may still be booked
            filtered_slots = []
            for slot in availability:
                slot_start_str = slot.get("start_time", "00:00:00")
                slot_end_str = slot.get("end_time", "23:59:59")
                slot_start = datetime.strptime(slot_start_str[:5], "%H:%M").time() if len(slot_start_str) >= 5 else datetime.strptime("00:00", "%H:%M").time()
                slot_end = datetime.strptime(slot_end_str[:5], "%H:%M").time() if len(slot_end_str) >= 5 else datetime.strptime("23:59", "%H:%M").time()
                
                # Check if this slot overlaps with any booked appointments
                has_conflict = False
                for booked in booked_slots:
                    booked_start_str = booked.get("start_time", "00:00:00")
                    booked_end_str = booked.get("end_time", "23:59:59")
                    booked_start = datetime.strptime(booked_start_str[:5], "%H:%M").time() if len(booked_start_str) >= 5 else datetime.strptime("00:00", "%H:%M").time()
                    booked_end = datetime.strptime(booked_end_str[:5], "%H:%M").time() if len(booked_end_str) >= 5 else datetime.strptime("23:59", "%H:%M").time()
                    
                    # Check for overlap
                    if not (slot_end <= booked_start or slot_start >= booked_end):
                        has_conflict = True
                        break
                
                if not has_conflict:
                    filtered_slots.append(slot)

            # Update agent state
            self.state_manager.update_appointment_state(
                session_id,
                available_slots=filtered_slots if filtered_slots else availability
            )

            return {
                "success": True,
                "date": date,
                "available_slots": filtered_slots if filtered_slots else availability,
                "booked_slots": booked_slots,
                "message": f"Found {len(filtered_slots if filtered_slots else availability)} available slot(s) on {date}. {len(booked_slots)} time slot(s) already booked."
            }

        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return {"error": str(e)}

    async def _execute_booking_workflow(
        self,
        session_id: str,
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Execute strict synchronous booking workflow with timeouts and verification.
        
        Workflow steps:
        1. Validate inputs
        2. Generate idempotency key
        3. Get clinic and appointment type
        4. Create appointment with idempotency key
        5. Final read verification from DB
        6. Return success only if verification passes
        
        Args:
            session_id: Session identifier
            patient_id: Patient ID
            doctor_id: Doctor ID
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            reason: Appointment reason
            correlation_id: Correlation ID for logging
        
        Returns:
            Dict with success status and appointment details or error
        """
        import time as time_module
        import asyncio
        from datetime import datetime as dt
        
        workflow_start = time_module.time()
        logger.info(f"🚀 BOOKING WORKFLOW START: correlation_id={correlation_id}")
        logger.info(f"   patient_id={patient_id}, doctor_id={doctor_id}, date={date}, time={time}")
        
        try:
            # Step 1: Validate patient_id
            if not patient_id or patient_id.strip() == "":
                logger.error(f"❌ Validation failed: patient_id is empty - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Patient registration required",
                    "error_code": "VALIDATION_FAILED",
                    "message": "You must complete registration before booking an appointment. Please register first."
                }
            
            # Step 2: Generate idempotency key
            idempotency_key = self._generate_idempotency_key(patient_id, doctor_id, date, time, reason)
            logger.info(f"🔑 Idempotency key: {idempotency_key} - correlation_id={correlation_id}")
            
            # Step 3: Get clinic ID
            clinic_info = self.db_client.get_clinic_info()
            if not clinic_info:
                all_clinics = self.db_client.get_all_clinics()
                if all_clinics and len(all_clinics) > 0:
                    clinic_id = all_clinics[0].get("id")
                    logger.info(f"Using first available clinic: {clinic_id} - correlation_id={correlation_id}")
                else:
                    clinic_id = "11111111-1111-1111-1111-111111111111"
                    logger.warning(f"No clinic found, using default: {clinic_id} - correlation_id={correlation_id}")
            else:
                clinic_id = clinic_info.get("id")
                logger.info(f"Using clinic: {clinic_id} - correlation_id={correlation_id}")
            
            # Step 4: Get appointment type
            appointment_types = self.db_client.get_appointment_types()
            appointment_type_id = appointment_types[0].get("id") if appointment_types else None
            
            if not appointment_type_id:
                logger.error(f"❌ Appointment type not found - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Appointment type not found",
                    "error_code": "APPOINTMENT_TYPE_NOT_FOUND",
                    "message": "Unable to determine appointment type. Please try again."
                }
            
            # Step 5: Calculate end time
            start_dt = dt.strptime(time, "%H:%M")
            end_dt = start_dt + timedelta(minutes=30)
            end_time = end_dt.strftime("%H:%M")
            
            # Step 6: Create appointment with idempotency key (with timeout and retry)
            logger.info(f"📋 Creating appointment with idempotency_key={idempotency_key} - correlation_id={correlation_id}")
            create_start = time_module.time()
            
            def create_appointment_sync():
                return self.db_client.create_appointment(
                    clinic_id=clinic_id,
                    patient_id=patient_id,
                    doctor_id=doctor_id,
                    appointment_type_id=appointment_type_id,
                    appointment_date=date,
                    start_time=time,
                    end_time=end_time,
                    reason=reason,
                    emergency_level="routine",
                    request_id=idempotency_key  # Pass idempotency key as request_id
                )
            
            try:
                appointment = await self._execute_with_timeout(
                    create_appointment_sync,
                    timeout_seconds=10.0,
                    max_retries=2,
                    correlation_id=correlation_id
                )
                create_duration_ms = (time_module.time() - create_start) * 1000
                logger.info(f"📊 Create appointment completed in {create_duration_ms:.0f}ms - correlation_id={correlation_id}")
            except TimeoutError:
                logger.error(f"❌ Create appointment timed out - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Create appointment timed out",
                    "error_code": "TIMEOUT_CREATE",
                    "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                }
            except Exception as e:
                error_str = str(e).lower()
                if "409" in error_str or "conflict" in error_str:
                    logger.warning(f"⚠️ Slot conflict detected - correlation_id={correlation_id}")
                    # Get alternatives for user
                    alternatives = self._get_alternative_slots(doctor_id, date, time)
                    return {
                        "success": False,
                        "error": "Slot unavailable",
                        "error_code": "SLOT_CONFLICT",
                        "message": f"I couldn't book that time — Dr. [name] is not available at {date} {time}. Would you like these alternatives: {alternatives}?"
                    }
                else:
                    logger.error(f"❌ Create appointment failed: {e} - correlation_id={correlation_id}")
                    return {
                        "success": False,
                        "error": str(e),
                        "error_code": "CREATE_FAILED",
                        "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                    }
            
            if not appointment:
                logger.error(f"❌ Create appointment returned None - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Create appointment returned None",
                    "error_code": "CREATE_RETURNED_NONE",
                    "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                }
            
            appointment_id = appointment.get('id')
            logger.info(f"✅ Appointment created: {appointment_id} - correlation_id={correlation_id}")
            
            # Step 7: Final read verification (CRITICAL - never skip this)
            logger.info(f"🔍 FINAL READ VERIFY: Starting verification - correlation_id={correlation_id}")
            verify_start = time_module.time()
            
            # Final read verify is synchronous, but we wrap it for timeout
            try:
                verified_appointment = await self._execute_with_timeout(
                    lambda: self._final_read_verify(
                        appointment_id=appointment_id,
                        patient_id=patient_id,
                        doctor_id=doctor_id,
                        date=date,
                        time=time,
                        correlation_id=correlation_id
                    ),
                    timeout_seconds=10.0,
                    max_retries=2,
                    correlation_id=correlation_id
                )
                verify_duration_ms = (time_module.time() - verify_start) * 1000
                logger.info(f"📊 Final read verify completed in {verify_duration_ms:.0f}ms - correlation_id={correlation_id}")
            except TimeoutError:
                logger.error(f"❌ Final read verify timed out - correlation_id={correlation_id}")
                # CRITICAL: Do NOT confirm if verification times out
                return {
                    "success": False,
                    "error": "Final read verify timed out",
                    "error_code": "TIMEOUT_VERIFY",
                    "message": "I tried to confirm the booking but couldn't verify it in our system. I did not book the appointment. Would you like me to try again?"
                }
            except Exception as e:
                logger.error(f"❌ Final read verify failed: {e} - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Final read verify failed",
                    "error_code": "VERIFY_FAILED",
                    "message": "I tried to confirm the booking but couldn't verify it in our system. I did not book the appointment. Would you like me to try again?"
                }
            
            if not verified_appointment:
                # CRITICAL: Verification failed - do NOT confirm
                logger.error(f"❌ FINAL READ VERIFY FAILED: Appointment not found in DB - correlation_id={correlation_id}")
                logger.error(f"   🚨 FALSE CONFIRMATION PREVENTED - NOT confirming to user")
                
                # Log metrics
                workflow_duration_ms = (time_module.time() - workflow_start) * 1000
                self._log_booking_metrics(
                    appointment_inserted=False,
                    db_row_id=appointment_id,
                    verification_latency_ms=verify_duration_ms,
                    scheduling_latency_ms=create_duration_ms,
                    dashboard_visible=False,
                    request_id=idempotency_key,
                    mismatch_detected=True,
                    error_code="FINAL_VERIFY_FAILED"
                )
                
                logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={workflow_duration_ms:.0f}")
                
                return {
                    "success": False,
                    "error": "Final read verify failed - appointment not in DB",
                    "error_code": "FINAL_VERIFY_FAILED",
                    "message": "I tried to confirm the booking but couldn't verify it in our system. I did not book the appointment. Would you like me to try again?"
                }
            
            # Step 8: SUCCESS - Verification passed
            logger.info(f"✅ FINAL READ VERIFY PASSED: Appointment confirmed in DB - correlation_id={correlation_id}")
            
            # Get doctor name for success message
            doctor = self.db_client.get_doctor_by_id(doctor_id)
            doctor_name = f"Dr. {doctor.get('first_name', '')} {doctor.get('last_name', '')}" if doctor else "Dr. [name]"
            
            # Format date for display
            try:
                date_obj = dt.strptime(date, "%Y-%m-%d")
                date_display = date_obj.strftime("%B %d, %Y")
            except:
                date_display = date
            
            # Format time for display
            try:
                time_obj = dt.strptime(time, "%H:%M")
                time_display = time_obj.strftime("%I:%M %p")
            except:
                time_display = time
            
            # Log success metrics
            workflow_duration_ms = (time_module.time() - workflow_start) * 1000
            self._log_booking_metrics(
                appointment_inserted=True,
                db_row_id=appointment_id,
                verification_latency_ms=verify_duration_ms,
                scheduling_latency_ms=create_duration_ms,
                dashboard_visible=True,
                request_id=idempotency_key,
                mismatch_detected=False,
                error_code=None
            )
            
            logger.info(f"📊 METRIC: appointment_booking.success correlation_id={correlation_id} latency_ms={workflow_duration_ms:.0f}")
            
            # Update state
            self.state_manager.update_appointment_state(
                session_id,
                workflow_step="completed",
                operation_type="booking"
            )
            
            # Return success with formatted message
            return {
                "success": True,
                "appointment": verified_appointment,
                "appointment_id": appointment_id,
                "verified": True,
                "idempotency_key": idempotency_key,
                "verification_latency_ms": round(verify_duration_ms, 2),
                "workflow_latency_ms": round(workflow_duration_ms, 2),
                "message": f"✅ Your appointment is confirmed.\n\n**Doctor:** {doctor_name}\n**Date:** {date_display}\n**Time:** {time_display}\n**Reason:** {reason}\n**Appointment ID:** {appointment_id}"
            }
            
        except Exception as e:
            workflow_duration_ms = (time_module.time() - workflow_start) * 1000
            logger.error(f"❌ WORKFLOW EXCEPTION: {e} - correlation_id={correlation_id}", exc_info=True)
            logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={workflow_duration_ms:.0f}")
            
            return {
                "success": False,
                "error": str(e),
                "error_code": "WORKFLOW_EXCEPTION",
                "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
            }
    
    def _get_alternative_slots(self, doctor_id: str, date: str, requested_time: str) -> str:
        """
        Get alternative time slots when requested time is unavailable.
        
        Args:
            doctor_id: Doctor ID
            date: Requested date
            requested_time: Requested time
        
        Returns:
            String with up to 3 alternative slots
        """
        try:
            availability = self.db_client.get_doctor_availability(doctor_id, date)
            if not availability:
                return "No alternative slots available"
            
            # Get available slots
            slots = []
            for slot in availability[:3]:  # Max 3 alternatives
                start = slot.get('start_time', '')[:5]
                if start:
                    slots.append(start)
            
            if slots:
                return ", ".join(slots)
            else:
                return "No alternative slots available"
        except Exception as e:
            logger.warning(f"Failed to get alternative slots: {e}")
            return "No alternative slots available"
    
    def tool_book_appointment(
        self,
        session_id: str,
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Book a new appointment using strict synchronous workflow.
        
        This method orchestrates the complete booking flow with timeouts,
        retries, idempotency, and mandatory final DB verification.
        """
        import asyncio
        import time as time_module
        
        correlation_id = session_id
        overall_start = time_module.time()
        
        logger.info(f"📋 BOOKING REQUEST: correlation_id={correlation_id}")
        logger.info(f"   patient_id={patient_id}, doctor_id={doctor_id}, date={date}, time={time}, reason={reason}")
        
        try:
            # Execute workflow with overall timeout (20 seconds)
            # Handle both sync and async contexts
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # If loop is already running, we need to use a different approach
                    # Create a task and wait for it
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                asyncio.wait_for(
                                    self._execute_booking_workflow(
                                        session_id=session_id,
                                        patient_id=patient_id,
                                        doctor_id=doctor_id,
                                        date=date,
                                        time=time,
                                        reason=reason,
                                        correlation_id=correlation_id
                                    ),
                                    timeout=20.0
                                )
                            )
                        )
                        result = future.result(timeout=25.0)  # Slightly longer than async timeout
                else:
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            self._execute_booking_workflow(
                                session_id=session_id,
                                patient_id=patient_id,
                                doctor_id=doctor_id,
                                date=date,
                                time=time,
                                reason=reason,
                                correlation_id=correlation_id
                            ),
                            timeout=20.0  # Overall orchestration timeout
                        )
                    )
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                overall_duration_ms = (time_module.time() - overall_start) * 1000
                logger.error(f"❌ OVERALL TIMEOUT: Workflow exceeded 20s - correlation_id={correlation_id}")
                logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={overall_duration_ms:.0f} error_code=OVERALL_TIMEOUT")
                
                result = {
                    "success": False,
                    "error": "Overall workflow timeout",
                    "error_code": "OVERALL_TIMEOUT",
                    "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                }
            
            overall_duration_ms = (time_module.time() - overall_start) * 1000
            logger.info(f"📊 Overall workflow completed in {overall_duration_ms:.0f}ms - correlation_id={correlation_id}")
            
            return result
            
        except Exception as e:
            overall_duration_ms = (time_module.time() - overall_start) * 1000
            logger.error(f"❌ BOOKING EXCEPTION: {e} - correlation_id={correlation_id}", exc_info=True)
            logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={overall_duration_ms:.0f}")
            
            return {
                "success": False,
                "error": str(e),
                "error_code": "BOOKING_EXCEPTION",
                "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
            }
    
    # Old tool_book_appointment method removed - replaced with new synchronous workflow
            # Validate patient_id exists
            if not patient_id or patient_id.strip() == "":
                return {
                    "success": False,
                    "error": "Patient registration required",
                    "message": "You must complete registration before booking an appointment. Please register first."
                }
            
            # Get clinic ID (use first available clinic or default)
            clinic_info = self.db_client.get_clinic_info()
            if not clinic_info:
                # Try to get all clinics and use the first one
                all_clinics = self.db_client.get_all_clinics()
                if all_clinics and len(all_clinics) > 0:
                    clinic_id = all_clinics[0].get("id")
                    logger.info(f"Using first available clinic: {clinic_id}")
                else:
                    # Use a known clinic ID from the database
                    clinic_id = "11111111-1111-1111-1111-111111111111"  # Al Dhait Branch
                    logger.warning(f"No clinic found, using default: {clinic_id}")
            else:
                clinic_id = clinic_info.get("id")
                logger.info(f"Using clinic from get_clinic_info: {clinic_id}")

            # Calculate end time (default 30 min appointments)
            from datetime import datetime
            start_dt = datetime.strptime(time, "%H:%M")
            end_dt = start_dt + timedelta(minutes=30)
            end_time = end_dt.strftime("%H:%M")

            # Get appointment type ID (use a default or fetch)
            appointment_types = self.db_client.get_appointment_types()
            appointment_type_id = appointment_types[0].get("id") if appointment_types else None
            
            if not appointment_type_id:
                return {
                    "success": False,
                    "error": "Appointment type not found",
                    "message": "Unable to determine appointment type. Please try again."
                }

            # Create appointment (status is set automatically by API, don't pass it)
            logger.info(f"📋 BOOKING APPOINTMENT: patient_id={patient_id}, doctor_id={doctor_id}, date={date}, time={time}")
            logger.info(f"   clinic_id={clinic_id}, appointment_type_id={appointment_type_id}")
            
            # Track scheduling API latency
            import time as time_module
            scheduling_start = time_module.time()
            
            appointment = self.db_client.create_appointment(
                clinic_id=clinic_id,
                patient_id=patient_id,
                doctor_id=doctor_id,
                appointment_type_id=appointment_type_id,
                appointment_date=date,
                start_time=time,
                end_time=end_time,
                reason=reason,
                emergency_level="routine"
            )
            
            scheduling_latency_ms = (time_module.time() - scheduling_start) * 1000
            logger.info(f"📊 Scheduling API call completed in {scheduling_latency_ms:.0f}ms")

            if appointment:
                appointment_id = appointment.get('id')
                logger.info(f"✅ API returned appointment: {appointment_id}")
                
                # ========== MANDATORY DB VERIFICATION ==========
                # CRITICAL: Never trust API response alone - verify appointment exists in DB
                # This prevents false confirmations when DB writes fail silently
                logger.info(f"🔍 VERIFICATION: Confirming appointment exists in database...")
                verification_start = time_module.time()
                
                # Step 1: Verify appointment exists in DB
                verified_appointment = self._verify_appointment_in_db(
                    appointment_id=appointment_id,
                    max_retries=3,
                    retry_delay_ms=500
                )
                
                verification_latency_ms = (time_module.time() - verification_start) * 1000
                logger.info(f"📊 DB verification completed in {verification_latency_ms:.0f}ms")
                
                if not verified_appointment:
                    # CRITICAL: API returned success but appointment NOT in DB
                    # This is the exact false confirmation scenario we must prevent
                    logger.error(f"❌ VERIFICATION FAILED: Appointment {appointment_id} not found in DB!")
                    logger.error(f"   API returned: {appointment}")
                    logger.error(f"   DB query returned: None")
                    logger.error(f"   🚨 This is a FALSE CONFIRMATION scenario - NOT confirming to user")
                    
                    # Log metrics for this mismatch
                    self._log_booking_metrics(
                        appointment_inserted=False,
                        db_row_id=appointment_id,
                        verification_latency_ms=verification_latency_ms,
                        scheduling_latency_ms=scheduling_latency_ms,
                        dashboard_visible=False,
                        request_id=None,
                        mismatch_detected=True,
                        error_code="DB_VERIFICATION_FAILED"
                    )
                    
                    return {
                        "success": False,
                        "error": "Verification failed - appointment not found in database",
                        "error_code": "DB_VERIFICATION_FAILED",
                        "message": "I couldn't complete your booking. The system encountered an issue. Please try again or contact support."
                    }
                
                # Step 2: Verify appointment details match
                if not self._verify_appointment_details(
                    appointment=verified_appointment,
                    expected_patient_id=patient_id,
                    expected_doctor_id=doctor_id,
                    expected_date=date,
                    expected_time=time
                ):
                    logger.error(f"❌ VERIFICATION FAILED: Appointment details mismatch!")
                    logger.error(f"   Expected: patient={patient_id}, doctor={doctor_id}, date={date}, time={time}")
                    logger.error(f"   Found: {verified_appointment}")
                    
                    # Log metrics for this mismatch
                    self._log_booking_metrics(
                        appointment_inserted=False,
                        db_row_id=appointment_id,
                        verification_latency_ms=verification_latency_ms,
                        scheduling_latency_ms=scheduling_latency_ms,
                        dashboard_visible=False,
                        request_id=None,
                        mismatch_detected=True,
                        error_code="DB_VERIFICATION_MISMATCH"
                    )
                    
                    return {
                        "success": False,
                        "error": "Verification failed - appointment details don't match",
                        "error_code": "DB_VERIFICATION_MISMATCH",
                        "message": "I couldn't verify your booking. Please try again or contact support."
                    }
                
                # ========== VERIFICATION PASSED ==========
                logger.info(f"✅ VERIFICATION PASSED: Appointment {appointment_id} confirmed in database")
                
                # Log successful metrics
                self._log_booking_metrics(
                    appointment_inserted=True,
                    db_row_id=appointment_id,
                    verification_latency_ms=verification_latency_ms,
                    scheduling_latency_ms=scheduling_latency_ms,
                    dashboard_visible=True,  # Assume visible if DB verification passed
                    request_id=None,
                    mismatch_detected=False,
                    error_code=None
                )
                
                # Update state
                self.state_manager.update_appointment_state(
                    session_id,
                    workflow_step="completed",
                    operation_type="booking"
                )

                return {
                    "success": True,
                    "appointment": verified_appointment,
                    "appointment_id": appointment_id,
                    "verified": True,
                    "verification_latency_ms": round(verification_latency_ms, 2),
                    "message": f"Appointment booked successfully for {date} at {time}"
                }
            else:
                logger.error(f"❌ Failed to create appointment - API returned None")
                
                # Log failed metrics
                self._log_booking_metrics(
                    appointment_inserted=False,
                    db_row_id=None,
                    verification_latency_ms=0,
                    scheduling_latency_ms=scheduling_latency_ms,
                    dashboard_visible=False,
                    request_id=None,
                    mismatch_detected=False,
                    error_code="API_RETURNED_NONE"
                )
                
                return {
                    "success": False,
                    "error": "Failed to create appointment",
                    "error_code": "API_RETURNED_NONE",
                    "message": "The appointment could not be created. Please try again or contact support."
                }

        except Exception as e:
            logger.error(f"Error booking appointment: {e}")
            return {"error": str(e)}

    def tool_check_patient_appointments(
        self,
        session_id: str,
        patient_id: str
    ) -> Dict[str, Any]:
        """Get patient's appointments."""
        try:
            appointments = self.db_client.get_patient_appointments(patient_id)

            if not appointments:
                return {
                    "success": True,
                    "appointments": [],
                    "message": "No appointments found"
                }

            return {
                "success": True,
                "appointments": appointments,
                "count": len(appointments)
            }

        except Exception as e:
            logger.error(f"Error fetching appointments: {e}")
            return {"error": str(e)}

    def tool_cancel_appointment(
        self,
        session_id: str,
        appointment_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """Cancel an appointment."""
        try:
            result = self.db_client.cancel_appointment(appointment_id, reason)

            if result:
                return {
                    "success": True,
                    "message": f"Appointment {appointment_id} cancelled successfully"
                }
            else:
                return {"error": "Failed to cancel appointment"}

        except Exception as e:
            logger.error(f"Error cancelling appointment: {e}")
            return {"error": str(e)}

    def tool_reschedule_appointment(
        self,
        session_id: str,
        appointment_id: str,
        new_date: str,
        new_time: str
    ) -> Dict[str, Any]:
        """Reschedule an appointment."""
        try:
            # Calculate new end time
            start_dt = datetime.strptime(new_time, "%H:%M")
            end_dt = start_dt + timedelta(minutes=30)
            new_end_time = end_dt.strftime("%H:%M")

            result = self.db_client.reschedule_appointment(
                appointment_id,
                new_date,
                new_time,
                new_end_time
            )

            if result:
                return {
                    "success": True,
                    "appointment": result,
                    "message": f"Appointment rescheduled to {new_date} at {new_time}"
                }
            else:
                return {"error": "Failed to reschedule appointment"}

        except Exception as e:
            logger.error(f"Error rescheduling appointment: {e}")
            return {"error": str(e)}

    # ========== SYNCHRONOUS BOOKING WORKFLOW INFRASTRUCTURE ==========
    # These methods support the strict synchronous booking workflow
    
    def _generate_idempotency_key(
        self,
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: str
    ) -> str:
        """
        Generate idempotency key using SHA256 hash.
        
        Args:
            patient_id: Patient ID
            doctor_id: Doctor ID
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            reason: Appointment reason
        
        Returns:
            32-character hex digest of SHA256 hash
        """
        import hashlib
        
        key_string = f"{patient_id}_{doctor_id}_{date}_{time}_{reason}"
        idempotency_key = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        
        logger.info(f"🔑 Generated idempotency_key: {idempotency_key} (from: {key_string[:50]}...)")
        return idempotency_key
    
    def _is_transient_error(self, error: Exception) -> bool:
        """
        Determine if an error is transient and should be retried.
        
        Args:
            error: The exception that occurred
        
        Returns:
            True if error is transient (5xx, timeout), False otherwise
        """
        error_str = str(error).lower()
        
        # Check for HTTP status codes in error message
        if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
            return True
        
        # Check for timeout errors
        if "timeout" in error_str or "timed out" in error_str:
            return True
        
        # Check for network errors
        if "connection" in error_str or "network" in error_str:
            return True
        
        # Do NOT retry on these errors
        if "409" in error_str or "conflict" in error_str:
            return False
        if "404" in error_str or "not found" in error_str:
            return False
        if "400" in error_str or "bad request" in error_str:
            return False
        
        # Default: treat as transient if it's a generic exception
        return True
    
    async def _execute_with_timeout(
        self,
        func,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        correlation_id: Optional[str] = None
    ) -> Any:
        """
        Execute a function with timeout and retry logic.
        
        Args:
            func: Async function to execute
            timeout_seconds: Timeout per attempt (default 10s)
            max_retries: Maximum retry attempts (default 2, so 3 total attempts)
            correlation_id: Session ID for logging
        
        Returns:
            Function result
        
        Raises:
            TimeoutError: If all attempts timeout
            Exception: If all retries fail
        """
        import asyncio
        import time as time_module
        
        correlation_id = correlation_id or "unknown"
        last_error = None
        
        for attempt in range(max_retries + 1):  # +1 because first attempt is not a retry
            try:
                logger.info(f"⏱️ Attempt {attempt + 1}/{max_retries + 1} (timeout={timeout_seconds}s) - correlation_id={correlation_id}")
                start_time = time_module.time()
                
                # Execute with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(), timeout=timeout_seconds)
                else:
                    # For sync functions, run in executor with timeout
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func),
                        timeout=timeout_seconds
                    )
                
                duration_ms = (time_module.time() - start_time) * 1000
                logger.info(f"✅ Attempt {attempt + 1} succeeded in {duration_ms:.0f}ms - correlation_id={correlation_id}")
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Operation timed out after {timeout_seconds}s")
                logger.warning(f"⏱️ Attempt {attempt + 1} timed out - correlation_id={correlation_id}")
                
                if attempt < max_retries:
                    backoff_ms = 200 * (2 ** attempt)  # 200ms, 400ms, 800ms...
                    logger.info(f"🔄 Retrying in {backoff_ms}ms (exponential backoff) - correlation_id={correlation_id}")
                    await asyncio.sleep(backoff_ms / 1000)
                else:
                    logger.error(f"❌ All {max_retries + 1} attempts timed out - correlation_id={correlation_id}")
                    raise last_error
                    
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e} - correlation_id={correlation_id}")
                
                # Check if error is transient
                if self._is_transient_error(e) and attempt < max_retries:
                    backoff_ms = 200 * (2 ** attempt)  # 200ms, 400ms, 600ms (capped at 600ms per requirements)
                    if backoff_ms > 600:
                        backoff_ms = 600
                    logger.info(f"🔄 Retrying transient error in {backoff_ms}ms - correlation_id={correlation_id}")
                    await asyncio.sleep(backoff_ms / 1000)
                else:
                    # Non-transient error or out of retries
                    logger.error(f"❌ Failed after {attempt + 1} attempts: {e} - correlation_id={correlation_id}")
                    raise e
        
        # Should not reach here, but just in case
        raise last_error or Exception("Unknown error in _execute_with_timeout")
    
    def _final_read_verify(
        self,
        appointment_id: Optional[str],
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        correlation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Final read verification: Query DB to confirm appointment exists with correct status.
        
        This is the CRITICAL final step - never confirm without this verification.
        
        Args:
            appointment_id: Appointment ID from create call (if available)
            patient_id: Patient ID
            doctor_id: Doctor ID
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            correlation_id: Session ID for logging
        
        Returns:
            Appointment dict if found and verified, None otherwise
        """
        correlation_id = correlation_id or "unknown"
        
        logger.info(f"🔍 FINAL READ VERIFY: correlation_id={correlation_id}")
        logger.info(f"   appointment_id={appointment_id}, patient_id={patient_id}, doctor_id={doctor_id}")
        logger.info(f"   date={date}, time={time}")
        
        # Try by appointment_id first (fastest)
        if appointment_id:
            try:
                appointment = self.db_client.get_appointment_by_id(appointment_id)
                if appointment:
                    logger.info(f"✅ Found appointment by ID: {appointment_id}")
                    # Verify details match
                    if self._verify_appointment_details(appointment, patient_id, doctor_id, date, time):
                        # Check status is confirmed/scheduled/pending
                        status = appointment.get('status', '')
                        if status in ['scheduled', 'confirmed', 'pending']:
                            logger.info(f"✅ FINAL READ VERIFY PASSED: Status={status}")
                            return appointment
                        else:
                            logger.error(f"❌ FINAL READ VERIFY FAILED: Invalid status={status}")
                            return None
            except Exception as e:
                logger.warning(f"⚠️ Failed to query by appointment_id: {e}")
        
        # Fallback: Query by patient + doctor + date + time
        try:
            logger.info(f"🔍 Fallback: Querying by patient+doctor+date+time")
            appointments = self.db_client.get_patient_appointments(patient_id)
            
            if appointments:
                for apt in appointments:
                    apt_date = str(apt.get('appointment_date', ''))
                    if 'T' in apt_date:
                        apt_date = apt_date.split('T')[0]
                    
                    apt_time = str(apt.get('start_time', ''))[:5]
                    expected_time_short = time[:5] if len(time) >= 5 else time
                    
                    if (apt.get('doctor_id') == doctor_id and
                        date in apt_date and
                        expected_time_short == apt_time):
                        logger.info(f"✅ Found appointment by query: {apt.get('id')}")
                        # Verify details
                        if self._verify_appointment_details(apt, patient_id, doctor_id, date, time):
                            status = apt.get('status', '')
                            if status in ['scheduled', 'confirmed', 'pending']:
                                logger.info(f"✅ FINAL READ VERIFY PASSED: Status={status}")
                                return apt
                            else:
                                logger.error(f"❌ FINAL READ VERIFY FAILED: Invalid status={status}")
                                return None
        except Exception as e:
            logger.error(f"❌ Failed to query by patient+doctor+date+time: {e}")
        
        logger.error(f"❌ FINAL READ VERIFY FAILED: Appointment not found in DB")
        return None
    
    # ========== DB VERIFICATION METHODS ==========
    # These methods ensure we never send false confirmations to users
    
    def _verify_appointment_in_db(
        self,
        appointment_id: str,
        max_retries: int = 3,
        retry_delay_ms: int = 500
    ) -> Optional[Dict[str, Any]]:
        """
        Verify appointment exists in database with retries.
        
        This is a CRITICAL safety check to prevent false confirmations.
        Never trust API response alone - always verify DB state.
        
        Args:
            appointment_id: The appointment ID to verify
            max_retries: Maximum retry attempts (default 3)
            retry_delay_ms: Delay between retries in milliseconds (default 500)
        
        Returns:
            Appointment dict if found and valid, None otherwise
        """
        import time as time_module
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔍 Verification attempt {attempt + 1}/{max_retries}: Querying DB for appointment {appointment_id}")
                
                # Query DB for the appointment
                appointment = self.db_client.get_appointment_by_id(appointment_id)
                
                if appointment:
                    logger.info(f"✅ Verification attempt {attempt + 1}: Found appointment {appointment_id}")
                    logger.info(f"   Status: {appointment.get('status')}")
                    logger.info(f"   Patient: {appointment.get('patient_id')}")
                    logger.info(f"   Doctor: {appointment.get('doctor_id')}")
                    return appointment
                
                # Not found - wait and retry
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Verification attempt {attempt + 1}: Appointment not found, retrying in {retry_delay_ms}ms...")
                    time_module.sleep(retry_delay_ms / 1000)
                
            except Exception as e:
                logger.error(f"❌ Verification attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time_module.sleep(retry_delay_ms / 1000)
        
        # All retries exhausted
        logger.error(f"❌ VERIFICATION FAILED: Appointment {appointment_id} not found after {max_retries} attempts")
        return None

    def _verify_appointment_details(
        self,
        appointment: Dict[str, Any],
        expected_patient_id: str,
        expected_doctor_id: str,
        expected_date: str,
        expected_time: str
    ) -> bool:
        """
        Verify appointment details match what was requested.
        
        This prevents scenarios where DB returns a different appointment
        or where data corruption occurs.
        
        Args:
            appointment: The appointment dict from DB
            expected_patient_id: Expected patient ID
            expected_doctor_id: Expected doctor ID
            expected_date: Expected date (YYYY-MM-DD)
            expected_time: Expected time (HH:MM)
        
        Returns:
            True if all details match, False otherwise
        """
        try:
            # Check patient
            actual_patient_id = appointment.get('patient_id')
            if actual_patient_id != expected_patient_id:
                logger.error(f"❌ Patient ID mismatch: expected={expected_patient_id}, actual={actual_patient_id}")
                return False
            
            # Check doctor
            actual_doctor_id = appointment.get('doctor_id')
            if actual_doctor_id != expected_doctor_id:
                logger.error(f"❌ Doctor ID mismatch: expected={expected_doctor_id}, actual={actual_doctor_id}")
                return False
            
            # Check date (handle different formats)
            actual_date = str(appointment.get('appointment_date', ''))
            # Extract just the date part if it includes time
            if 'T' in actual_date:
                actual_date = actual_date.split('T')[0]
            
            if expected_date not in actual_date and actual_date not in expected_date:
                logger.error(f"❌ Date mismatch: expected={expected_date}, actual={actual_date}")
                return False
            
            # Check time (compare HH:MM part)
            actual_time = str(appointment.get('start_time', ''))
            expected_time_short = expected_time[:5] if len(expected_time) >= 5 else expected_time
            actual_time_short = actual_time[:5] if len(actual_time) >= 5 else actual_time
            
            if expected_time_short != actual_time_short:
                logger.error(f"❌ Time mismatch: expected={expected_time_short}, actual={actual_time_short}")
                return False
            
            # Check status (should be scheduled, confirmed, or pending)
            status = appointment.get('status', '')
            valid_statuses = ['scheduled', 'confirmed', 'pending']
            if status not in valid_statuses:
                logger.error(f"❌ Invalid status: {status} (expected one of {valid_statuses})")
                return False
            
            logger.info(f"✅ All appointment details verified successfully")
            logger.info(f"   Patient: {actual_patient_id} ✓")
            logger.info(f"   Doctor: {actual_doctor_id} ✓")
            logger.info(f"   Date: {actual_date} ✓")
            logger.info(f"   Time: {actual_time_short} ✓")
            logger.info(f"   Status: {status} ✓")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying appointment details: {e}")
            return False

    def _log_booking_metrics(
        self,
        appointment_inserted: bool,
        db_row_id: Optional[str],
        verification_latency_ms: float,
        scheduling_latency_ms: float,
        dashboard_visible: bool,
        request_id: Optional[str],
        mismatch_detected: bool = False,
        error_code: Optional[str] = None
    ):
        """
        Log structured metrics for appointment booking.
        
        These metrics are used for monitoring and alerting on false confirmations.
        
        Args:
            appointment_inserted: Whether appointment was inserted in DB
            db_row_id: The appointment ID if inserted
            verification_latency_ms: Time taken for DB verification
            scheduling_latency_ms: Time taken for API scheduling call
            dashboard_visible: Whether appointment is visible in dashboard
            request_id: Idempotency request ID
            mismatch_detected: Whether a false confirmation was detected
            error_code: Error code if any
        """
        import json
        from datetime import datetime as dt
        
        metrics = {
            "appointment_inserted": appointment_inserted,
            "db_row_id": db_row_id,
            "verification_latency_ms": round(verification_latency_ms, 2),
            "scheduling_call_latency_ms": round(scheduling_latency_ms, 2),
            "dashboard_visible": dashboard_visible,
            "request_id": request_id,
            "mismatch_detected": mismatch_detected,
            "last_error_code": error_code,
            "timestamp": dt.now().isoformat()
        }
        
        # Log structured metrics
        logger.info(f"📊 METRICS: {json.dumps(metrics)}")
        
        # Log threshold alerts
        if verification_latency_ms > 15000:
            logger.error(f"🚨 THRESHOLD CRITICAL: verification_latency_ms ({verification_latency_ms:.0f}) > 15000ms")
        elif verification_latency_ms > 5000:
            logger.warning(f"⚠️ THRESHOLD WARN: verification_latency_ms ({verification_latency_ms:.0f}) > 5000ms")
        
        # Log mismatch alert
        if mismatch_detected:
            logger.error(f"🚨 MISMATCH DETECTED: API returned success but DB verification failed!")
            logger.error(f"   appointment_id={db_row_id}, request_id={request_id}")
        
        return metrics

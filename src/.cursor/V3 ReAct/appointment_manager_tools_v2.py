"""
Enhanced Appointment Manager Tools - Result Classification

This file shows how to update appointment_manager.py tools to return
proper result_type values and actionable fields.

Each tool now returns:
- result_type: SUCCESS, PARTIAL, USER_INPUT, RECOVERABLE, FATAL, SYSTEM_ERROR
- Actionable fields: alternatives, recovery_action, suggested_response, etc.
- Clear success/failure indicators
- Criteria satisfaction hints
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPE ENUM (import from base_agent_v2 in actual implementation)
# =============================================================================

class ToolResultType(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    USER_INPUT_NEEDED = "user_input"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"
    SYSTEM_ERROR = "system_error"


# =============================================================================
# ENHANCED TOOLS FOR APPOINTMENT MANAGER
# =============================================================================

class AppointmentManagerToolsMixin:
    """
    Mixin containing all enhanced tools for AppointmentManagerAgent.
    
    Add these methods to your AppointmentManagerAgent class.
    """
    
    # -------------------------------------------------------------------------
    # TOOL: List Doctors
    # -------------------------------------------------------------------------
    
    async def tool_list_doctors(
        self,
        specialty: Optional[str] = None,
        search_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List available doctors, optionally filtered by specialty or name.
        
        Args:
            specialty: Filter by specialty (e.g., "General Dentistry", "Orthodontics")
            search_name: Search by name (partial match)
        
        Returns:
            List of doctors with their details
        """
        try:
            # Call database
            doctors = await self.db_client.get_doctors(
                specialty=specialty,
                search=search_name
            )
            
            if not doctors:
                # No doctors found - provide recovery path
                if search_name:
                    # Name search failed - suggest listing all
                    return {
                        "success": True,  # Query succeeded, just no results
                        "result_type": ToolResultType.RECOVERABLE.value,
                        "doctors": [],
                        "count": 0,
                        "error": f"No doctor found matching '{search_name}'",
                        "recovery_action": "list_doctors",  # Call without search
                        "recovery_message": "Try listing all doctors with specialty instead",
                        "suggested_response": f"I couldn't find a doctor named '{search_name}'. Let me show you our available doctors."
                    }
                else:
                    # No doctors at all - system issue
                    return {
                        "success": False,
                        "result_type": ToolResultType.SYSTEM_ERROR.value,
                        "error": "No doctors found in system",
                        "should_retry": False
                    }
            
            # Format doctor results with useful extra info
            formatted_doctors = []
            for doc in doctors:
                formatted_doctors.append({
                    "id": doc.get("id"),
                    "name": doc.get("name") or f"Dr. {doc.get('first_name', '')} {doc.get('last_name', '')}",
                    "specialty": doc.get("specialty"),
                    "next_available": doc.get("next_available"),  # If available from DB
                })
            
            # Success - return PARTIAL because we still need to check availability and book
            return {
                "success": True,
                "result_type": ToolResultType.PARTIAL.value,
                "doctors": formatted_doctors,
                "count": len(formatted_doctors),
                "next_step": "check_availability",
                "message": f"Found {len(formatted_doctors)} doctor(s)"
            }
        
        except Exception as e:
            logger.error(f"Error listing doctors: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
    
    # -------------------------------------------------------------------------
    # TOOL: Check Availability
    # -------------------------------------------------------------------------
    
    async def tool_check_availability(
        self,
        doctor_id: str,
        date: str,
        requested_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check doctor's availability for a specific date and optionally time.
        
        Args:
            doctor_id: UUID of the doctor
            date: Date to check (YYYY-MM-DD format)
            requested_time: Specific time to check (HH:MM format, optional)
        
        Returns:
            Availability info with slots or alternatives
        """
        try:
            # Validate doctor exists
            doctor = await self.db_client.get_doctor(doctor_id)
            if not doctor:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "doctor_not_found",
                    "error_message": f"Doctor with ID {doctor_id} not found",
                    "recovery_action": "list_doctors",
                    "suggested_response": "I couldn't find that doctor. Let me show you our available doctors."
                }
            
            # Get available slots
            available_slots = await self.db_client.get_available_slots(doctor_id, date)
            
            # Check if clinic is open on this date
            if available_slots is None:
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": "doctor_unavailable",
                    "error_message": f"Doctor is unavailable on {date}",
                    "suggested_response": f"I'm sorry, doctor is unavailable on {date}. Would you like to try a different date?"
                }
            
            # No slots available
            if len(available_slots) == 0:
                # Try to get next available date
                next_date = await self._find_next_available_date(doctor_id, date)
                return {
                    "success": True,  # Query worked, just no availability
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "doctor_id": doctor_id,
                    "date": date,
                    "available": False,
                    "available_at_requested_time": False,
                    "available_slots": [],
                    "reason": "no_availability_on_date",
                    "alternatives": [],
                    "next_available_date": next_date,
                    "blocks_criteria": "appointment booked",
                    "suggested_response": f"Dr. {doctor.get('name', 'the doctor')} is fully booked on {date}. The next available date is {next_date}. Would that work?"
                }
            
            # Specific time requested
            if requested_time:
                # Normalize time format
                normalized_time = self._normalize_time(requested_time)
                
                if normalized_time in available_slots:
                    # ✅ Requested time IS available
                    return {
                        "success": True,
                        "result_type": ToolResultType.PARTIAL.value,  # Partial - still need to book
                        "doctor_id": doctor_id,
                        "date": date,
                        "requested_time": normalized_time,
                        "available": True,
                        "available_at_requested_time": True,
                        "can_proceed": True,
                        "message": f"Time {normalized_time} is available!",
                        "booking_params": {
                            "doctor_id": doctor_id,
                            "date": date,
                            "time": normalized_time
                        }
                    }
                else:
                    # ❌ Requested time NOT available - need user input
                    # Find closest alternatives
                    alternatives = self._find_closest_times(normalized_time, available_slots)
                    
                    return {
                        "success": True,  # Query succeeded!
                        "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                        "doctor_id": doctor_id,
                        "date": date,
                        "requested_time": normalized_time,
                        "available": False,
                        "available_at_requested_time": False,
                        "can_proceed": False,  # Cannot proceed without user choice
                        "reason": "requested_time_unavailable",
                        "alternatives": alternatives,
                        "all_available_slots": available_slots,
                        "blocks_criteria": "appointment booked",
                        "next_action": "ask_user_for_alternative",
                        "suggested_response": self._format_alternatives_message(
                            normalized_time, alternatives, doctor.get('name')
                        )
                    }
            
            # No specific time - return all slots
            return {
                "success": True,
                "result_type": ToolResultType.PARTIAL.value,
                "doctor_id": doctor_id,
                "date": date,
                "available": True,
                "available_slots": available_slots,
                "count": len(available_slots),
                "next_step": "ask_user_preferred_time_or_book",
                "message": f"{len(available_slots)} time slots available"
            }
        
        except Exception as e:
            logger.error(f"Error checking availability: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
    
    def _normalize_time(self, time_str: str) -> str:
        """Normalize time to HH:MM format."""
        time_str = time_str.strip().lower()
        
        # Handle various formats
        import re
        
        # "3pm", "3 pm", "15:00", "3:00pm", etc.
        patterns = [
            (r'^(\d{1,2}):(\d{2})\s*(am|pm)?$', lambda m: self._convert_12_to_24(m.group(1), m.group(2), m.group(3))),
            (r'^(\d{1,2})\s*(am|pm)$', lambda m: self._convert_12_to_24(m.group(1), "00", m.group(2))),
            (r'^(\d{1,2}):(\d{2})$', lambda m: f"{int(m.group(1)):02d}:{m.group(2)}"),
        ]
        
        for pattern, converter in patterns:
            match = re.match(pattern, time_str)
            if match:
                return converter(match)
        
        return time_str
    
    def _convert_12_to_24(self, hour: str, minute: str, period: Optional[str]) -> str:
        """Convert 12-hour time to 24-hour format."""
        h = int(hour)
        if period:
            if period.lower() == 'pm' and h != 12:
                h += 12
            elif period.lower() == 'am' and h == 12:
                h = 0
        return f"{h:02d}:{minute}"
    
    def _find_closest_times(self, requested: str, available: List[str], count: int = 5) -> List[str]:
        """Find times closest to the requested time."""
        try:
            req_minutes = int(requested.split(':')[0]) * 60 + int(requested.split(':')[1])
            
            def to_minutes(t):
                parts = t.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            
            sorted_slots = sorted(available, key=lambda t: abs(to_minutes(t) - req_minutes))
            return sorted_slots[:count]
        except:
            return available[:count]
    
    def _format_alternatives_message(
        self,
        requested: str,
        alternatives: List[str],
        doctor_name: Optional[str] = None
    ) -> str:
        """Format a user-friendly message with alternatives."""
        doctor_str = f"with Dr. {doctor_name} " if doctor_name else ""
        
        if not alternatives:
            return f"I'm sorry, {requested} isn't available {doctor_str}and there are no other times on this date."
        
        if len(alternatives) == 1:
            return f"I'm sorry, {requested} isn't available {doctor_str}. Would {alternatives[0]} work instead?"
        
        if len(alternatives) == 2:
            return f"I'm sorry, {requested} isn't available {doctor_str}. Would {alternatives[0]} or {alternatives[1]} work?"
        
        # 3 or more
        options = ", ".join(alternatives[:2]) + f", or {alternatives[2]}"
        return f"I'm sorry, {requested} isn't available {doctor_str}. I have {options} available. Which would you prefer?"
    
    async def _find_next_available_date(self, doctor_id: str, from_date: str) -> Optional[str]:
        """Find the next date with availability."""
        try:
            from datetime import datetime, timedelta
            current = datetime.strptime(from_date, "%Y-%m-%d")
            
            for i in range(1, 14):  # Check next 2 weeks
                next_date = current + timedelta(days=i)
                date_str = next_date.strftime("%Y-%m-%d")
                slots = await self.db_client.get_available_slots(doctor_id, date_str)
                if slots and len(slots) > 0:
                    return date_str
            
            return None
        except:
            return None
    
    # -------------------------------------------------------------------------
    # TOOL: Book Appointment
    # -------------------------------------------------------------------------
    
    async def tool_book_appointment(
        self,
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: Optional[str] = None,
        procedure: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Book an appointment for a patient.
        
        Args:
            patient_id: UUID of the patient
            doctor_id: UUID of the doctor
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            reason: Reason for visit
            procedure: Specific procedure (e.g., "root canal", "cleaning")
            notes: Additional notes
        
        Returns:
            Booking confirmation or error with recovery path
        """
        try:
            # Validate patient
            if not patient_id:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "patient_not_registered",
                    "error_message": "Patient must be registered before booking",
                    "recovery_action": "register_patient",
                    "recovery_message": "Please register the patient first",
                    "suggested_response": "I'll need to get you registered first. Can I have your full name and phone number?"
                }
            
            patient = await self.db_client.get_patient(patient_id)
            if not patient:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "patient_not_found",
                    "error_message": f"Patient {patient_id} not found",
                    "recovery_action": "register_patient",
                    "suggested_response": "I couldn't find your patient record. Let's get you registered."
                }
            
            # Validate doctor
            doctor = await self.db_client.get_doctor(doctor_id)
            if not doctor:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "doctor_not_found",
                    "recovery_action": "list_doctors",
                    "suggested_response": "I couldn't find that doctor. Let me show you our available doctors."
                }
            
            # Check for existing appointment at same time
            existing = await self.db_client.check_appointment_conflict(
                patient_id=patient_id,
                date=date,
                time=time
            )
            if existing:
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": "time_conflict",
                    "error_message": f"You already have an appointment at {time}",
                    "conflicting_appointment": {
                        "id": existing.get("id"),
                        "doctor": existing.get("doctor_name"),
                        "time": existing.get("time"),
                        "reason": existing.get("reason")
                    },
                    "alternatives": await self._get_alternative_times(doctor_id, date, time),
                    "suggested_response": f"You already have an appointment at {time}. Would you like to book for a different time, or reschedule your existing appointment?"
                }
            
            # Verify time is still available (double-check)
            available_slots = await self.db_client.get_available_slots(doctor_id, date)
            normalized_time = self._normalize_time(time)
            
            if normalized_time not in available_slots:
                alternatives = self._find_closest_times(normalized_time, available_slots)
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": "time_no_longer_available",
                    "error_message": f"The {normalized_time} slot is no longer available",
                    "alternatives": alternatives,
                    "blocks_criteria": f"{procedure or 'appointment'} booked",
                    "suggested_response": f"I'm sorry, that time was just taken. Would {alternatives[0] if alternatives else 'another time'} work?"
                }
            
            # Attempt to create the appointment
            appointment = await self.db_client.create_appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                date=date,
                time=normalized_time,
                reason=reason or procedure,
                notes=notes
            )
            
            if not appointment or not appointment.get("id"):
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "booking_failed",
                    "error_message": "Failed to create appointment in database",
                    "should_retry": True
                }
            
            # ✅ SUCCESS!
            doctor_name = doctor.get("name") or f"Dr. {doctor.get('last_name', '')}"
            procedure_str = procedure or reason or "appointment"
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "verified": True,
                "appointment_id": appointment.get("id"),
                "appointment": {
                    "id": appointment.get("id"),
                    "patient_id": patient_id,
                    "doctor_id": doctor_id,
                    "doctor_name": doctor_name,
                    "date": date,
                    "time": normalized_time,
                    "reason": reason or procedure,
                    "status": "scheduled"
                },
                "satisfies_criteria": [
                    f"{procedure_str} appointment booked",
                    f"{procedure_str} appointment booked with appointment_id"
                ],
                "confirmation_message": f"Your {procedure_str} with {doctor_name} is confirmed for {date} at {normalized_time}.",
                "suggested_response": f"✅ Your {procedure_str} is booked with {doctor_name} for {date} at {normalized_time}. Your confirmation number is {appointment.get('id')[:8]}."
            }
        
        except Exception as e:
            logger.error(f"Error booking appointment: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
    
    async def _get_alternative_times(
        self,
        doctor_id: str,
        date: str,
        around_time: str
    ) -> List[str]:
        """Get alternative times around a given time."""
        try:
            slots = await self.db_client.get_available_slots(doctor_id, date)
            return self._find_closest_times(around_time, slots)
        except:
            return []
    
    # -------------------------------------------------------------------------
    # TOOL: Cancel Appointment
    # -------------------------------------------------------------------------
    
    async def tool_cancel_appointment(
        self,
        appointment_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an existing appointment.
        
        Args:
            appointment_id: UUID of the appointment to cancel
            reason: Reason for cancellation
        
        Returns:
            Cancellation confirmation or error
        """
        try:
            # Get appointment
            appointment = await self.db_client.get_appointment(appointment_id)
            
            if not appointment:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "appointment_not_found",
                    "error_message": f"Appointment {appointment_id} not found",
                    "recovery_action": "get_patient_appointments",
                    "suggested_response": "I couldn't find that appointment. Let me show you your scheduled appointments."
                }
            
            # Check if already cancelled
            if appointment.get("status") == "cancelled":
                return {
                    "success": True,
                    "result_type": ToolResultType.SUCCESS.value,
                    "appointment_id": appointment_id,
                    "already_cancelled": True,
                    "message": "This appointment was already cancelled",
                    "suggested_response": "That appointment has already been cancelled. Is there anything else I can help with?"
                }
            
            # Check if appointment is in the past
            from datetime import datetime
            apt_datetime = datetime.strptime(
                f"{appointment['date']} {appointment['time']}",
                "%Y-%m-%d %H:%M"
            )
            if apt_datetime < datetime.now():
                return {
                    "success": False,
                    "result_type": ToolResultType.FATAL.value,
                    "error": "appointment_in_past",
                    "error_message": "Cannot cancel past appointments",
                    "suggested_response": "That appointment has already passed and cannot be cancelled."
                }
            
            # Perform cancellation
            result = await self.db_client.cancel_appointment(appointment_id, reason)
            
            if not result:
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "cancellation_failed",
                    "should_retry": True
                }
            
            # ✅ SUCCESS
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "cancelled": True,
                "appointment_id": appointment_id,
                "cancelled_appointment": {
                    "date": appointment.get("date"),
                    "time": appointment.get("time"),
                    "doctor": appointment.get("doctor_name"),
                    "reason": appointment.get("reason")
                },
                "satisfies_criteria": [
                    "appointment cancelled",
                    f"appointment {appointment_id} cancelled"
                ],
                "suggested_response": f"I've cancelled your appointment on {appointment.get('date')} at {appointment.get('time')}. Is there anything else I can help with?"
            }
        
        except Exception as e:
            logger.error(f"Error cancelling appointment: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
    
    # -------------------------------------------------------------------------
    # TOOL: Get Patient Appointments
    # -------------------------------------------------------------------------
    
    async def tool_get_patient_appointments(
        self,
        patient_id: str,
        status: Optional[str] = "scheduled",
        include_past: bool = False
    ) -> Dict[str, Any]:
        """
        Get appointments for a patient.
        
        Args:
            patient_id: UUID of the patient
            status: Filter by status ("scheduled", "cancelled", "completed", or None for all)
            include_past: Whether to include past appointments
        
        Returns:
            List of patient's appointments
        """
        try:
            if not patient_id:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "patient_not_registered",
                    "recovery_action": "register_patient",
                    "suggested_response": "I'll need to look up your patient record first. Can you confirm your name?"
                }
            
            appointments = await self.db_client.get_patient_appointments(
                patient_id=patient_id,
                status=status,
                include_past=include_past
            )
            
            if not appointments:
                return {
                    "success": True,
                    "result_type": ToolResultType.PARTIAL.value,
                    "appointments": [],
                    "count": 0,
                    "message": "No appointments found",
                    "suggested_response": "You don't have any upcoming appointments. Would you like to schedule one?"
                }
            
            # Format appointments
            formatted = []
            for apt in appointments:
                formatted.append({
                    "id": apt.get("id"),
                    "date": apt.get("date"),
                    "time": apt.get("time"),
                    "doctor_name": apt.get("doctor_name"),
                    "reason": apt.get("reason"),
                    "status": apt.get("status")
                })
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "appointments": formatted,
                "count": len(formatted),
                "satisfies_criteria": ["appointments listed", "show appointments"],
                "message": f"Found {len(formatted)} appointment(s)"
            }
        
        except Exception as e:
            logger.error(f"Error getting appointments: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
    
    # -------------------------------------------------------------------------
    # TOOL: Reschedule Appointment
    # -------------------------------------------------------------------------
    
    async def tool_reschedule_appointment(
        self,
        appointment_id: str,
        new_date: Optional[str] = None,
        new_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reschedule an existing appointment.
        
        Args:
            appointment_id: UUID of the appointment to reschedule
            new_date: New date (YYYY-MM-DD), or None to keep same date
            new_time: New time (HH:MM), or None to keep same time
        
        Returns:
            Rescheduling confirmation or error
        """
        try:
            # Get existing appointment
            appointment = await self.db_client.get_appointment(appointment_id)
            
            if not appointment:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "appointment_not_found",
                    "recovery_action": "get_patient_appointments",
                    "suggested_response": "I couldn't find that appointment. Let me show you your scheduled appointments."
                }
            
            # Use existing values if not provided
            target_date = new_date or appointment.get("date")
            target_time = self._normalize_time(new_time) if new_time else appointment.get("time")
            doctor_id = appointment.get("doctor_id")
            
            # Check new time is available
            available_slots = await self.db_client.get_available_slots(doctor_id, target_date)
            
            if target_time not in available_slots:
                alternatives = self._find_closest_times(target_time, available_slots)
                return {
                    "success": True,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": "new_time_unavailable",
                    "requested_time": target_time,
                    "requested_date": target_date,
                    "available": False,
                    "alternatives": alternatives,
                    "blocks_criteria": "appointment rescheduled",
                    "suggested_response": f"{target_time} on {target_date} isn't available. Would {alternatives[0] if alternatives else 'another time'} work?"
                }
            
            # Perform reschedule
            result = await self.db_client.reschedule_appointment(
                appointment_id=appointment_id,
                new_date=target_date,
                new_time=target_time
            )
            
            if not result:
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "reschedule_failed",
                    "should_retry": True
                }
            
            # ✅ SUCCESS
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "rescheduled": True,
                "appointment_id": appointment_id,
                "previous": {
                    "date": appointment.get("date"),
                    "time": appointment.get("time")
                },
                "new": {
                    "date": target_date,
                    "time": target_time
                },
                "satisfies_criteria": [
                    "appointment rescheduled",
                    f"appointment {appointment_id} rescheduled"
                ],
                "suggested_response": f"Done! I've rescheduled your appointment from {appointment.get('date')} {appointment.get('time')} to {target_date} at {target_time}."
            }
        
        except Exception as e:
            logger.error(f"Error rescheduling appointment: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
    
    # -------------------------------------------------------------------------
    # TOOL: Smart Book (Composite)
    # -------------------------------------------------------------------------
    
    async def tool_smart_book_appointment(
        self,
        patient_id: str,
        doctor_name: Optional[str] = None,
        doctor_id: Optional[str] = None,
        date: str = None,
        time: str = None,
        procedure: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Smart booking that handles the complete flow:
        1. Find doctor by name (if ID not provided)
        2. Check availability
        3. Book if available, or return alternatives
        
        Args:
            patient_id: UUID of the patient
            doctor_name: Name to search for (if no ID)
            doctor_id: Doctor UUID (if known)
            date: Desired date
            time: Desired time
            procedure: Type of appointment
        
        Returns:
            Booking result or alternatives needed
        """
        try:
            # Validate patient
            if not patient_id:
                return {
                    "success": False,
                    "result_type": ToolResultType.RECOVERABLE.value,
                    "error": "patient_not_registered",
                    "recovery_action": "register_patient",
                    "suggested_response": "I'll need to register you first before booking. Can I have your name and phone number?"
                }
            
            # Step 1: Resolve doctor
            if not doctor_id and doctor_name:
                doctors = await self.db_client.get_doctors(search=doctor_name)
                if not doctors:
                    all_doctors = await self.db_client.get_doctors()
                    return {
                        "success": False,
                        "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                        "error": "doctor_not_found",
                        "error_message": f"No doctor found matching '{doctor_name}'",
                        "available_doctors": [d.get("name") for d in all_doctors[:5]],
                        "suggested_response": f"I couldn't find Dr. {doctor_name}. Did you mean one of these doctors: {', '.join(d.get('name') for d in all_doctors[:3])}?"
                    }
                doctor_id = doctors[0].get("id")
                doctor = doctors[0]
            elif doctor_id:
                doctor = await self.db_client.get_doctor(doctor_id)
                if not doctor:
                    return {
                        "success": False,
                        "result_type": ToolResultType.RECOVERABLE.value,
                        "error": "doctor_not_found",
                        "recovery_action": "list_doctors"
                    }
            else:
                return {
                    "success": False,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "error": "no_doctor_specified",
                    "suggested_response": "Which doctor would you like to see? I can show you our available doctors."
                }
            
            # Step 2: Check availability
            available_slots = await self.db_client.get_available_slots(doctor_id, date)
            
            if not available_slots:
                next_date = await self._find_next_available_date(doctor_id, date)
                return {
                    "success": True,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "doctor_id": doctor_id,
                    "doctor_name": doctor.get("name"),
                    "date": date,
                    "available": False,
                    "reason": "no_availability",
                    "next_available_date": next_date,
                    "blocks_criteria": f"{procedure or 'appointment'} booked",
                    "suggested_response": f"Dr. {doctor.get('name')} is fully booked on {date}. The next available date is {next_date}. Would that work?"
                }
            
            normalized_time = self._normalize_time(time) if time else None
            
            if normalized_time and normalized_time not in available_slots:
                alternatives = self._find_closest_times(normalized_time, available_slots)
                return {
                    "success": True,
                    "result_type": ToolResultType.USER_INPUT_NEEDED.value,
                    "doctor_id": doctor_id,
                    "doctor_name": doctor.get("name"),
                    "date": date,
                    "requested_time": normalized_time,
                    "available": False,
                    "available_at_requested_time": False,
                    "alternatives": alternatives,
                    "blocks_criteria": f"{procedure or 'appointment'} booked",
                    "suggested_response": self._format_alternatives_message(
                        normalized_time, alternatives, doctor.get("name")
                    )
                }
            
            # If no specific time, use first available
            booking_time = normalized_time or available_slots[0]
            
            # Step 3: Book the appointment
            appointment = await self.db_client.create_appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                date=date,
                time=booking_time,
                reason=procedure
            )
            
            if not appointment or not appointment.get("id"):
                return {
                    "success": False,
                    "result_type": ToolResultType.SYSTEM_ERROR.value,
                    "error": "booking_failed",
                    "should_retry": True
                }
            
            # ✅ SUCCESS
            doctor_name = doctor.get("name", "the doctor")
            procedure_str = procedure or "appointment"
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "verified": True,
                "appointment_id": appointment.get("id"),
                "appointment": {
                    "id": appointment.get("id"),
                    "doctor_name": doctor_name,
                    "date": date,
                    "time": booking_time,
                    "procedure": procedure
                },
                "steps_completed": ["doctor_resolved", "availability_checked", "appointment_booked"],
                "satisfies_criteria": [
                    f"{procedure_str} appointment booked",
                    f"{procedure_str} appointment booked with appointment_id"
                ],
                "suggested_response": f"✅ Your {procedure_str} with {doctor_name} is confirmed for {date} at {booking_time}. Your confirmation number is {appointment.get('id')[:8]}."
            }
        
        except Exception as e:
            logger.error(f"Error in smart booking: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
    
    # -------------------------------------------------------------------------
    # TOOL: Get Patient Status
    # -------------------------------------------------------------------------
    
    async def tool_get_patient_status(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get patient's current status and what they can do.
        
        Args:
            session_id: Session ID to look up patient
        
        Returns:
            Patient status including registration state and capabilities
        """
        try:
            # Get patient from state
            global_state = self.state_manager.get_global_state(session_id)
            patient_profile = global_state.patient_profile
            
            if not patient_profile.patient_id:
                return {
                    "success": True,
                    "result_type": ToolResultType.PARTIAL.value,
                    "is_registered": False,
                    "patient_id": None,
                    "can_book": False,
                    "needs_action": ["register"],
                    "message": "Patient not registered",
                    "next_step": "register_patient"
                }
            
            # Get upcoming appointments
            appointments = await self.db_client.get_patient_appointments(
                patient_id=patient_profile.patient_id,
                status="scheduled"
            )
            
            return {
                "success": True,
                "result_type": ToolResultType.SUCCESS.value,
                "is_registered": True,
                "patient_id": patient_profile.patient_id,
                "patient_name": f"{patient_profile.first_name} {patient_profile.last_name}".strip(),
                "can_book": True,
                "upcoming_appointments": len(appointments) if appointments else 0,
                "needs_action": []
            }
        
        except Exception as e:
            logger.error(f"Error getting patient status: {e}", exc_info=True)
            return {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e)
            }


# =============================================================================
# TOOL REGISTRATION
# =============================================================================

def register_tools(agent):
    """
    Register all tools with the agent.
    
    Call this in your AppointmentManagerAgent._register_tools() method:
    
    def _register_tools(self):
        register_appointment_tools(self)
    """
    agent._tools = {
        "list_doctors": agent.tool_list_doctors,
        "check_availability": agent.tool_check_availability,
        "book_appointment": agent.tool_book_appointment,
        "cancel_appointment": agent.tool_cancel_appointment,
        "get_patient_appointments": agent.tool_get_patient_appointments,
        "reschedule_appointment": agent.tool_reschedule_appointment,
        "smart_book_appointment": agent.tool_smart_book_appointment,
        "get_patient_status": agent.tool_get_patient_status,
    }

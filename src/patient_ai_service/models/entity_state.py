"""
Entity State - Combined Patient + Derived Entities for a Session

This module provides the unified entity layer that:
- Holds both patient entities and derived entities
- Handles the linkage between them
- Provides easy access methods for agents
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from .patient_entities import PatientEntities, PatientEntityChange, EntitySource
from .derived_entities import DerivedEntitiesManager, DerivedEntity


class EntityState(BaseModel):
    """
    Combined entity state for a session.
    
    This is the main interface for entity management, containing:
    - patient: Source of truth for what patient wants
    - derived: System resolutions from tool calls
    
    It handles the coordination between them.
    """
    
    session_id: str
    
    # Patient entities (source of truth)
    patient: PatientEntities
    
    # Derived entities (system resolutions)
    derived: DerivedEntitiesManager
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(cls, session_id: str) -> 'EntityState':
        """Create new entity state for a session."""
        return cls(
            session_id=session_id,
            patient=PatientEntities(session_id=session_id),
            derived=DerivedEntitiesManager(session_id=session_id)
        )
    
    def update_patient_preference(
        self,
        field_path: str,
        value: Any,
        source: EntitySource = EntitySource.USER_STATED,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Update a patient preference and handle derived entity invalidation.
        
        Returns:
            Dict with:
            - changed: bool - whether value actually changed
            - change: PatientEntityChange if changed
            - invalidated: List of invalidated derived entity keys
        """
        # Get old value for comparison
        parts = field_path.split(".")
        obj = self.patient
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        
        old_value = getattr(obj, parts[-1], None) if obj else None
        
        # Update patient entity
        change = self.patient.update_preference(field_path, value, source, confidence)
        
        result = {
            "changed": change is not None,
            "change": change,
            "invalidated": []
        }
        
        # If changed, invalidate affected derived entities
        if change:
            result["invalidated"] = self.derived.invalidate_for_patient_change(
                field_path, old_value, value
            )
            self.last_updated_at = datetime.utcnow()
        
        return result
    
    def store_tool_result(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> List[DerivedEntity]:
        """
        Process a tool result and extract derived entities.
        
        This method knows how to extract entities from different tool results.
        
        Args:
            tool_name: Name of the tool that was called
            tool_params: Parameters used in the tool call
            tool_result: The result from the tool
            
        Returns:
            List of derived entities that were stored
        """
        stored = []
        
        # Skip if tool failed
        if not tool_result.get("success", False):
            return stored
        
        # Extract entities based on tool type
        if tool_name in ["find_doctor_by_name", "list_doctors"]:
            stored.extend(self._extract_doctor_entities(tool_name, tool_params, tool_result))
        
        elif tool_name == "check_availability":
            stored.extend(self._extract_availability_entities(tool_name, tool_params, tool_result))
        
        elif tool_name == "book_appointment":
            stored.extend(self._extract_booking_entities(tool_name, tool_params, tool_result))
        
        elif tool_name == "register_patient":
            stored.extend(self._extract_registration_entities(tool_name, tool_params, tool_result))
        
        if stored:
            self.last_updated_at = datetime.utcnow()
        
        return stored
    
    def _extract_doctor_entities(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[DerivedEntity]:
        """Extract doctor-related derived entities."""
        stored = []
        
        if tool_name == "find_doctor_by_name":
            doctor = result.get("doctor", {})
            if doctor and doctor.get("id"):
                entity = self.derived.store_entity(
                    key="doctor_uuid",
                    value=doctor["id"],
                    source_tool=tool_name,
                    source_params=params,
                    resolves_patient_entity="appointment.doctor_preference",
                    resolves_patient_value=self.patient.appointment.doctor_preference
                )
                stored.append(entity)
                
                # Also store full doctor info
                info_entity = self.derived.store_entity(
                    key="doctor_info",
                    value=doctor,
                    source_tool=tool_name,
                    source_params=params,
                    resolves_patient_entity="appointment.doctor_preference",
                    resolves_patient_value=self.patient.appointment.doctor_preference
                )
                stored.append(info_entity)
        
        elif tool_name == "list_doctors":
            doctors = result.get("doctors", [])
            # Store all doctors for quick lookup
            entity = self.derived.store_entity(
                key="doctors_list",
                value=doctors,
                source_tool=tool_name,
                source_params=params,
                valid_for=None  # Never expires
            )
            stored.append(entity)
            
            # If patient has doctor preference, try to match
            if self.patient.appointment.doctor_preference:
                pref_lower = self.patient.appointment.doctor_preference.lower()
                for doc in doctors:
                    doc_name = doc.get("name", "").lower()
                    if pref_lower in doc_name or doc_name in pref_lower:
                        uuid_entity = self.derived.store_entity(
                            key="doctor_uuid",
                            value=doc["id"],
                            source_tool=tool_name,
                            source_params=params,
                            resolves_patient_entity="appointment.doctor_preference",
                            resolves_patient_value=self.patient.appointment.doctor_preference
                        )
                        stored.append(uuid_entity)
                        break
        
        return stored
    
    def _extract_availability_entities(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[DerivedEntity]:
        """Extract availability-related derived entities."""
        stored = []
        
        # Store availability check result
        entity = self.derived.store_entity(
            key="availability_check",
            value={
                "available": result.get("available", False),
                "available_at_requested_time": result.get("available_at_requested_time", False),
                "requested_time": result.get("requested_time"),
                "alternatives": result.get("alternatives", []),
                "doctor_id": params.get("doctor_id"),
                "date": params.get("date")
            },
            source_tool=tool_name,
            source_params=params,
            resolves_patient_entity="appointment.time_preference",
            resolves_patient_value=self.patient.appointment.time_preference,
            valid_for=300  # 5 minutes
        )
        stored.append(entity)
        
        # Store available slots if present
        if result.get("all_available_slots"):
            slots_entity = self.derived.store_entity(
                key="available_slots",
                value=result["all_available_slots"],
                source_tool=tool_name,
                source_params=params,
                valid_for=300
            )
            stored.append(slots_entity)
        
        return stored
    
    def _extract_booking_entities(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[DerivedEntity]:
        """Extract booking-related derived entities."""
        stored = []
        
        if result.get("appointment_id"):
            entity = self.derived.store_entity(
                key="last_booking",
                value={
                    "appointment_id": result["appointment_id"],
                    "doctor_id": params.get("doctor_id"),
                    "date": params.get("date"),
                    "time": params.get("time"),
                    "procedure": params.get("reason")
                },
                source_tool=tool_name,
                source_params=params,
                valid_for=None  # Never expires within session
            )
            stored.append(entity)
        
        return stored
    
    def _extract_registration_entities(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[DerivedEntity]:
        """Extract registration-related derived entities."""
        stored = []
        
        if result.get("patient_id"):
            entity = self.derived.store_entity(
                key="patient_id",
                value=result["patient_id"],
                source_tool=tool_name,
                source_params=params,
                valid_for=None  # Never expires
            )
            stored.append(entity)
        
        return stored
    
    def get_booking_readiness(self) -> Dict[str, Any]:
        """
        Check if we have everything needed to book an appointment.
        
        Returns dict with:
        - ready: bool - can we book right now?
        - has_doctor: bool
        - has_availability: bool
        - has_patient_id: bool
        - missing: list of what's missing
        - doctor_uuid: str if available
        - availability_confirmed: bool
        """
        readiness = {
            "ready": False,
            "has_doctor": False,
            "has_availability": False,
            "has_patient_id": False,
            "missing": [],
            "doctor_uuid": None,
            "availability_confirmed": False
        }
        
        # Check doctor
        doctor_entity = self.derived.get_valid_entity("doctor_uuid", self.patient)
        if doctor_entity:
            readiness["has_doctor"] = True
            readiness["doctor_uuid"] = doctor_entity.value
        else:
            readiness["missing"].append("doctor_uuid")
        
        # Check availability
        avail_entity = self.derived.get_valid_entity("availability_check", self.patient)
        if avail_entity and avail_entity.value.get("available_at_requested_time"):
            readiness["has_availability"] = True
            readiness["availability_confirmed"] = True
        else:
            readiness["missing"].append("availability_check")
        
        # Check patient ID
        patient_entity = self.derived.get_valid_entity("patient_id", self.patient)
        if patient_entity:
            readiness["has_patient_id"] = True
        else:
            readiness["missing"].append("patient_id")
        
        # Ready if we have everything
        readiness["ready"] = (
            readiness["has_doctor"] and
            readiness["has_availability"] and
            readiness["has_patient_id"]
        )
        
        return readiness
    
    def get_agent_context_display(self) -> str:
        """
        Generate combined display for agent prompt.
        
        Shows both patient preferences and derived entity status.
        """
        lines = []
        
        # Patient preferences section
        lines.append("═" * 60)
        lines.append("PATIENT PREFERENCES (Source of Truth)")
        lines.append("═" * 60)
        
        prefs = self.patient.appointment
        lines.append(f"• Doctor: {prefs.doctor_preference or '(not specified)'}")
        lines.append(f"• Date: {prefs.date_preference or '(not specified)'}")
        lines.append(f"• Time: {prefs.time_preference or '(not specified)'}")
        lines.append(f"• Procedure: {prefs.procedure_preference or '(not specified)'}")
        
        # Derived entities section
        lines.append("")
        lines.append(self.derived.get_status_display(self.patient))
        
        # Booking readiness
        readiness = self.get_booking_readiness()
        lines.append("")
        lines.append("═" * 60)
        if readiness["ready"]:
            lines.append("STATUS: ✅ READY TO BOOK")
            lines.append("All required entities are valid. Call book_appointment directly.")
        else:
            lines.append("STATUS: ⚠️ NOT READY")
            lines.append(f"Missing: {', '.join(readiness['missing'])}")
        lines.append("═" * 60)
        
        return "\n".join(lines)


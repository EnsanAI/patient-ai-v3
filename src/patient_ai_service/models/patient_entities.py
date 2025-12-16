"""
Patient Entities - Source of Truth for Patient Preferences

This module defines what the patient WANTS. These entities are:
- Updated by: Situation Assessor extractions, direct user input
- Never expire: Only change when patient explicitly changes mind
- Source of truth: Derived entities resolve FROM these
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class EntitySource(str, Enum):
    """How an entity was obtained."""
    USER_STATED = "user_stated"           # Patient explicitly said it
    EXTRACTED = "extracted"               # Extracted from natural language
    INFERRED = "inferred"                 # Inferred from context
    SYSTEM_DEFAULT = "system_default"     # Default value
    CONTINUED = "continued"               # Carried from previous turn


class PatientEntityChange(BaseModel):
    """Record of a change to a patient entity."""
    field: str
    old_value: Optional[Any] = None
    new_value: Any
    changed_at: datetime = Field(default_factory=datetime.utcnow)
    source: EntitySource = EntitySource.USER_STATED
    confidence: float = 1.0


class AppointmentPreferences(BaseModel):
    """Patient preferences for appointment booking."""
    
    # Doctor preference
    doctor_preference: Optional[str] = None  # "Dr. Sarah", "Dr. Ahmed" - name, NOT UUID
    doctor_preference_source: EntitySource = EntitySource.USER_STATED
    doctor_preference_confidence: float = 1.0
    
    # Date preference
    date_preference: Optional[str] = None    # "tomorrow", "Monday", "2024-12-15"
    date_resolved: Optional[str] = None      # Resolved to YYYY-MM-DD format
    date_preference_source: EntitySource = EntitySource.USER_STATED
    
    # Time preference
    time_preference: Optional[str] = None    # "3pm", "afternoon", "morning"
    time_resolved: Optional[str] = None      # Resolved to HH:MM format
    time_preference_source: EntitySource = EntitySource.USER_STATED
    
    # Procedure preference
    procedure_preference: Optional[str] = None  # "cleaning", "root canal"
    procedure_preference_source: EntitySource = EntitySource.USER_STATED
    
    # Additional preferences
    urgency: Optional[str] = None            # "routine", "urgent", "emergency"
    special_requests: List[str] = Field(default_factory=list)
    
    def has_doctor_preference(self) -> bool:
        return self.doctor_preference is not None
    
    def has_time_preference(self) -> bool:
        return self.time_preference is not None or self.date_preference is not None


class PersonalInfo(BaseModel):
    """Patient personal information for registration."""
    
    first_name: Optional[str] = None
    first_name_source: EntitySource = EntitySource.USER_STATED
    
    last_name: Optional[str] = None
    last_name_source: EntitySource = EntitySource.USER_STATED
    
    phone: Optional[str] = None
    phone_source: EntitySource = EntitySource.USER_STATED
    
    date_of_birth: Optional[str] = None
    date_of_birth_source: EntitySource = EntitySource.USER_STATED
    
    gender: Optional[str] = None
    gender_source: EntitySource = EntitySource.USER_STATED
    
    email: Optional[str] = None
    email_source: EntitySource = EntitySource.USER_STATED
    
    def is_complete_for_registration(self) -> bool:
        """Check if we have minimum info for registration."""
        return all([
            self.first_name,
            self.last_name,
            self.phone
        ])
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields."""
        missing = []
        if not self.first_name:
            missing.append("first_name")
        if not self.last_name:
            missing.append("last_name")
        if not self.phone:
            missing.append("phone")
        return missing


class PatientEntities(BaseModel):
    """
    Complete patient entities - SOURCE OF TRUTH for what patient wants.
    
    These entities represent the patient's expressed preferences and info.
    They are NEVER derived from tool results - only from patient input.
    """
    
    # Session tracking
    session_id: str
    
    # Appointment preferences
    appointment: AppointmentPreferences = Field(default_factory=AppointmentPreferences)
    
    # Personal information
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    
    # Intent tracking
    current_intent: Optional[str] = None  # "book_appointment", "cancel", "inquire"
    intent_confidence: float = 1.0
    
    # History of changes (for debugging/audit)
    change_history: List[PatientEntityChange] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def update_preference(
        self,
        field_path: str,
        value: Any,
        source: EntitySource = EntitySource.USER_STATED,
        confidence: float = 1.0
    ) -> Optional[PatientEntityChange]:
        """
        Update a preference and track the change.
        
        Args:
            field_path: Dot-notation path like "appointment.doctor_preference"
            value: New value
            source: How we got this value
            confidence: How confident we are
            
        Returns:
            PatientEntityChange if value changed, None if same
        """
        parts = field_path.split(".")
        
        # Navigate to parent object
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        field_name = parts[-1]
        old_value = getattr(obj, field_name, None)
        
        # Only record change if value actually changed
        if old_value != value:
            change = PatientEntityChange(
                field=field_path,
                old_value=old_value,
                new_value=value,
                source=source,
                confidence=confidence
            )
            self.change_history.append(change)
            setattr(obj, field_name, value)
            
            # Update source tracking if available
            source_field = f"{field_name}_source"
            if hasattr(obj, source_field):
                setattr(obj, source_field, source)
            
            # Update confidence if available
            confidence_field = f"{field_name}_confidence"
            if hasattr(obj, confidence_field):
                setattr(obj, confidence_field, confidence)
            
            self.last_updated_at = datetime.utcnow()
            return change
        
        return None
    
    def get_recent_changes(self, since: datetime = None) -> List[PatientEntityChange]:
        """Get changes since a given time."""
        if since is None:
            return self.change_history[-10:]  # Last 10 changes
        return [c for c in self.change_history if c.changed_at >= since]
    
    def get_changed_fields(self, since: datetime) -> List[str]:
        """Get list of fields that changed since a given time."""
        return [c.field for c in self.get_recent_changes(since)]
    
    def to_context_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for agent context."""
        return {
            "intent": self.current_intent,
            "doctor_preference": self.appointment.doctor_preference,
            "date_preference": self.appointment.date_preference,
            "time_preference": self.appointment.time_preference,
            "procedure_preference": self.appointment.procedure_preference,
            "patient_name": f"{self.personal_info.first_name or ''} {self.personal_info.last_name or ''}".strip() or None,
            "patient_phone": self.personal_info.phone
        }


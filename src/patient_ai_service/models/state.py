"""
State management models for the multi-agent system.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator

from .enums import ConversationStage, TriageLevel


class PatientProfile(BaseModel):
    """Patient profile information."""
    patient_id: Optional[str] = None
    user_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    preferred_language: Optional[str] = Field(default="en")
    allergies: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    medical_conditions: List[str] = Field(default_factory=list)
    insurance_provider: Optional[str] = None
    insurance_id: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None

    @field_validator("preferred_language", mode="before")
    @classmethod
    def _default_preferred_language(cls, value: Optional[str]) -> str:
        """Ensure preferred language is always a string for serialization."""
        if value is None or str(value).strip() == "":
            return "en"
        return str(value)

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "550e8400-e29b-41d4-a716-446655440000",
                "first_name": "John",
                "last_name": "Doe",
                "phone": "+971501234567",
                "preferred_language": "en"
            }
        }


class AppointmentContext(BaseModel):
    """Context for current appointment being booked/managed."""
    appointment_id: Optional[str] = None
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    procedure_types: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None


class GlobalState(BaseModel):
    """Global state shared across all agents."""
    session_id: str
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    current_appointment: Optional[AppointmentContext] = None
    conversation_stage: ConversationStage = ConversationStage.INITIAL
    active_agent: Optional[str] = None
    intent_history: List[str] = Field(default_factory=list)
    entities_collected: Dict[str, Any] = Field(default_factory=dict)
    conversation_summary: str = ""
    detected_language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_12345",
                "conversation_stage": "initial",
                "detected_language": "en"
            }
        }


class AppointmentAgentState(BaseModel):
    """Local state for Appointment Agent."""
    workflow_step: str = "initial"
    preferred_doctor: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    available_slots: List[Dict[str, Any]] = Field(default_factory=list)
    operation_type: Optional[str] = None  # "booking", "rescheduling", "canceling"
    target_appointment_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_step": "collecting_details",
                "operation_type": "booking"
            }
        }


class MedicalAgentState(BaseModel):
    """Local state for Medical Agent."""
    symptoms: List[str] = Field(default_factory=list)
    symptom_duration: Optional[str] = None
    pain_level: Optional[int] = Field(None, ge=0, le=10)
    triage_level: TriageLevel = TriageLevel.ROUTINE
    recommended_action: Optional[str] = None
    assessment_complete: bool = False
    inquiry_type: Optional[str] = None  # "preventive_tip", "side_effect", "general"

    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["tooth pain", "sensitivity"],
                "pain_level": 7,
                "triage_level": "urgent"
            }
        }


class EmergencyAgentState(BaseModel):
    """Local state for Emergency Agent."""
    emergency_type: Optional[str] = None
    severity: str = "unknown"
    location_confirmed: bool = False
    ambulance_needed: bool = False
    immediate_actions: List[str] = Field(default_factory=list)
    emergency_contacts_notified: bool = False
    first_aid_provided: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "emergency_type": "severe_bleeding",
                "severity": "critical",
                "ambulance_needed": True
            }
        }


class RegistrationState(BaseModel):
    """Local state for Registration Agent."""
    form_completion: float = 0.0  # 0.0 to 1.0
    missing_fields: List[str] = Field(default_factory=list)
    current_section: str = "personal_info"
    required_fields: List[str] = Field(default_factory=lambda: [
        "first_name",
        "last_name",
        "date_of_birth",
        "phone",
        "emergency_contact_name",
        "emergency_contact_phone"
    ])
    collected_fields: Dict[str, Any] = Field(default_factory=dict)
    registration_complete: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "form_completion": 0.6,
                "current_section": "medical_history",
                "missing_fields": ["allergies", "medications"]
            }
        }


class TranslationState(BaseModel):
    """Local state for Translation Agent."""
    source_language: str = "en"
    target_language: str = "en"
    translation_cache: Dict[str, str] = Field(default_factory=dict)
    auto_detect: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "source_language": "ar",
                "target_language": "en",
                "auto_detect": True
            }
        }

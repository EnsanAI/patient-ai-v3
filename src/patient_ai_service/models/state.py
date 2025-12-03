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


class LanguageContext(BaseModel):
    """
    Unified language context for session.

    This information MUST be passed to all agents in minimal_context.

    Tracks:
    - Current detected language + dialect
    - User's preferred language + dialect
    - Language switch history
    - Translation metadata
    """
    # Current conversation language - MUST BE PASSED TO AGENTS
    current_language: str = "en"  # ISO 639-1 code (e.g., "en", "ar", "es")
    current_dialect: Optional[str] = None  # Dialect code (e.g., "US", "GB", "EG", "SA")

    # User's preferred language (from profile or auto-detected)
    preferred_language: str = "en"
    preferred_dialect: Optional[str] = None

    # Language switch tracking
    language_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Example: [{"from_language": "en", "to_language": "ar", "timestamp": "...", "turn": 1}, ...]

    # Metadata
    auto_detect_enabled: bool = True
    last_detected_at: Optional[datetime] = None
    turn_count: int = 0

    # Translation quality tracking
    translation_failures: int = 0
    last_translation_error: Optional[str] = None

    def get_full_language_code(self) -> str:
        """Get language code with dialect (e.g., 'en-US', 'ar-EG')."""
        if self.current_dialect:
            return f"{self.current_language}-{self.current_dialect}"
        return self.current_language

    def record_language_switch(self, new_language: str, new_dialect: Optional[str], turn: int):
        """Record when user switches language."""
        if self.current_language != new_language or self.current_dialect != new_dialect:
            self.language_history.append({
                "from_language": self.current_language,
                "from_dialect": self.current_dialect,
                "to_language": new_language,
                "to_dialect": new_dialect,
                "timestamp": datetime.utcnow().isoformat(),
                "turn": turn
            })
        self.current_language = new_language
        self.current_dialect = new_dialect
        self.last_detected_at = datetime.utcnow()

    def to_agent_context(self) -> Dict[str, Any]:
        """
        Format language context for passing to agents.

        THIS IS WHAT GETS PASSED IN MINIMAL_CONTEXT.
        """
        return {
            "current_language": self.current_language,
            "current_dialect": self.current_dialect,
            "full_code": self.get_full_language_code(),
            "recently_switched": len(self.language_history) > 0
        }

    class Config:
        json_schema_extra = {
            "example": {
                "current_language": "ar",
                "current_dialect": "EG",
                "preferred_language": "ar",
                "auto_detect_enabled": True
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

    # [NEW] Unified language context (replaces detected_language)
    language_context: LanguageContext = Field(default_factory=LanguageContext)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 0

    # Backward compatibility property
    @property
    def detected_language(self) -> str:
        """
        Deprecated: Use language_context.current_language instead.

        Maintained for backward compatibility.
        """
        return self.language_context.current_language

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_12345",
                "conversation_stage": "initial",
                "language_context": {
                    "current_language": "en",
                    "current_dialect": "US"
                }
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
    reasoning_plan: Optional[str] = None  # Plan from reasoning engine for agent to follow

    # Lightweight booking state tracking (Phase 2.1)
    booking_pending: bool = False
    last_booking_attempt: Optional[Dict[str, Any]] = None  # {doctor_id, date, time, timestamp}
    last_verified_appointment_id: Optional[str] = None

    def mark_booking_pending(self, doctor_id: str, date: str, time: str):
        """Mark that booking was attempted but not yet verified."""
        self.booking_pending = True
        self.last_booking_attempt = {
            "doctor_id": doctor_id,
            "date": date,
            "time": time,
            "timestamp": datetime.now().isoformat()
        }

    def mark_booking_verified(self, appointment_id: str):
        """Mark that booking was verified in database."""
        self.booking_pending = False
        self.last_verified_appointment_id = appointment_id
        self.last_booking_attempt = None

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

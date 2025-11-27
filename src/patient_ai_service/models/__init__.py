"""
Pydantic models for the dental clinic management system.
"""

from .enums import (
    IntentType,
    UrgencyLevel,
    AppointmentType,
    ProcedureType,
    AppointmentStatus,
    ConversationStage,
    MessageType,
    TriageLevel,
    EmergencyType,
    Language,
)

from .state import (
    GlobalState,
    AppointmentAgentState,
    MedicalAgentState,
    EmergencyAgentState,
    RegistrationState,
    PatientProfile,
)

from .messages import (
    Message,
    ChatRequest,
    ChatResponse,
    IntentClassification,
    AppointmentClassification,
)

from .appointments import (
    Appointment,
    Doctor,
    AppointmentContext,
)

__all__ = [
    # Enums
    "IntentType",
    "UrgencyLevel",
    "AppointmentType",
    "ProcedureType",
    "AppointmentStatus",
    "ConversationStage",
    "MessageType",
    "TriageLevel",
    "EmergencyType",
    "Language",
    # State
    "GlobalState",
    "AppointmentAgentState",
    "MedicalAgentState",
    "EmergencyAgentState",
    "RegistrationState",
    "PatientProfile",
    # Messages
    "Message",
    "ChatRequest",
    "ChatResponse",
    "IntentClassification",
    "AppointmentClassification",
    # Appointments
    "Appointment",
    "Doctor",
    "AppointmentContext",
]

"""
Message and communication models.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator

from .enums import IntentType, UrgencyLevel, AppointmentType, ProcedureType, MessageType


class Message(BaseModel):
    """Internal message for pub/sub system."""
    id: str = Field(default_factory=lambda: f"msg_{datetime.utcnow().timestamp()}")
    topic: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    priority: int = 0  # 0=normal, 1=high, 2=critical

    @validator('priority')
    def validate_priority(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError('Priority must be 0, 1, or 2')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "user.input",
                "payload": {"text": "I need an appointment"},
                "session_id": "user_123",
                "priority": 0
            }
        }


class ChatRequest(BaseModel):
    """HTTP request for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(..., min_length=1, max_length=100)  # Allow phone numbers and other formats
    language: Optional[str] = None

    @validator('message')
    def sanitize_message(cls, v):
        """Basic sanitization."""
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I need to book an appointment",
                "session_id": "user_12345",
                "language": "en"
            }
        }


class ChatResponse(BaseModel):
    """HTTP response for chat endpoint."""
    response: str
    session_id: str
    detected_language: str = "en"
    intent: Optional[str] = None
    urgency: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "response": "I'd be happy to help you book an appointment...",
                "session_id": "user_12345",
                "detected_language": "en",
                "intent": "appointment_booking",
                "urgency": "low"
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "message", "response", "typing", "state_update", "error"
    session_id: str
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "type": "message",
                "session_id": "user_12345",
                "content": "I need help"
            }
        }


class IntentClassification(BaseModel):
    """Result of intent classification."""
    intent: IntentType
    urgency: UrgencyLevel
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "intent": "appointment_booking",
                "urgency": "medium",
                "entities": {"date": "tomorrow", "time": "morning"},
                "confidence": 0.95
            }
        }


class AppointmentClassification(BaseModel):
    """Result of appointment classification."""
    appointment_type: AppointmentType
    procedure_types: List[ProcedureType]
    urgency_level: UrgencyLevel
    estimated_duration: int = Field(..., ge=15, le=240)  # minutes
    requires_specialist: bool = False
    specialist_type: Optional[str] = None
    reasoning: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "appointment_type": "existing_patient",
                "procedure_types": ["root_canal", "examination"],
                "urgency_level": "urgent",
                "estimated_duration": 120,
                "requires_specialist": True,
                "specialist_type": "endodontist",
                "reasoning": "Patient reported severe tooth pain...",
                "confidence": 0.95
            }
        }


class Topics:
    """Message broker topics."""
    USER_INPUT = "user.input"
    INTENT_CLASSIFIED = "intent.classified"
    APPOINTMENT_REQUEST = "appointment.request"
    APPOINTMENT_RESPONSE = "appointment.response"
    MEDICAL_INQUIRY = "medical.inquiry"
    EMERGENCY_ALERT = "emergency.alert"
    REGISTRATION_EVENT = "registration.event"
    TRANSLATION_REQUEST = "translation.request"
    STATE_UPDATE = "state.update"
    ERROR = "system.error"
    RESPONSE_READY = "response.ready"
    GENERAL_ASSISTANT = "general.assistant"
    GREETING = "greeting"

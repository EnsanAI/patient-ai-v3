"""
Appointment-related models.
"""

from datetime import datetime, date, time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from .enums import AppointmentStatus, AppointmentType, ProcedureType, UrgencyLevel


class Doctor(BaseModel):
    """Doctor information."""
    doctor_id: str
    first_name: str
    last_name: str
    specialty: Optional[str] = None
    languages: List[str] = Field(default_factory=lambda: ["en"])
    available_days: List[str] = Field(default_factory=list)
    bio: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"Dr. {self.first_name} {self.last_name}"

    class Config:
        json_schema_extra = {
            "example": {
                "doctor_id": "doc_001",
                "first_name": "Ahmed",
                "last_name": "Khan",
                "specialty": "General Dentistry",
                "languages": ["en", "ar"]
            }
        }


class Appointment(BaseModel):
    """Appointment model."""
    appointment_id: str
    patient_id: str
    doctor_id: str
    clinic_id: str
    appointment_type: AppointmentType
    procedure_types: List[ProcedureType] = Field(default_factory=list)
    appointment_date: str  # ISO format YYYY-MM-DD
    start_time: str  # HH:MM format
    end_time: str  # HH:MM format
    duration: int  # minutes
    status: AppointmentStatus
    reason: Optional[str] = None
    notes: Optional[str] = None
    urgency_level: Optional[UrgencyLevel] = None
    classification_reasoning: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "appointment_id": "apt_001",
                "patient_id": "pat_001",
                "doctor_id": "doc_001",
                "clinic_id": "clinic_001",
                "appointment_type": "existing_patient",
                "procedure_types": ["cleaning", "examination"],
                "appointment_date": "2025-11-20",
                "start_time": "10:00",
                "end_time": "10:30",
                "duration": 30,
                "status": "confirmed"
            }
        }


class AppointmentSlot(BaseModel):
    """Available appointment slot."""
    doctor_id: str
    doctor_name: str
    date: str  # YYYY-MM-DD
    start_time: str  # HH:MM
    end_time: str  # HH:MM
    available: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "doctor_id": "doc_001",
                "doctor_name": "Dr. Ahmed Khan",
                "date": "2025-11-20",
                "start_time": "10:00",
                "end_time": "10:30",
                "available": True
            }
        }


class AppointmentContext(BaseModel):
    """Context for current appointment being managed."""
    appointment_id: Optional[str] = None
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    procedure_types: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "doctor_id": "doc_001",
                "doctor_name": "Dr. Ahmed Khan",
                "date": "2025-11-20",
                "time": "10:00",
                "procedure_types": ["cleaning"]
            }
        }

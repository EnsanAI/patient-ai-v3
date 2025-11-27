"""
Specialized agents for the dental clinic system.
"""

from .base_agent import BaseAgent
from .appointment_manager import AppointmentManagerAgent
from .medical_inquiry import MedicalInquiryAgent
from .emergency_response import EmergencyResponseAgent
from .registration import RegistrationAgent
from .translation import TranslationAgent
from .general_assistant import GeneralAssistantAgent

__all__ = [
    "BaseAgent",
    "AppointmentManagerAgent",
    "MedicalInquiryAgent",
    "EmergencyResponseAgent",
    "RegistrationAgent",
    "TranslationAgent",
    "GeneralAssistantAgent",
]

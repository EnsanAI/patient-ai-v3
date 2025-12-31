"""
Enumerations for the dental clinic management system.
"""

from enum import Enum


class IntentType(str, Enum):
    """User intent types."""
    APPOINTMENT_BOOKING = "appointment_booking"
    APPOINTMENT_RESCHEDULE = "appointment_reschedule"
    APPOINTMENT_CANCEL = "appointment_cancel"
    APPOINTMENT_CHECK = "appointment_check"
    FOLLOW_UP = "follow_up"
    MEDICAL_INQUIRY = "medical_inquiry"
    EMERGENCY = "emergency"
    REGISTRATION = "registration"
    GENERAL_INQUIRY = "general_inquiry"
    GREETING = "greeting"


class UrgencyLevel(str, Enum):
    """Urgency levels for requests."""
    LOW = "low"           # Routine
    MEDIUM = "medium"     # Soon
    HIGH = "high"         # Urgent
    CRITICAL = "critical" # Emergency


class AppointmentType(str, Enum):
    """Types of appointments."""
    NEW_PATIENT = "new_patient"
    EXISTING_PATIENT = "existing_patient"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    CONSULTATION = "consultation"
    ROUTINE_CHECKUP = "routine_checkup"


class ProcedureType(str, Enum):
    """Types of dental procedures."""
    CLEANING = "cleaning"
    EXAMINATION = "examination"
    XRAY = "xray"
    FILLING = "filling"
    ROOT_CANAL = "root_canal"
    EXTRACTION = "extraction"
    CROWN = "crown"
    BRIDGE = "bridge"
    IMPLANT = "implant"
    BRACES = "braces"
    ALIGNERS = "aligners"
    WHITENING = "whitening"
    VENEERS = "veneers"
    GUM_TREATMENT = "gum_treatment"
    ORAL_SURGERY = "oral_surgery"
    DENTURES = "dentures"
    OTHER = "other"


class AppointmentStatus(str, Enum):
    """Status of appointments."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"


class ConversationStage(str, Enum):
    """Stages in the conversation flow."""
    INITIAL = "initial"
    COLLECTING_DETAILS = "collecting_details"
    CHECKING_AVAILABILITY = "checking_availability"
    CONFIRMING = "confirming"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class MessageType(str, Enum):
    """Types of messages in the system."""
    TEXT = "text"
    ACTION = "action"
    SYSTEM = "system"
    ERROR = "error"


class TriageLevel(str, Enum):
    """Medical triage levels."""
    EMERGENCY = "emergency"    # Call 911, immediate
    URGENT = "urgent"          # Same-day appointment
    SOON = "soon"             # Next-day appointment
    ROUTINE = "routine"        # Regular scheduling


class EmergencyType(str, Enum):
    """Types of dental emergencies."""
    SEVERE_BLEEDING = "severe_bleeding"
    SEVERE_PAIN = "severe_pain"
    FACIAL_SWELLING = "facial_swelling"
    KNOCKED_OUT_TOOTH = "knocked_out_tooth"
    BROKEN_JAW = "broken_jaw"
    INFECTION = "infection"
    TRAUMA = "trauma"
    OTHER = "other"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ARABIC = "ar"
    HINDI = "hi"
    RUSSIAN = "ru"
    ITALIAN = "it"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class AnthropicModel(str, Enum):
    """Anthropic Claude models."""
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_HAIKU = "claude-3-5-haiku-20241022"


class OpenAIModel(str, Enum):
    """OpenAI GPT models."""
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"

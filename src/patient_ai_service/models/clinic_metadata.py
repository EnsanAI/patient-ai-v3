"""Clinic metadata model for AI reasoning context."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ClinicMetadata(BaseModel):
    """
    Clinic metadata for AI reasoning context.

    Fetched from db-ops /clinics/:id endpoint and cached in StateManager.
    Provides clinic name, location, timezone-based current datetime for LLM prompts.

    Attributes:
        clinic_id: UUID of the clinic
        name: Clinic name (e.g., "Al Dhait Branch")
        address: Full clinic address (optional)
        timezone: IANA timezone identifier (e.g., "Asia/Dubai"), defaults to "UTC"
        phone_number: Main clinic phone number (optional)
        email: Clinic email address (optional)
        opening_hours: Dict of business hours by day (optional)
        fetched_at: Timestamp when this data was fetched/cached
    """

    clinic_id: str
    name: str
    address: Optional[str] = None
    timezone: str = "UTC"
    phone_number: Optional[str] = None
    email: Optional[str] = None
    opening_hours: dict = Field(default_factory=dict)

    # Timestamp when this data was fetched/cached
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    def get_current_datetime_str(self) -> str:
        """
        Get current datetime in clinic's timezone, formatted for LLMs.

        Converts current time to clinic's timezone and formats in human-readable form.
        Falls back to UTC if timezone is invalid or not found.

        Returns:
            Formatted datetime string, e.g.:
            - "January 26, 2026 3:45 PM" (clinic timezone)
            - "January 26, 2026 3:45 PM (UTC)" (fallback)

        Examples:
            >>> metadata = ClinicMetadata(clinic_id="123", name="Test", timezone="Asia/Dubai")
            >>> metadata.get_current_datetime_str()
            'January 26, 2026 7:45 PM'  # GST time (UTC+4)
        """
        try:
            tz = ZoneInfo(self.timezone)
            now = datetime.now(tz)
            # Format: "January 26, 2026 3:45 PM"
            return now.strftime("%B %d, %Y %I:%M %p")
        except (ZoneInfoNotFoundError, Exception) as e:
            # Fallback to UTC if timezone is invalid
            now = datetime.now(ZoneInfo("UTC"))
            formatted = now.strftime("%B %d, %Y %I:%M %p")
            return f"{formatted} (UTC)"

    def to_prompt_context(self) -> str:
        """
        Format clinic metadata for inclusion in LLM prompts.

        Generates a multi-line string with clinic name, current datetime,
        and location (if available) suitable for CONTEXT sections.

        Returns:
            Multi-line formatted string for prompt inclusion

        Examples:
            >>> metadata = ClinicMetadata(
            ...     clinic_id="123",
            ...     name="Dubai Dental Clinic",
            ...     address="123 Sheikh Zayed Road, Dubai",
            ...     timezone="Asia/Dubai"
            ... )
            >>> print(metadata.to_prompt_context())
            Clinic Name: Dubai Dental Clinic
            Current Date & Time: January 26, 2026 7:45 PM
            Clinic Location: 123 Sheikh Zayed Road, Dubai
        """
        current_time = self.get_current_datetime_str()

        parts = [
            f"Clinic Name: {self.name}",
            f"Current Date & Time: {current_time}"
        ]

        if self.address:
            parts.append(f"Clinic Location: {self.address}")

        return "\n".join(parts)

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

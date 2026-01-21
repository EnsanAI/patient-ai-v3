"""
Clinic Context Model for Multi-Tenant Architecture.

This context is created at the API entry point and propagated through
the entire request lifecycle. It provides:
1. Clinic identification for all downstream operations
2. Composite session key for clinic-scoped state isolation
3. Headers for db-ops RLS enforcement
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ClinicContext:
    """
    Immutable context object containing clinic information.

    Created once at request entry (from wa-service or API),
    propagated through orchestrator to all components.

    Attributes:
        clinic_id: UUID of the clinic this request belongs to
        session_id: Original session identifier (typically phone number)
        composite_key: Generated key for clinic-scoped state storage
    """
    clinic_id: str
    session_id: str
    composite_key: str = field(init=False)

    def __post_init__(self):
        """Generate composite session key after initialization."""
        self.composite_key = f"clinic:{self.clinic_id}:session:{self.session_id}"

    def to_headers(self) -> Dict[str, str]:
        """
        Generate headers for db-ops requests.

        These headers are used by db-ops RLS middleware to set
        PostgreSQL session variables for row-level security.
        """
        return {
            "X-Clinic-Id": self.clinic_id,
            "X-Session-Id": self.session_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "clinic_id": self.clinic_id,
            "session_id": self.session_id,
            "composite_key": self.composite_key,
        }

    def __repr__(self) -> str:
        return f"ClinicContext(clinic_id={self.clinic_id!r}, session_id={self.session_id!r})"

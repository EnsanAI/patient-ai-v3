"""
Dental Clinic AI Service v2.0

A multi-agent AI system for dental clinic management featuring:
- Appointment booking and management
- Medical inquiry handling (no medical advice)
- Emergency response
- Patient registration
- Multi-language support (12 languages)
- State management
- Pub/sub architecture
"""

__version__ = "2.0.0"
__author__ = "Dental Clinic AI Team"

from .core.config import settings

__all__ = ["settings", "__version__"]

"""
Script Detector - Linguistic property-based language identification.

Uses Unicode ranges to instantly identify Arabic vs Latin script.
No LLM calls needed for obvious cases.
"""

import logging
from typing import Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DetectedScript(Enum):
    """Detected script type."""
    LATIN = "latin"
    ARABIC = "arabic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ScriptDetector:
    """
    Fast script detection using Unicode ranges.
    
    Design Philosophy:
    - Uses linguistic properties (Unicode ranges), not keyword matching
    - Instant detection for obvious cases
    - Falls back to LLM only when necessary
    """
    
    # Unicode ranges for Arabic script
    # Basic Arabic: U+0600 to U+06FF
    # Arabic Supplement: U+0750 to U+077F
    # Arabic Extended-A: U+08A0 to U+08FF
    # Arabic Presentation Forms-A: U+FB50 to U+FDFF
    # Arabic Presentation Forms-B: U+FE70 to U+FEFF
    
    ARABIC_RANGES = [
        (0x0600, 0x06FF),   # Basic Arabic
        (0x0750, 0x077F),   # Arabic Supplement
        (0x08A0, 0x08FF),   # Arabic Extended-A
        (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    ]
    
    # Short message threshold - only apply fast path to short messages
    SHORT_MESSAGE_THRESHOLD = 100
    
    @classmethod
    def _is_arabic_char(cls, char: str) -> bool:
        """Check if a character is Arabic script."""
        code_point = ord(char)
        return any(start <= code_point <= end for start, end in cls.ARABIC_RANGES)
    
    @classmethod
    def _is_latin_char(cls, char: str) -> bool:
        """Check if a character is Latin script (ASCII letters)."""
        return char.isascii() and char.isalpha()
    
    @classmethod
    def detect_script(cls, text: str) -> DetectedScript:
        """
        Detect the primary script of the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectedScript enum value
        """
        text = text.strip()
        
        if not text:
            return DetectedScript.UNKNOWN
        
        arabic_count = 0
        latin_count = 0
        
        for char in text:
            if cls._is_arabic_char(char):
                arabic_count += 1
            elif cls._is_latin_char(char):
                latin_count += 1
        
        total_letters = arabic_count + latin_count
        
        if total_letters == 0:
            # Only numbers, punctuation, emojis
            return DetectedScript.UNKNOWN
        
        arabic_ratio = arabic_count / total_letters
        latin_ratio = latin_count / total_letters
        
        # Thresholds for classification
        if arabic_ratio > 0.8:
            return DetectedScript.ARABIC
        elif latin_ratio > 0.8:
            return DetectedScript.LATIN
        elif arabic_ratio > 0.2 and latin_ratio > 0.2:
            return DetectedScript.MIXED  # Code-switching
        elif arabic_ratio > latin_ratio:
            return DetectedScript.ARABIC
        else:
            return DetectedScript.LATIN
    
    @classmethod
    def fast_detect(cls, text: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Fast language detection for short, obvious messages.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language_code, dialect_code, should_skip_llm)
            - If should_skip_llm is True, use the returned codes directly
            - If should_skip_llm is False, fall back to LLM detection
        """
        text = text.strip()
        
        # Only apply fast path to short messages
        if len(text) > cls.SHORT_MESSAGE_THRESHOLD:
            logger.debug(f"[ScriptDetector] Message too long ({len(text)} chars), using LLM")
            return None, None, False
        
        script = cls.detect_script(text)
        
        if script == DetectedScript.LATIN:
            # In dental clinic context (UAE), Latin script = English
            logger.info(f"[ScriptDetector] ⚡ Fast detect: LATIN script → en (no LLM needed)")
            return "en", None, True
        
        elif script == DetectedScript.ARABIC:
            # Arabic detected - default to Emirati dialect (ae)
            # Skip dialect detection LLM call
            logger.info(f"[ScriptDetector] ⚡ Fast detect: ARABIC script → ar-ae (no LLM needed)")
            return "ar", "ae", True
        
        elif script == DetectedScript.MIXED:
            # Code-switching detected - need LLM for proper handling
            logger.info(f"[ScriptDetector] Mixed script detected, using LLM for dialect detection")
            return None, None, False
        
        else:
            # Unknown script - let LLM figure it out
            logger.debug(f"[ScriptDetector] Unknown script, using LLM")
            return None, None, False


# Convenience function
def fast_detect_language(text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Convenience function for fast language detection.
    
    Returns:
        Tuple of (language_code, dialect_code, should_skip_llm)
    """
    return ScriptDetector.fast_detect(text)


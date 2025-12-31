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
    def _is_numbers_only(cls, text: str) -> bool:
        """
        Check if text contains only digits, spaces, and punctuation.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text contains only numbers, spaces, and punctuation (no letters)
        """
        if not text:
            return False
        
        for char in text:
            # If we find any letter (Latin or Arabic), it's not numbers-only
            if cls._is_latin_char(char) or cls._is_arabic_char(char):
                return False
        
        # Check if there's at least one digit
        has_digit = any(char.isdigit() for char in text)
        return has_digit
    
    @classmethod
    def _has_latin_letters(cls, text: str) -> bool:
        """
        Check if text contains any Latin/ASCII letters.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text contains at least one Latin letter
        """
        return any(cls._is_latin_char(char) for char in text)
    
    @classmethod
    def _has_arabic_letters(cls, text: str) -> bool:
        """
        Check if text contains any Arabic script characters.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text contains at least one Arabic letter
        """
        return any(cls._is_arabic_char(char) for char in text)
    
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
        
        Simplified logic:
        1. Numbers only → Return None (keep previous language)
        2. Arabic letters present → Return ar-ae
        3. Latin letters present → Check Franco-Arabic, if not Franco-Arabic return en
        4. No letters → Return None (keep previous language)
        
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
        
        # 1. Check if only numbers/punctuation
        if cls._is_numbers_only(text):
            logger.info(f"[ScriptDetector] Numbers only → null (keeping previous language)")
            return None, None, False
        
        # 2. Check for Arabic letters
        if cls._has_arabic_letters(text):
            logger.info(f"[ScriptDetector] ⚡ Fast detect: ARABIC script → ar-ae")
            return "ar", "ae", True
        
        # 3. Check for Latin letters
        if cls._has_latin_letters(text):
            # Check for Franco-Arabic using LanguageDetector logic
            try:
                from patient_ai_service.agents.translation import LanguageDetector
                detector = LanguageDetector()
                is_non_english, detected_lang = detector.is_non_english(text)
                
                if detected_lang == 'ar-franco':
                    # Franco-Arabic detected - let LLM handle it
                    logger.info(f"[ScriptDetector] Franco-Arabic detected → null (using LLM)")
                    return None, None, False
                else:
                    # Not Franco-Arabic - it's English
                    logger.info(f"[ScriptDetector] ⚡ Fast detect: LATIN script → en")
                    return "en", None, True
            except Exception as e:
                # If LanguageDetector fails, default to English for Latin script
                logger.warning(f"[ScriptDetector] Error checking Franco-Arabic: {e}, defaulting to English")
                logger.info(f"[ScriptDetector] ⚡ Fast detect: LATIN script → en")
                return "en", None, True
        
        # 4. No letters found (only punctuation/emojis)
        logger.info(f"[ScriptDetector] No letters → null (keeping previous language)")
        return None, None, False


# Convenience function
def fast_detect_language(text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Convenience function for fast language detection.
    
    Returns:
        Tuple of (language_code, dialect_code, should_skip_llm)
    """
    return ScriptDetector.fast_detect(text)


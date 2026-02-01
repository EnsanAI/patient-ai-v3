"""
JSON parsing utilities for handling LLM responses.

Provides robust JSON extraction and parsing that handles:
- Markdown code blocks (```json ... ```)
- Multi-line strings with emojis
- Trailing commas
- Nested objects
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_and_parse_json(response: str, context: str = "response") -> dict:
    """
    Extract and parse JSON from LLM response with robust error handling.
    
    Args:
        response: The raw LLM response text
        context: Description of what's being parsed (for logging)
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If no valid JSON could be extracted
        json.JSONDecodeError: If JSON is malformed and can't be fixed
    """
    # Strategy 1: Extract from markdown code block (```json ... ```)
    json_str = None
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        logger.debug(f"[{context}] Found JSON in markdown code block")
    
    # Strategy 2: Find balanced braces (handles nested objects properly)
    if not json_str:
        # Find first opening brace
        start = response.find('{')
        if start != -1:
            # Count braces to find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(response[start:], start=start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if brace_count == 0:
                json_str = response[start:end]
                logger.debug(f"[{context}] Found JSON with balanced braces")
    
    if not json_str:
        logger.error(f"[{context}] No JSON found in response")
        raise ValueError(f"No JSON found in {context}")
    
    logger.debug(f"[{context}] Extracted JSON ({len(json_str)} chars)")
    
    # Parse JSON with error recovery
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        logger.warning(f"[{context}] Initial parse failed, attempting fixes...")
        
        # Remove trailing commas (common LLM mistake)
        fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Try parsing again
        try:
            data = json.loads(fixed_json)
            logger.info(f"[{context}] JSON parsed after fixing trailing commas")
            return data
        except json.JSONDecodeError:
            # If still fails, re-raise original error with context
            logger.error(f"[{context}] JSON parsing failed: {e}")
            logger.error(f"[{context}] Extracted JSON:\n{json_str}")
            raise


def safe_json_loads(json_str: str, context: str = "JSON") -> dict:
    """
    Safely load JSON string with error recovery.
    
    This is a lighter version that assumes json_str is already extracted
    and just needs parsing with error recovery.
    
    Args:
        json_str: JSON string to parse
        context: Description for logging
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If JSON is malformed and can't be fixed
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix trailing commas
        logger.warning(f"[{context}] Parse error, attempting fix...")
        fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        try:
            data = json.loads(fixed_json)
            logger.info(f"[{context}] Parsed after fixing trailing commas")
            return data
        except json.JSONDecodeError:
            logger.error(f"[{context}] Parse failed: {e}")
            raise

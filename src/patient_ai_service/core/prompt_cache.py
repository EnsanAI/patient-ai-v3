"""
Prompt caching support for Carebot agents.

This module provides:
1. Universal static content shared across ALL agents (cacheable)
2. Configuration for cache behavior
3. Utilities for building cached prompts

Cache Architecture:
- Layer 1 (Cached): Universal guides, decision rules, output format (~1,100 tokens)
- Layer 2 (Not Cached): Agent identity, instructions, tools (~500-800 tokens)  
- Layer 3 (Not Cached): Dynamic context - message, entities, observations
"""

import logging
from typing import Optional
from dataclasses import dataclass

from patient_ai_service.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for prompt caching."""
    enabled: bool = True
    min_tokens_for_caching: int = 1024  # Anthropic minimum
    ttl_seconds: int = 300  # 5 minute TTL (Anthropic default)


# Global cache configuration - can be overridden via settings
_cache_config = CacheConfig(
    enabled=getattr(settings, 'prompt_cache_enabled', True)
)


def get_cache_config() -> CacheConfig:
    """Get current cache configuration."""
    return _cache_config


def set_cache_enabled(enabled: bool):
    """Enable or disable caching globally."""
    _cache_config.enabled = enabled
    logger.info(f"Prompt caching {'enabled' if enabled else 'disabled'}")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL STATIC CONTENT (LAYER 1 - CACHED)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# CRITICAL: This content MUST be identical across all agents.
# Any agent-specific content here will break caching.
# ═══════════════════════════════════════════════════════════════════════════════

UNIVERSAL_RESULT_TYPE_GUIDE = """
TOOL RESULT TYPES & REQUIRED ACTIONS

| result_type   | Meaning                  | Action                                      |
|---------------|--------------------------|---------------------------------------------|
| SUCCESS       | Goal achieved            | Mark criterion COMPLETE, continue           |
| PARTIAL       | Progress made            | Continue to next step                       |
| USER_INPUT_NEEDED    | User decision required   | STOP → RESPOND_WITH_OPTIONS (present choices) IF AVAILABLE OTHERWISE GENERATE OPTIONS|
| RECOVERABLE   | Try different approach   | Follow recovery_action field                |
| FATAL         | Cannot complete          | RESPOND_IMPOSSIBLE                          |
| SYSTEM_ERROR  | Infrastructure failure   | RETRY once, then RESPOND_IMPOSSIBLE         |

⚠️ USER_INPUT_NEEDED: Not a failure. Tool worked but user must choose. DO NOT retry other tools.

"""

UNIVERSAL_DECISION_GUIDE = """
DECISION TYPES

CALL_TOOL → Have required params, no blockers
RESPOND_COMPLETE → ALL tasks completed with evidence (is_task_complete=true)
RESPOND_WITH_OPTIONS → result_type=USER_INPUT_NEEDED, present alternatives
RESPOND_IMPOSSIBLE → result_type=FATAL, explain + suggest alternatives
CLARIFY → Missing or unclear required info (specify: clarification_question, awaiting_info, entities)
COLLECT_INFORMATION → Need user data before continuing (specify: awaiting_info, entities)
REQUEST_CONFIRMATION → BEFORE critical actions. Stages tool call (confirmation_action=tool_name, confirmation_details=tool_input) + confirmation_question for user
RETRY → result_type=SYSTEM_ERROR, within retry limit
ENTITIES:

Your short-term memory of:
- user preferences (e.g., preferred doctor, times)
- provided information needed to execute tool
- Tool outputs that update user data (e.g., doctor_id, appointment_id, available_times, etc)

How to update COLLECTED ENTITIES:
- Only output NEW or CHANGED entities in entities_to_update
- For CHANGED entities: use same key, new value
    // for example, to update preferred_time from "morning" to "afternoon":
    INPUT: ENTITIES: "preferred_time": "morning"
    OUTPUT: "entities_to_update": {"preferred_time": "afternoon"} NOTE: must use same key
- For NEW entities: use new key, value
- Nothing new → return empty: {}

FALLBACK: Complete state format (optional, if you find that that ENTITIES non-relevant to IDENTIFIED INTENT)
    "entities": {{}},

"""


UNIVERSAL_OUTPUT_FORMAT = """
OUTPUT JSON SCHEMA

{
        
    "decision": "<CALL_TOOL|RESPOND_COMPLETE|RESPOND_WITH_OPTIONS|RESPOND_IMPOSSIBLE|CLARIFY|COLLECT_INFORMATION|REQUEST_CONFIRMATION|RETRY>",
    
    
}
═══════════════════════════════════════════════════════════════
If CALL_TOOL - specify tool to call
═══════════════════════════════════════════════════════════════
    "tool_name": "name of tool",
    "tool_input": {{}},

// ═══════════════════════════════════════════════════════════════
    // If NOT CALL_TOOL, GENERATE RESPONSE DATA - Fill based on decision type
    // ═══════════════════════════════════════════════════════════════
    "response": {{

        "entities_to_update": {{}} // Only new or changed entities here

        CLARIFY:
            "clarification_needed": "<what's unclear>",
            "clarification_question": "<specific question>",    

        IF COLLECT_INFORMATION:
        // When you need to collect information from the user, you need to specify all the information needed in the information_needed field. These will be used to call tools.
        // Example: "information_needed": "first_name, last_name, date_of_birth, symptoms, etc."
            "response": {
            "information_needed": "",
            }

        IF RESPOND_WITH_OPTIONS:

            "options": [],
            "options_context": "<available_times|doctors|dates>",


        IF RESPOND_IMPOSSIBLE:

            "failure_reason": "<why>",
            "failure_suggestion": "<alternative>"


        IF RESPOND_COMPLETE:
            "completion_summary": "<what was done>",
            "completion_details": {
                "appointment_id": "",
                "doctor": "",
                "date": "",
                "time": ""
            }

        IF REQUEST_CONFIRMATION (staged tool call awaiting user approval):
            "confirmation_action": "<tool_name to execute after confirmation>",
            "confirmation_details": {<tool_inputs to execute after confirmation>},
            "confirmation_question": "<human-readable summary for user>"
    }

    // ═══════════════════════════════════════════════════════════════
    // SUGGESTED_RESPONSE (REQUIRED for all decisions)
    // ═══════════════════════════════════════════════════════════════
    // 1- Generate a real, informed user-facing response. 
    // 2- Include all details needed (names, dates, times).
    // 3- Keep it simple and concise.
    // 4- Keep it conversational and natural.
    // 5- Do not include any techincal details or IDs.
    // 6- Write in English.

    "suggested_response": "<complete user-facing message in English>"
}
"""


UNIVERSAL_DECISION_RULES = """
═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════════════════

DECISION RULES:
1. If reasoning engine provided clear guidance → EXECUTE IT (call appropriate tools)
2. If all criteria are ✅ → RESPOND_COMPLETE with is_task_complete=true
3. If last result has alternatives and available=false → RESPOND_WITH_OPTIONS
4. If task impossible → RESPOND_IMPOSSIBLE
5. Only CLARIFY if execution reveals missing CRITICAL data not in context
6. Use COLLECT_INFORMATION when you need to exit agentic loop to gather information from user
"""


def get_universal_system_content() -> str:
    """
    Get the universal static content for prompt caching.
    
    This content is IDENTICAL across all agents and is the cacheable portion.
    Approximately ~1,100 tokens.
    
    Returns:
        Static system prompt content suitable for caching
    """
    return f"""You are a thinking module for an AI agent system.

Your job is to:
1. Analyze the current situation
2. Decide the best next action
3. Know when to stop

{UNIVERSAL_RESULT_TYPE_GUIDE}

{UNIVERSAL_DECISION_GUIDE}

{UNIVERSAL_OUTPUT_FORMAT}

Always respond with valid JSON in the specified format."""


def estimate_universal_tokens() -> int:
    """Estimate token count of universal content (rough: 4 chars per token)."""
    content = get_universal_system_content()
    return len(content) // 4


# Verify at module load that universal content meets minimum requirements
_estimated_tokens = estimate_universal_tokens()
if _estimated_tokens < 1024:
    logger.warning(
        f"Universal content (~{_estimated_tokens} tokens) may be below "
        f"Anthropic's 1024 token minimum for caching"
    )
else:
    logger.debug(f"Universal content estimated at ~{_estimated_tokens} tokens")



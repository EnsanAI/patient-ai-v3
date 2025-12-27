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
"""


UNIVERSAL_OUTPUT_FORMAT = """
OUTPUT JSON SCHEMA

{
    
    "plan_status": {
        "completed_tasks": [],
        "pending_tasks": [],
        "blocked_tasks": []
    },
    
    "decision": "<CALL_TOOL|RESPOND_COMPLETE|RESPOND_WITH_OPTIONS|RESPOND_IMPOSSIBLE|CLARIFY|COLLECT_INFORMATION|REQUEST_CONFIRMATION|RETRY>",
    "reasoning": "<decision rationale>",
    "is_task_complete": <boolean>,
    
    
    // Include fields ONLY for your decision type:
    // See DECISION_RESPONSE_FIELDS below
    
}
═══════════════════════════════════════════════════════════════
If CALL_TOOL - specify tool to call
 ═══════════════════════════════════════════════════════════════
    "tool_name": "name of tool",
    "tool_input": {{}},

// ═══════════════════════════════════════════════════════════════
    // RESPONSE DATA - Fill based on decision type
    // ═══════════════════════════════════════════════════════════════
    "response": {{
        // ✅ DELTA UPDATE FORMAT (preferred): Only output entities that are NEW or CHANGED
        // - Do NOT output entities that haven't changed
        // - Only include what's different from the previous turn
        // - If nothing changed, leave entities_to_update empty: {{}}
        //
        // Examples:
        // - User says "3pm" → {{"time_preference": "3pm"}}
        // - User changes doctor → {{"doctor_preference": "Dr. Jones"}}
        // - Nothing new → {{}}
        "entities_to_update": {{
            // Only new or changed entities here
            // Examples:
               "time_preference": "3pm"  
               "doctor_preference": "Dr. Jones"  
        }},
        
        // FALLBACK: Complete state format (optional, if you find that that resolved_enities are too old or not useful at all)
        // System will prefer entities_to_update if present
        "entities": {{}},

        CLARIFY:
            "clarification_needed": "<what's unclear>",
            "clarification_question": "<specific question>",    

        COLLECT_INFORMATION:
            "response": {
            "information_needed": "<all pieces needed>",
            "awaiting_info": "<what you need>",
            }

        RESPOND_WITH_OPTIONS:

            "options": [],
            "options_context": "<available_times|doctors|dates>",


        RESPOND_IMPOSSIBLE:

            "failure_reason": "<why>",
            "failure_suggestion": "<alternative>"


        RESPOND_COMPLETE:
            "completion_summary": "<what was done>",
            "completion_details": {
                "appointment_id": "",
                "doctor": "",
                "date": "",
                "time": ""
            }

        REQUEST_CONFIRMATION (staged tool call awaiting user approval):
            "confirmation_action": "<tool_name to execute after confirmation>",
            "confirmation_details": {<tool_input to execute after confirmation>},
            "confirmation_question": "<human-readable summary for user>"
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


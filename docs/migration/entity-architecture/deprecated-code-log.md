# Deprecated Code Log - Entity Migration

This document tracks all code that is deprecated/removed during the entity architecture migration.

## Purpose

Track what code is removed so we can:
- Verify nothing critical is lost
- Understand migration impact
- Plan cleanup in Stage 11
- Document for future reference

## Format

For each deprecated item:
- **File:** Path to file
- **Code:** Specific code removed
- **Stage:** When it was removed
- **Reason:** Why it was removed
- **Replacement:** What replaces it

---

## Deprecated Items

### Stage 11: Final Cutover

#### ContinuationContext.entities field
- **File:** `src/patient_ai_service/core/state_manager.py`
- **Code:** `entities: Dict[str, Any]` field in ContinuationContext
- **Stage:** 11
- **Reason:** Replaced by ConversationEntitiesManager
- **Replacement:** `ConversationEntitiesManager` with FIFO eviction
- **Status:** Pending removal

#### Old entity merging logic
- **File:** `src/patient_ai_service/agents/base_agent.py`
- **Code:** `updated_resolved.update(response_data.entities)` pattern
- **Stage:** 11
- **Reason:** Replaced by delta-based merging
- **Replacement:** `merge_entity_delta()` function
- **Status:** Pending removal

#### Old entity extraction
- **File:** `src/patient_ai_service/core/orchestrator.py`
- **Code:** Direct dict merging from continuation_context.entities
- **Stage:** 11
- **Reason:** Replaced by scoped entity loading
- **Replacement:** `get_agent_entity_context()` helper
- **Status:** Pending removal

#### Fallback parsing logic
- **File:** `src/patient_ai_service/agents/base_agent.py`
- **Code:** Old format parsing fallback
- **Stage:** 11
- **Reason:** LLM should follow delta format after validation
- **Replacement:** Delta format only
- **Status:** Pending removal

#### Feature flags
- **File:** `src/patient_ai_service/core/config.py`
- **Code:** `USE_DELTA_ENTITIES`, `USE_AGENT_SCOPED_DERIVED` flags
- **Stage:** 11
- **Reason:** Migration complete, no longer needed
- **Replacement:** New system is default
- **Status:** Pending removal

---

## Notes

- This log will be updated as code is removed
- Final cleanup happens in Stage 11
- All removals are verified with tests before deletion


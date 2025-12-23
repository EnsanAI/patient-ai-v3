# Entity Migration Rollback Procedures

This document outlines rollback procedures for the entity architecture migration.

## Quick Rollback (Git Tag)

If you need to revert to the migration start point:

```bash
# Revert to migration-start tag
git reset --hard migration-start

# Or create a new branch from the tag
git checkout -b rollback-$(date +%Y%m%d) migration-start
```

## Feature Flag Rollback

If issues are detected during migration, disable feature flags:

```bash
# Disable via environment variables
export USE_DELTA_ENTITIES=false
export USE_AGENT_SCOPED_DERIVED=false

# Or in .env file
USE_DELTA_ENTITIES=false
USE_AGENT_SCOPED_DERIVED=false
```

This will revert to the old entity handling system without code changes.

## Stage-Specific Rollback

### Stage 1-2: Foundation
- **Rollback:** Revert model files
- **Impact:** None (not integrated yet)
- **Command:** `git checkout migration-start -- src/patient_ai_service/models/conversation_entities.py src/patient_ai_service/models/agent_scoped_derived.py src/patient_ai_service/models/entity_delta.py`

### Stage 3: LLM Testing
- **Rollback:** No rollback needed (isolated testing)
- **Impact:** None (doesn't affect production)

### Stage 4: State Manager Integration
- **Rollback:** Remove new methods from StateManager
- **Impact:** Low (old system still works)
- **Command:** Revert `src/patient_ai_service/core/state_manager.py` changes

### Stage 5-6: LLM Prompts & Parser
- **Rollback:** Revert prompt and parser changes
- **Impact:** Medium (affects agent responses)
- **Command:** Revert `src/patient_ai_service/core/prompt_cache.py` and `src/patient_ai_service/agents/base_agent.py` changes

### Stage 7-8: Context & Storage
- **Rollback:** Revert context building and storage changes
- **Impact:** High (affects agent execution)
- **Command:** Revert orchestrator and base_agent changes

### Stage 9: Continuation Migration
- **Rollback:** Restore from backup, revert code
- **Impact:** Critical (affects conversation continuity)
- **Procedure:**
  1. Restore ContinuationContext backup
  2. Revert `state_manager.py` changes
  3. Verify entities restored correctly

### Stage 10: Validation
- **Rollback:** Disable feature flags
- **Impact:** Low (parallel operation)
- **Command:** Set `USE_DELTA_ENTITIES=false`

### Stage 11: Cutover
- **Rollback:** Revert to Stage 10, re-enable feature flags
- **Impact:** Critical (production system)
- **Procedure:**
  1. Revert final cutover commit
  2. Re-enable feature flags
  3. Restore old system code paths

## Data Backup & Restoration

### Before Migration (Stage 9)
```bash
# Backup all ContinuationContext data
# Implementation depends on backend (Redis/InMemory)
# For Redis:
redis-cli --scan --pattern "session:*:agentic_state" | xargs redis-cli MGET > continuation_backup.json

# For InMemory: Export state before migration
```

### Restore from Backup
```bash
# Restore ContinuationContext data
# For Redis:
cat continuation_backup.json | redis-cli --pipe

# Verify restoration
# Check entity counts match
```

## Emergency Rollback Checklist

- [ ] Identify issue severity (low/medium/high/critical)
- [ ] Determine rollback scope (feature flag vs code revert)
- [ ] Backup current state (if not already done)
- [ ] Execute rollback procedure
- [ ] Verify system functionality
- [ ] Document issue and rollback reason
- [ ] Plan fix for next attempt

## Rollback Decision Tree

```
Issue Detected
    │
    ├─> Feature flag enabled?
    │   ├─> YES → Disable flags (fastest)
    │   └─> NO → Code revert needed
    │
    ├─> Data loss?
    │   ├─> YES → Restore from backup
    │   └─> NO → Code revert sufficient
    │
    └─> Production impact?
        ├─> Critical → Full rollback + restore
        ├─> High → Code revert
        └─> Low → Feature flag disable
```

## Post-Rollback Actions

1. **Document the issue:**
   - What went wrong?
   - Why did rollback occur?
   - What stage were we at?

2. **Analyze root cause:**
   - Review logs
   - Check metrics
   - Identify failure point

3. **Plan fix:**
   - Update migration plan if needed
   - Add additional safeguards
   - Re-test before retry

4. **Update rollback procedures:**
   - Add new scenarios if discovered
   - Improve documentation

## Contact & Support

For critical issues during migration:
- Check logs: `logs/migration/`
- Review metrics dashboard
- Consult ENTITY_MIGRATION_PLAN.md for stage-specific procedures


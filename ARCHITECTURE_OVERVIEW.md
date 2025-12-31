# Patient AI Service v3 - Architecture Overview

## System Flow

```
User Message
    ↓
[Translation] (Input) - Detect language & translate to English
    ↓
[Unified Reasoning] - Single LLM call for classification, routing, plan decisions
    ↓
    ├─ Fast Path → Conversational Response (greetings, farewells, unclear)
    └─ Standard Path → Agent Execution (appointments, registration, etc.)
        ↓
    [Agentic Loop] - Think → Act → Observe → Repeat
        ├─ No-Brainer: Auto-execute (confirmations, info collection)
        └─ Deliberative: Full thinking process
    ↓
[Response Generation] - Humanized, context-aware
    ↓
[Translation] (Output) - DISABLED (agents generate in target language)
    ↓
User Response
```

## Core Components

### Unified Reasoning
**File:** [`src/patient_ai_service/core/unified_reasoning.py`](src/patient_ai_service/core/unified_reasoning.py)

Single-pass LLM call that replaces SituationAssessor + ReasoningEngine. Performs message classification, routing decisions, agent selection, plan lifecycle management, and routing actions. Uses Claude Haiku 4.5 with Extended Thinking (1024 token budget).

### Fast Path / Standard Path

**Fast Path** (`route_type="fast_path"`): Handles greetings, farewells, thanks, pleasantries, and unclear requests. Bypasses agent loop entirely with single LLM call. Generates response in target language (no translation needed).

**Standard Path** (`route_type="agent"`): Routes to specialized agents for full agentic loop execution with tool calling, result handling, and completion verification.

### Agent Framework
**File:** [`src/patient_ai_service/agents/base_agent.py`](src/patient_ai_service/agents/base_agent.py)

ReAct pattern implementation: THINK → ACT → OBSERVE → REPEAT until task complete. Agent types: `appointment_manager` (booking/rescheduling), `registration` (patient registration), `emergency_response` (urgent medical), `general_assistant` (clinic info), `medical_inquiry` (log questions), `translation` (language detection).

### Action Router

Classification system embedded in Unified Reasoning output. **No-Brainer Actions** (auto-execute): `execute_confirmed_action` (user confirms → execute immediately), `collect_information` (lightweight multi-turn info collection). **Deliberative Actions**: Full agentic loop with `_think()` for complex operations requiring planning and tool orchestration.

## Action Types

### No-Brainers (Auto-Execute)

**Confirmation Flow:** User confirms with "yes/yeah/sure" → `routing_action="execute_confirmed_action"` → pending tool executed immediately before agentic loop.

**Information Collection:** Agent needs user data → `routing_action="collect_information"` → lightweight LLM call generates follow-up question until all info collected.

### Deliberative (Thinking Required)

**Standard Agentic Loop:** `_think()` analyzes situation and decides action, tool execution with result validation, completion criteria checking, response generation when all tasks complete.

**Enhanced Tool Usage:** Multi-step planning (e.g., book multiple appointments), result-aware execution (reads tool outputs before deciding), dynamic tool selection (LLM decides tools, not hardcoded).

## LLM Configuration

**File:** [`config/llm_config.yaml`](config/llm_config.yaml)

**Hierarchical Configuration:** Global defaults → Agent-level overrides → Function-level overrides.

**Primary Model: Claude Haiku 4.5** - Used for unified_reasoning, appointment_manager, registration, general_assistant. Extended Thinking enabled for reasoning functions (1024 token budget). Temperature: 0.1-1.0 depending on function.

**Model Selection Logic:**
- **Unified Reasoning**: Claude Haiku 4.5 (fast, cheap, extended thinking)
- **Agent Thinking**: Claude Haiku 4.5 or Claude Sonnet 4.5 (depending on complexity)
- **Response Generation**: GPT-4o-mini (natural language, lower cost)
- **Translation**: GPT-4.1-mini (OpenAI, optimized for translation)

**Fallback Mechanisms:** Unified Reasoning errors → fallback to general_assistant. Tool execution failures → retry with circuit breaker. LLM API failures → error response with graceful degradation.

## Active vs Inactive

### Currently Used

**Active Components:**
- ✅ Unified Reasoning (replaces SituationAssessor + ReasoningEngine)
- ✅ Fast Path routing (conversational responses)
- ✅ Agent Framework (ReAct pattern)
- ✅ Prompt Caching (Anthropic, Layer 1 universal content)
- ✅ Extended Thinking (Anthropic models, 1024 token budget)
- ✅ Information Collection non-brainer flow
- ✅ Confirmation non-brainer flow
- ✅ Native Language Memory (original language preservation)
- ✅ Conversation Memory (English, for reasoning)

**Partially Active:**
- ⚠️ ReasoningEngine: Still used for `validate_response` and `finalize_response` functions (legacy)
- ⚠️ Translation Output: DISABLED (agents generate in target language directly)

### Disabled/Legacy

**Deprecated Components:**
- ❌ [INACTIVE] SituationAssessor: Replaced by Unified Reasoning
- ❌ [INACTIVE] Intent Router: Replaced by Unified Reasoning routing
- ❌ [INACTIVE] Validation Layer: Can be disabled via config
- ❌ [INACTIVE] Finalization Layer: Can be disabled via config

**Legacy Code (Backward Compatibility):**
- `process_message_legacy()` in BaseAgent (kept for compatibility)
- Legacy entity fields in agentic models (deprecated, use response.entities)
- Old LLM config env vars (deprecated, use llm_config.yaml)

## Key Mechanisms

### Extended Thinking
**Anthropic-specific feature** for deeper reasoning. Enabled for unified_reasoning.reason and appointment_manager._think. Budget: 1024 tokens (configurable). Forces temperature=1.0 (API requirement). Provides internal reasoning before response.

### Prompt Caching
**Anthropic optimization** to reduce token costs. Layer 1 (Cached): Universal guides, decision rules (~1,100 tokens). Layer 2 (Not Cached): Agent identity, tools (~500-800 tokens). Layer 3 (Not Cached): Dynamic context (message, entities, observations). Minimum 1024 tokens required for caching.

### Plan Lifecycle Management
**Planning System** (A/B testable via feature flags). `no_plan`: Fast path or general assistant (no plan needed). `create_new`: New intent, no existing plan. `resume`: Continue existing plan unchanged. `abandon_create`: New intent with existing plan to abandon. `complete`: Plan is complete, clear it. Planning can be disabled per session via feature flags (removes plan sections from prompts).

### State Management
**Multi-layered state:** Global State (patient profile, language context, active agent). Agentic State (continuation context, pending actions, information collection). Plan State (agent plans with tasks, entities, completion status). Memory (Conversation memory in English + Native language memory in original).

## Architecture Decisions

**Why Single-Pass Reasoning?** Reduces latency (~300ms saved), reduces cost (~25% reduction), maintains accuracy with Claude Haiku + Extended Thinking.

**Why Fast Path?** Simple conversational exchanges don't need agent overhead. Faster response times for greetings/farewells. Lower cost for high-frequency messages.

**Why No-Brainer Actions?** User confirmations are deterministic → no thinking needed. Information collection is lightweight → avoid full agent loop. Reduces latency and cost for common flows.

**Why Disable Translation Output?** Agents and fast path generate responses in target language. Eliminates translation step (latency + cost savings). Better context preservation (no translation artifacts).


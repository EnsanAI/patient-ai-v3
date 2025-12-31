# v3 Architecture and Changes

## Overview

This document describes the v3 codebase architecture, key features, and implementation details for alignment purposes.

### Codebase Structure

**Total Python Files:** 35 (excluding `__pycache__` and `.cursor` reference files)

### Directory Organization

#### agents/ (7 files)
- `appointment_manager.py` - Appointment booking, rescheduling, cancellation
- `base_agent.py` - Base agent class with ReAct pattern
- `emergency_response.py` - Emergency handling
- `general_assistant.py` - General inquiries
- `medical_inquiry.py` - Medical questions
- `registration.py` - Patient registration
- `translation.py` - Language detection and translation

#### core/ (12 files)
- **Core Services:** config.py, llm.py, message_broker.py, orchestrator.py, state_manager.py
- **New Services:** conversation_memory.py, cost_calculator.py, observability.py, observability_broadcaster.py, reasoning.py, transaction_logger.py
- **Deprecated:** intent_router.py (replaced by reasoning.py)

#### infrastructure/ (2 files)
- `db_ops_client.py` - Database operations
- `notification_client.py` - Notification services

#### models/ (7 files)
- **Core Models:** appointments.py, enums.py, messages.py, state.py
- **New Models:** agentic.py, observability.py, validation.py

#### api/ (1 file)
- `server.py` - API server with observability and validation

---

## agents/appointment_manager.py

**Implementation Details:**

- **`on_activated` method** - Integrates with reasoning engine to receive routing, understanding, and response guidance. Initializes appointment workflow state based on reasoning output (`patient_ai_service/agents/appointment_manager.py:L41-103`)

- **`__init__` method** - Accepts and passes `max_iterations` parameter (default 15) to base class for complex booking flows (`patient_ai_service/agents/appointment_manager.py:L35-39`)

- **Imports** - Uses `ToolResultType` from `patient_ai_service.models.agentic` for structured tool result types (`patient_ai_service/agents/appointment_manager.py:L18`)

- **`tool_list_doctors`** - Supports `specialty` and `search_name` optional parameters for filtering. Returns structured results with `ToolResultType`, recovery actions, and suggested responses (`patient_ai_service/agents/appointment_manager.py:L598-711`)

- **`tool_check_availability`** - Implements two modes:
  - **Range Mode** (no `requested_time`): Returns `availability_ranges` - continuous time blocks where doctor is available
  - **Specific Time Mode** (with `requested_time`): Checks if specific time is available, returns alternatives if unavailable
  - Uses `get_available_time_slots` API endpoint
  - Returns structured results with `ToolResultType`, recovery actions, and next steps (`patient_ai_service/agents/appointment_manager.py:L1239-1434`)

- **Availability calculation helpers**:
  - `_merge_timeslots_to_ranges` - Merges consecutive timeslots into time ranges (`patient_ai_service/agents/appointment_manager.py:L773-825`)
  - `_normalize_time_to_24hr` - Normalizes various time formats to 24-hour HH:MM (`patient_ai_service/agents/appointment_manager.py:L827-897`)
  - `_is_time_in_available_slots` - Checks if time falls within available slots (`patient_ai_service/agents/appointment_manager.py:L898-929`)
  - `_normalize_time` - Time normalization wrapper (`patient_ai_service/agents/appointment_manager.py:L994-1000`)
  - `_convert_12_to_24` - Converts 12-hour to 24-hour format (`patient_ai_service/agents/appointment_manager.py:L1001-1010`)
  - `_find_closest_times` - Finds closest alternative times when requested time unavailable (`patient_ai_service/agents/appointment_manager.py:L1011-1024`)
  - `_format_alternatives_message` - Formats user-friendly alternative time messages (`patient_ai_service/agents/appointment_manager.py:L1025-1046`)
  - `_find_next_available_date` - Finds next available date when current date fully booked (`patient_ai_service/agents/appointment_manager.py:L1047-1068`)
  - `_check_specific_time_availability` - Checks specific time availability with conflict detection (`patient_ai_service/agents/appointment_manager.py:L1069-1167`)
  - `_calculate_availability_ranges` - Calculates availability ranges from timeslots (`patient_ai_service/agents/appointment_manager.py:L1168-1238`)

- **`tool_check_patient_appointments`** - Supports optional `appointment_date` and `start_time` parameters for filtering appointments (`patient_ai_service/agents/appointment_manager.py:L2419-2531`)

- **`tool_book_multiple_appointments`** - Books multiple appointments sequentially with rollback on failure. Supports booking different doctors, dates, times, and reasons in a single call (`patient_ai_service/agents/appointment_manager.py:L2057-2418`)

- **`tool_update_appointment`** - Updates appointment fields (doctor_id, date, time, status, reason, notes, etc.) with validation. Supports partial updates (`patient_ai_service/agents/appointment_manager.py:L2816-3002`)

- **`_rollback_appointments`** - Async method to rollback successfully booked appointments if later bookings fail in multi-appointment scenarios (`patient_ai_service/agents/appointment_manager.py:L2010-2056`)

- **Tool descriptions** - Comprehensive tool registration descriptions with instructions, examples, and mode explanations (`patient_ai_service/agents/appointment_manager.py:L105-345`)

**Key Features:**
- Integration with reasoning engine for context-aware workflows
- Dual-mode availability checking (range and specific time)
- Multiple appointment booking with transaction-like rollback
- Structured tool results with recovery actions and suggestions
- Flexible doctor search and appointment filtering
- Comprehensive appointment update functionality

---

## agents/base_agent.py

**Architecture:** ReAct (Reasoning, Action, Observation) Pattern

**Core Components:**

- **`CircuitBreaker` class** - Implements circuit breaker pattern to prevent cascading failures with states: CLOSED, OPEN, HALF_OPEN (`patient_ai_service/agents/base_agent.py:L57-123`)

- **`ExecutionContext` class** - Central hub tracking execution state:
  - Observations (tool results, events)
  - Criteria and their states (pending, complete, blocked, failed)
  - Continuation context for blocked flows
  - Metrics (tool calls, LLM calls, iteration count)
  - Error tracking and retry logic (`patient_ai_service/agents/base_agent.py:L134-460`)

- **`__init__` method** - Accepts `max_iterations` parameter (default 15) for controlling agentic loop iterations. Maintains `_context` dict for minimal context from reasoning engine. Maintains `_execution_log` for execution tracking (`patient_ai_service/agents/base_agent.py:L482-510`)

- **`on_activated` hook** - Abstract method called when agent is selected for a session, receives reasoning output (`patient_ai_service/agents/base_agent.py:L526-528`)

- **`set_context` method** - Sets minimal context for session from reasoning engine (`patient_ai_service/agents/base_agent.py:L530-533`)

- **`register_tool`** - Parameter schema generation identifies required parameters from schema (`patient_ai_service/agents/base_agent.py:L538-562`)

- **`process_message`** - Implements agentic loop:
  - Returns `Tuple[str, ExecutionLog]`
  - Initializes `ExecutionContext` with criteria from reasoning
  - Runs Think → Act → Observe loop until completion or max iterations
  - Handles different `AgentDecision` types: CALL_TOOL, RESPOND, RESPOND_WITH_OPTIONS, RESPOND_COMPLETE, RESPOND_IMPOSSIBLE, CLARIFY, RETRY
  - Verifies task completion before responding
  - Processes tool results with result type awareness (`patient_ai_service/agents/base_agent.py:L578-743`)

- **`process_message_with_log`** - Wrapper method that accepts and returns `ExecutionLog` for observability (`patient_ai_service/agents/base_agent.py:L566-576`)

- **`_think` method** - Core reasoning step that analyzes situation and decides next action. Uses lower temperature (0.2) for consistent reasoning. Returns `ThinkingResult` with decision, reasoning, and tool call details (`patient_ai_service/agents/base_agent.py:L906-995`)

- **`_get_thinking_prompt`** - Builds system prompt for thinking step with ReAct instructions, execution history, and available tools (`patient_ai_service/agents/base_agent.py:L997-1100`)

- **`_format_observations`** - Formats execution observations for prompt inclusion (`patient_ai_service/agents/base_agent.py:L1102-1120`)

- **`_format_tools_for_prompt`** - Formats tool schemas for prompt inclusion (`patient_ai_service/agents/base_agent.py:L1121-1150`)

- **`_build_thinking_messages`** - Builds message list for thinking step with user message and context (`patient_ai_service/agents/base_agent.py:L1151-1182`)

- **`_parse_thinking_response`** - Parses LLM thinking response into `ThinkingResult` with decision and reasoning (`patient_ai_service/agents/base_agent.py:L1183-1225`)

- **`_interpret_unstructured_response`** - Fallback for parsing unstructured LLM responses (`patient_ai_service/agents/base_agent.py:L1226-1263`)

- **`_verify_task_completion`** - Safety check before generating final response. Uses LLM to verify task completion based on execution history (`patient_ai_service/agents/base_agent.py:L1264-1382`)

- **`_generate_final_response`** - Generates final response after all tools executed. Only called after task completion verified (`patient_ai_service/agents/base_agent.py:L1386-1505`)

- **`_execute_tool`** - Accepts `execution_log` parameter. Records tool execution to observability logger and execution log. Returns structured results (`patient_ai_service/agents/base_agent.py:L1506-1581`)

- **`process_message_legacy`** - Legacy method preserved for backward compatibility (`patient_ai_service/agents/base_agent.py:L745-905`)

- **Dependencies** - Imports from:
  - `patient_ai_service.models.agentic` - `ToolResultType`, `CriterionState`, `Criterion`, `Observation`, `CompletionCheck`, `ThinkingResult`, `AgentDecision`
  - `patient_ai_service.models.validation` - `ExecutionLog`, `ToolExecution`
  - `patient_ai_service.models.observability` - Observability models
  - `patient_ai_service.core.observability` - `get_observability_logger`
  - `patient_ai_service.core.config` - `settings` (`patient_ai_service/agents/base_agent.py:L30-52`)

**Key Features:**
- Agentic loop with Think → Act → Observe pattern
- Error handling and recovery through result type awareness
- Task completion verification prevents false confirmations
- Observability integration for monitoring and debugging
- Support for complex multi-step workflows with criteria tracking
- Continuation context support for blocked flows
- Circuit breaker pattern prevents cascading failures

---

## agents/emergency_response.py

**Implementation:** Standard emergency response handling (279 lines)

**Note:** No significant changes from previous version

---

## agents/general_assistant.py

**Implementation:** General inquiry handling (540 lines)

**Note:** No significant changes from previous version

---

## agents/medical_inquiry.py

**Implementation:** Medical question handling (333 lines)

**Note:** Minor formatting updates only

---

## agents/registration.py

**Implementation Details:**

- **`on_activated` method** - Integrates with reasoning engine to initialize registration workflow. Determines missing fields and updates registration state (`patient_ai_service/agents/registration.py:L39-71`)

- **`REQUIRED_FIELDS`** - Required fields: first_name, last_name, phone, date_of_birth, gender (`patient_ai_service/agents/registration.py:L27-33`)

**Key Features:**
- Integration with reasoning engine for workflow initialization
- Simplified registration requirements (emergency contacts optional)

---

## agents/translation.py

**Implementation Details:**

- **`detect_language_and_dialect` method** - Detects both language and regional dialect (e.g., "ar-EG", "en-US", "es-MX"). Returns tuple of (language_code, dialect_code) with confidence level (`patient_ai_service/agents/translation.py:L112-177`)

- **`translate_to_english_with_dialect` method** - Translates to English with dialect awareness. Adapts region-specific terms to US English medical standard (`patient_ai_service/agents/translation.py:L218-267`)

- **`translate_from_english_with_dialect` method** - Translates from English to target language with dialect awareness for more natural regional translations (`patient_ai_service/agents/translation.py:L309-360`)

- **Legacy methods** - `detect_language`, `translate_to_english`, `translate_from_english` maintained for backward compatibility but recommend using dialect-aware versions (`patient_ai_service/agents/translation.py:L73-110, L179-217, L269-307`)

- **Configuration** - Uses `settings.translation_temperature` from `patient_ai_service.core.config` for configurable translation temperature (`patient_ai_service/agents/translation.py:L12`)

**Key Features:**
- Dialect-aware translation for regional language variations
- Support for Egyptian vs Gulf Arabic, US vs UK English, etc.
- Natural translations adapted to regional medical terminology standards
- Configurable translation temperature

---

## core/orchestrator.py

**Architecture:** Unified Pipeline Orchestration

**Core Components:**

- **`__init__` method** - Initializes `reasoning_engine` and `memory_manager` alongside other core services (`patient_ai_service/core/orchestrator.py:L57-68`)

- **`process_message`** - Implements 8-step pipeline:
  - Step 1: Load patient
  - Step 2: Translation (input) with dialect detection
  - Step 3: Add to conversation memory
  - Step 4: Unified reasoning
  - Step 5: Agent execution with agentic loop
  - Step 6: Translation (output) with dialect awareness
  - Step 7: Response validation
  - Step 8: Publish to message broker
  - Comprehensive observability logging throughout (`patient_ai_service/core/orchestrator.py:L103-756`)

- **`_extract_task_context` method** - Extracts task context from reasoning output, merges with continuation context for resuming workflows (`patient_ai_service/core/orchestrator.py:L848-919`)

- **`_build_agent_context` method** - Builds context dict for agents, injects critical parameters like `patient_id` to prevent hallucinations (`patient_ai_service/core/orchestrator.py:L921-988`)

- **`_handle_agentic_completion` method** - Handles completion of agentic execution, updates state, processes continuation context (`patient_ai_service/core/orchestrator.py:L990-1091`)

- **`_build_response_metadata` method** - Builds response metadata with execution details, validation results, and observability info (`patient_ai_service/core/orchestrator.py:L1092-1137`)

- **`_get_validation_fallback` method** - Generates fallback response when validation fails (`patient_ai_service/core/orchestrator.py:L757-771`)

- **Translation integration** - Uses `detect_language_and_dialect` and `translate_to_english_with_dialect` for dialect-aware translation (`patient_ai_service/core/orchestrator.py:L156-195`)

- **Dependencies** - Imports from:
  - `patient_ai_service.core.reasoning` - `get_reasoning_engine`
  - `patient_ai_service.core.conversation_memory` - `get_conversation_memory_manager`
  - `patient_ai_service.core.observability` - `get_observability_logger`, `clear_observability_logger`
  - `patient_ai_service.models.validation` - `ExecutionLog`, `ValidationResult`
  - `patient_ai_service.core.config` - `settings` (`patient_ai_service/core/orchestrator.py:L20-40`)

**Key Features:**
- Unified reasoning engine for intelligent routing
- Conversation memory integration for context awareness
- Dialect-aware translation for multilingual support
- Comprehensive observability throughout pipeline
- Agentic loop execution with completion handling
- Response validation before returning to user

---

## core/config.py

**Implementation:** Configuration management (263 lines)

**Features:**
- Configuration settings for reasoning engine
- Observability configuration
- Conversation memory settings
- LLM provider and model configuration
- Temperature settings for different components

---

## core/intent_router.py

**Status:** Deprecated

**Note:** This file is maintained for backward compatibility but is no longer the primary routing mechanism. Intent routing is handled by `core/reasoning.py` (unified reasoning engine).

---

## core/llm.py

**Implementation:** LLM client with enhanced capabilities (444 lines)

**Features:**
- Token usage tracking
- Observability integration
- Reasoning engine support
- Multiple provider support (OpenAI, Anthropic, etc.)

---

## core/message_broker.py

**Implementation:** Message broker for pub/sub messaging (193 lines)

**Note:** Standard message broker implementation

---

## core/state_manager.py

**Implementation:** Enhanced state management (899 lines)

**Features:**
- Agentic execution state tracking
- Conversation memory state management
- Reasoning context state
- Language context with dialect support
- Continuation context for blocked flows
- Enhanced state management for complex workflows

---

## core/reasoning.py

**Architecture:** Unified Reasoning Engine (Replaces `core/intent_router.py`)

**Core Components:**

- **`ReasoningEngine` class** - Main engine that processes user messages and returns structured `ReasoningOutput`:
  - `UnderstandingResult` - What user means, continuation detection, sentiment
  - `RoutingResult` - Agent selection, action, urgency
  - `MemoryUpdate` - New facts, system actions, awaiting state
  - `ResponseGuidance` - Tone, task context, minimal context, plan
  - `reasoning_chain` - Step-by-step reasoning process (`patient_ai_service/core/reasoning.py:L274-1538`)

- **`ContinuationDetector` class** - Detects when user message is a continuation/response to previous options (`patient_ai_service/core/reasoning.py:L98-272`)

- **Pydantic models**:
  - `UnderstandingResult` - User understanding with continuation detection
  - `RoutingResult` - Agent routing decision
  - `MemoryUpdate` - Memory update instructions
  - `TaskContext` - Structured context for agent execution with entities, success criteria, constraints
  - `ResponseGuidance` - Response guidance for agents
  - `ReasoningOutput` - Complete reasoning output (`patient_ai_service/core/reasoning.py:L37-91`)

**Architectural Philosophy:**

- **Reasoning-based understanding** - Treats routing as holistic reasoning: understand user intent in full conversation context → determine best action → provide structured guidance
- **Chain-of-thought reasoning** - Single LLM call performs all reasoning tasks: understanding, routing, memory updates, response guidance (`reasoning.py:L320-593`)
- **Stateful reasoning** - Integrates with conversation memory, continuation context, and system state (`reasoning.py:L349-404`)
- **Multi-purpose component** - Handles understanding, routing, memory updates, validation, and finalization (`reasoning.py:L1047-1265`)

**Design Pattern:**

- **Input:** User message + conversation memory + patient info + continuation context (`reasoning.py:L320-325`)
- **Process:** Memory retrieval → continuation detection → unified LLM reasoning → structured parsing → memory updates (`reasoning.py:L349-556`)
- **Output:** `ReasoningOutput` with understanding, routing, memory updates, response guidance, reasoning chain (`reasoning.py:L85-91`)
- **Integration:** Direct memory manager integration, state manager integration, observability integration (`reasoning.py:L285-301`)

**Decision Flow:**

1. Retrieve conversation memory and continuation context (`reasoning.py:L349-358`)
2. Pre-detect continuation using pattern matching (`reasoning.py:L361-374`)
3. Check for conversation restart (`reasoning.py:L376-396`)
4. Build unified reasoning prompt with full context (`reasoning.py:L399-783`)
5. Single LLM call with chain-of-thought reasoning (`reasoning.py:L443-458`)
6. Parse and enhance with continuation detection (`reasoning.py:L484-949`)
7. Update memory with extracted facts (`reasoning.py:L541-556`)
8. Trigger summarization if needed (`reasoning.py:L553-556`)

**State Management:**

- **Rich context** - Retrieves conversation memory with user facts, summary, recent turns, system state (`reasoning.py:L350`)
- **Continuation context** - Retrieves and merges continuation context from state manager (`reasoning.py:L353-358`)
- **Memory integration** - Updates memory with new facts and system state (`reasoning.py:L541-551`)
- **State persistence** - All context stored in state manager and conversation memory manager (`reasoning.py:L285-301`)

**Integration Points:**

- **Orchestrator** - Called via `get_reasoning_engine()` singleton (`reasoning.py:L1529-1534`)
- **LLM Client** - Direct LLM calls with token usage tracking (`reasoning.py:L446-458`)
- **Memory Manager** - Direct integration for memory retrieval and updates (`reasoning.py:L22-25, L350, L541-551`)
- **State Manager** - Retrieves continuation context and global state (`reasoning.py:L26, L353-358, L494`)
- **Observability** - Records reasoning steps, LLM calls, and reasoning details (`reasoning.py:L411-474, L506-539`)
- **Validation Layer** - Provides `validate_response()` and `finalize_response()` methods (`reasoning.py:L1047-1265`)

**Key Features:**
- Unified reasoning replaces separate intent classification
- Context understanding with continuation detection
- Structured task context for agents with entities and success criteria
- Single LLM call for all reasoning tasks
- Memory integration enables persistent conversation context
- Validation layer prevents false confirmations and hallucinations

---

## core/conversation_memory.py

**Architecture:** Conversation Memory System

**Core Components:**

- **`ConversationMemoryManager` class** - Manages conversation history with:
  - Persistent facts storage (never summarized)
  - Smart summarization based on token count
  - Recent turns tracking (last 6 kept raw)
  - Conversation boundary detection
  - Redis persistence support (`patient_ai_service/core/conversation_memory.py:L51-388`)

- **Pydantic models**:
  - `ConversationTurn` - Single turn in conversation (role, content, timestamp)
  - `ConversationMemory` - Memory structure with user_facts, summary, recent_turns, system state (`patient_ai_service/core/conversation_memory.py:L21-49`)

**Key Features:**
- Conversation context tracking across turns
- Efficient memory management with summarization
- Persistent facts prevent information loss
- Supports long conversations without token bloat

---

## core/observability.py

**Architecture:** Comprehensive Observability System

**Core Components:**

- **`ObservabilityLogger` class** - Main logger with:
  - `TokenTracker` - Tracks token usage
  - `AgentFlowTracker` - Tracks agent execution flow
  - `ReasoningTracker` - Tracks reasoning engine calls
  - `ToolExecutionTracker` - Tracks tool executions
  - Pipeline step tracking
  - Cost accumulation (`patient_ai_service/core/observability.py:L815`)

- **Helper classes**:
  - `TokenTracker` - Token usage tracking
  - `AgentFlowTracker` - Agent flow tracking
  - `ReasoningTracker` - Reasoning tracking
  - `ToolExecutionTracker` - Tool execution tracking (`patient_ai_service/core/observability.py`)

**Key Features:**
- Comprehensive observability for debugging and monitoring
- Token usage tracking for cost management
- Agent flow tracking for understanding execution paths
- Cost accumulation for budget management

---

## core/observability_broadcaster.py

**Architecture:** Observability Event Broadcasting

**Core Components:**

- **`ObservabilityBroadcaster` class** - Broadcasts observability data to configured endpoints (`patient_ai_service/core/observability_broadcaster.py:L93`)

**Key Features:**
- External observability integration for monitoring systems
- Fire-and-forget broadcasting for performance

---

## core/cost_calculator.py

**Architecture:** LLM Cost Calculation

**Core Components:**

- **`CostCalculator` class** - Calculates costs for different LLM providers (OpenAI, Anthropic, etc.) based on input/output tokens and model pricing (`patient_ai_service/core/cost_calculator.py:L158`)

**Key Features:**
- Cost tracking and budget management for LLM API usage
- Provider-specific pricing models

---

## core/transaction_logger.py

**Architecture:** Transaction Logging System

**Core Components:**

- **`TransactionLogger` class** - Logs transactions with:
  - Transaction type (appointment_booking, registration, etc.)
  - Transaction metrics (latency, success/failure)
  - Detailed execution logs
  - Session-based logging (`patient_ai_service/core/transaction_logger.py:L504`)

- **Enums and models**:
  - `TransactionType` - Enum for transaction types
  - `TransactionMetrics` - Metrics for transactions
  - `Transaction` - Transaction model (`patient_ai_service/core/transaction_logger.py`)

**Key Features:**
- Transaction logging for audit and debugging
- Metrics tracking for performance monitoring
- Detailed execution logs for troubleshooting

---

## infrastructure/db_ops_client.py

**Implementation:** Database operations client (1,742 lines)

**Key Methods:**
- `get_available_time_slots` - New endpoint for availability checking
- `update_appointment` - Appointment update functionality
- Enhanced error handling and observability integration

**Key Features:**
- Enhanced database operations with new appointment management features
- Observability integration for monitoring

---

## infrastructure/notification_client.py

**Implementation:** Notification services client (292 lines)

**Note:** Standard notification client implementation

---

## models/appointments.py

**Implementation:** Appointment data models (121 lines)

**Note:** Standard appointment models

---

## models/enums.py

**Implementation:** Enum definitions (149 lines)

**Features:**
- Enum values for v3 features
- Urgency levels, agent types, and other system enums

---

## models/messages.py

**Implementation:** Message data models (158 lines)

**Note:** Standard message models

---

## models/state.py

**Implementation:** State management models (304 lines)

**Features:**
- Agentic execution state models
- Conversation memory state models
- Reasoning context state models
- Language context with dialect support
- Enhanced state tracking for complex workflows

---

## models/agentic.py

**Architecture:** Agentic Execution Models

**Core Models:**

- **Enums:**
  - `ToolResultType` - Tool result types (SUCCESS, PARTIAL, RECOVERABLE, USER_INPUT_NEEDED, SYSTEM_ERROR, FATAL)
  - `CriterionState` - Criterion states (PENDING, COMPLETE, BLOCKED, FAILED)
  - `AgentDecision` - Agent decisions (CALL_TOOL, RESPOND, RESPOND_WITH_OPTIONS, RESPOND_COMPLETE, RESPOND_IMPOSSIBLE, CLARIFY, RETRY)

- **Models:**
  - `Criterion` - Success criteria tracking
  - `Observation` - Tool/event observations
  - `ThinkingResult` - Thinking step results
  - `CompletionCheck` - Task completion verification (`patient_ai_service/models/agentic.py`)

**Key Features:**
- Structured models for agentic execution and ReAct pattern implementation
- Type-safe enums for agent decisions and tool results

---

## models/observability.py

**Architecture:** Observability Data Models

**Core Models:**

- `TokenUsage` - Token usage tracking
- `CostInfo` - Cost information
- `LLMCall` - LLM call details
- `ToolExecutionDetail` - Tool execution details
- `AgentContext` - Agent context information
- `ReasoningStep` - Reasoning step details
- `ReasoningDetails` - Reasoning details
- `PipelineStep` - Pipeline step tracking
- `AgentExecutionDetails` - Agent execution details
- `ValidationDetails` - Validation details
- `FinalizationDetails` - Finalization details
- `SessionObservability` - Session-level observability (`patient_ai_service/models/observability.py`)

**Key Features:**
- Comprehensive observability models for monitoring and debugging
- Structured data for cost tracking and performance analysis

---

## models/validation.py

**Architecture:** Validation Models

**Core Models:**

- `ToolExecution` - Tool execution record
- `ExecutionLog` - Execution log with tools used
- `ValidationResult` - Validation result with issues and suggestions (`patient_ai_service/models/validation.py`)

**Key Features:**
- Structured validation models for execution logging and validation
- Supports response validation and retry logic

---

## api/server.py

**Architecture:** API Server with Observability and Validation

**Implementation:** Enhanced API server (1,506 lines)

**Features:**
- Enhanced endpoint handlers
- Observability integration throughout
- Execution logging
- Response validation
- New endpoints for v3 features
- Enhanced error handling
- Request/response validation

**Key Features:**
- Comprehensive observability integration
- Response validation before returning to clients
- Execution logging for debugging and audit
- Enhanced error handling and recovery

---





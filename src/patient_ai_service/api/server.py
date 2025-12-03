"""
FastAPI server with HTTP and WebSocket endpoints.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from patient_ai_service.core.config import settings
from patient_ai_service.core.orchestrator import Orchestrator
from patient_ai_service.core.observability_broadcaster import get_observability_broadcaster
from patient_ai_service.models.messages import ChatRequest, ChatResponse, WebSocketMessage
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient
from fastapi.responses import HTMLResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global orchestrator

    # Startup
    logger.info("Starting Dental Clinic AI Service v2.0")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.get_llm_model()}")

    # Initialize orchestrator
    db_client = DbOpsClient()
    orchestrator = Orchestrator(db_client=db_client)
    await orchestrator.start()

    logger.info("Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down service...")
    await orchestrator.stop()
    logger.info("Service stopped")


# Create FastAPI app
app = FastAPI(
    title="Dental Clinic AI Service",
    description="Multi-agent AI system for dental clinic management",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "service": "Dental Clinic AI Service",
        "version": "2.0.0",
        "status": "running",
        "llm_provider": settings.llm_provider,
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Could add more health checks here
        return {
            "status": "healthy",
            "service": "patient-ai-service-v2",
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post("/api/message")
async def api_message(request: Request):
    """
    Legacy endpoint for WhatsApp service compatibility.
    
    Accepts the same format as patient-ai-service v1:
    {
        "message": "Hello",
        "phone_number": "+971501234567",
        "production": false,
        "source": "whatsapp"
    }
    
    Returns:
    {
        "response": "Hello! How can I help?",
        "intent": "greeting",
        "patient_identified": true,
        "patient_id": "uuid",
        "emergency_flag": false
    }
    """
    try:
        data = await request.json()
        message = data.get("message") or data.get("text", "")
        phone_number = data.get("phone_number") or data.get("phoneNumber")
        session_id = phone_number or data.get("session_id") or f"session_{hash(str(data))}"
        
        logger.info(f"API message request: phone={phone_number}, session={session_id}")
        
        # Process message through orchestrator
        response = await orchestrator.process_message(
            session_id=session_id,
            message=message,
            language=None
        )
        
        # Get patient info from state
        state = orchestrator.get_session_state(session_id)
        patient_id = state.get("patient_profile", {}).get("patient_id")
        patient_identified = patient_id is not None
        
        # Return v1-compatible format
        return {
            "response": response.response,
            "intent": str(response.intent) if response.intent else "general_inquiry",
            "patient_identified": patient_identified,
            "patient_id": patient_id,
            "emergency_flag": response.urgency == "critical",
            "session_id": session_id,
            "metadata": response.metadata
        }
        
    except Exception as e:
        logger.error(f"Error in /api/message endpoint: {e}", exc_info=True)
        return {
            "response": "I'm sorry, but I encountered an error processing your message. Please try again.",
            "intent": "error",
            "patient_identified": False,
            "patient_id": None,
            "emergency_flag": False,
            "error": str(e) if settings.debug else None
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message.

    Args:
        request: ChatRequest with message and session_id

    Returns:
        ChatResponse with agent's reply
    """
    try:
        logger.info(f"Chat request from session: {request.session_id}")

        response = await orchestrator.process_message(
            session_id=request.session_id,
            message=request.message,
            language=request.language
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Get session state.

    Args:
        session_id: Session identifier

    Returns:
        Complete session state
    """
    try:
        state = orchestrator.get_session_state(session_id)
        return {
            "session_id": session_id,
            "state": state
        }

    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session state"
        )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session state.

    Args:
        session_id: Session identifier

    Returns:
        Success message
    """
    try:
        orchestrator.clear_session(session_id)
        return {
            "success": True,
            "message": f"Session {session_id} cleared"
        }

    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear session"
        )


@app.get("/config")
async def get_config():
    """Get public configuration info."""
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.get_llm_model(),
        "supported_languages": ["en", "ar", "es", "fr", "hi", "zh", "pt", "ru"],
        "translation_enabled": settings.enable_translation,
        "environment": settings.environment
    }


@app.post("/notifications/push")
async def push_notification(request: Request):
    """
    Push a notification to an active WebSocket connection.
    
    This endpoint is called by the notification service to send notifications
    to patients who have active web chat sessions.
    
    Request body:
    {
        "session_id": "phone_number_or_session_id",
        "message": "Notification message",
        "notification_type": "appointment_reminder",
        "metadata": {}
    }
    
    Returns:
    {
        "success": true/false,
        "delivered": true/false,
        "session_id": "...",
        "message": "..."
    }
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        message = data.get("message")
        notification_type = data.get("notification_type", "general")
        metadata = data.get("metadata", {})
        
        if not session_id or not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id and message are required"
            )
        
        # Check if there's an active WebSocket connection for this session
        websocket = websocket_connections.get(session_id)
        
        if websocket:
            try:
                # Send notification via WebSocket
                await websocket.send_json({
                    "type": "notification",
                    "session_id": session_id,
                    "content": message,
                    "notification_type": notification_type,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"✅ Notification pushed to web chat session: {session_id}")
                return {
                    "success": True,
                    "delivered": True,
                    "session_id": session_id,
                    "message": "Notification delivered to web chat"
                }
            except Exception as e:
                logger.error(f"❌ Error sending notification via WebSocket: {e}")
                # Remove stale connection
                if session_id in websocket_connections:
                    del websocket_connections[session_id]
                return {
                    "success": False,
                    "delivered": False,
                    "session_id": session_id,
                    "message": f"Failed to deliver: {str(e)}"
                }
        else:
            # No active WebSocket connection - this is normal if patient is not using web chat
            logger.debug(f"ℹ️ No active web chat session for {session_id} - notification will be sent via WhatsApp only")
            return {
                "success": True,
                "delivered": False,
                "session_id": session_id,
                "message": "No active web chat session - notification sent via WhatsApp"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pushing notification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to push notification: {str(e)}"
        )


# ============================================================================
# WebSocket Endpoint
# ============================================================================

# Active WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.

    Supports session_id in query parameters for immediate registration:
    ws://host/ws?session_id=+971501234567

    Client sends:
    {
        "type": "message",
        "session_id": "user_123",
        "content": "Hello"
    }

    Or to register a session:
    {
        "type": "register",
        "session_id": "user_123"
    }

    Server responds:
    {
        "type": "response",
        "session_id": "user_123",
        "content": "Hi! How can I help you?",
        "data": {...}
    }
    """
    import asyncio
    
    await websocket.accept()
    session_id: str = None

    try:
        # Get session_id from query parameters
        query_params = dict(websocket.query_params)
        session_id = query_params.get("session_id")
        
        if session_id:
            websocket_connections[session_id] = websocket
            logger.info(f"WebSocket connection established: {session_id}")
            
            # Send registration confirmation
            await websocket.send_json({
                "type": "registered",
                "session_id": session_id,
                "message": "Session registered successfully"
            })
        else:
            logger.info("WebSocket connection established (no session_id)")

        # Keep connection alive with proper timeout handling
        while True:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=300.0  # 5 minute timeout for idle connections
                )

                message_type = data.get("type")
                content = data.get("content")
                message_session_id = data.get("session_id")
                
                if message_session_id:
                    session_id = message_session_id
                    websocket_connections[session_id] = websocket

                if not session_id:
                    await websocket.send_json({
                        "type": "error",
                        "content": "session_id required"
                    })
                    continue

                if message_type == "register":
                    websocket_connections[session_id] = websocket
                    logger.info(f"Session registered: {session_id}")
                    await websocket.send_json({
                        "type": "registered",
                        "session_id": session_id,
                        "message": "Session registered successfully"
                    })
                    continue

                if message_type == "message":
                    # Send typing indicator
                    await websocket.send_json({
                        "type": "typing",
                        "session_id": session_id,
                        "is_typing": True
                    })

                    try:
                        # Process message with extended timeout
                        response = await asyncio.wait_for(
                            orchestrator.process_message(
                                session_id=session_id,
                                message=content,
                                language=data.get("language")
                            ),
                            timeout=90.0  # 90 second timeout for processing
                        )

                        # Send response
                        await websocket.send_json({
                            "type": "response",
                            "session_id": session_id,
                            "content": response.response,
                            "data": {
                                "intent": response.intent,
                                "urgency": response.urgency,
                                "detected_language": response.detected_language,
                                "metadata": response.metadata
                            }
                        })
                    except asyncio.TimeoutError:
                        await websocket.send_json({
                            "type": "error",
                            "session_id": session_id,
                            "content": "Request timeout. Please try again with a shorter message."
                        })
                    finally:
                        # Stop typing
                        await websocket.send_json({
                            "type": "typing",
                            "session_id": session_id,
                            "is_typing": False
                        })

                elif message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "session_id": session_id
                    })

                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except asyncio.TimeoutError:
                # Idle timeout - close connection gracefully
                logger.info(f"WebSocket idle timeout: {session_id}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "session_id": session_id,
                "content": "An error occurred"
            })
        except:
            pass
    finally:
        if session_id and session_id in websocket_connections:
            del websocket_connections[session_id]
        # Close with proper code (1000 = normal closure)
        try:
            await websocket.close(code=1000)
        except:
            pass


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed logging."""
    errors = exc.errors()
    body = await request.body()
    
    logger.warning(f"Validation error on {request.method} {request.url.path}")
    logger.warning(f"Request body: {body.decode('utf-8') if body else 'None'}")
    logger.warning(f"Validation errors: {errors}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": errors,
            "body": body.decode('utf-8') if body else None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.get("/observability", response_class=HTMLResponse)
async def observability_ui():
    """Serve the observability UI with terminal-like experience."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Observability - Real-Time</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Monaco', 'Menlo', 'Consolas', 'Courier New', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            font-size: 13px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: #252526;
            padding: 15px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #007acc;
        }
        
        .header h1 {
            color: #4ec9b0;
            font-size: 18px;
            margin-bottom: 5px;
        }
        
        .status {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .status.connected {
            background: #4ec9b0;
            color: #1e1e1e;
        }
        
        .status.disconnected {
            background: #f48771;
            color: #1e1e1e;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: #252526;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007acc;
        }
        
        .stat-label {
            color: #858585;
            font-size: 11px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .stat-value {
            color: #4ec9b0;
            font-size: 20px;
            font-weight: bold;
        }
        
        .stat-sub {
            color: #858585;
            font-size: 10px;
            margin-top: 3px;
        }
        
        .terminal-output {
            background: #1e1e1e;
            border: 1px solid #3e3e42;
            border-radius: 5px;
            padding: 15px;
            max-height: 600px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.8;
            margin-bottom: 20px;
        }
        
        .terminal-line {
            margin-bottom: 8px;
            padding: 4px 0;
            border-bottom: 1px solid #2d2d30;
        }
        
        .terminal-prompt {
            color: #4ec9b0;
            font-weight: bold;
        }
        
        .terminal-timestamp {
            color: #858585;
            margin-right: 10px;
        }
        
        .terminal-event-type {
            color: #569cd6;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .terminal-data {
            color: #d4d4d4;
        }
        
        .terminal-success {
            color: #b5cea8;
        }
        
        .terminal-error {
            color: #f48771;
        }
        
        .terminal-warning {
            color: #ce9178;
        }
        
        .terminal-info {
            color: #9cdcfe;
        }
        
        .sections {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .section {
            background: #252526;
            border-radius: 5px;
            padding: 15px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .section-title {
            color: #4ec9b0;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #3e3e42;
        }
        
        .pipeline-steps {
            grid-column: 1 / -1;
            max-height: 400px;
        }
        
        .step {
            background: #1e1e1e;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 3px;
            border-left: 3px solid #007acc;
            font-size: 12px;
        }
        
        .step-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .step-number {
            color: #007acc;
            font-weight: bold;
        }
        
        .step-name {
            color: #d4d4d4;
            font-weight: bold;
        }
        
        .step-component {
            color: #858585;
            font-size: 10px;
        }
        
        .step-duration {
            color: #4ec9b0;
            font-size: 11px;
        }
        
        .step-details {
            color: #858585;
            font-size: 11px;
            margin-top: 5px;
        }
        
        .llm-call {
            background: #1e1e1e;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 3px;
            border-left: 3px solid #4ec9b0;
            font-size: 11px;
        }
        
        .llm-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .llm-component {
            color: #4ec9b0;
            font-weight: bold;
        }
        
        .llm-model {
            color: #858585;
            font-size: 10px;
        }
        
        .llm-tokens {
            color: #ce9178;
            font-size: 11px;
        }
        
        .llm-cost {
            color: #b5cea8;
            font-size: 11px;
        }
        
        .tool-exec {
            background: #1e1e1e;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 3px;
            border-left: 3px solid #ce9178;
            font-size: 11px;
        }
        
        .tool-name {
            color: #ce9178;
            font-weight: bold;
        }
        
        .tool-duration {
            color: #4ec9b0;
            font-size: 10px;
        }
        
        .tool-success {
            color: #b5cea8;
        }
        
        .tool-error {
            color: #f48771;
        }
        
        .reasoning-step {
            background: #1e1e1e;
            padding: 8px;
            margin-bottom: 5px;
            border-radius: 3px;
            border-left: 3px solid #b5cea8;
            font-size: 11px;
        }
        
        .reasoning-desc {
            color: #d4d4d4;
        }
        
        .json-view {
            background: #1e1e1e;
            padding: 15px;
            border-radius: 5px;
            max-height: 600px;
            overflow-y: auto;
            font-size: 11px;
        }
        
        .json-key {
            color: #9cdcfe;
        }
        
        .json-string {
            color: #ce9178;
        }
        
        .json-number {
            color: #b5cea8;
        }
        
        .json-boolean {
            color: #569cd6;
        }
        
        .timestamp {
            color: #858585;
            font-size: 10px;
        }
        
        .error {
            color: #f48771;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1e1e1e;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #424242;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #4e4e4e;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>System Observability <span class="status disconnected" id="status">DISCONNECTED</span></h1>
            <div class="timestamp" id="lastUpdate">Waiting for connection...</div>
        </div>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="stat-label">Total Tokens</div>
                <div class="stat-value" id="totalTokens">0</div>
                <div class="stat-sub" id="tokenBreakdown">Input: 0 | Output: 0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Cost</div>
                <div class="stat-value" id="totalCost">$0.000000</div>
                <div class="stat-sub" id="costBreakdown">Input: $0 | Output: $0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pipeline Steps</div>
                <div class="stat-value" id="pipelineSteps">0</div>
                <div class="stat-sub" id="pipelineDuration">Duration: 0ms</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">LLM Calls</div>
                <div class="stat-value" id="llmCalls">0</div>
                <div class="stat-sub" id="llmTokens">Total: 0 tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tool Executions</div>
                <div class="stat-value" id="toolExecs">0</div>
                <div class="stat-sub" id="toolSuccess">Success: 0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Session</div>
                <div class="stat-value" id="activeSession" style="font-size: 14px;">-</div>
                <div class="stat-sub" id="sessionTime">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Memory</div>
                <div class="stat-value" id="memoryHistoryLength">0</div>
                <div class="stat-sub" id="memoryUpdates">Updates: 0</div>
            </div>
        </div>
        
        <div class="terminal-output" id="terminalOutput">
            <div class="terminal-line">
                <span class="terminal-prompt">$</span>
                <span class="terminal-timestamp">[SYSTEM]</span>
                <span class="terminal-info">Waiting for observability events...</span>
            </div>
        </div>
        
        <div class="sections">
            <div class="section pipeline-steps">
                <div class="section-title">Pipeline Steps</div>
                <div id="pipelineStepsList"></div>
            </div>
        </div>
        
        <div class="sections">
            <div class="section">
                <div class="section-title">LLM Calls</div>
                <div id="llmCallsList"></div>
            </div>
            <div class="section">
                <div class="section-title">Tool Executions</div>
                <div id="toolExecsList"></div>
            </div>
        </div>
        
        <div class="sections">
            <div class="section">
                <div class="section-title">Memory</div>
                <div id="memoryList"></div>
            </div>
            <div class="section">
                <div class="section-title">Reasoning Chain</div>
                <div id="reasoningChain"></div>
            </div>
        </div>
        
        <div class="sections">
            <div class="section">
                <div class="section-title">Session Summary (JSON)</div>
                <div class="json-view" id="sessionSummary"></div>
            </div>
        </div>
    </div>
    
    <script>
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/observability`;
        let ws = null;
        let sessionData = {
            totalTokens: { input: 0, output: 0, total: 0 },
            totalCost: { input: 0, output: 0, total: 0 },
            pipelineSteps: [],
            llmCalls: [],
            toolExecs: [],
            reasoningSteps: [],
            memoryUpdates: [],
            conversationHistoryLength: 0,
            sessionId: null,
            startTime: null,
            currentRequest: {
                startTime: null,
                pipelineSteps: [],
                llmCalls: [],
                toolExecs: [],
                tokens: { input: 0, output: 0, total: 0 },
                cost: { input: 0, output: 0, total: 0 }
            }
        };
        
        function connect() {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                document.getElementById('status').textContent = 'CONNECTED';
                document.getElementById('status').className = 'status connected';
                addTerminalLine('system', {message: 'Connected to observability stream'}, 'terminal-success');
                console.log('Connected to observability stream');
            };
            
            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    handleMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    console.error('Raw message:', event.data);
                    addTerminalLine('error', {message: 'Failed to parse message: ' + error.message}, 'terminal-error');
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                addTerminalLine('error', {message: 'WebSocket error occurred'}, 'terminal-error');
            };
            
            ws.onclose = () => {
                document.getElementById('status').textContent = 'DISCONNECTED';
                document.getElementById('status').className = 'status disconnected';
                addTerminalLine('system', {message: 'Disconnected from observability stream. Reconnecting...'}, 'terminal-error');
                console.log('Disconnected from observability stream');
                setTimeout(connect, 3000);
            };
        }
        
        function addTerminalLine(type, data, colorClass = 'terminal-data') {
            const terminal = document.getElementById('terminalOutput');
            const timestamp = new Date().toLocaleTimeString();
            const line = document.createElement('div');
            line.className = 'terminal-line';
            
            let content = '';
            if (type === 'pipeline_step') {
                const durationMs = data.duration_ms || 0;
                const durationSec = (durationMs / 1000).toFixed(2);
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[PIPELINE]</span> <span class="terminal-info">Step #${data.step_number}: ${data.step_name}</span> <span class="${colorClass}">[${data.component}]</span> <span class="terminal-warning">${durationMs.toFixed(2)}ms = ${durationSec}s</span>`;
            } else if (type === 'llm_call') {
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[LLM]</span> <span class="terminal-info">${data.component}</span> <span class="${colorClass}">${data.model}</span> <span class="terminal-warning">${data.tokens?.total_tokens || 0} tokens</span> <span class="terminal-success">$${data.cost?.total_cost_usd?.toFixed(6) || '0.000000'}</span>`;
            } else if (type === 'tool_execution') {
                const status = data.success ? '<span class="terminal-success">✓</span>' : '<span class="terminal-error">✗</span>';
                const durationMs = data.duration_ms || 0;
                const durationSec = (durationMs / 1000).toFixed(2);
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[TOOL]</span> <span class="terminal-info">${data.tool_name}</span> ${status} <span class="terminal-warning">${durationMs.toFixed(2)}ms = ${durationSec}s</span>`;
            } else if (type === 'memory_update') {
                const updates = data.updates || {};
                const newFacts = updates.new_facts || {};
                const systemAction = updates.system_action || '';
                const awaiting = updates.awaiting || '';
                let memInfo = [];
                if (Object.keys(newFacts).length > 0) {
                    memInfo.push(`Facts: ${Object.keys(newFacts).length}`);
                }
                if (systemAction) memInfo.push(`Action: ${systemAction}`);
                if (awaiting) memInfo.push(`Awaiting: ${awaiting}`);
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[MEMORY]</span> <span class="terminal-info">${memInfo.join(' | ')}</span>`;
            } else if (type === 'reasoning_step') {
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[REASONING]</span> <span class="${colorClass}">${data.step_number}. ${data.description}</span>`;
            } else if (type === 'memory_update') {
                const updates = data.updates || {};
                const newFacts = updates.new_facts || {};
                const systemAction = updates.system_action || '';
                const awaiting = updates.awaiting || '';
                let memInfo = [];
                if (Object.keys(newFacts).length > 0) {
                    memInfo.push(`Facts: ${Object.keys(newFacts).length}`);
                }
                if (systemAction) memInfo.push(`Action: ${systemAction}`);
                if (awaiting) memInfo.push(`Awaiting: ${awaiting}`);
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[MEMORY]</span> <span class="terminal-info">${memInfo.join(' | ')}</span>`;
            } else if (type === 'session_summary') {
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[SUMMARY]</span> <span class="terminal-info">Session: ${data.session_id}</span> <span class="terminal-success">Complete</span>`;
            } else {
                content = `<span class="terminal-prompt">$</span> <span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-event-type">[${type.toUpperCase()}]</span> <span class="${colorClass}">${JSON.stringify(data).substring(0, 100)}</span>`;
            }
            
            line.innerHTML = content;
            terminal.appendChild(line);
            
            // Auto-scroll to bottom
            terminal.scrollTop = terminal.scrollHeight;
            
            // Keep only last 500 lines
            while (terminal.children.length > 500) {
                terminal.removeChild(terminal.firstChild);
            }
        }
        
        function handleMessage(message) {
            try {
                if (message.timestamp) {
                    document.getElementById('lastUpdate').textContent = `Last update: ${new Date(message.timestamp).toLocaleTimeString()}`;
                }
                
                // Add to terminal output
                addTerminalLine(message.type, message.data || {});
            } catch (error) {
                console.error('Error handling message:', error);
                console.error('Message:', message);
            }
            
            switch(message.type) {
                case 'pipeline_step':
                    handlePipelineStep(message.data);
                    break;
                case 'llm_call':
                    handleLLMCall(message.data);
                    break;
                case 'tool_execution':
                    handleToolExecution(message.data);
                    break;
                case 'reasoning_step':
                    handleReasoningStep(message.data);
                    break;
                case 'session_summary':
                    handleSessionSummary(message.data);
                    break;
            }
        }
        
        function handlePipelineStep(step) {
            // Check if this is the start of a new request (step_number = 1)
            if (step.step_number === 1 && sessionData.currentRequest.pipelineSteps.length > 0) {
                // End previous request and show summary
                endRequest();
            }
            
            // Track start time of new request
            if (step.step_number === 1) {
                sessionData.currentRequest.startTime = new Date();
                sessionData.currentRequest.pipelineSteps = [];
                sessionData.currentRequest.llmCalls = [];
                sessionData.currentRequest.toolExecs = [];
                sessionData.currentRequest.tokens = { input: 0, output: 0, total: 0 };
                sessionData.currentRequest.cost = { input: 0, output: 0, total: 0 };
            }
            
            sessionData.pipelineSteps.push(step);
            sessionData.currentRequest.pipelineSteps.push(step);
            updatePipelineSteps();
            updateStats();
        }
        
        function handleLLMCall(call) {
            sessionData.llmCalls.push(call);
            sessionData.totalTokens.input += call.tokens.input_tokens || 0;
            sessionData.totalTokens.output += call.tokens.output_tokens || 0;
            sessionData.totalTokens.total += call.tokens.total_tokens || 0;
            sessionData.totalCost.input += call.cost.input_cost_usd || 0;
            sessionData.totalCost.output += call.cost.output_cost_usd || 0;
            sessionData.totalCost.total += call.cost.total_cost_usd || 0;
            
            // Track for current request
            sessionData.currentRequest.llmCalls.push(call);
            sessionData.currentRequest.tokens.input += call.tokens.input_tokens || 0;
            sessionData.currentRequest.tokens.output += call.tokens.output_tokens || 0;
            sessionData.currentRequest.tokens.total += call.tokens.total_tokens || 0;
            sessionData.currentRequest.cost.input += call.cost.input_cost_usd || 0;
            sessionData.currentRequest.cost.output += call.cost.output_cost_usd || 0;
            sessionData.currentRequest.cost.total += call.cost.total_cost_usd || 0;
            
            updateLLMCalls();
            updateStats();
        }
        
        function handleToolExecution(tool) {
            sessionData.toolExecs.push(tool);
            sessionData.currentRequest.toolExecs.push(tool);
            updateToolExecs();
            updateStats();
        }
        
        function handleReasoningStep(step) {
            sessionData.reasoningSteps.push(step);
            
            // Extract memory updates if present
            if (step.memory_updates && Object.keys(step.memory_updates).length > 0) {
                const memoryUpdate = {
                    timestamp: new Date().toISOString(),
                    updates: step.memory_updates
                };
                sessionData.memoryUpdates.push(memoryUpdate);
                
                // Add to terminal output
                addTerminalLine('memory_update', memoryUpdate);
                
                updateMemory();
                updateStats();
            }
            
            updateReasoningChain();
        }
        
        function handleSessionSummary(summary) {
            sessionData.sessionId = summary.session_id;
            sessionData.startTime = summary.timestamp;
            updateSessionSummary(summary);
            updateStats();
            // End current request when summary is received
            endRequest();
        }
        
        function endRequest() {
            if (sessionData.currentRequest.pipelineSteps.length === 0) {
                return; // No request to end
            }
            
            const terminal = document.getElementById('terminalOutput');
            const separator = document.createElement('div');
            separator.className = 'terminal-line';
            separator.style.borderTop = '2px solid #3e3e42';
            separator.style.marginTop = '10px';
            separator.style.marginBottom = '10px';
            separator.style.paddingTop = '10px';
            
            const duration = sessionData.currentRequest.startTime ? 
                ((new Date() - sessionData.currentRequest.startTime) / 1000).toFixed(2) : '0.00';
            const durationMs = (parseFloat(duration) * 1000).toFixed(2);
            
            const reqDuration = sessionData.currentRequest.pipelineSteps.reduce((sum, s) => sum + (s.duration_ms || 0), 0);
            const reqDurationSec = (reqDuration / 1000).toFixed(2);
            separator.innerHTML = `
                <div style="color: #4ec9b0; font-weight: bold; margin-bottom: 8px; text-align: center;">
                    ════════════════════════════════════════════════════════════════════════════════
                </div>
                <div style="color: #ce9178; margin-bottom: 5px;">
                    <strong>Request Summary:</strong> Steps: ${sessionData.currentRequest.pipelineSteps.length} | 
                    LLM Calls: ${sessionData.currentRequest.llmCalls.length} | 
                    Tools: ${sessionData.currentRequest.toolExecs.length} | 
                    Duration: ${reqDuration.toFixed(2)}ms = ${reqDurationSec}s | 
                    Tokens: ${sessionData.currentRequest.tokens.total.toLocaleString()} | 
                    Cost: $${sessionData.currentRequest.cost.total.toFixed(6)}
                </div>
            `;
            terminal.appendChild(separator);
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        function updateStats() {
            document.getElementById('totalTokens').textContent = sessionData.totalTokens.total.toLocaleString();
            document.getElementById('tokenBreakdown').textContent = 
                `Input: ${sessionData.totalTokens.input.toLocaleString()} | Output: ${sessionData.totalTokens.output.toLocaleString()}`;
            
            document.getElementById('totalCost').textContent = `$${sessionData.totalCost.total.toFixed(6)}`;
            document.getElementById('costBreakdown').textContent = 
                `Input: $${sessionData.totalCost.input.toFixed(6)} | Output: $${sessionData.totalCost.output.toFixed(6)}`;
            
            document.getElementById('pipelineSteps').textContent = sessionData.pipelineSteps.length;
            const totalDuration = sessionData.pipelineSteps.reduce((sum, s) => sum + (s.duration_ms || 0), 0);
            const totalDurationSec = (totalDuration / 1000).toFixed(2);
            document.getElementById('pipelineDuration').textContent = `Duration: ${totalDuration.toFixed(2)}ms = ${totalDurationSec}s`;
            
            document.getElementById('llmCalls').textContent = sessionData.llmCalls.length;
            const llmTotalTokens = sessionData.llmCalls.reduce((sum, c) => sum + (c.tokens?.total_tokens || 0), 0);
            document.getElementById('llmTokens').textContent = `Total: ${llmTotalTokens.toLocaleString()} tokens`;
            
            document.getElementById('toolExecs').textContent = sessionData.toolExecs.length;
            const successCount = sessionData.toolExecs.filter(t => t.success).length;
            document.getElementById('toolSuccess').textContent = `Success: ${successCount}`;
            
            document.getElementById('memoryHistoryLength').textContent = sessionData.conversationHistoryLength || 0;
            document.getElementById('memoryUpdates').textContent = `Updates: ${sessionData.memoryUpdates.length}`;
            
            if (sessionData.sessionId) {
                document.getElementById('activeSession').textContent = sessionData.sessionId.substring(0, 20) + '...';
                if (sessionData.startTime) {
                    const elapsed = (new Date() - new Date(sessionData.startTime)) / 1000;
                    document.getElementById('sessionTime').textContent = `Elapsed: ${elapsed.toFixed(1)}s`;
                }
            }
        }
        
        function updatePipelineSteps() {
            const container = document.getElementById('pipelineStepsList');
            container.innerHTML = sessionData.pipelineSteps.slice(-20).reverse().map(step => `
                <div class="step">
                    <div class="step-header">
                        <div>
                            <span class="step-number">#${step.step_number}</span>
                            <span class="step-name">${step.step_name}</span>
                            <span class="step-component">[${step.component}]</span>
                        </div>
                        <div class="step-duration">${(step.duration_ms || 0).toFixed(2)}ms = ${((step.duration_ms || 0) / 1000).toFixed(2)}s</div>
                    </div>
                    ${step.error ? `<div class="error">Error: ${step.error}</div>` : ''}
                    ${step.metadata && Object.keys(step.metadata).length > 0 ? 
                        `<div class="step-details">${JSON.stringify(step.metadata, null, 2)}</div>` : ''}
                </div>
            `).join('');
        }
        
        function updateLLMCalls() {
            const container = document.getElementById('llmCallsList');
            container.innerHTML = sessionData.llmCalls.slice(-10).reverse().map(call => `
                <div class="llm-call">
                    <div class="llm-header">
                        <div>
                            <span class="llm-component">${call.component}</span>
                            <span class="llm-model">[${call.model}]</span>
                        </div>
                        <div class="llm-duration">${(call.duration_ms || 0).toFixed(2)}ms = ${((call.duration_ms || 0) / 1000).toFixed(2)}s</div>
                    </div>
                    <div class="llm-tokens">
                        Tokens: ${call.tokens?.input_tokens || 0} in / ${call.tokens?.output_tokens || 0} out 
                        (${call.tokens?.total_tokens || 0} total)
                    </div>
                    ${call.cost?.total_cost_usd > 0 ? 
                        `<div class="llm-cost">Cost: $${call.cost.total_cost_usd.toFixed(6)}</div>` : ''}
                </div>
            `).join('');
        }
        
        function updateToolExecs() {
            const container = document.getElementById('toolExecsList');
            container.innerHTML = sessionData.toolExecs.slice(-10).reverse().map(tool => `
                <div class="tool-exec">
                    <div class="tool-name">${tool.tool_name}</div>
                    <div class="tool-duration">${(tool.duration_ms || 0).toFixed(2)}ms = ${((tool.duration_ms || 0) / 1000).toFixed(2)}s</div>
                    ${tool.success ? 
                        `<div class="tool-success">✓ Success</div>` : 
                        `<div class="tool-error">✗ Error: ${tool.error || 'Unknown'}</div>`}
                </div>
            `).join('');
        }
        
        function updateReasoningChain() {
            const container = document.getElementById('reasoningChain');
            container.innerHTML = sessionData.reasoningSteps.slice(-15).reverse().map(step => `
                <div class="reasoning-step">
                    <div class="reasoning-desc">${step.step_number}. ${step.description}</div>
                </div>
            `).join('');
        }
        
        function updateMemory() {
            const container = document.getElementById('memoryList');
            if (sessionData.memoryUpdates.length === 0) {
                container.innerHTML = '<div style="color: #858585; padding: 10px;">No memory updates yet...</div>';
                return;
            }
            
            container.innerHTML = sessionData.memoryUpdates.slice(-10).reverse().map((update, idx) => {
                const updates = update.updates || {};
                const newFacts = updates.new_facts || {};
                const systemAction = updates.system_action || '';
                const awaiting = updates.awaiting || '';
                
                let content = '<div class="memory-update" style="background: #1e1e1e; padding: 10px; margin-bottom: 8px; border-radius: 3px; border-left: 3px solid #b5cea8; font-size: 11px;">';
                content += `<div style="color: #858585; font-size: 10px; margin-bottom: 5px;">${new Date(update.timestamp).toLocaleTimeString()}</div>`;
                
                if (Object.keys(newFacts).length > 0) {
                    content += `<div style="color: #4ec9b0; margin-bottom: 5px;"><strong>New Facts:</strong></div>`;
                    content += `<div style="color: #d4d4d4; margin-left: 10px;">${JSON.stringify(newFacts, null, 2)}</div>`;
                }
                
                if (systemAction) {
                    content += `<div style="color: #ce9178; margin-top: 5px;"><strong>System Action:</strong> ${systemAction}</div>`;
                }
                
                if (awaiting) {
                    content += `<div style="color: #569cd6; margin-top: 5px;"><strong>Awaiting:</strong> ${awaiting}</div>`;
                }
                
                content += '</div>';
                return content;
            }).join('');
        }
        
        function updateSessionSummary(summary) {
            const container = document.getElementById('sessionSummary');
            container.textContent = JSON.stringify(summary, null, 2);
            
            // Extract conversation history length from agent context if available
            if (summary.agent && summary.agent.conversation_history_length) {
                sessionData.conversationHistoryLength = summary.agent.conversation_history_length;
                updateStats();
            }
        }
        
        connect();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws/observability")
async def observability_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time observability streaming."""
    await websocket.accept()
    broadcaster = get_observability_broadcaster()
    broadcaster.add_connection(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to observability stream"
        })
        
        # Keep connection alive
        while True:
            # Wait for ping or just keep connection open
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except:
                break
                
    except Exception as e:
        logger.error(f"Observability WebSocket error: {e}")
    finally:
        broadcaster.remove_connection(websocket)


@app.get("/observability/status")
async def observability_status():
    """Diagnostic endpoint to check observability system status."""
    from patient_ai_service.core.observability_broadcaster import get_observability_broadcaster
    
    broadcaster = get_observability_broadcaster()
    
    return {
        "observability_enabled": settings.enable_observability,
        "cost_tracking_enabled": settings.cost_tracking_enabled,
        "output_format": settings.observability_output_format,
        "broadcaster_initialized": broadcaster is not None,
        "active_connections": len(broadcaster._connections) if broadcaster else 0,
        "status": "operational" if settings.enable_observability else "disabled"
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An error occurred"
        }
    )


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "patient_ai_service.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

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
from patient_ai_service.models.messages import ChatRequest, ChatResponse, WebSocketMessage
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

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

    Client sends:
    {
        "type": "message",
        "session_id": "user_123",
        "content": "Hello"
    }

    Server responds:
    {
        "type": "response",
        "session_id": "user_123",
        "content": "Hi! How can I help you?",
        "data": {...}
    }
    """
    await websocket.accept()
    session_id: str = None

    try:
        logger.info("WebSocket connection established")

        while True:
            # Receive message
            data = await websocket.receive_json()

            message_type = data.get("type")
            session_id = data.get("session_id")
            content = data.get("content")

            if not session_id:
                await websocket.send_json({
                    "type": "error",
                    "content": "session_id required"
                })
                continue

            # Store connection
            websocket_connections[session_id] = websocket

            if message_type == "message":
                # Send typing indicator
                await websocket.send_json({
                    "type": "typing",
                    "session_id": session_id,
                    "is_typing": True
                })

                # Process message
                response = await orchestrator.process_message(
                    session_id=session_id,
                    message=content,
                    language=data.get("language")
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

                # Stop typing
                await websocket.send_json({
                    "type": "typing",
                    "session_id": session_id,
                    "is_typing": False
                })

            elif message_type == "ping":
                # Heartbeat
                await websocket.send_json({
                    "type": "pong",
                    "session_id": session_id
                })

            else:
                logger.warning(f"Unknown message type: {message_type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        if session_id and session_id in websocket_connections:
            del websocket_connections[session_id]

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

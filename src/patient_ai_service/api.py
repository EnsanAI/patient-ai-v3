from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import time

from patient_ai_service.core.orchestrator import Orchestrator
from patient_ai_service.models.messages import ChatResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Patient AI Service", version="3.0")

# Initialize orchestrator
orchestrator = Orchestrator()

class MessageRequest(BaseModel):
    session_id: str
    message: str
    language: Optional[str] = "en"

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Patient AI Service...")
    await orchestrator.start()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Patient AI Service...")
    await orchestrator.stop()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "patient-ai", "version": "3.0"}

@app.post("/message", response_model=ChatResponse)
async def process_message(request: MessageRequest):
    """Process a user message and return the agent's response."""
    start_time = time.time()
    try:
        logger.info(f"Received message for session {request.session_id}")
        
        response = await orchestrator.process_message(
            session_id=request.session_id,
            message=request.message,
            language=request.language
        )
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"Processed message in {duration:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
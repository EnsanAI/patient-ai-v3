import uvicorn
import argparse
import os
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Patient AI Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Patient AI Service on {args.host}:{args.port}")
    
    # Run the application
    # Point explicitly to the server module to avoid ambiguity
    uvicorn.run("patient_ai_service.api.server:app", host=args.host, port=args.port, reload=True)
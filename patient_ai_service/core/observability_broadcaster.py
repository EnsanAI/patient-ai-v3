"""
Broadcaster for real-time observability events to WebSocket clients.
"""

import json
import logging
from typing import Set, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ObservabilityBroadcaster:
    """Manages WebSocket connections for real-time observability streaming."""
    
    def __init__(self):
        self._connections: Set[Any] = set()  # Set of WebSocket connections
    
    def add_connection(self, websocket: Any):
        """Add a WebSocket connection."""
        self._connections.add(websocket)
        logger.info(f"Observability connection added. Total connections: {len(self._connections)}")
    
    def remove_connection(self, websocket: Any):
        """Remove a WebSocket connection."""
        self._connections.discard(websocket)
        logger.info(f"Observability connection removed. Total connections: {len(self._connections)}")
    
    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Broadcast an event to all connected clients."""
        if not self._connections:
            logger.debug(f"No connections to broadcast {event_type} event")
            return
        
        message = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        logger.debug(f"Broadcasting {event_type} event to {len(self._connections)} connection(s)")
        
        # Send to all connections (remove disconnected ones)
        disconnected = set()
        for websocket in self._connections:
            try:
                await websocket.send_json(message)
                logger.debug(f"Successfully sent {event_type} event to websocket")
            except Exception as e:
                logger.warning(f"Error sending {event_type} event to websocket: {e}", exc_info=True)
                disconnected.add(websocket)
        
        # Remove disconnected connections
        for ws in disconnected:
            self._connections.discard(ws)
            logger.info(f"Removed disconnected websocket. Remaining connections: {len(self._connections)}")
    
    async def broadcast_pipeline_step(self, step: Dict[str, Any]):
        """Broadcast a pipeline step event."""
        await self.broadcast("pipeline_step", step)
    
    async def broadcast_llm_call(self, llm_call: Dict[str, Any]):
        """Broadcast an LLM call event."""
        await self.broadcast("llm_call", llm_call)
    
    async def broadcast_tool_execution(self, tool_exec: Dict[str, Any]):
        """Broadcast a tool execution event."""
        await self.broadcast("tool_execution", tool_exec)
    
    async def broadcast_reasoning_step(self, step: Dict[str, Any]):
        """Broadcast a reasoning step event."""
        await self.broadcast("reasoning_step", step)
    
    async def broadcast_session_summary(self, summary: Dict[str, Any]):
        """Broadcast a session summary."""
        await self.broadcast("session_summary", summary)
    
    async def broadcast_custom_metric(self, metric: Dict[str, Any]):
        """Broadcast a custom metric event."""
        await self.broadcast("custom_metric", metric)


# Global broadcaster instance
_observability_broadcaster: ObservabilityBroadcaster = None


def get_observability_broadcaster() -> ObservabilityBroadcaster:
    """Get the global observability broadcaster."""
    global _observability_broadcaster
    if _observability_broadcaster is None:
        _observability_broadcaster = ObservabilityBroadcaster()
    return _observability_broadcaster


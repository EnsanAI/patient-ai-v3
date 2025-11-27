"""
Message Broker for pub/sub architecture.

Provides asynchronous message routing between system components.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from patient_ai_service.models.messages import Message, Topics

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    Asynchronous message broker for pub/sub messaging.

    Allows components to publish and subscribe to topics for loosely
    coupled communication.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        logger.info("MessageBroker initialized")

    def subscribe(self, topic: str, handler: Callable):
        """
        Subscribe to a topic.

        Args:
            topic: Topic name to subscribe to
            handler: Async or sync function to handle messages
        """
        if handler not in self._subscribers[topic]:
            self._subscribers[topic].append(handler)
            logger.info(f"Subscribed to topic '{topic}': {handler.__name__}")

    def unsubscribe(self, topic: str, handler: Callable):
        """
        Unsubscribe from a topic.

        Args:
            topic: Topic name
            handler: Handler function to remove
        """
        if handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)
            logger.info(f"Unsubscribed from topic '{topic}': {handler.__name__}")

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        session_id: str,
        priority: int = 0
    ):
        """
        Publish a message to a topic.

        Args:
            topic: Topic name
            payload: Message data
            session_id: Session identifier
            priority: Message priority (0=normal, 1=high, 2=critical)
        """
        message = Message(
            topic=topic,
            payload=payload,
            session_id=session_id,
            priority=priority,
            timestamp=datetime.utcnow()
        )

        await self._message_queue.put(message)
        logger.debug(f"Published to '{topic}': session={session_id}, priority={priority}")

    async def start(self):
        """Start the message broker worker."""
        if self._running:
            logger.warning("MessageBroker already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._process_messages())
        logger.info("MessageBroker started")

    async def stop(self):
        """Stop the message broker worker."""
        if not self._running:
            return

        self._running = False

        # Wait for worker to finish
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("MessageBroker stopped")

    async def _process_messages(self):
        """Worker coroutine to process messages from queue."""
        logger.info("Message processing worker started")

        while self._running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                # Get subscribers for this topic
                handlers = self._subscribers.get(message.topic, [])

                if not handlers:
                    logger.warning(f"No subscribers for topic: {message.topic}")
                    continue

                # Call all handlers
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        logger.error(
                            f"Error in handler {handler.__name__} for topic "
                            f"'{message.topic}': {e}",
                            exc_info=True
                        )

            except asyncio.TimeoutError:
                # No message, continue
                continue
            except asyncio.CancelledError:
                logger.info("Message worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

        logger.info("Message processing worker stopped")

    def get_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers for a topic."""
        return len(self._subscribers.get(topic, []))

    def get_all_topics(self) -> List[str]:
        """Get list of all topics with subscribers."""
        return list(self._subscribers.keys())

    def clear_topic(self, topic: str):
        """Remove all subscribers from a topic."""
        if topic in self._subscribers:
            del self._subscribers[topic]
            logger.info(f"Cleared all subscribers from topic: {topic}")

    def clear_all(self):
        """Remove all subscribers from all topics."""
        self._subscribers.clear()
        logger.info("Cleared all subscribers")


# Global message broker instance
_message_broker: Optional[MessageBroker] = None


def get_message_broker() -> MessageBroker:
    """Get or create the global message broker instance."""
    global _message_broker
    if _message_broker is None:
        _message_broker = MessageBroker()
    return _message_broker


def reset_message_broker():
    """Reset the global message broker (useful for testing)."""
    global _message_broker
    if _message_broker and _message_broker._running:
        # Can't easily await here, so just log warning
        logger.warning("Resetting running message broker - call stop() first")
    _message_broker = None

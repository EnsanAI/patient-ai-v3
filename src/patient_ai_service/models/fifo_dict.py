"""
FIFO (First In, First Out) dictionary with automatic eviction.

When the dict exceeds max_size, the oldest entry is automatically removed.
This prevents unbounded entity growth in long conversations.
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class FIFODict(OrderedDict):
    """
    Dictionary with FIFO eviction when max_size is exceeded.

    Maintains insertion order and automatically evicts oldest entries
    when new items would exceed the maximum size.

    Example:
        >>> entities = FIFODict(max_size=3)
        >>> entities['a'] = 1
        >>> entities['b'] = 2
        >>> entities['c'] = 3
        >>> entities['d'] = 4  # 'a' is automatically evicted
        >>> 'a' in entities
        False
        >>> list(entities.keys())
        ['b', 'c', 'd']
    """

    def __init__(self, max_size: int = 8, *args, **kwargs):
        """
        Initialize FIFO dictionary.

        Args:
            max_size: Maximum number of items before eviction (default: 8)
        """
        self.max_size = max_size
        self._eviction_log: List[Tuple[str, Any, datetime]] = []
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Any):
        """Set item with FIFO eviction if needed."""

        # Check if this is an update (key exists) or new insertion
        is_update = key in self

        if is_update:
            # Update: Move to end (most recent) and update value
            self.move_to_end(key)
            super().__setitem__(key, value)
            logger.debug(f"Updated entity '{key}' (moved to end)")
        else:
            # New insertion: Check if we need to evict
            if len(self) >= self.max_size:
                # Evict oldest (first item in OrderedDict)
                oldest_key = next(iter(self))
                oldest_value = self[oldest_key]

                # Log eviction
                self._eviction_log.append((
                    oldest_key,
                    oldest_value,
                    datetime.utcnow()
                ))

                # Track eviction metrics
                try:
                    from patient_ai_service.core.observability import (
                        fifo_eviction_count,
                        entity_count_at_eviction
                    )
                    fifo_eviction_count.labels(evicted_key=oldest_key).inc()
                    entity_count_at_eviction.observe(len(self))
                except Exception:
                    pass  # Don't fail on metrics errors

                del self[oldest_key]

                logger.warning(
                    f"⚠️ FIFO evicted entity '{oldest_key}' = {oldest_value} "
                    f"(limit: {self.max_size}, new key: '{key}')"
                )

            # Add new item (goes to end)
            super().__setitem__(key, value)
            logger.debug(f"Added new entity '{key}' (count: {len(self)}/{self.max_size})")

    def get_eviction_log(self) -> List[Dict[str, Any]]:
        """
        Get log of evicted entities.

        Returns:
            List of dicts with 'key', 'value', 'evicted_at'
        """
        return [
            {
                "key": key,
                "value": value,
                "evicted_at": evicted_at
            }
            for key, value, evicted_at in self._eviction_log
        ]

    def clear_eviction_log(self):
        """Clear the eviction log."""
        self._eviction_log.clear()

    def get_oldest_key(self) -> Optional[str]:
        """Get the oldest (first) key without removing it."""
        return next(iter(self), None)

    def get_newest_key(self) -> Optional[str]:
        """Get the newest (last) key without removing it."""
        return next(reversed(self), None)

    def get_age_ordered_items(self) -> List[Tuple[str, Any]]:
        """
        Get items ordered by age (oldest first).

        Returns:
            List of (key, value) tuples ordered from oldest to newest
        """
        return list(self.items())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to regular dict (loses ordering)."""
        return dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_size: int = 8) -> 'FIFODict':
        """
        Create FIFODict from regular dict.

        Args:
            data: Dictionary to convert
            max_size: Maximum size for the FIFODict

        Returns:
            FIFODict with items from data

        Note:
            If data has more items than max_size, oldest items
            (arbitrary order from dict) will be evicted.
        """
        fifo = cls(max_size=max_size)
        for key, value in data.items():
            fifo[key] = value
        return fifo



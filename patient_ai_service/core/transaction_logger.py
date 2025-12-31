"""
Transaction Logger - Detailed logging for appointment manager and orchestrator.

Provides comprehensive logging for:
- Appointment Manager: inputs, outputs, tools, tokens, timing
- Orchestrator: inputs, outputs, agent communication, timing
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TransactionType(str, Enum):
    """Types of transactions to log."""
    ORCHESTRATOR_PROCESS = "orchestrator_process"
    AGENT_ACTIVATION = "agent_activation"
    AGENT_EXECUTION = "agent_execution"
    TOOL_EXECUTION = "tool_execution"
    LLM_CALL = "llm_call"
    AGENT_TRANSITION = "agent_transition"
    APPOINTMENT_BOOKING = "appointment_booking"
    APPOINTMENT_WORKFLOW = "appointment_workflow"


@dataclass
class TransactionMetrics:
    """Metrics for a transaction."""
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tokens_used: Dict[str, int] = field(default_factory=dict)  # {"input": 0, "output": 0}
    cost_usd: float = 0.0

    def complete(self):
        """Mark transaction as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6)
        }


@dataclass
class Transaction:
    """Represents a single transaction (operation) in the system."""
    transaction_id: str
    transaction_type: TransactionType
    session_id: str
    component: str  # e.g., "orchestrator", "appointment_manager", "reasoning_engine"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: TransactionMetrics = field(default_factory=lambda: TransactionMetrics(start_time=time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True
    sub_transactions: List['Transaction'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary for logging."""
        return {
            "transaction_id": self.transaction_id,
            "transaction_type": self.transaction_type,
            "session_id": self.session_id,
            "component": self.component,
            "inputs": self._sanitize_dict(self.inputs),
            "outputs": self._sanitize_dict(self.outputs),
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
            "error": self.error,
            "success": self.success,
            "sub_transactions": [st.to_dict() for st in self.sub_transactions]
        }

    def _sanitize_dict(self, d: Dict[str, Any], max_str_len: int = 500) -> Dict[str, Any]:
        """Sanitize dictionary for logging (truncate long strings, handle non-serializable)."""
        result = {}
        for key, value in d.items():
            if isinstance(value, str):
                result[key] = value[:max_str_len] + "..." if len(value) > max_str_len else value
            elif isinstance(value, (dict, list)):
                # For nested structures, convert to JSON string and truncate if needed
                try:
                    json_str = json.dumps(value, default=str)
                    if len(json_str) > max_str_len:
                        result[key] = json_str[:max_str_len] + "... (truncated)"
                    else:
                        result[key] = value
                except:
                    result[key] = str(value)[:max_str_len]
            elif isinstance(value, (int, float, bool, type(None))):
                result[key] = value
            else:
                result[key] = str(value)[:max_str_len]
        return result


class TransactionLogger:
    """
    Main transaction logger for detailed operation tracking.

    Usage:
        logger = TransactionLogger()

        # Start a transaction
        tx_id = logger.start_transaction(
            transaction_type=TransactionType.APPOINTMENT_BOOKING,
            session_id="session_123",
            component="appointment_manager",
            inputs={"patient_id": "...", "doctor_id": "..."}
        )

        # Record outputs and complete
        logger.complete_transaction(
            transaction_id=tx_id,
            outputs={"appointment_id": "..."},
            success=True
        )

        # Or use as context manager
        with logger.transaction(
            TransactionType.TOOL_EXECUTION,
            session_id="session_123",
            component="appointment_manager",
            inputs={"tool": "book_appointment"}
        ) as tx:
            # Do work
            result = do_work()
            tx.outputs = {"result": result}
    """

    def __init__(self, session_id: Optional[str] = None, enable_detailed_logging: bool = True):
        """
        Initialize transaction logger.

        Args:
            session_id: Optional session ID for all transactions
            enable_detailed_logging: Whether to enable detailed logging
        """
        self.session_id = session_id
        self.enable_detailed_logging = enable_detailed_logging
        self._transactions: Dict[str, Transaction] = {}
        self._current_transaction_stack: List[str] = []
        self._transaction_counter = 0

    def _generate_transaction_id(self, transaction_type: TransactionType) -> str:
        """Generate unique transaction ID."""
        self._transaction_counter += 1
        timestamp = int(time.time() * 1000)
        return f"{transaction_type}_{timestamp}_{self._transaction_counter}"

    def start_transaction(
        self,
        transaction_type: TransactionType,
        session_id: str,
        component: str,
        inputs: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Start a new transaction.

        Args:
            transaction_type: Type of transaction
            session_id: Session identifier
            component: Component name
            inputs: Input parameters
            metadata: Additional metadata

        Returns:
            Transaction ID
        """
        if not self.enable_detailed_logging:
            return None

        tx_id = self._generate_transaction_id(transaction_type)

        transaction = Transaction(
            transaction_id=tx_id,
            transaction_type=transaction_type,
            session_id=session_id or self.session_id,
            component=component,
            inputs=inputs or {},
            metadata=metadata or {}
        )

        self._transactions[tx_id] = transaction
        self._current_transaction_stack.append(tx_id)

        # Log transaction start
        logger.info(
            f"[TX_START] {transaction_type} | "
            f"tx_id={tx_id} | "
            f"session={session_id} | "
            f"component={component}"
        )

        return tx_id

    def complete_transaction(
        self,
        transaction_id: str,
        outputs: Dict[str, Any] = None,
        success: bool = True,
        error: Optional[str] = None,
        tokens: Optional[Dict[str, int]] = None,
        cost_usd: float = 0.0
    ):
        """
        Complete a transaction.

        Args:
            transaction_id: Transaction ID
            outputs: Output data
            success: Whether transaction succeeded
            error: Error message if failed
            tokens: Token usage dict ({"input": X, "output": Y})
            cost_usd: Cost in USD
        """
        if not self.enable_detailed_logging or not transaction_id:
            return

        if transaction_id not in self._transactions:
            logger.warning(f"Transaction {transaction_id} not found")
            return

        tx = self._transactions[transaction_id]
        tx.outputs = outputs or {}
        tx.success = success
        tx.error = error
        tx.metrics.complete()

        if tokens:
            tx.metrics.tokens_used = tokens
        if cost_usd:
            tx.metrics.cost_usd = cost_usd

        # Remove from stack
        if transaction_id in self._current_transaction_stack:
            self._current_transaction_stack.remove(transaction_id)

        # Log transaction complete
        logger.info(
            f"[TX_COMPLETE] {tx.transaction_type} | "
            f"tx_id={transaction_id} | "
            f"duration={tx.metrics.duration_ms:.2f}ms | "
            f"success={success} | "
            f"tokens={tokens} | "
            f"cost=${cost_usd:.6f}"
        )

        # If there's a parent transaction, add this as sub-transaction
        if self._current_transaction_stack:
            parent_id = self._current_transaction_stack[-1]
            parent_tx = self._transactions.get(parent_id)
            if parent_tx:
                parent_tx.sub_transactions.append(tx)

    @contextmanager
    def transaction(
        self,
        transaction_type: TransactionType,
        session_id: str,
        component: str,
        inputs: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Context manager for transactions.

        Usage:
            with logger.transaction(
                TransactionType.TOOL_EXECUTION,
                session_id="session_123",
                component="appointment_manager",
                inputs={"tool": "book_appointment"}
            ) as tx:
                result = do_work()
                tx.outputs = {"result": result}
                tx.metrics.tokens_used = {"input": 100, "output": 50}
        """
        tx_id = self.start_transaction(
            transaction_type=transaction_type,
            session_id=session_id,
            component=component,
            inputs=inputs,
            metadata=metadata
        )

        tx = self._transactions.get(tx_id) if tx_id else None
        error = None

        try:
            yield tx
        except Exception as e:
            error = str(e)
            if tx:
                tx.success = False
                tx.error = error
            raise
        finally:
            if tx_id:
                self.complete_transaction(
                    transaction_id=tx_id,
                    outputs=tx.outputs if tx else {},
                    success=tx.success if tx else False,
                    error=error,
                    tokens=tx.metrics.tokens_used if tx else None,
                    cost_usd=tx.metrics.cost_usd if tx else 0.0
                )

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        return self._transactions.get(transaction_id)

    def get_all_transactions(self) -> List[Transaction]:
        """Get all transactions."""
        return list(self._transactions.values())

    def get_root_transactions(self) -> List[Transaction]:
        """Get only root-level transactions (not sub-transactions)."""
        # Find transactions that are not in any other transaction's sub_transactions
        all_sub_tx_ids = set()
        for tx in self._transactions.values():
            all_sub_tx_ids.update(st.transaction_id for st in tx.sub_transactions)

        return [tx for tx in self._transactions.values() if tx.transaction_id not in all_sub_tx_ids]

    def log_summary(self):
        """Log summary of all transactions."""
        if not self.enable_detailed_logging:
            return

        root_transactions = self.get_root_transactions()

        logger.info("=" * 80)
        logger.info("TRANSACTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Session: {self.session_id}")
        logger.info(f"Total Transactions: {len(self._transactions)}")
        logger.info(f"Root Transactions: {len(root_transactions)}")
        logger.info("")

        total_duration = 0
        total_tokens_input = 0
        total_tokens_output = 0
        total_cost = 0.0

        for tx in root_transactions:
            if tx.metrics.duration_ms:
                total_duration += tx.metrics.duration_ms
            if tx.metrics.tokens_used:
                total_tokens_input += tx.metrics.tokens_used.get("input", 0)
                total_tokens_output += tx.metrics.tokens_used.get("output", 0)
            total_cost += tx.metrics.cost_usd

            self._log_transaction_tree(tx, indent=0)

        logger.info("")
        logger.info("TOTALS:")
        logger.info(f"  Total Duration: {total_duration:.2f}ms")
        logger.info(f"  Total Tokens: {total_tokens_input + total_tokens_output} "
                   f"(Input: {total_tokens_input}, Output: {total_tokens_output})")
        logger.info(f"  Total Cost: ${total_cost:.6f}")
        logger.info("=" * 80)

    def _log_transaction_tree(self, tx: Transaction, indent: int = 0):
        """Recursively log transaction tree."""
        prefix = "  " * indent
        status = "✓" if tx.success else "✗"

        logger.info(
            f"{prefix}{status} [{tx.transaction_type}] {tx.component} - "
            f"{tx.metrics.duration_ms:.2f}ms"
        )

        if tx.metrics.tokens_used:
            tokens_total = sum(tx.metrics.tokens_used.values())
            logger.info(f"{prefix}  Tokens: {tokens_total}")

        if tx.error:
            logger.info(f"{prefix}  Error: {tx.error}")

        # Log sub-transactions
        for sub_tx in tx.sub_transactions:
            self._log_transaction_tree(sub_tx, indent + 1)

    def export_json(self) -> str:
        """Export all transactions as JSON."""
        root_transactions = self.get_root_transactions()
        data = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "transactions": [tx.to_dict() for tx in root_transactions]
        }
        return json.dumps(data, indent=2, default=str)

    def clear(self):
        """Clear all transactions."""
        self._transactions.clear()
        self._current_transaction_stack.clear()
        self._transaction_counter = 0


# Global transaction loggers per session
_session_transaction_loggers: Dict[str, TransactionLogger] = {}


def get_transaction_logger(session_id: str, enable_detailed_logging: bool = True) -> TransactionLogger:
    """
    Get or create transaction logger for a session.

    Args:
        session_id: Session identifier
        enable_detailed_logging: Whether to enable detailed logging

    Returns:
        TransactionLogger instance
    """
    if session_id not in _session_transaction_loggers:
        _session_transaction_loggers[session_id] = TransactionLogger(
            session_id=session_id,
            enable_detailed_logging=enable_detailed_logging
        )
    return _session_transaction_loggers[session_id]


def clear_transaction_logger(session_id: str):
    """Clear transaction logger for a session."""
    if session_id in _session_transaction_loggers:
        del _session_transaction_loggers[session_id]


# Helper decorators for easy integration

def log_transaction(
    transaction_type: TransactionType,
    component: str,
    include_args: bool = True,
    include_result: bool = True
):
    """
    Decorator to automatically log function as a transaction.

    Args:
        transaction_type: Type of transaction
        component: Component name
        include_args: Whether to include function arguments in inputs
        include_result: Whether to include return value in outputs

    Usage:
        @log_transaction(TransactionType.TOOL_EXECUTION, "appointment_manager")
        def book_appointment(session_id: str, patient_id: str, ...):
            # Function implementation
            return result
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Extract session_id from kwargs or args
            session_id = kwargs.get('session_id') or (args[1] if len(args) > 1 else None)

            if not session_id:
                # If no session_id, just call function normally
                return func(*args, **kwargs)

            tx_logger = get_transaction_logger(session_id)

            # Prepare inputs
            inputs = {}
            if include_args:
                inputs = {
                    "args": str(args)[:200],
                    "kwargs": {k: str(v)[:200] for k, v in kwargs.items()}
                }

            with tx_logger.transaction(
                transaction_type=transaction_type,
                session_id=session_id,
                component=component,
                inputs=inputs
            ) as tx:
                result = func(*args, **kwargs)

                if include_result and tx:
                    tx.outputs = {"result": str(result)[:500]}

                return result

        return wrapper
    return decorator

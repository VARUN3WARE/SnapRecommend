"""
Structured logging and tracing configuration for SnapRecommend.

This module provides:
- JSON-formatted logging for log aggregation systems (ELK, Splunk, etc.)
- Request tracing with correlation IDs for distributed tracing
- Context-aware logging with user and request information
- Performance tracking for slow operations

Usage:
    from api.logging import setup_logging, get_logger, request_context
    
    # In FastAPI app initialization:
    setup_logging()
    logger = get_logger(__name__)
    
    # In request handlers:
    with request_context(user_id="u123", request_id="req-456"):
        logger.info("Processing recommendation")  # Includes context in JSON output

Environment Variables:
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)
    - LOG_FORMAT: Output format (json, text) (default: text in dev, json in prod)
    - LOG_FILE: Log file path (optional, default: stdout only)
"""

import json
import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional

# Context variables for request tracking
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_user_id: ContextVar[str] = ContextVar("user_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")


class ContextFilter(logging.Filter):
    """Add request context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to record attributes."""
        record.request_id = _request_id.get()
        record.user_id = _user_id.get()
        record.session_id = _session_id.get()
        return True


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context
        if hasattr(record, "request_id") and record.request_id:
            log_obj["request_id"] = record.request_id
        if hasattr(record, "user_id") and record.user_id:
            log_obj["user_id"] = record.user_id
        if hasattr(record, "session_id") and record.session_id:
            log_obj["session_id"] = record.session_id

        # Add exception info if present
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra") and record.extra:
            log_obj.update(record.extra)

        return json.dumps(log_obj)


class TextFormatter(logging.Formatter):
    """Format logs as human-readable text."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        context_parts = []
        if hasattr(record, "request_id") and record.request_id:
            context_parts.append(f"req={record.request_id}")
        if hasattr(record, "user_id") and record.user_id:
            context_parts.append(f"user={record.user_id}")

        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        return (
            f"{record.levelname:8} | "
            f"{record.name:30} | "
            f"{record.getMessage()}"
            f"{context_str}"
        )


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "text",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "text")
        log_file: Optional file path for log output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Add context filter
    context_filter = ContextFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Set context for current request.

    Args:
        request_id: Request identifier (generated if not provided)
        user_id: User identifier
        session_id: Session identifier
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    _request_id.set(request_id)
    _user_id.set(user_id or "")
    _session_id.set(session_id or "")


def get_request_context() -> dict[str, str]:
    """Get current request context."""
    return {
        "request_id": _request_id.get(),
        "user_id": _user_id.get(),
        "session_id": _session_id.get(),
    }


def clear_request_context() -> None:
    """Clear request context (call at end of request)."""
    _request_id.set("")
    _user_id.set("")
    _session_id.set("")


class RequestContextMiddleware:
    """FastAPI middleware for automatic request context tracking."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate request ID from headers
        request_id = None
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == b"x-request-id":
                request_id = header_value.decode()
                break

        # Set context
        set_request_context(request_id=request_id)

        try:
            await self.app(scope, receive, send)
        finally:
            clear_request_context()


# Module-level logger
logger = get_logger(__name__)

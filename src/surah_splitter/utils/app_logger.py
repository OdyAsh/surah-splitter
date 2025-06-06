"""
Application-wide logging utility based on Loguru.

This module provides a simple, configurable logger for the Surah Splitter project.
It uses Loguru for rich, structured logging with minimal setup but powerful features.
"""

from pathlib import Path
import sys
from typing import Optional, Union
from loguru import logger


def setup_logger(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "1 week",
    format_string: Optional[str] = None,
) -> None:
    """
    Setup the application logger with the specified configuration.

    Args:
        log_file: Path to the log file. If None, logs are only sent to stderr.
        log_level: Minimum log level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate the log file (e.g., "10 MB", "1 day")
        retention: How long to keep log files (e.g., "1 week", "1 month")
        format_string: Custom format string for log messages. If None, use default format.

    Returns:
        None
    """
    # Default format - production ready with timestamps and structured info
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<blue>{file.name}</blue>:<cyan>{line}</cyan> <blue>[{function}()]</blue> | "
            "<level>{message}</level>"
        )

    # Remove default logger and add stderr with custom format
    logger.remove()
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
    )

    # Add file logger if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            enqueue=True,  # Thread-safe logging
            backtrace=True,  # Better exception logging
            diagnose=True,  # Exception diagnostics
        )

    logger.info(f"Logger initialized with level {log_level}")


# Create a context manager to track timing of operations
class TimingContext:
    """Context manager for timing operations and logging the duration."""

    def __init__(self, operation_name: str, level: str = "DEBUG"):
        """
        Initialize a new timing context.

        Args:
            operation_name: Name of the operation being timed
            level: Log level to use when logging the timing (default: INFO)
        """
        self.operation_name = operation_name
        self.level = level

    def __enter__(self):
        """Start the timer when entering the context."""
        import time

        self.start_time = time.time()
        logger.log(self.level, f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the elapsed time when exiting the context."""
        import time

        elapsed = time.time() - self.start_time
        if exc_type:
            logger.error(f"Operation {self.operation_name} failed after {elapsed:.2f}s: {exc_val}")
        else:
            logger.log(self.level, f"Completed: {self.operation_name} in {elapsed:.2f}s")


# Set up a basic default logger to stderr
setup_logger()

# Export the main logger instance and timing context
__all__ = ["logger", "setup_logger", "TimingContext"]

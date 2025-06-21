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
    Configure the logger for the application.

    Args:
        log_file: Path to log file. If None, logs only go to stderr.
        log_level: Minimum level for logging messages.
        rotation: When to rotate the log file (e.g., "10 MB", "1 day")
        retention: How long to keep log files (e.g., "1 week", "30 days")
        format_string: Format string for log messages. If None, a default format is used.

    Returns:
        None
    """
    # First, remove any existing handlers
    logger.remove()

    # Default format string for structured yet readable logs
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<blue>{file.name}</blue>:<cyan>{line}</cyan> <magenta>[{function}()]</magenta> | "
            "<level>{message}</level>"
        )

    # Add stderr handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=format_string,
        colorize=True,
    )

    # Add file handler if log_file is provided
    if log_file:
        # Convert to Path if it's a string
        if isinstance(log_file, str):
            log_file = Path(log_file)

        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_file),
            level=log_level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8",
        )


# Create a context manager to track timing of operations
class LoggerTimingContext:
    """Context manager for timing operations and logging the duration."""

    def __init__(
        self, operation_name: str, level: str = "DEBUG", succ_when_complete: bool = False, show_started_log: bool = False
    ):
        """
        Initialize a new timing context.

        Args:
            operation_name: Name of the operation being timed
            level: Log level to use when logging the timing (default: INFO)
        """
        self.operation_name = operation_name
        self.level = level
        self.succ_when_complete = succ_when_complete
        self.show_started_log = show_started_log

    def __enter__(self):
        """Start the timer when entering the context."""
        import time

        self.start_time = time.time()
        if self.show_started_log:
            logger.log(self.level, f'Started: "{self.operation_name}"')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the elapsed time when exiting the context."""
        import time

        elapsed = time.time() - self.start_time

        # Format elapsed time as seconds, minutes, or hours
        if elapsed < 60:  # Less than a minute
            time_str = f"{elapsed:.2f}s"
        elif elapsed < 3600:  # Less than an hour
            time_str = f"{elapsed / 60:.2f}m"
        else:  # Hours or more
            time_str = f"{elapsed / 3600:.2f}h"

        if exc_type:
            logger.error(f'Operation "{self.operation_name}" failed after {time_str}: {exc_val}')
        else:
            self.level = "SUCCESS" if self.succ_when_complete else self.level
            # TODO soon: <magenta>in {time_str}</magenta>
            logger.log(self.level, f'Completed: "{self.operation_name}" in {time_str}')


# Initialize the logger with default settings at import time
# This can be reconfigured later using setup_logger() if needed
setup_logger()

# Usage examples in the project:
# logger.debug("Detailed debug information")
# logger.info("Normal processing events")
# logger.success("Operation completed successfully")
# logger.warning("Something might cause issues")
# logger.error("Something failed but execution continues")
# logger.critical("Application is about to crash or has a significant issue")
# logger.exception("Log an exception with traceback")

# This allows importing the logger directly from this module
__all__ = ["logger", "setup_logger"]

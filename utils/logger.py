"""
Logger module with timestamps for the image processing pipeline.
"""

import logging
import sys
from datetime import datetime


class TimestampFormatter(logging.Formatter):
    """Custom formatter that adds timestamps to log messages."""

    def format(self, record):
        # Add timestamp to the log record
        # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp = datetime.now().strftime('%H:%M:%S')
        record.timestamp = timestamp
        return super().format(record)


def setup_logger(name: str = 'paint_by_numbers') -> logging.Logger:
    """
    Set up and return a logger with timestamp formatting.

    Args:
        name: Name for the logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = TimestampFormatter(
            fmt='[%(timestamp)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


# Create default logger instance
logger = setup_logger()

import logging
from typing import (
    Optional,
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
DEFAULT_LOG_LEVEL_NAME = (
    logging.getLevelName(logging.INFO)
)
DEFAULT_LOG_LEVEL = getattr(logging, DEFAULT_LOG_LEVEL_NAME)


def get_log_level_name(level: int) -> str:
    """Get log level string from log level integer
    """
    return logging.getLevelName(level)


def is_valid_log_level_name(name: str) -> bool:
    """Check if the log level name is valid
    Args:
        name: log level name
    Returns: bool
    """
    return hasattr(logging, name)


def get_log_level(name: str) -> int:
    """Get log level integer from log level name
    Args:
        name: log level name
    Returns: logging level integer
    """
    assert is_valid_log_level_name(name), f"Invalid log level name {name}."
    return getattr(logging, name)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Logger instance factory method
    See https://docs.python.org/2/howto/logging.html#logging-advanced-tutorial
    The logger name should follow the package/module hierarchy

    Args:
        name: logger name following the package/module hierarchy
        level: optional log level
    Returns:
        logger instance
    """
    _logger = logging.getLogger(name=name)
    if level:
        _logger.setLevel(level)
    else:
        _logger.setLevel(DEFAULT_LOG_LEVEL)
    return _logger

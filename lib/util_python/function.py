"""Module for Python function utilities"""
from functools import (
    wraps
)
import logging
import random
import time
from typing import (
    Callable
)

from util_logging import (
    get_logger
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def retry_with_exponential_backoff(
        proactive_delay: float = 0.0,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple = (Exception,)
) -> Callable:
    """Retry a function with exponential backoff.
    See https://pypi.org/project/backoff/ for PyPi module as an alternative.
    Usage:

    @retry_with_exponential_backoff(args...)
    def func_to_retry():

    Args:
        proactive_delay: delay before calling the function
        initial_delay: initial time in seconds to wait before retry
        exponential_base:
        jitter:
        max_retries: number of retries
        errors: errors for which attempt the retry

    Return a decorator.
    """
    def decorator_factory(func: Callable) -> Callable:
        @wraps(func)
        def decorator(*args, **kwargs):
            num_retries: int = 0
            delay: float = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    time.sleep(proactive_delay)
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors as error:
                    msg: str = f"function {func.__name__}() failed due to [{error}] "

                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        msg += f"and maximum number of retries ({max_retries}) exceeded."
                        _logger.error("%s", msg)
                        raise RuntimeError(msg)

                    delay *= exponential_base * (1 + jitter * random.random())
                    msg += f"and retry in {delay} seconds."
                    _logger.error(msg)
                    time.sleep(delay)

        return decorator

    return decorator_factory

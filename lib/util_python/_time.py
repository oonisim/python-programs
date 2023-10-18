"""Module for time utility"""
import logging
import functools
import time


logger: logging.Logger = logging.getLogger(__name__)


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        print("-" * 80)
        print(f"calling {func.__name__}()...", flush=True)
        start_time = time.perf_counter()
        value = func(*args, **kwargs)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"{func.__name__!r} took {run_time:.4f} secs")
        print("-" * 80, flush=True)

        return value
    return wrapper_timer

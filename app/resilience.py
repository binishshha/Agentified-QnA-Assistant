import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def retry_with_backoff(attempts: int = 3, base_delay: float = 0.5, max_wait: float = 10.0):
    """Retry transient provider failures with exponential backoff and jitter."""
    def decorator(function: Callable[..., T]) -> Callable[..., T]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None
            for attempt in range(max(1, attempts)):
                try:
                    return function(*args, **kwargs)
                except Exception as error:
                    last_error = error
                    if attempt == attempts - 1:
                        raise
                    delay = min(max_wait, base_delay * (2**attempt) + random.uniform(0, 0.25))
                    time.sleep(delay)
            raise last_error or RuntimeError("retry failed")
        return wrapped
    return decorator

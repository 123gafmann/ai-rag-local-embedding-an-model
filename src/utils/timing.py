import time
import logging
from contextlib import contextmanager


@contextmanager
def log_duration(logger: logging.Logger, label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{label} took {elapsed:.2f}s")

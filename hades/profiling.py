import time
import logging
import functools

logger = logging.getLogger(__name__)


try:
    perf_counter = time.perf_counter
except AttributeError:
    # python 2
    perf_counter = time.time


class tictoc(object):
    """Instrument and log function execution duration."""

    def __init__(self, name=None, loglevel=logging.DEBUG):
        self.name = name
        self.loglevel = loglevel

    def __call__(self, fn):
        if self.name is None:
            self.name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            logger.log(self.loglevel, "tic(%r)", self.name, extra={"tic": self.name})
            try:
                return fn(*args, **kwargs)
            finally:
                end = perf_counter()
                dt = end - start
                logger.log(self.loglevel, "toc(%r) = %r", self.name, dt,
                           extra={"toc": self.name, "duration": dt})

        return wrapper

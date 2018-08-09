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

    def start(self):
        self.tick = perf_counter()
        logger.log(self.loglevel, "tic(%r)", self.name, extra={"tic": self.name})

    def stop(self):
        dt = perf_counter() - self.tick
        logger.log(self.loglevel, "toc(%r) = %r", self.name, dt,
                   extra={"toc": self.name, "duration": dt})

    def __init__(self, name=None, loglevel=logging.DEBUG):
        self.name = name
        self.loglevel = loglevel

    def __call__(self, fn):
        if self.name is None:
            self.name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            self.start()
            try:
                return fn(*args, **kwargs)
            finally:
                self.stop()

        return wrapper

    def __enter__(self, name=None, loglevel=logging.DEBUG):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

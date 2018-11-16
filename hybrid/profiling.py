# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import logging
import functools
from contextlib import contextmanager

logger = logging.getLogger(__name__)


try:
    perf_counter = time.perf_counter
except AttributeError:
    # python 2
    perf_counter = time.time


class tictoc(object):
    """Instrument and log function execution duration.

    Examples:
        @tictoc('function', loglevel=logging.INFO)
        def f(x, y):
            a = x * y
            with tictoc('block'):
                return [a * a for _ in range(x)]

        class Example(object):
            @tictoc()
            def method(self, args):
                # ...
    """

    def start(self):
        self.tick = perf_counter()
        logger.log(self.loglevel, "tic(%r)", self.name, extra={"tic": self.name})

    def stop(self):
        self.dt = perf_counter() - self.tick
        logger.log(self.loglevel, "toc(%r) = %r", self.name, self.dt,
                   extra={"toc": self.name, "duration": self.dt})

    def __init__(self, name=None, loglevel=logging.DEBUG):
        if name is None:
            # TODO: use file/lineno
            pass
        self.name = name

        if loglevel is None:
            loglevel = logging.NOTSET
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class trace(tictoc):
    """`tictoc` with TRACE loglevel default."""

    def __init__(self, name=None, loglevel=logging.TRACE):
        super(trace, self).__init__(name, loglevel)


def make_count(counters):
    """Generate counter increment context manager specialized for handling
    counters in `counters` dictionary.

    Args:
        counters (dict): Counters storage.

    Example:
        counters = {}
        count = make_count(counters)
        with count('first'):
            f()
        # counters['first'] is now a list holding f's runtime
    """

    @contextmanager
    def _counted_mgr(counter_name):
        with tictoc(name=None, loglevel=None) as timer:
            try:
                yield None
            finally:
                pass
        counters.setdefault(counter_name, []).append(timer.dt)

    return _counted_mgr


def iter_inorder(runnable):
    """Inorder DFS traversal of `runnable`, as an iterator."""
    yield runnable
    for child in runnable:
        for node in iter_inorder(child):
            yield node


def walk_inorder(runnable, visit, level=0):
    """Inorder DFS traversal of `runnable`, as a callback `visit`."""
    visit(runnable, level)
    for child in runnable:
        walk_inorder(child, visit, level+1)


def pprint(runnable, indent=2):
    """Pretty print `runnable` tree with `indent` spaces level indentation."""
    walk_inorder(runnable, lambda r, d: print(" "*indent*d, r.name, sep=''))

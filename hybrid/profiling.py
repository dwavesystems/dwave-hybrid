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

from __future__ import print_function

import logging
import functools
from time import perf_counter

__all__ = ['perf_counter', 'tictoc', 'trace', 'print_structure', 'print_counters']

logger = logging.getLogger(__name__)


class tictoc(object):
    """Instrument and log function execution duration.

    Examples::

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


class make_count(object):
    """Generate counter increment callable object specialized for handling
    (bound to) counters in the provided `counters` dictionary.

    Args:
        counters (dict): Counters storage.

    Example::

        counters = {}
        count = make_count(counters)

        for _ in range(10):
            count('f')

        # counters['f'] == 10
    """

    def __init__(self, counters, prefix=None, loglevel=logging.NOTSET):
        self.counters = counters
        self.prefix = prefix
        self.loglevel = loglevel

    def __call__(self, counter_name, inc=1):
        self.counters[counter_name] = self.counters.get(counter_name, 0) + inc
        val = self.counters[counter_name]

        prefixed_name = '.'.join([self.prefix or '', counter_name])
        logger.log(self.loglevel, "counter(%r) -> %r", prefixed_name, val,
                   extra={"counter": prefixed_name, "value": val})


class make_timeit(object):
    """Generate timer increment context manager specialized for handling
    (bound to) timers in the provided `timers` dictionary.

    Args:
        timers (dict): Timers storage.

    Example::

        timers = {}
        timeit = make_timeit(timers)

        for _ in range(10):
            with timeit('f'):
                f()

        # timers['f'] is now a list holding 10 runtimes of `f`
    """

    class _local_timer_mgr(object):
        def __init__(self, timers, timer_name, prefix=None, loglevel=None):
            self.timers = timers
            self.timer_name = timer_name
            prefixed_name = '.'.join([prefix or '', timer_name])
            self.timer = tictoc(name=prefixed_name, loglevel=loglevel)

        def __enter__(self):
            self.timer.start()
            return self.timer

        def __exit__(self, exc_type, exc_value, traceback):
            self.timer.stop()
            self.timers.setdefault(self.timer_name, []).append(self.timer.dt)

    def __init__(self, timers, prefix=None, loglevel=None):
        self.timers = timers
        self.prefix = prefix
        self.loglevel = loglevel

    def __call__(self, timer_name):
        return make_timeit._local_timer_mgr(
            self.timers, timer_name, self.prefix, self.loglevel)


def iter_inorder(runnable):
    """Inorder DFS traversal of `runnable`, as an iterator."""
    yield runnable
    for child in runnable:
        for node in iter_inorder(child):
            yield node


def walk_inorder(runnable, visit, level=0):     # pragma: no cover
    """Inorder DFS traversal of `runnable`, as a callback `visit`."""
    visit(runnable, level)
    for child in runnable:
        walk_inorder(child, visit, level+1)


def print_structure(runnable, indent=2):
    """Pretty print `runnable` tree with `indent` spaces level indentation."""
    walk_inorder(runnable, lambda r, d: print(" "*indent*d, r.name, sep=''))


def print_counters(runnable, indent=4):         # pragma: no cover
    """Pretty print timers and counters, recursively starting with `runnable`."""

    def visit(runnable, level):
        tab = " " * indent * level
        print(tab, "* ", runnable.name, sep='')

        if runnable.timers:
            print(tab, "  (timers)", sep='')
            for timer, val in sorted(runnable.timers.items()):
                line = ("{tab}  - {timer!r}: cnt = {cnt}, cumtime = {cumtime:.3f} s, "
                        "avgtime = {avgtime:.3f} s").format(
                            tab=tab, timer=timer, cnt=len(val), cumtime=sum(val),
                            avgtime=sum(val)/len(val))
                print(line, sep='')

        if runnable.counters:
            print(tab, "  (counters)", sep='')
            for counter, val in sorted(runnable.counters.items()):
                line = "{tab}  - {counter!r}: {val}".format(
                    tab=tab, counter=counter, val=val)
                print(line, sep='')

        print()

    walk_inorder(runnable, visit)

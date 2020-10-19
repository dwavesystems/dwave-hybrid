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

import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, Executor

from hybrid.utils import cpu_count

__all__ = [
    'immediate_executor', 'thread_executor'
]

logger = logging.getLogger(__name__)


class Present(Future):
    """Already resolved :class:`~concurrent.futures.Future` object.

    Users should treat this class as just another
    :class:`~concurrent.futures.Future`, the difference being an implementation
    detail: :class:`Present` is "resolved" at construction time.

    See the example of the :meth:`~hybrid.core.Runnable.run` method.
    """

    def __init__(self, result=None, exception=None):
        super(Present, self).__init__()
        if result is not None:
            self.set_result(result)
        elif exception is not None:
            self.set_exception(exception)
        else:
            raise ValueError("can't provide both 'result' and 'exception'")


class ImmediateExecutor(Executor):

    def submit(self, fn, *args, **kwargs):
        """Blocking version of `Executor.submit()`. Returns a resolved
        `Future`.
        """

        try:
            return Present(result=fn(*args, **kwargs))
        except Exception as exc:
            return Present(exception=exc)


immediate_executor = ImmediateExecutor()
thread_executor = ThreadPoolExecutor(max_workers=cpu_count() * 5)

# make `process_executor` optional, since multiprocessing features might not
# be available in some environments (like AWS Lambda)
try:
    process_executor = ProcessPoolExecutor(max_workers=cpu_count())
except:
    pass

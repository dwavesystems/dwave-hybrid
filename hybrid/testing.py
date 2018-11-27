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

"""Testing utils."""

import os
import contextlib

__all__ = ['mock', 'isolated_environ']


# py2/3 mock support
try:
    # python 3
    from unittest import mock

except ImportError:  # pragma: no cover
    # python 2
    import mock


@contextlib.contextmanager
def isolated_environ(add=None, remove=None, remove_dwave=False, empty=False):
    """Context manager for modified process environment isolation.

    Environment variables can be updated, added and removed. Complete
    environment can be cleared, or cleared only of a subset of variables
    that affect config/loading (``DWAVE_*`` and ``DW_INTERNAL__*`` vars).

    On context clear, original `os.environ` is restored.

    Args:
        add (dict/Mapping):
            Values to add (or update) in the isolated `os.environ`.

        remove (dict/Mapping, or set/Iterable):
            Values to remove in the isolated `os.environ`.

        remove_dwave (bool, default=False):
            Remove dwave tools' specific variables that affect config and
            loading (prefixed with ``DWAVE_`` or ``DW_INTERNAL__``)

        empty (bool, default=False):
            Return empty environment.

    Context:
        Modified copy of global `os.environ`. Restored on context exit.
    """

    if add is None:
        add = {}

    if remove is None:
        remove = {}

    with mock.patch.dict(os.environ, values=add, clear=empty):
        for key in remove:
            os.environ.pop(key, None)

        for key in frozenset(os.environ.keys()):
            if remove_dwave and (key.startswith("DWAVE_") or key.startswith("DW_INTERNAL__")):
                os.environ.pop(key, None)

        yield os.environ

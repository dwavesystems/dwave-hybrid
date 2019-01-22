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


class RunnableError(Exception):
    """Generic Runnable exception error that includes the error context, in
    particular, the `State` that caused the runnable component to fail."""

    def __init__(self, message, state):
        super(RunnableError, self).__init__(message)
        self.state = state


class InvalidStateError(Exception):
    """General state error."""


class StateTraitMissingError(InvalidStateError):
    """State missing a trait."""


class StateDimensionalityError(InvalidStateError):
    """Single state expected instead of a state sequence, or vice versa."""


class EndOfStream(StopIteration):
    """Signals end of stream for streaming runnables."""

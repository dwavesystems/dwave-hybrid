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

"""
State traits validation base class and related state validation mixins (i/o
validation toggle, i/o dimensionality, state structure).

When subclassing (combining with :class:`~hybrid.core.Runnable`), list them in
the following order, left to right:

    - structure mixins (e.g. `SubsamplesIntaking` and `SubproblemSampler`)
    - dimensionality mixins (e.g. `MultiInputStates` and `MISO`)
    - validation toggles (e.g. `InputValidated` and `NotValidated`)
    - StateTraits base class (not required if any of the above is used)
    - Runnable base class

For example:

    class MyRunnable(hybrid.traits.SubsamplesIntaking, hybrid.traits.MISO, hybid.Runnable):
        pass

"""

from collections.abc import Sequence, Mapping

from hybrid.exceptions import StateTraitMissingError, StateDimensionalityError


class StateTraits(object):
    """Set of traits imposed on State. By default, **not validated**."""

    def __init__(self):
        self.inputs = set()
        self.outputs = set()
        self.multi_input = False
        self.multi_output = False
        self.validate_input = False
        self.validate_output = False

    def validate_state_trait(self, state, trait, io):
        """Validate single input/output (`io`) `state` `trait`."""
        if trait not in state:
            raise StateTraitMissingError(
                "{} state is missing {!r} on {!r}".format(io, trait, self))

    def validate_input_state_traits(self, inp):
        if not self.validate_input:
            return

        if self.multi_input:
            if not isinstance(inp, Sequence):
                raise StateDimensionalityError(
                    "state sequence required on input to {!r}".format(self))

            for state in inp:
                for trait in self.inputs:
                    self.validate_state_trait(state, trait, "input")

        else:
            if not isinstance(inp, Mapping):
                raise StateDimensionalityError(
                    "single state required on input to {!r}".format(self))

            for trait in self.inputs:
                self.validate_state_trait(inp, trait, "input")

    def validate_output_state_traits(self, out):
        if not self.validate_output:
            return

        if self.multi_output:
            if not isinstance(out, Sequence):
                raise StateDimensionalityError(
                    "state sequence required on output from {!r}".format(self))

            for state in out:
                for trait in self.outputs:
                    self.validate_state_trait(state, trait, "output")

        else:
            if not isinstance(out, Mapping):
                raise StateDimensionalityError(
                    "single state required on output from {!r}".format(self))

            for trait in self.outputs:
                self.validate_state_trait(out, trait, "output")


#
# I/O validation mixins
#

class InputValidated(StateTraits):
    def __init__(self):
        super(InputValidated, self).__init__()
        self.validate_input = True

class OutputValidated(StateTraits):
    def __init__(self):
        super(OutputValidated, self).__init__()
        self.validate_output = True

class InputNotValidated(StateTraits):
    def __init__(self):
        super(InputNotValidated, self).__init__()
        self.validate_input = False

class OutputNotValidated(StateTraits):
    def __init__(self):
        super(OutputNotValidated, self).__init__()
        self.validate_output = False


class Validated(InputValidated, OutputValidated):
    """Validated input state(s) and output state(s)."""

class NotValidated(InputNotValidated, OutputNotValidated):
    """Input state(s) and output state(s) are not validated."""


#
# I/O dimensionality mixins. Imply I/O validation.
#

class SingleInputState(InputValidated, StateTraits):
    def __init__(self):
        super(SingleInputState, self).__init__()
        self.multi_input = False

class MultiInputStates(InputValidated, StateTraits):
    def __init__(self):
        super(MultiInputStates, self).__init__()
        self.multi_input = True

class SingleOutputState(OutputValidated, StateTraits):
    def __init__(self):
        super(SingleOutputState, self).__init__()
        self.multi_output = False

class MultiOutputStates(OutputValidated, StateTraits):
    def __init__(self):
        super(MultiOutputStates, self).__init__()
        self.multi_output = True


class SISO(SingleInputState, SingleOutputState):
    """Single Input, Single Output."""

class SIMO(SingleInputState, MultiOutputStates):
    """Single Input, Multiple Outputs."""

class MIMO(MultiInputStates, MultiOutputStates):
    """Multiple Inputs, Multiple Outputs."""

class MISO(MultiInputStates, SingleOutputState):
    """Multiple Inputs, Single Output."""


#
# State structure mixins. Imply I/O validation.
#

class ProblemIntaking(InputValidated, StateTraits):
    def __init__(self):
        super(ProblemIntaking, self).__init__()
        self.inputs.add('problem')

class ProblemProducing(OutputValidated, StateTraits):
    def __init__(self):
        super(ProblemProducing, self).__init__()
        self.outputs.add('problem')


class SamplesIntaking(InputValidated, StateTraits):
    def __init__(self):
        super(SamplesIntaking, self).__init__()
        self.inputs.add('samples')

class SamplesProducing(OutputValidated, StateTraits):
    def __init__(self):
        super(SamplesProducing, self).__init__()
        self.outputs.add('samples')


class SubproblemIntaking(InputValidated, StateTraits):
    def __init__(self):
        super(SubproblemIntaking, self).__init__()
        self.inputs.add('subproblem')

class SubproblemProducing(OutputValidated, StateTraits):
    def __init__(self):
        super(SubproblemProducing, self).__init__()
        self.outputs.add('subproblem')


class SubsamplesIntaking(InputValidated, StateTraits):
    def __init__(self):
        super(SubsamplesIntaking, self).__init__()
        self.inputs.add('subsamples')

class SubsamplesProducing(OutputValidated, StateTraits):
    def __init__(self):
        super(SubsamplesProducing, self).__init__()
        self.outputs.add('subsamples')


class EmbeddingIntaking(InputValidated, StateTraits):
    def __init__(self):
        super(EmbeddingIntaking, self).__init__()
        self.inputs.add('embedding')

class EmbeddingProducing(OutputValidated, StateTraits):
    def __init__(self):
        super(EmbeddingProducing, self).__init__()
        self.outputs.add('embedding')


class ProblemDecomposer(ProblemIntaking, SamplesIntaking, SubproblemProducing):
    pass

class SubsamplesComposer(SamplesIntaking, SubsamplesIntaking, ProblemIntaking, SamplesProducing):
    pass

class ProblemSampler(ProblemIntaking, SamplesProducing):
    pass

class SubproblemSampler(SubproblemIntaking, SubsamplesProducing):
    pass


class SamplesProcessor(SamplesIntaking, SamplesProducing):
    pass

class SubsamplesProcessor(SubsamplesIntaking, SubsamplesProducing):
    pass

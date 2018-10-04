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

from hades.exceptions import InvalidStateError


class StateTraitMissingError(InvalidStateError):
    pass


class StateTraits(object):
    """Set of traits imposed on State."""

    def __init__(self):
        self.inputs = set()
        self.outputs = set()

    def validate_state_trait(self, state, trait):
        if trait not in state:
            raise StateTraitMissingError("state is missing %r" % trait)

    def validate_input_state_traits(self, state):
        for trait in self.inputs:
            self.validate_state_trait(state, trait)

    def validate_output_state_traits(self, state):
        for trait in self.outputs:
            self.validate_state_trait(state, trait)


class ProblemActing(StateTraits):
    def __init__(self):
        super(ProblemActing, self).__init__()
        self.inputs.add('problem')


class SubproblemActing(StateTraits):
    def __init__(self):
        super(SubproblemActing, self).__init__()
        self.inputs.add('subproblem')


class ProblemProducing(StateTraits):
    def __init__(self):
        super(ProblemProducing, self).__init__()
        self.outputs.add('problem')


class SubproblemProducing(StateTraits):
    def __init__(self):
        super(SubproblemProducing, self).__init__()
        self.outputs.add('subproblem')


class SampleProducing(StateTraits):
    def __init__(self):
        super(SampleProducing, self).__init__()
        self.outputs.add('samples')


class SubsampleProducing(StateTraits):
    def __init__(self):
        super(SubsampleProducing, self).__init__()
        self.outputs.add('subsamples')


class ProblemDecomposer(ProblemActing, SubproblemProducing):
    pass


class SubproblemComposer(SubproblemActing, ProblemActing, ProblemProducing):
    pass


class ProblemSampler(ProblemActing, SampleProducing):
    pass


class SubproblemSampler(SubproblemActing, SubsampleProducing):
    pass

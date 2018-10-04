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

class ProblemProducing(StateTraits):
    def __init__(self):
        super(ProblemProducing, self).__init__()
        self.outputs.add('problem')


class SamplesActing(StateTraits):
    def __init__(self):
        super(SamplesActing, self).__init__()
        self.inputs.add('samples')

class SamplesProducing(StateTraits):
    def __init__(self):
        super(SamplesProducing, self).__init__()
        self.outputs.add('samples')


class SubproblemActing(StateTraits):
    def __init__(self):
        super(SubproblemActing, self).__init__()
        self.inputs.add('subproblem')

class SubproblemProducing(StateTraits):
    def __init__(self):
        super(SubproblemProducing, self).__init__()
        self.outputs.add('subproblem')


class SubsamplesActing(StateTraits):
    def __init__(self):
        super(SubsamplesActing, self).__init__()
        self.inputs.add('subsamples')

class SubsamplesProducing(StateTraits):
    def __init__(self):
        super(SubsamplesProducing, self).__init__()
        self.outputs.add('subsamples')


class EmbeddingActing(StateTraits):
    def __init__(self):
        super(EmbeddingActing, self).__init__()
        self.inputs.add('embedding')

class EmbeddingProducing(StateTraits):
    def __init__(self):
        super(EmbeddingProducing, self).__init__()
        self.outputs.add('embedding')


class ProblemDecomposer(ProblemActing, SamplesActing, SubproblemProducing):
    pass

class SubproblemComposer(SubproblemActing, SubsamplesActing, ProblemActing, SamplesProducing):
    pass

class ProblemSampler(ProblemActing, SamplesProducing):
    pass

class SubproblemSampler(SubproblemActing, SubsamplesProducing):
    pass

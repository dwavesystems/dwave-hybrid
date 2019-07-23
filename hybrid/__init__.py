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

import os
import logging


def _create_trace_loglevel(logging):
    "Add TRACE log level and Logger.trace() method."

    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, "TRACE")

    def _trace(logger, message, *args, **kwargs):
        if logger.isEnabledFor(logging.TRACE):
            logger._log(logging.TRACE, message, args, **kwargs)

    logging.Logger.trace = _trace


def _configure_logger(logger):
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    return logger


def _apply_loglevel_from_env(logger, env='DWAVE_HYBRID_LOG_LEVEL'):
    name = os.getenv(env) or os.getenv(env.upper()) or os.getenv(env.lower())
    if not name:
        return
    levels = {'trace': logging.TRACE, 'debug': logging.DEBUG, 'info': logging.INFO,
              'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
    requested_level = levels.get(name.lower())
    if requested_level:
        logger.setLevel(requested_level)


logger = logging.getLogger(__name__)
_create_trace_loglevel(logging)
_configure_logger(logger)
_apply_loglevel_from_env(logger)


# convenience top-level import
from hybrid.core import *
from hybrid.flow import *
from hybrid import traits
from hybrid.exceptions import *
from hybrid.concurrency import *

from hybrid.samplers import *
from hybrid.decomposers import *
from hybrid.composers import *

from hybrid.utils import *
from hybrid.profiling import *

from hybrid.reference import *

from hybrid.package_info import *

# convenience submodules import
import hybrid.core
import hybrid.flow
import hybrid.traits
import hybrid.exceptions
import hybrid.concurrency

import hybrid.samplers
import hybrid.decomposers
import hybrid.composers

import hybrid.utils
import hybrid.profiling

import hybrid.reference

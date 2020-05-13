# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import IntEnum


class TimingMeasType(IntEnum):
    """Timing measurement type.

    RISE: (x, 1 - x) percent rise time criterion at the output
    FALL: (x, 1 - x) percent fall time criterion at the output
    DLEAY_H2L: the time it takes to go from 50% of input to 50% of output, when input goes H2L
    DLEAY_L2H: the time it takes to go from 50% of input to 50% of output, when input goes L2H
    """
    RISE = 0
    FALL = 1
    DELAY_H2L = 2
    DELAY_L2H = 3

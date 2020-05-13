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

from __future__ import annotations

from enum import Enum


class LUTType(Enum):
    CONSTRAINT = 0
    DELAY = 1
    MAX_CAP = 2
    DRIVE_WVFM = 3


class TermType(Enum):
    input = 0
    output = 1
    inout = 2


class TimingType(Enum):
    """
    Notes
    -----
    bit 0: whether it is falling
    bit 3: whether it is timing type associated with output pin
    bit 4: whether it is non-sequential
    bit 5: whether it is combinational
    bit 6: whether it is falling although bit 0 is 0
    """
    setup_rising = 0
    setup_falling = 1
    hold_rising = 2
    hold_falling = 3
    recovery_rising = 4
    recovery_falling = 5
    removal_rising = 6
    removal_falling = 7
    rising_edge = 8
    falling_edge = 9
    non_seq_setup_rising = 16
    non_seq_setup_falling = 17
    non_seq_hold_rising = 18
    non_seq_hold_falling = 19
    combinational_rise = 40
    combinational_fall = 41
    combinational = 104

    @property
    def is_rising(self) -> bool:
        return (self.value & 0x01) == 0

    @property
    def is_output(self) -> bool:
        return (self.value & 0x08) != 0

    @property
    def is_non_seq(self) -> bool:
        return (self.value & 0x10) != 0

    @property
    def is_combinational(self) -> bool:
        return (self.value & 0x20) != 0

    @property
    def is_falling(self) -> bool:
        val = self.value
        return (val & 0x01) != 0 or (val & 0x40) != 0

    def with_non_seq(self, non_seq: bool) -> TimingType:
        return TimingType((self.value & 0xEF) | (int(non_seq) << 4))


class LogicType(Enum):
    COMB = 0
    SEQ = 1


class ConditionType(Enum):
    WHEN = 0
    SDF = 1


class TimingSenseType(Enum):
    positive_unate = 0
    negative_unate = 1
    non_unate = 2
    undefined = 3

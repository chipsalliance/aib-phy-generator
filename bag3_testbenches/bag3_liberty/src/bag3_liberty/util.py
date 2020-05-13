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

from typing import Tuple, Optional, Iterable, IO

import re
from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class BusRange:
    """Note: stop is inclusive"""
    start: int
    stop: int

    @property
    def name(self) -> str:
        return f'bus_{self.start}_to_{self.stop}'

    def __len__(self) -> int:
        return abs(self.start - self.stop) + 1

    def __iter__(self) -> Iterable[int]:
        if self.stop >= self.start:
            return iter(range(self.start, self.stop + 1))
        else:
            return iter(range(self.start, self.stop - 1, -1))

    def __getitem__(self, idx: int) -> int:
        num_idx = len(self)
        step = 1 if self.stop >= self.start else -1
        if idx < 0:
            idx += num_idx
        if idx < 0 or idx >= num_idx:
            raise IndexError(f'Invalid bus index: {idx}')

        return self.start + step * idx

    def stream_out(self, stream: IO, indent: int) -> None:
        pad = ' ' * indent
        stream.write(f'{pad}type ({self.name}) {{\n')
        indent += 4
        pad = ' ' * indent

        stream.write(f'{pad}base_type : array;\n')
        stream.write(f'{pad}data_type : bit;\n')
        stream.write(f'{pad}bit_width : {len(self)};\n')
        stream.write(f'{pad}bit_from : {self.start};\n')
        stream.write(f'{pad}bit_to : {self.stop};\n')
        stream.write(f'{pad}downto : {str(self.stop < self.start).lower()};\n')

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}}\n')


def parse_cdba_name(name: str) -> Tuple[str, Optional[BusRange]]:
    if not name:
        raise ValueError(f'Cannot have empty string as pin name.')

    if '[' in name or ']' in name:
        raise ValueError(f'Illegal pin name: {name}, must use angle brackets for bus.')

    if name[-1] == '>':
        idx = name.find('<')
        if idx < 0:
            raise ValueError(f'Illegal pin name: {name}')
        basename = name[:idx]
        if ':' in basename:
            raise ValueError(f'Illegal pin name: {name}')

        range_list = name[idx + 1:-1].split(':')
        num = len(range_list)
        try:
            start = int(range_list[0])
            if num == 1:
                return basename, BusRange(start, start)
            else:
                stop = int(range_list[1])
                if num == 2:
                    return basename, BusRange(start, stop)
                else:
                    step = int(range_list[2])
                    if stop > start and step != 1 or start > stop and step != -1:
                        raise ValueError(f'Illegal pin name: {name}, bus must have step of 1.')
                    return basename, BusRange(start, stop)
        except ValueError:
            raise ValueError(f'Illegal name: {name}')
    else:
        if '<' in name or ':' in name:
            raise ValueError(f'Illegal name: {name}')
        return name, None


def get_bus_bit_name(name: str, idx: int, cdba: bool = False) -> str:
    if cdba:
        return f'{name}<{idx}>'
    return f'{name}[{idx}]'


def cdba_to_unusal(name: str) -> str:
    basename, bus_range = parse_cdba_name(name)
    if bus_range is None:
        return f'{basename}_'
    if len(bus_range) != 1:
        raise ValueError('This method only works for one-bit bus names.')
    return f'{basename}_{bus_range[0]}_'


def remove_comments(text: str) -> str:
    """From https://stackoverflow.com/questions/241327/remove-c-and-c-comments-using-python"""
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

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

from typing import Dict, Any, List, Tuple, Optional, Iterable, Union, IO, Set, Sequence, Mapping

import math
from pathlib import Path
from itertools import chain
from datetime import datetime
from dataclasses import dataclass

import numpy as np

from .enum import LUTType, TermType, LogicType, ConditionType, TimingType, TimingSenseType
from .util import parse_cdba_name, get_bus_bit_name, BusRange
from .boolean import parse_timing_cond_expr


@dataclass(eq=True, frozen=True)
class Units:
    fmt: str
    voltage: float
    current: float
    time: float
    resistance: float
    capacitance: float
    power: float

    def format_v(self, val: float) -> str:
        return self.fmt.format(val / self.voltage)

    def format_i(self, val: float) -> str:
        return self.fmt.format(val / self.current)

    def format_t(self, val: float) -> str:
        return self.fmt.format(val / self.time)

    def format_f(self, val: float) -> str:
        return self.fmt.format(val * self.time)

    def format_r(self, val: float) -> str:
        return self.fmt.format(val / self.resistance)

    def format_c(self, val: float) -> str:
        return self.fmt.format(val / self.capacitance)

    def format_p(self, val: float) -> str:
        return self.fmt.format(val / self.power)

    def format(self, val: Union[str, float], val_type: str) -> str:
        if val_type == 'str':
            return val
        if not val_type:
            return self.fmt.format(val)
        if val_type == 'frequency':
            return self.format_f(val)
        return self.fmt.format(val / getattr(self, val_type))


class SimEnv:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._name: str = config['name']
        self._process: float = config.get('process', 1.0)
        self._temp: float = config['temperature']
        self._volt: float = config['voltage']
        self._bag_name: str = config.get('bag_name', self._name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def bag_name(self) -> str:
        return self._bag_name

    @property
    def process(self) -> float:
        return self._process

    @property
    def temperature(self) -> float:
        return self._temp

    @property
    def voltage(self) -> float:
        return self._volt

    def stream_out(self, stream: IO, indent: int, units: Units) -> None:
        pad = ' ' * indent
        stream.write(f'{pad}operating_conditions ({_to_string(self._name)}) {{\n')
        indent += 4
        pad = ' ' * indent

        stream.write(f'{pad}process : {self.process:.1f};\n')
        stream.write(f'{pad}temperature : {self.temperature:.1f};\n')
        stream.write(f'{pad}voltage : {units.format_v(self.voltage)};\n')
        stream.write(f'{pad}tree_type : "balanced_tree";\n')

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}}\n')


class LUT:
    def __init__(self, lut_type: LUTType, val_table: Dict[str, List[float]]) -> None:
        self._type = lut_type
        if lut_type is LUTType.CONSTRAINT:
            self._names = ['trf_in', 'trf_src']
            self._descs = ['constrained_pin_transition', 'related_pin_transition']
            self._units = ['time', 'time']
        elif lut_type is LUTType.DELAY:
            self._names = ['trf_src', 'cload']
            self._descs = ['input_net_transition', 'total_output_net_capacitance']
            self._units = ['time', 'capacitance']
        elif lut_type is LUTType.MAX_CAP:
            self._names = ['freq']
            self._descs = ['frequency']
            self._units = ['frequency']
        elif lut_type is LUTType.DRIVE_WVFM:
            self._names = ['trf_src', 'vout']
            self._descs = ['input_net_transition', 'normalized_voltage']
            self._units = ['time', '']
        else:
            raise ValueError(f'Unknown LUT Type: {lut_type}')

        self._values = [val_table[name] for name in self._names]

    @property
    def lut_type(self) -> LUTType:
        return self._type

    @property
    def variables(self) -> List[str]:
        return self._names

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple((len(val_list) for val_list in self._values))

    def __getitem__(self, name: str) -> Sequence[float]:
        return self._values[self._names.index(name)]

    def get_swp_order(self, name_dict: Mapping[str, str]) -> List[str]:
        return [name_dict[name] for name in self._names]

    def get_swp_info(self, name_dict: Mapping[str, str]) -> List[Tuple[str, Mapping[str, Any]]]:
        ans = [(name_dict[name], dict(type='LIST', values=arr))
               for name, arr in zip(self._names, self._values)]
        return ans

    def stream_out(self, stream: IO, indent: int, units: Units) -> None:
        pad = ' ' * indent
        if self._type is LUTType.MAX_CAP:
            stream.write(f'{pad}maxcap_lut_template ({self._type.name}) {{\n')
        else:
            stream.write(f'{pad}lu_table_template ({self._type.name}) {{\n')
        indent += 4
        pad = ' ' * indent

        for idx, desc in enumerate(self._descs):
            stream.write(f'{pad}variable_{idx + 1} : {desc};\n')
        self.stream_out_values(stream, pad, units)

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}}\n')

    def stream_out_values(self, stream: IO, pad: str, units: Units) -> None:
        for idx, val_list in enumerate(self._values):
            num_list = _num_list_to_string(val_list, units, self._units[idx])
            stream.write(f'{pad}index_{idx + 1} ({num_list});\n')


class NormDrvWvfm:
    def __init__(self, ndw_dict: Dict[str, Union[str, List[float], np.ndarray]], lut: LUT) -> None:
        self._name: str = ndw_dict['name']
        self._val: np.ndarray = np.asarray(ndw_dict['val'])
        self._lut = lut
        assert self._val.shape == lut.shape, 'Shape of NormDrvWvfm values are wrong'

    @property
    def name(self) -> str:
        return self._name

    def stream_out(self, stream: IO, indent: int, units: Units) -> None:
        pad = ' ' * indent
        stream.write(f'{pad}normalized_driver_waveform({self._lut.lut_type.name}) {{\n')
        indent += 4
        pad = ' ' * indent
        stream.write(f'{pad}driver_waveform_name : {_to_string(self._name)};\n')

        # write data values
        self._lut.stream_out_values(stream, pad, units)
        stream.write(f'{pad}values( \\\n')
        indent += 4
        pad = ' ' * indent
        num_rows = self._val.shape[0]
        for row_idx in range(num_rows):
            num_list = _num_list_to_string(self._val[row_idx, :], units, 'time')
            if row_idx == num_rows - 1:
                stream.write(f'{pad}{num_list} \\\n')
            else:
                stream.write(f'{pad}{num_list}, \\\n')
        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad});\n')

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}}\n')


class TimingCond:
    def __init__(self, cond: str) -> None:
        self._str_dict = parse_timing_cond_expr(cond)

    def get_cond(self, cond_type: ConditionType, lut_type: LUTType = LUTType.DELAY) -> str:
        if cond_type is ConditionType.WHEN:
            return _to_string(self._str_dict['when_str'])
        if lut_type is LUTType.DELAY:
            return _to_string(self._str_dict['sdf_out_str'])
        return _to_string(self._str_dict['sdf_in_str'])


class Timing:
    def __init__(self, lut: LUT, related: str, data: Dict[str, Any],
                 timing_type: TimingType, cond: str = '', sense: str = '',
                 clk_rising: bool = True) -> None:
        self._lut = lut
        self._related = related
        self._type = timing_type
        self._tc = TimingCond(cond) if cond else None
        self._clk_type = 'rising' if clk_rising else 'falling'
        if timing_type.is_combinational:
            self._sense = TimingSenseType[sense] if sense else TimingSenseType.non_unate
        else:
            self._sense = TimingSenseType.undefined

        data_list = [(key, np.asarray(data[key])) for key in sorted(data.keys())]
        self._data: Sequence[Tuple[str, np.ndarray]] = data_list

    def stream_out(self, stream: IO, indent: int, units: Units) -> None:
        lut_type = self._lut.lut_type
        lut_shape = self._lut.shape

        pad = ' ' * indent
        stream.write(f'{pad}timing () {{\n')
        indent += 4
        pad = ' ' * indent

        type_str = self._type.name
        stream.write(f'{pad}related_pin : {_to_string(self._related)};\n')
        stream.write(f'{pad}timing_type : {type_str};\n')
        if self._tc is not None:
            sdf_cond = self._tc.get_cond(ConditionType.SDF, lut_type)
            stream.write(f'{pad}sdf_cond : {sdf_cond};\n')
            stream.write(f'{pad}when : {self._tc.get_cond(ConditionType.WHEN)};\n')
        if self._sense is not TimingSenseType.undefined:
            stream.write(f'{pad}timing_sense : {self._sense.name};\n')

        for name, data in self._data:
            stream.write('\n')

            stream.write(f'{pad}{name}({lut_type.name}) {{\n')
            indent += 4
            pad = ' ' * indent

            # write data values
            if data.shape != lut_shape:
                raise ValueError(f'Timing data shape = {data.shape} != {lut_shape}')
            if len(data.shape) != 2:
                raise ValueError('Cannot write non-2D data.  See developer')
            self._lut.stream_out_values(stream, pad, units)
            stream.write(f'{pad}values( \\\n')
            indent += 4
            pad = ' ' * indent
            num_rows = data.shape[0]
            for row_idx in range(num_rows):
                num_list = _num_list_to_string(data[row_idx, :], units, 'time')
                if row_idx == num_rows - 1:
                    stream.write(f'{pad}{num_list} \\\n')
                else:
                    stream.write(f'{pad}{num_list}, \\\n')

            indent -= 4
            pad = ' ' * indent
            stream.write(f'{pad});\n')

            indent -= 4
            pad = ' ' * indent
            stream.write(f'{pad}}} /* end data {name} */\n')

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}} /* end timing {type_str} */\n')


class Pin:
    def __init__(self, lib: Library, cell: Cell, name: str, logic: LogicType, pin_type: TermType,
                 is_clock: bool, cap_dict: Optional[Dict[str, Union[float, Sequence[float]]]],
                 max_trf: Optional[float], pwr_pin: str, gnd_pin: str, is_bus: bool,
                 func: str = '', dw_rise: str = '', dw_fall: str = '', max_fanout: float = -1.0
                 ) -> None:
        pwr_type = cell.get_voltage_type(pwr_pin)

        self._lib = lib
        self._name = name
        self._logic = logic
        self._type = pin_type
        self._cap_dict = cap_dict
        self._timing_table: Dict[Tuple[int, str], Timing] = {}

        self._props: List[Tuple[str, Any]] = []
        if not is_bus:
            self._props.append(('direction', pin_type.name))
        if is_clock:
            self._props.append(('clock', 'true'))

        self._props.append(('related_power_pin', pwr_pin))
        self._props.append(('related_ground_pin', gnd_pin))

        if pin_type is TermType.input:
            self._props.append(('input_voltage', pwr_type))
            self._dw_rise = lib.get_driver_waveform(dw_rise)
            self._props.append(('driver_waveform_rise', _to_string(self._dw_rise.name)))
            self._dw_fall = lib.get_driver_waveform(dw_fall)
            self._props.append(('driver_waveform_fall', _to_string(self._dw_fall.name)))

            if max_trf is None:
                # set to default
                self._max_trf = lib.get_max_input_transition(logic, is_clock=is_clock)
            else:
                self._max_trf = max_trf
        else:
            self._max_trf = None
            self._dw_rise = self._dw_fall = None

        if pin_type is TermType.output or pin_type is TermType.inout:
            self._props.append(('output_voltage', pwr_type))
            if func:
                self._props.append(('function', _to_string(func)))
            if max_fanout <= 0:
                raise ValueError('output pin max fanout is not defined.')
            self._props.append(('power_down_function', _to_string(f'!{pwr_pin}+{gnd_pin}')))
            self._props.append(('max_fanout', max_fanout))

    def get_waveform(self, is_rise: bool = True) -> Optional[NormDrvWvfm]:
        return self._dw_rise if is_rise else self._dw_fall

    def add_timing(self, related: str, data: Dict[str, np.ndarray],
                   timing_type: Union[TimingType, str], cond: str = '', sense: str = '',
                   clk_rising: bool = True) -> None:
        timing_type_val = TimingType[timing_type] if isinstance(timing_type, str) else timing_type
        key = (timing_type_val.value, related)
        if key in self._timing_table:
            raise ValueError(f'Pin {self._name} already has timing data for '
                             f'({timing_type_val.name}, {related}).')

        if self._type is TermType.input:
            if self._logic is LogicType.COMB and not timing_type_val.is_non_seq:
                raise ValueError(f'Combinational input pin {self._name} cannot have sequential '
                                 f'timing info.')
            lut = self._lib.get_lut(LUTType.CONSTRAINT)
            timing = Timing(lut, related, data, timing_type_val, cond, sense, clk_rising)
        else:
            lut = self._lib.get_lut(LUTType.DELAY)
            timing = Timing(lut, related, data, timing_type_val, cond, sense, clk_rising)

        self._timing_table[key] = timing

    def stream_out(self, stream: IO, indent: int, units: Units) -> None:
        pad = ' ' * indent

        stream.write(f'{pad}pin ({self._name}) {{\n')
        indent += 4
        pad = ' ' * indent

        for k, v in self._props:
            stream.write(f'{pad}{k} : {v};\n')

        cap_dict = self._cap_dict
        if cap_dict is not None:
            if self._type is TermType.input:
                stream.write(f'{pad}capacitance : {units.format_c(cap_dict["cap"])};\n')
                stream.write(f'{pad}fall_capacitance : {units.format_c(cap_dict["cap_fall"])};\n')
                stream.write(f'{pad}fall_capacitance_range('
                             f'{units.format_c(cap_dict["cap_fall_range"][0])}, '
                             f'{units.format_c(cap_dict["cap_fall_range"][1])});\n')
                stream.write(f'{pad}rise_capacitance : {units.format_c(cap_dict["cap_rise"])};\n')
                stream.write(f'{pad}rise_capacitance_range('
                             f'{units.format_c(cap_dict["cap_rise_range"][0])}, '
                             f'{units.format_c(cap_dict["cap_rise_range"][1])});\n')
            elif self._type is TermType.output or self._type is TermType.inout:
                stream.write(f'{pad}max_capacitance : {units.format_c(cap_dict["cap_max"])};\n')
                stream.write(f'{pad}min_capacitance : {units.format_c(cap_dict["cap_min"])};\n')
                cap_table = cap_dict.get('cap_max_table', None)
                if cap_table is not None:
                    cap_lut = self._lib.get_lut(LUTType.MAX_CAP)
                    stream.write('\n')
                    num_list = _num_list_to_string(cap_table, units, 'capacitance')
                    stream.write(f'{pad}max_cap({cap_lut.lut_type.name}) {{\n')
                    indent += 4
                    pad = ' ' * indent
                    cap_lut.stream_out_values(stream, pad, units)
                    stream.write(f'{pad}values({num_list});\n')
                    stream.write(f'{pad}}}\n')
                    indent -= 4
                    pad = ' ' * indent

        if self._max_trf is not None:
            stream.write(f'{pad}max_transition : {units.format_t(self._max_trf)};\n')

        for key in sorted(self._timing_table.keys()):
            stream.write('\n')
            self._timing_table[key].stream_out(stream, indent, units)

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}} /* end pin {self._name} */\n')


class Bus:
    def __init__(self, lib: Library, cell: Cell, name: str, pin_type: TermType) -> None:
        tmp = parse_cdba_name(name)
        if tmp[1] is None:
            raise ValueError(f'{name} is an invalid bus name.')

        self._lib = lib
        self._cell = cell
        self._name = tmp[0]
        self._range = tmp[1]
        self._type = pin_type
        self._pins: List[Pin] = []
        self._props: List[Tuple[str, Any]] = [('direction', pin_type.name)]

    @property
    def full_name(self) -> str:
        return f'{self._name}[{self._range.start}:{self._range.stop}]'

    @property
    def bus_range(self) -> BusRange:
        return self._range

    def create_pin(self, idx: int, pin_type: TermType, params: Dict[str, Any]) -> Pin:
        if pin_type is TermType.input:
            return self.create_input_pin(idx, **params)
        if pin_type is TermType.output:
            return self.create_output_pin(idx, **params)
        if pin_type is TermType.inout:
            return self.create_inout_pin(idx, **params)
        raise ValueError(f'Unsupported terminal type: {pin_type}')

    def create_input_pin(self, idx: int, pwr_pin: str, gnd_pin: str, dw_rise: str, dw_fall: str,
                         cap_dict: Optional[Dict[str, Union[float, Tuple[float]]]] = None,
                         logic: str = 'COMB', is_clock: bool = False,
                         max_trf: Optional[float] = None) -> Pin:
        logic_type = LogicType[logic]
        pin_name = get_bus_bit_name(self._name, self._range[idx])
        pin = Pin(self._lib, self._cell, pin_name, logic_type, self._type, is_clock, cap_dict,
                  max_trf, pwr_pin, gnd_pin, True, dw_rise=dw_rise, dw_fall=dw_fall)
        self._pins.append(pin)
        return pin

    def create_output_pin(self, idx: int, func: str, pwr_pin: str, gnd_pin: str,
                          max_fanout: float, logic: str = 'COMB',
                          cap_dict: Optional[Dict[str, Union[float, Tuple[float]]]] = None,
                          is_clock: bool = False) -> Pin:
        logic_type = LogicType[logic]
        pin_name = get_bus_bit_name(self._name, self._range[idx])
        pin = Pin(self._lib, self._cell, pin_name, logic_type, self._type, is_clock, cap_dict,
                  None, pwr_pin, gnd_pin, True, func=func, max_fanout=max_fanout)
        self._pins.append(pin)
        return pin

    def create_inout_pin(self, idx: int, func: str, pwr_pin: str, gnd_pin: str, max_fanout: float,
                         cap_dict: Optional[Dict[str, Union[float, Tuple[float]]]] = None,
                         logic: str = 'COMB', is_clock: bool = False) -> Pin:
        logic_type = LogicType[logic]
        pin_name = get_bus_bit_name(self._name, self._range[idx])
        pin = Pin(self._lib, self._cell, pin_name, logic_type, self._type, is_clock, cap_dict,
                  None, pwr_pin, gnd_pin, True, func=func, max_fanout=max_fanout)
        self._pins.append(pin)
        return pin

    def stream_out(self, stream: IO, indent: int, units: Units) -> None:
        pad = ' ' * indent

        stream.write(f'{pad}bus ({self._name}) {{\n')
        indent += 4
        pad = ' ' * indent
        stream.write(f'{pad}bus_type : {self._range.name};\n')
        for k, v in self._props:
            stream.write(f'{pad}{k} : {v};\n')

        stream.write('\n')
        stream.write(f'{pad}/* IO pins */\n')
        for idx, pin in enumerate(self._pins):
            if idx != 0:
                stream.write('\n')
            pin.stream_out(stream, indent, units)

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}} /* end bus {self.full_name} */\n')


class Cell:
    def __init__(self, lib: Library, name: str, pwr_pins: Dict[str, str], gnd_pins: Dict[str, str],
                 props: Dict[str, Any], stdcell_pwr_pins: Optional[List[str]] = None) -> None:
        # error checking
        for ptype in chain(pwr_pins.values(), gnd_pins.values()):
            if not lib.has_voltage_type(ptype):
                raise ValueError(f'Unknown voltage type: {ptype}')

        self._lib = lib
        self._name = name
        self._pwr_pins = pwr_pins
        self._gnd_pins = gnd_pins
        self._props = [ele for ele in props.items()]
        self._pins: List[Pin] = []
        self._buses: List[Bus] = []
        if stdcell_pwr_pins:
            self._std_pwr_pins = set(stdcell_pwr_pins)
        else:
            self._std_pwr_pins = set()

    @property
    def bus_range_iter(self) -> Iterable[BusRange]:
        return (bus.bus_range for bus in self._buses)

    def get_voltage_type(self, pin_name: str) -> str:
        ans = self._pwr_pins.get(pin_name, None)
        if ans is None:
            return self._gnd_pins[pin_name]
        return ans

    def create_bus(self, name: str, pin_type: TermType) -> Bus:
        bus = Bus(self._lib, self, name, pin_type)
        self._buses.append(bus)
        return bus

    def create_pin(self, pin_type: TermType, params: Dict[str, Any]) -> Pin:
        if pin_type is TermType.input:
            return self.create_input_pin(**params)
        if pin_type is TermType.output:
            return self.create_output_pin(**params)
        if pin_type is TermType.inout:
            return self.create_inout_pin(**params)
        raise ValueError(f'Unsupported terminal type: {pin_type}')

    def create_input_pin(self, name: str, pwr_pin: str, gnd_pin: str, dw_rise: str, dw_fall: str,
                         cap_dict: Optional[Dict[str, Union[float, Tuple[float]]]] = None,
                         logic: str = 'COMB', is_clock: bool = False,
                         max_trf: Optional[float] = None) -> Pin:
        basename, rang = parse_cdba_name(name)
        if rang is not None:
            raise ValueError('Cannot add a bus pin.')

        logic_type = LogicType[logic]
        pin = Pin(self._lib, self, name, logic_type, TermType.input, is_clock, cap_dict, max_trf,
                  pwr_pin, gnd_pin, False, dw_rise=dw_rise, dw_fall=dw_fall)
        self._pins.append(pin)
        return pin

    def create_inout_pin(self, name: str, func: str, pwr_pin: str, gnd_pin: str, max_fanout: float,
                         cap_dict: Optional[Dict[str, Union[float, Tuple[float]]]] = None,
                         logic: str = 'COMB', is_clock: bool = False) -> Pin:
        basename, rang = parse_cdba_name(name)
        if rang is not None:
            raise ValueError('Cannot add a bus pin.')

        logic_type = LogicType[logic]
        # if not func:
        #     raise ValueError(f'Inout pin {name} needs a pin function.')
        pin = Pin(self._lib, self, name, logic_type, TermType.inout, is_clock, cap_dict, None,
                  pwr_pin, gnd_pin, False, func=func, max_fanout=max_fanout)
        self._pins.append(pin)
        return pin

    def create_output_pin(self, name: str, func: str, pwr_pin: str, gnd_pin: str,
                          max_fanout: float, logic: str = 'COMB', is_clock: bool = False,
                          cap_dict: Optional[Dict[str, Union[float, Tuple[float]]]] = None
                          ) -> Pin:
        basename, rang = parse_cdba_name(name)
        if rang is not None:
            raise ValueError('Cannot add a bus pin.')

        logic_type = LogicType[logic]
        pin = Pin(self._lib, self, name, logic_type, TermType.output, is_clock, cap_dict, None,
                  pwr_pin, gnd_pin, False, func=func, max_fanout=max_fanout)
        self._pins.append(pin)
        return pin

    def stream_out(self, stream: IO, indent: int, units: Units) -> None:
        pad = ' ' * indent
        stream.write(f'{pad}cell ({self._name}) {{\n')
        indent += 4
        pad = ' ' * indent

        # write attributes
        if self._props:
            stream.write(f'{pad}/* attributes */\n')
            for k, v in self._props:
                if k in ['pin_opposite', 'pin_equal']:
                    # different syntax, handle specially
                    separator = ' '
                    for entry in v:
                        if len(entry) == 2:
                            v0 = entry[0]
                            v1 = entry[1]
                            if isinstance(v0, str):
                                v0_str = v0
                            else:
                                v0_str = separator.join(v0)
                            if isinstance(v1, str):
                                v1_str = v1
                            else:
                                v1_str = separator.join(v1)
                            stream.write(f'{pad}{k} ("{v0_str}", "{v1_str}");\n')
                        else:
                            v_str = separator.join(entry)
                            stream.write(f'{pad}{k} ("{v_str}");\n')
                else:
                    if isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                        stream.write(f'{pad}{k} : {_to_string(v)};\n')
                    else:
                        raise ValueError('Unsupported type for value in stream_out for lib '
                                         'generation')
        # write supply pins
        stream.write('\n')
        stream.write(f'{pad}/* supply pins */\n')
        self._stream_out_pg_pin(stream, indent, self._pwr_pins, 'primary_power')
        self._stream_out_pg_pin(stream, indent, self._gnd_pins, 'primary_ground')

        # write IO pins
        if self._pins:
            stream.write('\n')
            stream.write(f'{pad}/* IO pins */\n')
            for idx, pin in enumerate(self._pins):
                if idx != 0:
                    stream.write('\n')
                pin.stream_out(stream, indent, units)

        # write bus pins
        if self._buses:
            stream.write('\n')
            stream.write(f'{pad}/* IO buses */\n')
            for idx, pin in enumerate(self._buses):
                if idx != 0:
                    stream.write('\n')
                pin.stream_out(stream, indent, units)

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}} /* end cell {self._name} */\n')

    def _stream_out_pg_pin(self, stream: IO, indent: int, table: Dict[str, str], type_str: str
                           ) -> None:
        pad = ' ' * indent
        for name in sorted(table.keys()):
            vname = table[name]
            stream.write(f'{pad}pg_pin ({name}) {{\n')
            indent += 4
            pad = ' ' * indent

            stream.write(f'{pad}pg_type : {type_str};\n')
            stream.write(f'{pad}voltage_name : {vname};\n')
            stream.write(f'{pad}direction : inout;\n')
            if name in self._std_pwr_pins:
                stream.write(f'{pad}std_cell_main_rail : true;\n')
            indent -= 4
            pad = ' ' * indent
            stream.write(f'{pad}}}\n')


class Library:
    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        self._name = name
        self._precision: int = config.get('precision', 4)

        self._props = [
            ('date', datetime.now().strftime("%c")),
            ('comment', 'Generated by bag3_liberty.'),
            ('revision', config.get('revision', '0')),
            ('in_place_swap_mode', 'match_footprint'),
        ]

        fmt_str = '{:.%dg}' % self._precision
        self._units: Units = Units(fmt=fmt_str, **config['units'])
        self._voltage_types: Dict[str, float] = config['voltages']
        self._thresholds: Dict[str, int] = config['thresholds']
        self._sim_envs = [SimEnv(v_) for v_ in config['sim_envs']]
        self._defaults = dict(
            threshold_voltage_group=['', 'str'],
            fanout_load=[1, ''],
            cell_leakage_power=[0.0, 'power'],
            inout_pin_cap=[0.0, 'capacitance'],
            input_pin_cap=[0.0, 'capacitance'],
            output_pin_cap=[0.0, 'capacitance'],
            leakage_power_density=[0.0, 'power'],
            max_transition=[0.0, 'time'],
        )
        user_defaults = config['defaults']
        for key in self._defaults.keys():
            val = user_defaults.get(key, None)
            if val is not None:
                self._defaults[key][0] = val

        lut_dict: Dict[str, Dict[str, List[float]]] = config['lut']
        self._luts: Dict[LUTType, LUT] = {}
        for lut_type in LUTType:
            v: Dict[str, List[float]] = lut_dict.get(lut_type.name, None)
            if v is not None:
                self._luts[lut_type] = LUT(lut_type, v)

        self._drv_wvfms: Dict[str, NormDrvWvfm] = {}
        drv_wvfm_list: Optional[List[Dict[str, Any]]] = config.get('norm_drv_wvfm', None)
        if drv_wvfm_list:
            drv_wvfm_lut = self._luts.get(LUTType.DRIVE_WVFM, None)
            if drv_wvfm_lut is None:
                raise ValueError('Cannot add NormDrvWvfm with no waveform LUT defined.')
            for info in drv_wvfm_list:
                name = info['name']
                self._drv_wvfms[name] = NormDrvWvfm(info, drv_wvfm_lut)

        self._cells: List[Cell] = []

        tl0 = self._thresholds['lower_fall']
        tl1 = self._thresholds['lower_rise']
        th0 = self._thresholds['upper_fall']
        th1 = self._thresholds['upper_rise']
        if tl0 != tl1:
            raise ValueError('Different low threshold for fall/rise is not supported.')
        if th0 != th1:
            raise ValueError('Different high threshold for fall/rise is not supported.')
        th_d = self._thresholds['input_fall']
        for key in ['input_rise', 'output_fall', 'output_rise']:
            if th_d != self._thresholds[key]:
                raise ValueError('Different delay threshold for fall/rise/in/out is not supported.')

        self._thres_lo = tl0 / 100
        self._thres_hi = th0 / 100
        self._thres_delay = th_d / 100

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def thres_lo(self) -> float:
        return self._thres_lo

    @property
    def thres_hi(self) -> float:
        return self._thres_hi

    @property
    def thres_delay(self) -> float:
        return self._thres_delay

    @property
    def sim_envs(self) -> Sequence[str]:
        ans = [env.bag_name for env in self._sim_envs]
        ans.sort()
        return ans

    def get_driver_waveform(self, name: str) -> NormDrvWvfm:
        return self._drv_wvfms[name]

    def has_voltage_type(self, name: str) -> bool:
        return name in self._voltage_types

    def get_voltage(self, name: str) -> float:
        return self._voltage_types[name]

    def get_lut(self, lut_type: LUTType) -> LUT:
        return self._luts[lut_type]

    def get_max_input_transition(self, logic_type: LogicType, is_clock: bool = False) -> float:
        if logic_type is LogicType.COMB:
            return self._luts[LUTType.DELAY]['trf_src'][-1]
        elif logic_type is LogicType.SEQ:
            if is_clock:
                return self._luts[LUTType.CONSTRAINT]['trf_src'][-1]
            else:
                return self._luts[LUTType.CONSTRAINT]['trf_in'][-1]
        else:
            raise ValueError(f'Unknown LogicType: {logic_type}')

    def create_cell(self, name: str, pwr_pins: Dict[str, str], gnd_pins: Dict[str, str],
                    props: Optional[Dict[str, Any]] = None,
                    stdcell_pwr_pins: Optional[List[str]] = None) -> Cell:
        if props is None:
            props = {}
        ans = Cell(self, name, pwr_pins, gnd_pins, props, stdcell_pwr_pins=stdcell_pwr_pins)
        self._cells.append(ans)
        return ans

    def generate(self, fname: Union[str, Path]) -> None:
        if isinstance(fname, str):
            fname = Path(fname)

        if fname.is_dir() or fname.is_symlink():
            raise ValueError(f'Cannot write to {fname}')

        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(fname, 'w') as f:
            self.stream_out(f, 0)

    def stream_out(self, stream: IO, indent: int) -> None:
        pad = ' ' * indent
        stream.write(f'{pad}library ({self._name}) {{\n')
        indent += 4
        pad = ' ' * indent

        # write properties
        stream.write(f'{pad}/* misc. attributes */\n')
        stream.write(f'{pad}technology (cmos);\n')
        stream.write(f'{pad}delay_model : table_lookup;\n')
        stream.write(
            f'{pad}library_features(report_delay_calculation, report_power_calculation);\n')
        for k, v in self._props:
            stream.write(f'{pad}{k} : {_to_string(v)};\n')
        stream.write('\n')

        # write units
        cap_si = _get_si_prefix(self._units.capacitance)
        volt_si = _get_si_prefix(self._units.voltage)
        current_si = _get_si_prefix(self._units.current)
        time_si = _get_si_prefix(self._units.time)
        res_si = _get_si_prefix(self._units.resistance)
        power_si = _get_si_prefix(self._units.power)
        stream.write(f'{pad}/* units */\n')
        stream.write(f'{pad}capacitive_load_unit(1, {cap_si}f);\n')
        stream.write(f'{pad}voltage_unit : "1{volt_si}V";\n')
        stream.write(f'{pad}current_unit : "1{current_si}A";\n')
        stream.write(f'{pad}time_unit : "1{time_si}s";\n')
        stream.write(f'{pad}pulling_resistance_unit : "1{res_si}ohm";\n')
        stream.write(f'{pad}leakage_power_unit : "1{power_si}W";\n')
        stream.write('\n')

        # write delay thresholds
        stream.write(f'{pad}/* delay thresholds */\n')
        stream.write(f'{pad}slew_derate_from_library : {self._thresholds["slew_derate"]};\n')
        stream.write(f'{pad}input_threshold_pct_fall : {self._thresholds["input_fall"]};\n')
        stream.write(f'{pad}input_threshold_pct_rise : {self._thresholds["input_rise"]};\n')
        stream.write(f'{pad}output_threshold_pct_fall : {self._thresholds["output_fall"]};\n')
        stream.write(f'{pad}output_threshold_pct_rise : {self._thresholds["output_rise"]};\n')
        stream.write(f'{pad}slew_lower_threshold_pct_fall : {self._thresholds["lower_fall"]};\n')
        stream.write(f'{pad}slew_lower_threshold_pct_rise : {self._thresholds["lower_rise"]};\n')
        stream.write(f'{pad}slew_upper_threshold_pct_fall : {self._thresholds["upper_fall"]};\n')
        stream.write(f'{pad}slew_upper_threshold_pct_rise : {self._thresholds["upper_rise"]};\n')
        stream.write('\n')

        # write default values
        stream.write(f'{pad}/* default values */\n')
        for k, (v, vtype) in self._defaults.items():
            if vtype != 'str' or v:
                stream.write(f'{pad}default_{k} : {self._units.format(v, vtype)};\n')
        stream.write('\n')

        # write voltage maps
        stream.write(f'{pad}/* voltage types */\n')
        pad2 = ' ' * indent * 2
        for name in sorted(self._voltage_types.keys()):
            val = self._voltage_types[name]
            stream.write(f'{pad}voltage_map({name}, {self._units.format_v(val)});\n')
            if val != 0:
                # write input voltage
                stream.write(f'{pad}input_voltage({name}) {{\n')
                stream.write(f'{pad2}vil : 0;\n')
                stream.write(f'{pad2}vih : {self._units.format_v(val)};\n')
                stream.write(f'{pad2}vimin : 0;\n')
                stream.write(f'{pad2}vimax : {self._units.format_v(val)};\n')
                stream.write(f'{pad}}}\n')
                # write output voltage
                stream.write(f'{pad}output_voltage({name}) {{\n')
                stream.write(f'{pad2}vol : 0;\n')
                stream.write(f'{pad2}voh : {self._units.format_v(val)};\n')
                stream.write(f'{pad2}vomin : 0;\n')
                stream.write(f'{pad2}vomax : {self._units.format_v(val)};\n')
                stream.write(f'{pad}}}\n')
            stream.write('\n')

        # write operating conditions
        stream.write(f'{pad}/* operating conditions */\n')
        for sim_env in self._sim_envs:
            sim_env.stream_out(stream, indent, self._units)
        sim_env_default = self._sim_envs[0]
        stream.write(f'{pad}default_operating_conditions : {sim_env_default.name};\n')
        stream.write(f'{pad}nom_process : {sim_env_default.process:.1f};\n')
        stream.write(f'{pad}nom_temperature : {sim_env_default.temperature:.1f};\n')
        stream.write(f'{pad}nom_voltage : {self._units.format_v(sim_env_default.voltage)};\n')
        stream.write('\n')

        # write lookup tables
        stream.write(f'{pad}/* lookup tables */\n')
        for lut_type in LUTType:
            self._luts[lut_type].stream_out(stream, indent, self._units)
        stream.write('\n')

        # write normalized driver waveforms
        stream.write(f'{pad}/* normalized driver waveforms */\n')
        for ndw in self._drv_wvfms.values():
            ndw.stream_out(stream, indent, self._units)

        # write bus types
        bus_types: Set[BusRange] = {r_ for c_ in self._cells for r_ in c_.bus_range_iter}
        if bus_types:
            stream.write('\n')
            stream.write(f'{pad}/* bus types */\n')
            for bus_range in bus_types:
                bus_range.stream_out(stream, indent)

        # write custom attribute definitions
        stream.write('\n')
        stream.write(f'{pad}/* custom attributes */\n')
        stream.write(f'{pad}define(cell_description, cell, string);\n')

        # write cells
        if self._cells:
            stream.write('\n')
            stream.write(f'{pad}/* cells */\n')
            for idx, c in enumerate(self._cells):
                if idx != 0:
                    stream.write('\n')

                c.stream_out(stream, indent, self._units)

        indent -= 4
        pad = ' ' * indent
        stream.write(f'{pad}}} /* end library {self._name} */\n')


_si_mag = [-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12]
_si_pre = ['a', 'f', 'p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T']


def _get_si_prefix(num: float) -> str:
    if abs(num) < 1e-21:
        raise ValueError(f'unsupported number: {num}')
    exp = math.log10(abs(num))

    pre_idx = len(_si_mag) - 1
    for idx in range(len(_si_mag)):
        if exp < _si_mag[idx]:
            pre_idx = idx - 1
            break

    res = 10.0 ** (_si_mag[pre_idx])
    mag = num / res
    if abs(mag - 1.0) > 1e-6:
        raise ValueError(f'unsupported number: {num}')
    return _si_pre[pre_idx]


def _to_string(obj: Any) -> str:
    if isinstance(obj, str):
        if '"' in obj:
            raise ValueError('Cannot write a string with double quotes in it.')
        if '<' in obj:
            # check to see if its using angled brackets, if so convert them to square brackets
            obj = obj.replace('<', '[')
            obj = obj.replace('>', ']')
        return f'"{obj}"'
    if isinstance(obj, bool):
        return 'true' if obj else 'false'

    raise ValueError(f'Unknown object: {obj}')


def _num_list_to_string(values: Iterable[float], units: Units, val_type: str) -> str:
    inner = ', '.join((units.format(val, val_type) for val in values))
    return f'"{inner}"'

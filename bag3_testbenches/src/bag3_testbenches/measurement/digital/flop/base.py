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

from typing import Any, Union, Tuple, Optional, Mapping, Dict, Sequence, Set, Iterable

import abc
import pprint
from enum import Enum, Flag, auto

import numpy as np

from bag.simulation.data import SimData

from bag3_liberty.enum import TimingType

from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import EdgeType


class FlopInputMode(Enum):
    IN = 0
    SI = 1
    SE = 2
    RECOVERY = 3
    REMOVAL = 4


class FlopMeasFlag(Flag):
    IN_RISING = auto()
    SETUP_RISING = auto()
    HOLD_RISING = auto()
    MEAS_SETUP = auto()

    @property
    def name(self) -> str:
        return f'{self.value}'

    @classmethod
    def from_str(cls, val: str) -> FlopMeasFlag:
        return FlopMeasFlag(int(val))


class FlopMeasMode:
    def __init__(self, in_mode: Union[str, FlopInputMode] = FlopInputMode.IN,
                 in_rising: bool = True, setup_rising: bool = True, hold_rising: bool = True,
                 meas_setup: bool = True) -> None:
        if isinstance(in_mode, str):
            self._in_mode = FlopInputMode[in_mode]
        else:
            self._in_mode = in_mode

        self._meas_flag = FlopMeasFlag(0)
        if in_rising:
            self._meas_flag |= FlopMeasFlag.IN_RISING
        if setup_rising:
            self._meas_flag |= FlopMeasFlag.SETUP_RISING
        if hold_rising:
            self._meas_flag |= FlopMeasFlag.HOLD_RISING
        if meas_setup:
            self._meas_flag |= FlopMeasFlag.MEAS_SETUP

    @property
    def input_mode_name(self) -> str:
        return self._in_mode.name

    @property
    def name(self) -> str:
        return f'{self._in_mode.name}_{self._meas_flag.name}'

    @property
    def is_input(self) -> bool:
        return self._in_mode is FlopInputMode.IN

    @property
    def is_scan_in(self) -> bool:
        return self._in_mode is FlopInputMode.SI

    @property
    def is_scan_en(self) -> bool:
        return self._in_mode is FlopInputMode.SE

    @property
    def is_recovery(self) -> bool:
        return self._in_mode is FlopInputMode.RECOVERY

    @property
    def is_removal(self) -> bool:
        return self._in_mode is FlopInputMode.REMOVAL

    @property
    def is_reset(self) -> bool:
        return self.is_recovery or self.is_removal

    @property
    def input_rising(self) -> bool:
        return not self.is_reset and FlopMeasFlag.IN_RISING in self._meas_flag

    @property
    def is_pos_edge_clk(self) -> bool:
        return FlopMeasFlag.SETUP_RISING in self._meas_flag

    @property
    def hold_opposite_clk(self) -> bool:
        return ((FlopMeasFlag.SETUP_RISING in self._meas_flag) !=
                (FlopMeasFlag.HOLD_RISING in self._meas_flag))

    @property
    def meas_setup(self) -> bool:
        return FlopMeasFlag.MEAS_SETUP in self._meas_flag

    @property
    def opposite_clk(self) -> FlopMeasMode:
        new_meas_flag = (self._meas_flag ^ FlopMeasFlag.SETUP_RISING) ^ FlopMeasFlag.HOLD_RISING
        return FlopMeasMode(self._in_mode, new_meas_flag)

    @classmethod
    def from_str(cls, val: str) -> FlopMeasMode:
        in_mode, code = val.rsplit('_', maxsplit=1)
        flag = FlopMeasFlag(int(code))
        return FlopMeasMode(in_mode=FlopInputMode[in_mode],
                            in_rising=FlopMeasFlag.IN_RISING in flag,
                            setup_rising=FlopMeasFlag.SETUP_RISING in flag,
                            hold_rising=FlopMeasFlag.HOLD_RISING in flag,
                            meas_setup=FlopMeasFlag.MEAS_SETUP in flag)

    @classmethod
    def from_dict(cls, val: Mapping[str, Any]) -> FlopMeasMode:
        return FlopMeasMode(**val)

    def get_out_edge(self, rst_to_high: bool, out_invert: bool) -> EdgeType:
        rising = (self.is_scan_en or self.input_rising or (self.is_reset and not rst_to_high))
        return EdgeType.RISE if (rising ^ out_invert) else EdgeType.FALL


class FlopTimingBase(DigitalTranTB, abc.ABC):
    """Base class of all flop timing TestbenchManagers.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    flop_params : Mapping[str, Any]
        Dictionary describing the flop data structure.
    meas_mode : Union[str, FlopMeasMode]
        the measurement mode.
    sim_env_name : str
        Use to query for sim_env dependent timing offset.
    sim_params : Mapping[str, float]
        Required entries are listed below.

        t_rst :
            the duration of reset signals.
        t_rst_rf :
            the reset signals rise/fall time, measured from thres_lo to thres_hi.
        t_clk_per :
            the clock period.
        t_clk_rf :
            the clock rise/fall time, measured from thres_lo to thres_hi.
        t_clk_delay :
            the clock delay, measured from end of reset period to 50% point.
        t_rf :
            the input rise/fall time, measured from thres_lo to thres_hi.
        t_recovery_<reset> :
            the recovery time.  Only defined during recovery simulation.
        t_removal_<reset> :
            the removal time.  Only defined during removal simulation.
        c_load :
            the load capacitance parameter.
        Furthermore, setup time for each input pin will have the variable t_setup_<pin>_<index>.
        For example, pin "foo<2>" has the variable "t_setup_foo_2", and pin "bar" has the
        variable "t_setup_bar_".  The same naming scheme applies to hold time.

    pwr_domain : Mapping[str, Tuple[str, str]]
        Dictionary from individual pin names or base names to (ground, power) pin name tuple.
    sup_values : Mapping[str, Union[float, Mapping[str, float]]]
        Dictionary from supply pin name to voltage values.
    dut_pins : Sequence[str]
        list of DUT pins.
    pin_values : Mapping[str, int]
        Dictionary from bus pin or scalar pin to the bit value as binary integer.
    reset_list : Sequence[Tuple[str, bool]]
        Optional.  List of reset pin name and reset type tuples.  Reset type is True for
        active-high, False for active-low.  Does not include flop reset pins.
    diff_list : Sequence[Tuple[Sequence[str], Sequence[str]]]
        Optional.  List of groups of differential pins.
    rtol : float
        Optional.  Relative tolerance for equality checking in timing measurement.
    atol : float
        Optional.  Absolute tolerance for equality checking in timing measurement.
    thres_lo : float
        Optional.  Low threshold value for rise/fall time calculation.  Defaults to 0.1
    thres_hi : float
        Optional.  High threshold value for rise/fall time calculation.  Defaults to 0.9
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._meas_mode = FlopMeasMode()
        self._flop_params: Dict[str, Any] = {}
        self._pulses: Sequence[Mapping[str, Any]] = []
        self._biases: Dict[str, int] = {}
        self._var_list: Sequence[str] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        val: Union[str, FlopMeasMode] = specs['meas_mode']
        flop_params_val: Mapping[str, Any] = specs['flop_params']

        self._meas_mode = val if isinstance(val, FlopMeasMode) else FlopMeasMode.from_str(val)
        self._flop_params = self.get_default_flop_params()
        self._flop_params.update(flop_params_val)

        self._pulses, self._biases, out_set, self._var_list = self.get_stimuli()
        specs['save_outputs'] = list(out_set)

    @property
    @abc.abstractmethod
    def num_cycles(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def c_load_pins(self) -> Iterable[str]:
        pass

    @property
    def t_clk_expr(self) -> str:
        mode = self.meas_mode
        clk_idx = self.num_cycles - 1
        if (not mode.meas_setup) and mode.hold_opposite_clk:
            clk_idx += 0.5
        return f'{self.t_rst_end_expr}+t_clk_delay+{clk_idx}*t_clk_per'

    @property
    def t_start_expr(self) -> str:
        return f'{self.t_clk_expr}-(t_clk_rf/{2 * self.trf_scale:.2f})'

    @property
    def meas_mode(self) -> FlopMeasMode:
        return self._meas_mode

    @property
    def flop_params(self) -> Dict[str, Any]:
        return self._flop_params

    @property
    def timing_variables(self) -> Sequence[str]:
        return self._var_list

    @classmethod
    @abc.abstractmethod
    def get_default_flop_params(cls) -> Dict[str, Any]:
        pass

    @classmethod
    @abc.abstractmethod
    def get_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        pass

    @classmethod
    @abc.abstractmethod
    def get_output_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        pass

    @classmethod
    def get_clk_pulse(cls, clk_pin: str, clk_rising: bool) -> Mapping[str, Any]:
        return dict(pin=clk_pin, tper='t_clk_per', tpw='t_clk_per/2',
                    trf='t_clk_rf', td='t_clk_delay', pos=clk_rising)

    @classmethod
    def get_input_pulse(cls, pin: str, var_setup: str, var_hold: str, pos: bool,
                        cycle_idx: int = 1, hold_opposite: bool = False) -> Mapping[str, Any]:
        clk_mid = 't_clk_delay' if cycle_idx == 0 else f't_clk_delay+{cycle_idx}*t_clk_per'
        if hold_opposite:
            tpw = f'{var_setup}+{var_hold}+(t_clk_per/2)'
        else:
            tpw = f'{var_setup}+{var_hold}'
        return dict(pin=pin, tper='2*t_sim', tpw=tpw, trf='t_rf',
                    td=f'{clk_mid}-{var_setup}', pos=pos)

    @abc.abstractmethod
    def get_stimuli(self) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, int], Set[str],
                                   Sequence[str]]:
        pass

    @abc.abstractmethod
    def get_output_map(self, output_timing: bool
                       ) -> Mapping[str, Tuple[Mapping[str, Any],
                                               Sequence[Tuple[EdgeType, Sequence[str]]]]]:
        pass

    def get_timing_type(self, non_seq: bool) -> TimingType:
        mode = self.meas_mode
        if mode.is_reset:
            ans = TimingType.recovery_rising if mode.is_recovery else TimingType.removal_rising
        else:
            if mode.meas_setup:
                ans = TimingType.setup_rising if mode.is_pos_edge_clk else TimingType.setup_falling
            elif mode.is_pos_edge_clk ^ mode.hold_opposite_clk:
                ans = TimingType.hold_rising
            else:
                ans = TimingType.hold_falling
            ans = ans.with_non_seq(non_seq)

        return ans

    def get_timing_info(self, meas_mode: FlopMeasMode, in_pins: Sequence[str],
                        clk_pin: str, cond_str: str, rst_active_high: bool,
                        inc_delay: bool = True, non_seq: bool = False,
                        offset: Union[float, Mapping[str, float]] = 0) -> Mapping[str, Any]:
        data_types = ['rise_constraint', 'fall_constraint']
        if meas_mode.is_reset:
            data_idx = int(rst_active_high)
        elif meas_mode.meas_setup:
            data_idx = int(not meas_mode.input_rising)
        else:
            # NOTE: meas_mode.is_rising returns True if the setup edge of data is rising,
            # so for hold measurement, we need to flip the rise/fall type.
            data_idx = int(meas_mode.input_rising)

        ttype_str = self.get_timing_type(non_seq).name
        pin_data_list = []
        for in_pin in in_pins:
            in_diff_grp = self.get_diff_groups(in_pin)
            pin_data_list.extend(((pn_, data_types[data_idx]) for pn_ in in_diff_grp[0]))
            pin_data_list.extend(((pn_, data_types[data_idx ^ 1]) for pn_ in in_diff_grp[1]))

        if isinstance(offset, Mapping):
            offset_val = offset[self.specs['sim_env_name']]
        else:
            offset_val = offset
        return dict(pin_data_list=pin_data_list, related=clk_pin, cond=cond_str,
                    timing_type=ttype_str, inc_delay=inc_delay, offset=offset_val)

    def get_t_clk(self, data: SimData) -> np.ndarray:
        return self.get_calculator(data).eval(self.t_clk_expr)

    def get_t_start(self, data: SimData) -> np.ndarray:
        return self.get_calculator(data).eval(self.t_start_expr)

    def get_rst_pulse(self, rst_pin: str, rst_active_high: bool, var_name: str = '',
                      is_recovery: bool = False) -> Mapping[str, Any]:
        if var_name:
            if is_recovery:
                td = f't_clk_delay-{var_name}'
            else:
                td = f't_clk_delay+{var_name}'
            return dict(pin=rst_pin, tper='2*t_sim', tpw='t_sim', trf='t_rf',
                        td=td, pos=not rst_active_high)
        return dict(pin=rst_pin, tper='2*t_sim', tpw='t_sim', trf='t_rst_rf',
                    td=f't_rst+(t_rst_rf/{2 * self.trf_scale:.2f})', pos=not rst_active_high,
                    td_after_rst=False)

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """Set up PWL waveform files."""
        if sch_params is None:
            return None

        specs = self.specs
        sup_values: Mapping[str, Union[float, Mapping[str, float]]] = specs['sup_values']
        pwr_domain: Mapping[str, Tuple[str, str]] = specs['pwr_domain']
        dut_pins: Sequence[str] = specs['dut_pins']
        reset_list: Sequence[Tuple[str, bool]] = specs.get('reset_list', [])

        self.sim_params['t_sim'] = f't_rst+t_rst_rf+t_clk_delay+{self.num_cycles}*t_clk_per'

        src_list = []
        src_pins = set()

        # add DC voltage sources for stimuli
        for pin_name, value in self._biases.items():
            sup_tuple = self.get_pin_supplies(pin_name, pwr_domain)
            pos_var = self.sup_var_name(sup_tuple[value])
            neg_var = self.sup_var_name(sup_tuple[value ^ 1])
            diff_grp = self.get_diff_groups(pin_name)
            for ppin in diff_grp[0]:
                src_pins.add(ppin)
                src_list.append(dict(type='vdc', lib='analogLib', value=pos_var,
                                     conns=dict(PLUS=ppin, MINUS='VSS')))
            for npin in diff_grp[1]:
                src_pins.add(npin)
                src_list.append(dict(type='vdc', lib='analogLib', value=neg_var,
                                     conns=dict(PLUS=npin, MINUS='VSS')))

        self.get_bias_sources(sup_values, src_list, src_pins)
        self.get_pulse_sources(self._pulses, src_list, src_pins)
        # NOTE: make sure we exclude flop reset sources
        self.get_reset_sources(reset_list, src_list, src_pins, skip_src=True)
        # add capacitance loads
        load_list = [dict(pin=opin, type='cap', value='c_load') for opin in self.c_load_pins]
        self.get_loads(load_list, src_list)

        dut_conns = self.get_dut_conns(dut_pins, src_pins)
        return dict(
            dut_lib=sch_params.get('dut_lib', ''),
            dut_cell=sch_params.get('dut_cell', ''),
            dut_params=sch_params.get('dut_params', None),
            dut_conns=dut_conns,
            vbias_list=[],
            src_list=src_list,
        )

    def print_results(self, data: SimData) -> None:
        fun = print if self.logger is None else self.logger.info
        fun(f'meas mode: {self.meas_mode.name}')
        out_map = self.get_output_map(False)
        for timing_info, edge_out_list in out_map.values():
            fun(pprint.pformat(timing_info, width=100))
            for edge_type, out_list in edge_out_list:
                for out_pin in out_list:
                    td = self.calc_clk_to_q(data, out_pin, edge_type)
                    trf = self.calc_out_trf(data, out_pin, edge_type)
                    fun(f'{out_pin} {edge_type.name}:\ntd:\n{td}\ntrf:\n{trf}')

    def calc_clk_to_q(self, data: SimData, out_name: str, out_edge: EdgeType) -> np.ndarray:
        """Get clk-to-q- delay.

        Currently assumes clock parameters and voltage parameters are not swept.

        Parameters
        ----------
        data : SimData
            the simulation result data structure.
        out_name : str
            the output pin name.
        out_edge : EdgeType
            the output edge type.

        Returns
        -------
        ans : np.ndarray
            the clk-to-q delay.
        """
        calc = self.get_calculator(data)
        t_start = calc.eval(self.t_start_expr)
        t_clk = calc.eval(self.t_clk_expr)
        ans = self.calc_cross(data, out_name, out_edge, t_start=t_start)
        ans -= t_clk
        return ans

    def calc_out_trf(self, data: SimData, out_name: str, out_edge: EdgeType) -> np.ndarray:
        """Get output rise/fall time.

        Parameters
        ----------
        data : SimData
            the simulation result data structure.
        out_name : str
            the output pin name.
        out_edge : EdgeType
            the output edge type.

        Returns
        -------
        ans : np.ndarray
            the output rise/fall time.
        """
        return self.calc_trf(data, out_name, out_edge is EdgeType.RISE,
                             t_start=self.get_t_start(data))

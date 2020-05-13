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

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, Iterable, List, Set

from itertools import chain

import numpy as np

from pybag.core import get_cdba_name_bits

from bag.simulation.data import SimData, AnalysisType

from bag3_liberty.data import parse_cdba_name

from ..data.tran import EdgeType, get_first_crossings
from .base import TranTB


class DigitalTranTB(TranTB):
    """A transient testbench with digital stimuli.  All pins are connected to either 0 or 1.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, float]
        Required entries are listed below.

        t_sim : float
            the total simulation time.
        t_rst : float
            the duration of reset signals.
        t_rst_rf : float
            the reset signals rise/fall time.
    dut_pins : Sequence[str]
        list of DUT pins.
    pulse_list : Sequence[Mapping[str, Any]]
        Optional.  List of pulse sources.  Each dictionary has the following entries:

        pin : str
            the pin to connect to.
        tper : Union[float, str]
            period.
        tpw : Union[float, str]
            the pulse width, measures from 50% to 50%, i.e. it is tper/2 for 50% duty cycle.
        trf : Union[float, str]
            rise/fall time as defined by thres_lo and thres_hi.
        td : Union[float, str]
            Optional.  Pulse delay in addition to any reset period,  Measured from the end of
            reset period to the 50% point of the first edge.
        pos : bool
            Defaults to True.  True if this is a positive pulse (010).
        td_after_rst: bool
            Defaults to True.  True if td is measured from the end of reset period, False
            if td is measured from t=0.

    load_list : Sequence[Mapping[str, Any]]
        Optional.  List of loads.  Each dictionary has the following entries:

        pin: str
            the pin to connect to.
        nin: str
            Optional, the negative pin to connect to
        type : str
            the load device type.
        value : Union[float, str]
            the load parameter value.
    pwr_domain : Mapping[str, Tuple[str, str]]
        Dictionary from individual pin names or base names to (ground, power) pin name tuple.
    sup_values : Mapping[str, Union[float, Mapping[str, float]]]
        Dictionary from supply pin name to voltage values.
    pin_values : Mapping[str, Union[int, str]]
        Dictionary from bus pin or scalar pin to the bit value as binary integer, or a pin name
        to short pins to nets.
    reset_list : Sequence[Tuple[str, bool]]
        Optional.  List of reset pin name and reset type tuples.  Reset type is True for
        active-high, False for active-low.
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
    skip_src : bool
        Defaults to True.  If True, ignore multiple stimuli on same pin (only use the
        first stimuli).
    subclasses' specs dictionary must have pwr_domain, rtol, atol, thres_lo, and thres_hi.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._diff_lookup: Mapping[str, Tuple[Sequence[str], Sequence[str]]] = {}
        self._bit_values: Mapping[str, Union[int, str]] = {}
        self._thres_lo: float = 0.1
        self._thres_hi: float = 0.9

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        diff_list: Sequence[Tuple[Sequence[str], Sequence[str]]] = specs.get('diff_list', [])
        pin_values: Mapping[str, Union[int, str]] = specs.get('pin_values', {})

        self._diff_lookup = self.get_diff_lookup(diff_list)
        self._bit_values = self._get_pin_bit_values(pin_values)
        self._thres_lo = specs.get('thres_lo', 0.1)
        self._thres_hi = specs.get('thres_hi', 0.9)
        thres_delay = specs.get('thres_delay', 0.5)
        if abs(thres_delay - 0.5) > 1e-4:
            raise ValueError('only thres_delay = 0.5 is supported.')

    @property
    def save_outputs(self) -> Sequence[str]:
        """Sequence[str]: list of saved output nodes.

        Override parent class property to automatically add differential pins.
        """
        save_outputs: Optional[List[str]] = self.specs.get('save_outputs', None)
        if save_outputs is None:
            return []

        out_set = set()
        for pin in save_outputs:
            pos_pins, neg_pins = self.get_diff_groups(pin)
            out_set.update(pos_pins)
            out_set.update(neg_pins)

        return list(out_set)

    @property
    def t_rst_end_expr(self) -> str:
        return f't_rst+t_rst_rf/{self.trf_scale:.2f}'

    @property
    def thres_lo(self) -> float:
        return self._thres_lo

    @property
    def thres_hi(self) -> float:
        return self._thres_hi

    @property
    def trf_scale(self) -> float:
        return self._thres_hi - self._thres_lo

    @classmethod
    def get_pin_supplies(cls, pin_name: str, pwr_domain: Mapping[str, Tuple[str, str]]
                         ) -> Tuple[str, str]:
        ans = pwr_domain.get(pin_name, None)
        if ans is None:
            # check if this is a r_src pin
            pin_base = cls.get_r_src_pin_base(pin_name)
            if pin_base:
                return pwr_domain[pin_base]

            # check if this is a bus pin, and pwr_domain is specified for the whole bus
            basename = parse_cdba_name(pin_name)[0]
            return pwr_domain[basename]
        return ans

    @classmethod
    def get_diff_lookup(cls, diff_list: Sequence[Tuple[Sequence[str], Sequence[str]]]
                        ) -> Mapping[str, Tuple[Sequence[str], Sequence[str]]]:
        ans = {}
        for pos_pins, neg_pins in diff_list:
            ppin_bits = [bit_name for ppin in pos_pins for bit_name in get_cdba_name_bits(ppin)]
            npin_bits = [bit_name for npin in neg_pins for bit_name in get_cdba_name_bits(npin)]
            pos_pair = (ppin_bits, npin_bits)
            neg_pair = (npin_bits, ppin_bits)
            for ppin in ppin_bits:
                ans[ppin] = pos_pair
            for npin in npin_bits:
                ans[npin] = neg_pair
        return ans

    @classmethod
    def get_r_src_pin(cls, in_pin: str) -> str:
        return in_pin + '_rs_'

    @classmethod
    def get_r_src_pin_base(cls, pin_name: str) -> str:
        return pin_name[:-4] if pin_name.endswith('_rs_') else ''

    def get_t_rst_end(self, data: SimData) -> np.ndarray:
        t_rst = self.get_param_value('t_rst', data)
        t_rst_rf = self.get_param_value('t_rst_rf', data)
        return t_rst + t_rst_rf / self.trf_scale

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """Set up PWL waveform files."""
        if sch_params is None:
            return None

        specs = self.specs
        sup_values: Mapping[str, Union[float, Mapping[str, float]]] = specs['sup_values']
        dut_pins: Sequence[str] = specs['dut_pins']
        pulse_list: Sequence[Mapping[str, Any]] = specs.get('pulse_list', [])
        reset_list: Sequence[Tuple[str, bool]] = specs.get('reset_list', [])
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])

        src_list = []
        src_pins = set()
        self.get_pulse_sources(pulse_list, src_list, src_pins)
        self.get_bias_sources(sup_values, src_list, src_pins)
        self.get_reset_sources(reset_list, src_list, src_pins, skip_src=True)
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

    def get_dut_conns(self, dut_pins: Iterable[str], src_pins: Set[str]) -> Mapping[str, str]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        ans = {}
        for pin_name in dut_pins:
            # build net expression list
            last_bit = ''
            cur_cnt = 0
            net_list = []
            for bit_name in get_cdba_name_bits(pin_name):
                if bit_name not in src_pins:
                    bit_val = self._bit_values.get(bit_name, None)
                    if bit_val is not None:
                        if isinstance(bit_val, int):
                            bit_name = self.get_pin_supplies(bit_name, pwr_domain)[bit_val]
                        else:
                            bit_name = bit_val
                if bit_name == last_bit:
                    cur_cnt += 1
                else:
                    if last_bit:
                        net_list.append(last_bit if cur_cnt == 1 else f'<*{cur_cnt}>{last_bit}')
                    last_bit = bit_name
                    cur_cnt = 1

            if last_bit:
                net_list.append(last_bit if cur_cnt == 1 else f'<*{cur_cnt}>{last_bit}')
            ans[pin_name] = ','.join(net_list)

        return ans

    def get_loads(self, load_list: Iterable[Mapping[str, Any]],
                  src_load_list: List[Mapping[str, Any]]) -> None:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        for params in load_list:
            pin: str = params['pin']
            value: Union[float, str] = params['value']
            dev_type: str = params['type']
            pos_pins, neg_pins = self.get_diff_groups(pin)
            if 'nin' in params:
                nin: str = params['nin']
                npos_pins, nneg_pins = self.get_diff_groups(nin)

                for pin_name, nin_name in zip(chain(pos_pins, neg_pins),
                                              chain(npos_pins, nneg_pins)):
                    src_load_list.append(dict(type=dev_type, lib='analogLib', value=value,
                                              conns=dict(PLUS=pin_name, MINUS=nin_name)))
            else:
                gnd_name = self.get_pin_supplies(pin, pwr_domain)[0]

                for pin_name in chain(pos_pins, neg_pins):
                    src_load_list.append(dict(type=dev_type, lib='analogLib', value=value,
                                              conns=dict(PLUS=pin_name, MINUS=gnd_name)))

    def get_reset_sources(self, reset_list: Iterable[Tuple[str, bool]],
                          src_list: List[Mapping[str, Any]], src_pins: Set[str],
                          skip_src: bool = False) -> None:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        trf_scale = self.trf_scale
        for pin, active_high in reset_list:
            if pin in src_pins:
                if skip_src:
                    continue
                else:
                    raise ValueError(f'Cannot add reset source on pin {pin}, already used.')

            gnd_name, pwr_name = self.get_pin_supplies(pin, pwr_domain)
            if active_high:
                v1 = f'v_{pwr_name}'
                v2 = f'v_{gnd_name}'
            else:
                v1 = f'v_{gnd_name}'
                v2 = f'v_{pwr_name}'

            trf_str = f't_rst_rf/{trf_scale:.2f}'
            pval_dict = dict(v1=v1, v2=v2, td='t_rst', per='2*t_sim', pw='t_sim',
                             tr=trf_str, tf=trf_str)
            self._add_diff_sources(pin, [pval_dict], '', src_list, src_pins)

    def get_pulse_sources(self, pulse_list: Iterable[Mapping[str, Any]],
                          src_list: List[Mapping[str, Any]], src_pins: Set[str]) -> None:
        specs = self.specs
        pwr_domain: Mapping[str, Tuple[str, str]] = specs['pwr_domain']
        skip_src: bool = specs.get('skip_src', False)

        trf_scale = self.trf_scale
        td_rst = f't_rst+(t_rst_rf/{trf_scale:.2f})'
        for pulse_params in pulse_list:
            pin: str = pulse_params['pin']
            rs: Union[float, str] = pulse_params.get('rs', '')
            vadd_list: Optional[Sequence[Mapping[str, Any]]] = pulse_params.get('vadd_list', None)

            if pin in src_pins:
                if skip_src:
                    continue
                else:
                    raise ValueError(f'Cannot add pulse source on pin {pin}, already used.')

            if not vadd_list:
                vadd_list = [pulse_params]

            gnd_name, pwr_name = self.get_pin_supplies(pin, pwr_domain)
            ptable_list = []
            for table in vadd_list:
                tper: Union[float, str] = table['tper']
                tpw: Union[float, str] = table['tpw']
                trf: Union[float, str] = table['trf']
                td: Union[float, str] = table.get('td', '')
                pos: bool = table.get('pos', True)
                td_after_rst: bool = table.get('td_after_rst', True)
                extra: Mapping[str, Union[float, str]] = table.get('extra', {})

                if pos:
                    v1 = f'v_{gnd_name}'
                    v2 = f'v_{pwr_name}'
                else:
                    v1 = f'v_{pwr_name}'
                    v2 = f'v_{gnd_name}'

                if isinstance(trf, float):
                    trf /= trf_scale
                    trf2 = self.get_sim_param_string(trf / 2)
                    trf = self.get_sim_param_string(trf)
                else:
                    trf2 = f'({trf})/{2 * trf_scale:.2f}'
                    trf = f'({trf})/{trf_scale:.2f}'

                if not td:
                    td = td_rst if td_after_rst else '0'
                else:
                    td = self.get_sim_param_string(td)
                    if td_after_rst:
                        td = f'{td_rst}+{td}-{trf2}'
                    else:
                        td = f'{td}-{trf2}'

                tpw = self.get_sim_param_string(tpw)
                ptable_list.append(dict(v1=v1, v2=v2, td=td, per=tper, pw=f'{tpw}-{trf}',
                                        tr=trf, tf=trf, **extra))

            self._add_diff_sources(pin, ptable_list, rs, src_list, src_pins)

    def get_pin_supply_values(self, pin_name: str, data: SimData) -> Tuple[np.ndarray, np.ndarray]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        gnd_pin, pwr_pin = self.get_pin_supplies(pin_name, pwr_domain)
        gnd_var = self.sup_var_name(gnd_pin)
        pwr_var = self.sup_var_name(pwr_pin)

        return self.get_param_value(gnd_var, data), self.get_param_value(pwr_var, data)

    def get_diff_groups(self, pin_name: str) -> Tuple[Sequence[str], Sequence[str]]:
        pin_base = self.get_r_src_pin_base(pin_name)
        if pin_base:
            diff_grp = self._diff_lookup.get(pin_base, None)
            if diff_grp is None:
                return [pin_name], []
            pos_pins = [self.get_r_src_pin(p_) for p_ in diff_grp[0]]
            neg_pins = [self.get_r_src_pin(p_) for p_ in diff_grp[1]]
            return pos_pins, neg_pins
        else:
            diff_grp = self._diff_lookup.get(pin_name, None)
            if diff_grp is None:
                return [pin_name], []
            return diff_grp

    def calc_cross(self, data: SimData, out_name: str, out_edge: EdgeType,
                   t_start: Union[np.ndarray, float, str] = 0,
                   t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        out_vec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vth_out = (out_1 - out_0) * thres_delay + out_0
        out_c = get_first_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                    stop=t_stop, rtol=rtol, atol=atol)
        return out_c

    def calc_delay(self, data: SimData, in_name: str, out_name: str, in_edge: EdgeType,
                   out_edge: EdgeType, t_start: Union[np.ndarray, float, str] = 0,
                   t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        in_0, in_1 = self.get_pin_supply_values(in_name, data)
        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        in_vec = data[in_name]
        out_vec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vth_in = (in_1 - in_0) * thres_delay + in_0
        vth_out = (out_1 - out_0) * thres_delay + out_0
        in_c = get_first_crossings(tvec, in_vec, vth_in, etype=in_edge, start=t_start, stop=t_stop,
                                   rtol=rtol, atol=atol)
        out_c = get_first_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                    stop=t_stop, rtol=rtol, atol=atol)
        out_c -= in_c
        return out_c

    def calc_trf(self, data: SimData, out_name: str, out_rise: bool, allow_inf: bool = False,
                 t_start: Union[np.ndarray, float, str] = 0,
                 t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        specs = self.specs
        logger = self.logger
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        yvec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vdiff = out_1 - out_0
        vth_0 = out_0 + self._thres_lo * vdiff
        vth_1 = out_0 + self._thres_hi * vdiff
        if out_rise:
            edge = EdgeType.RISE
            t0 = get_first_crossings(tvec, yvec, vth_0, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
            t1 = get_first_crossings(tvec, yvec, vth_1, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
        else:
            edge = EdgeType.FALL
            t0 = get_first_crossings(tvec, yvec, vth_1, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
            t1 = get_first_crossings(tvec, yvec, vth_0, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)

        has_nan = np.isnan(t0).any() or np.isnan(t1).any()
        has_inf = np.isinf(t0).any() or np.isinf(t1).any()
        if has_nan or (has_inf and not allow_inf):
            logger.warn(f'Got invalid value(s) in computing {edge.name} time of pin {out_name}.\n'
                        f't0:\n{t0}\nt1:\n{t1}')
            t1.fill(np.inf)
        else:
            t1 -= t0

        return t1

    def _get_pin_bit_values(self, pin_values: Mapping[str, Union[int, str]]
                            ) -> Mapping[str, Union[int, str]]:
        ans = {}
        for pin_name, pin_val in pin_values.items():
            bit_list = get_cdba_name_bits(pin_name)
            nlen = len(bit_list)
            if isinstance(pin_val, str):
                # user specify another pin to short to
                val_list = get_cdba_name_bits(pin_val)
                if len(val_list) != len(bit_list):
                    raise ValueError(
                        f'Cannot connect pin {pin_name} to {pin_val}, length mismatch.')

                for bit_name, net_name in zip(bit_list, val_list):
                    pos_bits, neg_bits = self.get_diff_groups(bit_name)
                    pos_nets, neg_nets = self.get_diff_groups(net_name)
                    for p_ in pos_bits:
                        ans[p_] = pos_nets[0]
                    for p_ in neg_bits:
                        ans[p_] = neg_nets[0]
            else:
                # user specify pin values
                bin_str = bin(pin_val)[2:].zfill(nlen)
                for bit_name, val_char in zip(bit_list, bin_str):
                    pin_val = int(val_char == '1')
                    pos_bits, neg_bits = self.get_diff_groups(bit_name)
                    for p_ in pos_bits:
                        ans[p_] = pin_val
                    for p_ in neg_bits:
                        ans[p_] = pin_val ^ 1

        return ans

    def _add_diff_sources(self, pin: str, ptable_list: Sequence[Mapping[str, Any]],
                          rs: Union[float, str], src_list: List[Mapping[str, Any]],
                          src_pins: Set[str]) -> None:
        pos_pins, neg_pins = self.get_diff_groups(pin)
        self._add_diff_sources_helper(pos_pins, ptable_list, rs, src_list, src_pins)
        if neg_pins:
            ntable_list = []
            for ptable in ptable_list:
                ntable = dict(**ptable)
                ntable['v1'] = ptable['v2']
                ntable['v2'] = ptable['v1']
                ntable_list.append(ntable)

            self._add_diff_sources_helper(neg_pins, ntable_list, rs, src_list, src_pins)

    def _add_diff_sources_helper(self, pin_list: Sequence[str],
                                 table_list: Sequence[Mapping[str, Any]],
                                 rs: Union[float, str], src_list: List[Mapping[str, Any]],
                                 src_pins: Set[str]) -> None:
        num_pulses = len(table_list)
        for pin_name in pin_list:
            if pin_name in src_pins:
                raise ValueError(f'Cannot add pulse source on pin {pin_name}, '
                                 f'already used.')
            if rs:
                pulse_pin = self.get_r_src_pin(pin_name)
                src_list.append(dict(type='res', lib='analogLib', value=rs,
                                     conns=dict(PLUS=pin_name, MINUS=pulse_pin)))
            else:
                pulse_pin = pin_name

            bot_pin = 'VSS'
            for idx, table in enumerate(table_list):
                top_pin = pulse_pin if idx == num_pulses - 1 else f'{pin_name}_vadd{idx}_'
                src_list.append(dict(type='vpulse', lib='analogLib', value=table,
                                     conns=dict(PLUS=top_pin, MINUS=bot_pin)))
                bot_pin = top_pin
            src_pins.add(pin_name)

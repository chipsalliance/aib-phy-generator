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

from typing import Any, Dict, Tuple, Optional, List, Mapping, Union, Sequence, cast, Set

import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_digital.measurement.util import get_digital_wrapper_params
from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.digital.util import setup_digital_tran
from bag3_testbenches.measurement.search import IntervalSearchMM, AcceptMode
from bag3_testbenches.measurement.tran.digital import DigitalTranTB


class PhaseDetMeasManagerUnit(IntervalSearchMM):
    """
    This measurement manager performs an interval search on the CLKA or CLKB to figure out what is the minimum
    difference between the two edges that can be discerned
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.adj_dict: Mapping[str, Any] = {}

    def process_init(self, cur_info: MeasInfo, sim_results: SimResults) -> Tuple[Dict[str, Any], bool]:
        return {}, False

    def init_search(self, sim_db: SimulationDB, dut: DesignInstance
                    ) -> Tuple[TestbenchManager, Mapping[str, Any], Mapping[str, Mapping[str, Any]],
                               Mapping[str, Any], bool, bool]:
        specs = self.specs
        assert specs['sigma_avt'] >= 0
        intv_params: Mapping[str, Any] = specs['intv_params']
        out_pins: Sequence[str] = specs.get('out_pins', ['t_up', 't_down'])  # specified as ['up', 'down']
        early_clk: str = specs['early_clk']
        late_clk: str = specs['late_clk']
        reset_list: Sequence[Tuple[str, bool]] = specs.get('rst_pins', [('RSTb', False)])
        invert_clk: bool = specs['invert_clk']
        clk_pol = not invert_clk

        clk_pins = [early_clk, late_clk]
        load_list = [dict(pin=out_pin, value='cload', type='cap') for out_pin in out_pins]
        early_clk_pulse = dict(pin=early_clk, tper='5/4*t_bit+t_delay', tpw='3/4*t_bit', trf='t_rf', td='t_bit+t_bit/4',
                               pos=clk_pol)
        late_clk_pulse = dict(pin=late_clk, tper='2*t_bit', tpw='t_bit', trf='t_rf', td='t_bit', pos=clk_pol)
        pulse_list = [early_clk_pulse, late_clk_pulse]
        digital_tran_tb_params = dict(
            pulse_list=pulse_list,
            reset_list=reset_list,
            load_list=load_list,
        )
        wrapper_params = get_digital_wrapper_params(specs, dut, clk_pins)
        tbm_specs, tb_params = setup_digital_tran(specs, dut, wrapper_params=wrapper_params,
                                                  **digital_tran_tb_params)
        # TODO Add this back in
        # for v_offset in self.specs['strongarm_offset_params'][mos]:
        #     tbm.sim_params[v_offset] = 3 * self.specs['sigma_avt'] * (1 if pol == 'pos' else -1)

        tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, tbm_specs))
        self.adj_dict = {'t_delay': intv_params}

        return tbm, tb_params, self.adj_dict, {}, False, True

    def process_output_helper(self, cur_info: MeasInfo, sim_results: SimResults, remaining: Set[str]
                              ) -> Mapping[str, Tuple[Tuple[float, float], Dict[str, Any], bool]]:
        specs = self.specs
        early_clk: str = specs['early_clk']
        adj_up: bool = specs['adj_up']
        out_sig: str = specs['out_sig']
        out_edge = EdgeType.RISE if specs['out_rising'] else EdgeType.FALL
        accept_mode = AcceptMode.POSITIVE if specs['accept_mode'] else AcceptMode.NEGATIVE
        cur_state = cur_info.state
        data = sim_results.data
        tbm = cast(DigitalTranTB, sim_results.tbm)

        t0 = tbm.get_t_rst_end(data)
        td = tbm.calc_delay(data, early_clk, out_sig, EdgeType.RISE, out_edge, t_start=t0)[0, ...]
        diff = 1-td
        # We want td to look for when td never settles, (where it will be inf), 1 is used because td should never be 1s
        arg, accept, low, high = self.get_adj_interval('t_delay', adj_up, data['t_delay'], diff,
                                                       accept_mode=accept_mode)
        delay = data['t_delay'][arg]
        new_result = {'value': delay, 'low': low, 'high': high, accept: 'accept'}
        self.log_result(cur_state, new_result)
        return {'t_delay': ((low, high), new_result, accept)}


class PhaseDetMeasManager(MeasurementManager):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        return False, MeasInfo("up_rising", {})

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        delay = sim_results.data['t_delay']['value']
        t_bit = self.specs['tbm_specs']['sim_params']['t_bit']
        delay = abs(delay-t_bit/2)
        if cur_info.state == 'up_rising':
            return False, MeasInfo("down_falling", dict(up_delay=delay))
        else:
            dct = cur_info.prev_results
            dct['down_delay'] = delay
            return True, MeasInfo("", dct)

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]], MeasurementManager], bool]:
        unit_specs = self._specs.copy()
        if cur_info.state == 'up_rising':
            unit_specs['early_clk'] = 'CLKB'
            unit_specs['late_clk'] = 'CLKA'
            unit_specs['out_rising'] = False
            unit_specs['out_sig'] = 't_up'
            unit_specs['adj_up'] = False
            unit_specs['invert_clk'] = False
        elif cur_info.state == 'down_falling':
            unit_specs['early_clk'] = 'CLKA'
            unit_specs['late_clk'] = 'CLKB'
            unit_specs['out_rising'] = False
            unit_specs['out_sig'] = 't_down'
            unit_specs['adj_up'] = False
            unit_specs['invert_clk'] = True
        mm = self.make_mm(PhaseDetMeasManagerUnit, unit_specs)
        return mm, True

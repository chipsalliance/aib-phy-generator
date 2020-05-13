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

from typing import Any, Union, Tuple, Optional, Mapping, Dict, cast

import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.util.math import Calculator

from bag3_testbenches.measurement.dc.base import DCTB


class DriverPullUpDownMM(MeasurementManager):
    default_gate_bias = dict(pu='full', pd='full')

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._tbm_info: Optional[Tuple[DCTB, Mapping[str, Any]]] = None
        self._mos_mapping = {}

        super().__init__(*args, **kwargs)

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        specs = self.specs
        vdd = specs['vdd']
        v_offset_map = specs['v_offset_map']

        gate_bias = self.default_gate_bias.copy()
        if 'gate_bias' in specs:
            gate_bias.update(specs['gate_bias'])

        specs['stack_pu'] = dut.sch_master.params['stack_p']
        specs['stack_pd'] = dut.sch_master.params['stack_n']
        self._dut = dut

        mos_mapping = specs['mos_mapping']['lay' if specs['extract'] else 'sch']

        # Find all transistors that match the passed in transistor names
        # The passed in transistor name can match multiple "transistors" in the netlist
        # in cases like seg > 1 or stack > 1
        # TODO: allow for more complex name matching (e.g., via regex)
        for mos, term in mos_mapping.items():
            voff_list = [v for k, v in v_offset_map.items() if term in k and k.endswith('_d')]
            if len(voff_list) == 0:
                raise ValueError(f"No matching transistor found for {term}")
            self._mos_mapping[mos] = voff_list

        for k, v in gate_bias.items():
            if v == 'full':
                gate_bias[k] = vdd if k == 'pd' else 0
            else:
                gate_bias = self._eval_expr(v)

        sup_values = dict(VDD=vdd, pden=gate_bias['pd'], puenb=gate_bias['pu'], out=vdd / 2)

        tbm_specs = dict(**specs['tbm_specs'])
        tbm_specs['sweep_var'] = 'v_out'
        tbm_specs['sweep_options'] = dict(type='LINEAR')
        for k in ['pwr_domain', 'sim_params', 'pin_values']:
            if k not in tbm_specs:
                tbm_specs[k] = {}
        tbm_specs['dut_pins'] = list(dut.sch_master.pins.keys())
        tbm_specs['load_list'] = []
        tbm_specs['sup_values'] = sup_values

        # Set all internal DC sources to 0 (since these are just used for current measurements)
        tbm_specs['sim_params'].update({k: 0 for k in v_offset_map.values()})

        tbm = cast(DCTB, sim_db.make_tbm(DCTB, tbm_specs))
        self._tbm_info = tbm, {}

        return False, MeasInfo('pd', {})

    def _eval_expr(self, expr) -> float:
        if isinstance(expr, str):
            return Calculator.evaluate(expr, dict(vdd=self.specs['vdd']))
        else:
            return expr

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        tbm = self._tbm_info[0]
        drain_bias = self.specs['drain_bias'][cur_info.state]
        drain_bias = list(map(self._eval_expr, drain_bias))

        swp_options = dict(num=len(drain_bias))
        if len(drain_bias) == 1:
            swp_options['start'] = swp_options['stop'] = drain_bias[0]
        elif len(drain_bias) == 2:
            swp_options['start'], swp_options['stop'] = drain_bias
        else:
            raise ValueError("Either 1 or 2 values must be given for drain_bias")
        tbm.specs['sweep_options'].update(**swp_options)
        return self._tbm_info, True

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        specs = self.specs
        state = cur_info.state
        data = cast(SimResults, sim_results).data
        mos_mapping = self._mos_mapping[state]
        stack = specs[f'stack_{state}']

        # Compute the total current by the following:
        # Find the drain current of all the unit transistors (the number of these transistors is seg * stack)
        # Add all these currents together, then divide by the number of stacks (to approximate
        # the sum of currents for each "finger"/parallel path of stacked transistors).
        # This calculation assumes that the junction leakage is negligible compared to the measured current.
        total_current = np.zeros(data.data_shape)
        for sig in data.signals:
            if any(filter(lambda x: x in sig, mos_mapping)):
                total_current += data._cur_ana[sig]
        total_current /= stack

        # Calculate resistance
        swp_options = self._tbm_info[0].specs['sweep_options']
        if swp_options['num'] == 1:
            res = swp_options['start'] / total_current[:, 0]
        else:
            res = (swp_options['stop'] - swp_options['start']) / (total_current[:, 1] - total_current[:, 0])
        res = np.abs(res)

        result = cur_info.prev_results.copy()
        if state == 'pd':
            result['sim_envs'] = data.sim_envs
        result[f'res_{state}'] = res

        next_state = 'done' if state == 'pu' else 'pu'
        return next_state == 'done', MeasInfo(next_state, result)

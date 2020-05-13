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

from typing import Any, Union, Tuple, Set, Mapping, Dict, Type, Sequence, cast

import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults
from bag.simulation.measure import MeasInfo

from ...data.tran import EdgeType
from ...search import IntervalSearchMM
from ..util import setup_digital_tran
from .base import FlopMeasMode, FlopTimingBase


class FlopConstraintTimingMM(IntervalSearchMM):
    """Measures setup/hold/recovery/removal time.

    Notes
    -----
    specification dictionary has the following entries in addition to IntervalSearchMM:

    meas_mode : Union[str, FlopMeasMode]
        the measurement mode.
    flop_params : Mapping[str, Any]
        flop parameters.
    delay_thres : float
        Defaults to 0.05.  Percent increase in delay for setup/hold constraints.  Use infinity
        to disable.  At least one of delay_thres or delay_inc must be specified.  If both are
        given, both constraints will be satisfied.
    delay_inc : float
        Defaults to infinity.  Increase in delay in seconds for setup/hold constraints.  At least
        one  delay_thres or delay_inc must be specified.  If both are given, both constraints
        will be satisfied.
    constraint_min : float
        Defaults to negative infinity.  If the timing constraint is less than this value, return
        this value instead.  Used to speed up simulation.
    sim_env_name : str
        Use to query for sim_env dependent timing offset.
    tbm_cls : Union[str, Type[FlopTimingBase]]
        The testbench class.
    tbm_specs : Mapping[str, Any]
        TestbenchManager specifications.
    search_params : Mapping[str, Any]
        interval search parameters, with the following entries:

        max_margin : float
            Optional.  maximum timing margin in seconds.  Defaults to t_clk_per/4.
        tol : float
            tolerance of the binary search.  Terminate the search when it is below this value.
        overhead_factor : float
            ratio of simulation startup time to time it takes to simulate one sweep point.
    fake: bool
        Defaults to False.  True to output fake data for debugging.
    use_dut : bool
        Defaults to True.  True to instantiate DUT.
    wrapper_params : Mapping[str, Any]
        Used only if simulated with a DUT wrapper.  Contains the following entries:

        lib : str
            wrapper library name.
        cell : str
            wrapper cell name.
        params : Mapping[str, Any]
            DUT wrapper schematic parameters.
        pins : Sequence[str]
            wrapper pin list.
        power_domain : Mapping[str, Tuple[str, str]
            power domain of wrapper.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._delay_targ: Dict[Tuple[str, str], float] = {}
        self._output_map: Mapping[str, Tuple[Mapping[str, Any],
                                             Sequence[Tuple[EdgeType, Sequence[str]]]]] = {}

        super().__init__(*args, **kwargs)

    def init_search(self, sim_db: SimulationDB, dut: DesignInstance
                    ) -> Tuple[TestbenchManager, Mapping[str, Any],
                               Mapping[str, Mapping[str, Any]], Mapping[str, Any],
                               bool, bool]:
        neg_inf = float('-inf')

        specs = self.specs
        meas_mode: Union[str, FlopMeasMode] = specs['meas_mode']
        flop_params: Mapping[str, Any] = specs['flop_params']
        tbm_cls: Union[str, Type[FlopTimingBase]] = specs['tbm_cls']
        search_params: Mapping[str, Any] = specs['search_params']
        fake: bool = specs.get('fake', False)
        use_dut: bool = specs.get('use_dut', True)
        constraint_min: float = specs.get('constraint_min', neg_inf)
        sim_env_name: str = specs.get('sim_env_name', '')

        tbm_specs, tb_params = setup_digital_tran(specs, dut, meas_mode=meas_mode,
                                                  flop_params=flop_params,
                                                  sim_env_name=sim_env_name)

        tbm = cast(FlopTimingBase, sim_db.make_tbm(tbm_cls, tbm_specs))
        sim_params = tbm.sim_params
        t_clk_per = sim_params['t_clk_per']
        t_rf = sim_params['t_rf'] / tbm.trf_scale

        tol: float = search_params['tol']
        overhead_factor: float = search_params['overhead_factor']
        max_margin: float = search_params.get('max_margin', t_clk_per / 4)

        min_val = max_margin if fake else -max_margin
        # NOTE: max_timing_value makes sure PWL width is always non-negative
        max_timing_value = max_margin + t_rf
        if constraint_min > neg_inf:
            # perform unbounded binary search instead
            high = constraint_min if fake else None
            defaults = dict(low=constraint_min, high=high, tol=tol, step=1.0e-12,
                            max_err=max_margin, overhead_factor=overhead_factor,
                            single_first=True)
        else:
            defaults = dict(low=min_val, high=max_margin, tol=tol, step=1.0e-12,
                            max_err=float('inf'), overhead_factor=overhead_factor,
                            single_first=False)

        self._output_map = tbm.get_output_map(False)
        for var in tbm.timing_variables:
            sim_params[var] = max_timing_value

        adj_table = {k: defaults for k in self._output_map.keys()}
        return tbm, tb_params, adj_table, defaults, True, use_dut

    def get_init_result(self, adj_name: str) -> Dict[str, Any]:
        specs = self.specs
        fake: bool = specs.get('fake', False)
        if fake:
            neg_inf = float('-inf')
            constraint_min: float = specs.get('constraint_min', neg_inf)
            value = constraint_min if constraint_min != neg_inf else 40.0e-12
            timing_info = self._output_map[adj_name][0]
            return dict(value=value + timing_info['offset'], margin=0.0,
                        delay=20.0e-12, timing_info=timing_info)
        return {}

    def process_init(self, cur_info: MeasInfo, sim_results: SimResults
                     ) -> Tuple[Dict[str, Any], bool]:
        specs = self.specs
        delay_thres: float = specs.get('delay_thres', 0.05)
        delay_inc: float = specs.get('delay_inc', float('inf'))

        tbm = cast(FlopTimingBase, sim_results.tbm)

        data = sim_results.data
        self._delay_targ.clear()
        result = {}
        for var_name, (timing_info, edge_out_list) in self._output_map.items():
            if timing_info['inc_delay']:
                scale = 1 + delay_thres
                inc = delay_inc
                pick_fun = min
            else:
                scale = 1 - delay_thres
                inc = -delay_inc
                pick_fun = max

            cur_results = {}
            for edge, out_list in edge_out_list:
                for out_pin in out_list:
                    delay = tbm.calc_clk_to_q(data, out_pin, edge).item()
                    if delay == float('inf'):
                        raise ValueError('Cannot measure reference delay.')
                    cur_results[out_pin] = (edge.name, delay)
                    delay_targ = pick_fun(scale * delay, delay + inc)
                    self._delay_targ[(var_name, out_pin)] = delay_targ
                    self.log(f'var={var_name}, out={out_pin}, '
                             f'delay = {delay:.4g}, delay_targ = {delay_targ:.4g}')
            result[var_name] = dict(delay=cur_results, timing_info=timing_info)

        self.log_result(cur_info.state, result)
        return result, False

    def process_output_helper(self, cur_info: MeasInfo, sim_results: SimResults,
                              remaining: Set[str]) -> Mapping[str, Tuple[Tuple[float, float],
                                                                         Dict[str, Any], bool]]:
        prev_results = cur_info.prev_results
        tbm = cast(FlopTimingBase, sim_results.tbm)
        sim_data = sim_results.data
        info_table = {}
        meas_mode = tbm.meas_mode
        is_removal = meas_mode.is_removal
        for var_name in remaining:
            cur_result = prev_results[var_name]
            if var_name in sim_data:
                adj_values = sim_data[var_name]
            else:
                adj_values = np.array([tbm.sim_params[var_name]])

            timing_info, edge_out_list = self._output_map[var_name]
            cons_offset: float = timing_info['offset']
            if is_removal:
                # NOTE: for removal test, we want to make sure clk-to-q delay is infinite for all
                # outputs.
                adj_sign = True
                diff = np.full(adj_values.shape, True)
                for edge, out_list in edge_out_list:
                    for out_pin in out_list:
                        # NOTE: remove corners
                        arr = tbm.calc_clk_to_q(sim_data, out_pin, edge)[0, ...]
                        diff = np.logical_and(diff, np.isinf(arr))
                diff = 2 * diff.astype(int) - 1
            else:
                adj_sign = timing_info['inc_delay']
                diff = np.full(adj_values.shape, np.inf)
                for edge, out_list in edge_out_list:
                    for out_pin in out_list:
                        # NOTE: remove corners
                        arr = tbm.calc_clk_to_q(sim_data, out_pin, edge)[0, ...]
                        targ = self._delay_targ[(var_name, out_pin)]

                        diff = np.minimum(diff, targ - arr)

            # find sign change
            res_idx, accept, low, high = self.get_adj_interval(var_name, adj_sign, adj_values, diff)
            cur_result['value'] = adj_values[res_idx] + cons_offset
            cur_result['margin'] = diff[res_idx]
            info_table[var_name] = ((low, high), cur_result, accept)

        self.log_result(cur_info.state, info_table)
        return info_table

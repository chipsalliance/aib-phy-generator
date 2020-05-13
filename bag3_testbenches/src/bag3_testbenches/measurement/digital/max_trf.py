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

from typing import Any, Tuple, Sequence, Mapping, Dict, Set, cast

import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults
from bag.simulation.measure import MeasInfo

from ..search import IntervalSearchMM
from ..tran.digital import DigitalTranTB
from .util import setup_digital_tran


class MaxRiseFallTime(IntervalSearchMM):
    """Tweaks a parameter as far as possible without exceeding max rise/fall time.

    Notes
    -----
    specification dictionary has the following entries:

    adj_name : str
        the adjust parameter name.
    in_pin : str
        input pin name.
    out_pin : str
        output pin name.
    adj_sign : bool
        True if increasing parameter value increases rise/fall time, False otherwise.
    max_trf : float
        maximum rise/fall time, in seconds.
    use_dut : bool
        Optional.  True to instantiate DUT.  Defaults to True.
    search_params : Mapping[str, Any]
        interval search parameters, with the following entries:

        low : float
            lower bound.
        high : Optional[float]
            upper bound.  If None, perform a unbounded binary search.
        step : float
            initial step size for unbounded binary search.
        tol : float
            tolerance of the binary search.  Terminate the search when it is below this value.
        max_err : float
            Used only in unbounded binary search.  If unbounded binary search exceeds this value,
            raise an error.
        overhead_factor : float
            ratio of simulation startup time to time it takes to simulate one sweep point.
    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

        t_rst :
            reset duration.
        t_rst_rf :
            reset rise/fall time.
        t_bit :
            bit value duration.
        t_rf :
            input pulse rise/fall time.
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
    load_list : Sequence[Mapping[str, Any]]
        Optional.  List of loads.  Each dictionary has the following entries:

        pin: str
            the pin to connect to.
        type : str
            the load device type.
        value : Union[float, str]
            the load parameter value.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def init_search(self, sim_db: SimulationDB, dut: DesignInstance
                    ) -> Tuple[TestbenchManager, Mapping[str, Any],
                               Mapping[str, Mapping[str, Any]], Mapping[str, Any],
                               bool, bool]:
        specs = self.specs
        adj_name: str = specs['adj_name']
        in_pin: str = specs['in_pin']
        out_pin: str = specs['out_pin']
        search_params: Mapping[str, Any] = specs['search_params']
        use_dut: bool = specs.get('use_dut', True)
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])

        # create pulse list
        pulse_list = [dict(pin=in_pin, tper='2*t_bit', tpw='t_bit', trf='t_rf',
                           td='t_bit', pos=True)]

        tbm_specs, tb_params = setup_digital_tran(specs, dut, pulse_list=pulse_list,
                                                  load_list=load_list)
        tbm_specs['save_outputs'] = [out_pin]
        tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, tbm_specs))
        tbm.sim_params['t_sim'] = f'{tbm.t_rst_end_expr}+3*t_bit'

        return tbm, tb_params, {adj_name: search_params}, {}, False, use_dut

    def process_init(self, cur_info: MeasInfo, sim_results: SimResults
                     ) -> Tuple[Dict[str, Any], bool]:
        return {}, False

    def process_output_helper(self, cur_info: MeasInfo, sim_results: SimResults,
                              remaining: Set[str]) -> Mapping[str, Tuple[Tuple[float, float],
                                                                         Dict[str, Any], bool]]:
        specs = self.specs
        out_pin: str = specs['out_pin']
        adj_name: str = specs['adj_name']
        adj_sign: bool = specs['adj_sign']
        max_trf: float = specs['max_trf']

        cur_state = cur_info.state
        data = sim_results.data
        tbm = cast(DigitalTranTB, sim_results.tbm)
        t0 = tbm.get_t_rst_end(data)
        tr = tbm.calc_trf(data, out_pin, True, allow_inf=True, t_start=t0)
        tf = tbm.calc_trf(data, out_pin, False, allow_inf=True, t_start=t0)
        adj_values = data[adj_name]
        # remove corners
        tr = tr[0, ...]
        tf = tf[0, ...]
        # replace NaN with inf to force to use smaller values
        tr[np.isnan(tr)] = np.inf
        tf[np.isnan(tf)] = np.inf

        # find sign change
        trf_max = np.maximum(tr, tf)
        diff = max_trf - trf_max
        res_idx, accept, low, high = self.get_adj_interval(adj_name, not adj_sign, adj_values, diff)
        new_result = {'value': adj_values[res_idx], 'tr': tr[res_idx], 'tf': tf[res_idx]}
        self.log_result(cur_state, new_result)
        return {adj_name: ((low, high), new_result, accept)}

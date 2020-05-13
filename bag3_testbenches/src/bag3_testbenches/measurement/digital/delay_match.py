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

from typing import Any, Tuple, Mapping, Dict, Set, Optional, Sequence, cast

import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults
from bag.simulation.measure import MeasInfo

from ..search import IntervalSearchMM, AcceptMode
from ..tran.digital import DigitalTranTB
from .util import setup_digital_tran


class DelayMatch(IntervalSearchMM):
    """This class tweaks a given parameter to match two delays.

    This class adjusts a given parameter using binary search to try and match the delay of
    two different paths, or the delay of a single path to a target number.
    Most often this is used to extract input capacitance.

    Assumptions:

    1. There are no corner or parameter sweeps.
    2. the reference and adjust delay paths have the same out_invert value.

    Notes
    -----
    specification dictionary has the following entries:

    adj_name : str
        the adjust parameter name.
    adj_sign : bool
        True if increasing parameter value increases delay, False otherwise.
    adj_params : Mapping[str, Any]
        adjust delay path parameters.
    ref_delay : float
        Optional.  The target delay value.
    ref_params : Mapping[str, Any]
        reference delay path parameters.  Used only if ref_delay is not present.
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

        t_sim :
            total simulation time.
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
    pulse_list : Sequence[Mapping[str, Any]]
        List of pulse sources.  Each dictionary has the following entries:

        pin : str
            the pin to connect to.
        tper : Union[float, str]
            period.
        tpw : Union[float, str]
            the flat region duration.
        trf : Union[float, str]
            rise/fall time.
        td : Union[float, str]
            Optional.  Pulse delay in addition to any reset period.  Defaults to 0.
        pos : bool
            Optional.  True if this is a positive pulse (010).  Defaults to True.
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
        self._value: Optional[float] = None

    def init_search(self, sim_db: SimulationDB, dut: DesignInstance
                    ) -> Tuple[TestbenchManager, Mapping[str, Any],
                               Mapping[str, Mapping[str, Any]], Mapping[str, Any],
                               bool, bool]:
        specs = self.specs
        adj_name: str = specs['adj_name']
        search_params: Mapping[str, Any] = specs['search_params']
        pulse_list: Sequence[Mapping[str, Any]] = specs['pulse_list']
        adj_params: Mapping[str, Any] = specs['adj_params']
        self._value: Optional[float] = specs.get('ref_delay', None)
        use_dut: bool = specs.get('use_dut', True)
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])

        # get output list
        save_outputs = [adj_params['in_name'], adj_params['out_name']]
        if self._value is None:
            ref_params: Mapping[str, Any] = specs['ref_params']
            save_outputs.append(ref_params['in_name'])
            save_outputs.append(ref_params['out_name'])

        tbm_specs, tb_params = setup_digital_tran(specs, dut, pulse_list=pulse_list,
                                                  load_list=load_list)
        tbm_specs['save_outputs'] = save_outputs
        tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, tbm_specs))

        return tbm, tb_params, {adj_name: search_params}, {}, False, use_dut

    def process_init(self, cur_info: MeasInfo, sim_results: SimResults
                     ) -> Tuple[Dict[str, Any], bool]:
        return {}, False

    def process_output_helper(self, cur_info: MeasInfo, sim_results: SimResults,
                              remaining: Set[str]) -> Mapping[str, Tuple[Tuple[float, float],
                                                                         Dict[str, Any], bool]]:
        specs = self.specs
        adj_name: str = specs['adj_name']
        adj_sign: bool = specs['adj_sign']
        adj_params: Mapping[str, Any] = specs['adj_params']

        cur_state = cur_info.state
        data = sim_results.data
        tbm = cast(DigitalTranTB, sim_results.tbm)
        td_adj = tbm.calc_delay(data, **adj_params)
        adj_values = data[adj_name]
        # remove corners
        td_adj = td_adj[0, ...]

        if self._value is None:
            ref_params: Mapping[str, Any] = specs['ref_params']
            td_ref = tbm.calc_delay(data, **ref_params)
            td_ref = td_ref[0, ...]
        else:
            td_ref = np.full(td_adj.shape, self._value)

        # find sign change
        diff = td_ref - td_adj
        arg, accept, low, high = self.get_adj_interval(adj_name, not adj_sign, adj_values, diff,
                                                       accept_mode=AcceptMode.BOTH)
        new_result = {'value': adj_values[arg].item(), 'td_ref': td_ref[arg].item(),
                      'td_adj': td_adj[arg].item(), 'accept': accept}
        self.log_result(cur_state, new_result)
        return {adj_name: ((low, high), new_result, accept)}

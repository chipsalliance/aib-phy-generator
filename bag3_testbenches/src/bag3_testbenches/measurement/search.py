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

from typing import Any, Tuple, Mapping, Optional, Dict, Set, Union

import abc
import pprint
from enum import Flag, auto

import numpy as np

from bag.util.search import FloatIntervalSearch
from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, SimResults, MeasureResult, DesignInstance
from bag.simulation.measure import MeasurementManager, MeasInfo


class AcceptMode(Flag):
    POSITIVE = auto()
    NEGATIVE = auto()
    BOTH = POSITIVE | NEGATIVE


class IntervalSearchMM(MeasurementManager, abc.ABC):
    """A Measurement manager that performs binary search for you.

    Assumes that no parameters/corners are swept.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._tbm_info: Optional[Tuple[TestbenchManager, Mapping[str, Any]]] = None
        self._table: Dict[str, Tuple[FloatIntervalSearch, bool]] = {}
        self._remaining: Set[str] = set()
        self._result: Dict[str, Dict[str, Any]] = {}
        self._use_dut = True

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {name: (val[0].low, val[0].high) for name, val in self._table.items()}

    @abc.abstractmethod
    def process_init(self, cur_info: MeasInfo, sim_results: SimResults
                     ) -> Tuple[Dict[str, Any], bool]:
        pass

    @abc.abstractmethod
    def init_search(self, sim_db: SimulationDB, dut: DesignInstance
                    ) -> Tuple[TestbenchManager, Mapping[str, Any],
                               Mapping[str, Mapping[str, Any]], Mapping[str, Any],
                               bool, bool]:
        """Initialize this MeasurementManager.

        Returns
        -------
        tbm : TestbenchManager
            the TestbenchManager object.
        tb_params : Mapping[str, Any]
            the testbench schematic parameters dictionary.
        intv_params : Mapping[str, Mapping[str, Any]]
            A dictionary from search parameter name to its configuration dictionary.
            The values have the following entries:

            low : float
                lower bound.
            high : Optional[float]
                upper bound.  If None, perform a unbounded binary search.
            step : float
                initial step size for unbounded binary search.
            tol : float
                tolerance of the binary search.  Terminate the search when it is below this value.
            max_err : float
                Used only in unbounded binary search.  If unbounded binary search exceeds this
                value, raise an error.
            overhead_factor : float
                ratio of simulation startup time to time it takes to simulate one sweep point.

        intv_defaults : Mapping[str, Any]
            If any interval configuration are not specified, the value is taken from
            this dictionary.
        has_init : bool
            True to run an initialization step.
        use_dut : bool
            True to instantiate DUT.
        """
        pass

    @abc.abstractmethod
    def process_output_helper(self, cur_info: MeasInfo, sim_results: SimResults,
                              remaining: Set[str]) -> Mapping[str, Tuple[Tuple[float, float],
                                                                         Dict[str, Any], bool]]:
        pass

    def get_init_result(self, adj_name: str) -> Dict[str, Any]:
        return {}

    def get_bound(self, adj_name: str) -> Tuple[float, float]:
        search = self._table[adj_name][0]
        return search.low, search.high

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        tmp = self.init_search(sim_db, dut)
        tbm, tb_params, intv_params, intv_defaults, has_init, use_dut = tmp
        if tbm.swp_info:
            self.error('Parameter sweep is not supported.')
        if tbm.num_sim_envs != 1:
            self.error('Corner sweep is not supported.')

        self._table.clear()
        self._remaining.clear()
        self._result.clear()
        self._use_dut = use_dut
        any_next = False
        for adj_name, search_params in intv_params.items():
            low: float = _get('low', search_params, intv_defaults)
            high: Optional[float] = _get('high', search_params, intv_defaults)
            step: float = _get('step', search_params, intv_defaults)
            tol: float = _get('tol', search_params, intv_defaults)
            max_err: float = _get('max_err', search_params, intv_defaults)
            overhead_factor: float = _get('overhead_factor', search_params, intv_defaults)
            guess: Optional[Union[float, Tuple[float, float]]] = _get(
                'guess', search_params, intv_defaults, None)
            single_first: bool = _get('single_first', search_params, intv_defaults, False)

            tmp = FloatIntervalSearch(low, high, overhead_factor, tol=tol, guess=guess,
                                      search_step=step, max_err=max_err)
            any_next = any_next or tmp.has_next()
            self._table[adj_name] = (tmp, single_first)
            self._result[adj_name] = self.get_init_result(adj_name)
            self._remaining.add(adj_name)

        if not any_next:
            return True, MeasInfo('done', self._result)

        self._tbm_info = (tbm, tb_params)
        return False, MeasInfo('init' if has_init else 'bin_0', {})

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        state = cur_info.state
        if state.startswith('bin'):
            if len(self._table) == 1:
                # can do sweeps
                adj_name = next(iter(self._table.keys()))
                search, single_first = self._table[adj_name]
                if state == 'bin_0' and single_first:
                    val = search.get_value()
                    self.log(f'Set {adj_name} to: {val:.4g}')
                    self._tbm_info[0].sim_params[adj_name] = val
                else:
                    swp_specs = search.get_sweep_specs()
                    self.log(f'{adj_name} sweep: {swp_specs}')
                    self._tbm_info[0].set_swp_info([(adj_name, swp_specs)])
            else:
                # no sweeps
                update_dict = {adj_name: self._table[adj_name][0].get_value()
                               for adj_name in self._remaining}
                self.log(f'Setting sim parameters:\n{pprint.pformat(update_dict, width=100)}')
                self._tbm_info[0].sim_params.update(update_dict)
        return self._tbm_info, self._use_dut

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        cur_state = cur_info.state
        if cur_state.startswith('bin'):
            info_table = self.process_output_helper(cur_info, sim_results, self._remaining)

            any_next = False
            for key, ((low, high), result, accept) in info_table.items():
                search = self._table[key][0]
                search.set_interval(low, high=high)
                any_next = any_next or search.has_next()
                if accept:
                    self._result[key] = result

            if any_next:
                return False, MeasInfo(f'bin_{int(cur_state[4:]) + 1}', self._result)
            else:
                return True, MeasInfo('done', self._result)
        else:
            self._result, done = self.process_init(cur_info, sim_results)
            if done:
                return True, MeasInfo('done', self._result)
            else:
                return False, MeasInfo('bin_0', self._result)

    def get_adj_interval(self, adj_name: str, adj_sign: bool, adj_values: np.ndarray,
                         diff: np.ndarray, accept_mode: AcceptMode = AcceptMode.POSITIVE
                         ) -> Tuple[int, bool, float, float]:
        num_values = adj_values.size
        # find sign change
        idx_arr = np.nonzero(np.diff((diff >= 0).astype(int)))[0]
        # idx_arr[0] will be the index before sign change (if it exists)
        if idx_arr.size == 0:
            # no sign change
            test_val = diff[0]
            if test_val == 0:
                # everything is 0, pick middle point
                res_idx = num_values // 2
                low = high = adj_values[res_idx]
                accept = True
            else:
                accept = (((test_val > 0) and AcceptMode.POSITIVE in accept_mode) or
                          ((test_val < 0) and AcceptMode.NEGATIVE in accept_mode))
                cur_bnds = self.get_bound(adj_name)
                if (test_val > 0) ^ adj_sign:
                    # increase parameter
                    low = adj_values[num_values - 1]
                    high = cur_bnds[1]
                    res_idx = num_values - 1
                else:
                    # decrease parameter
                    low = cur_bnds[0]
                    high = adj_values[0]
                    res_idx = 0
        else:
            res_idx = idx_arr[0]
            accept = True
            test_val = diff[res_idx + 1]
            if test_val == 0:
                res_idx += 1
                low = high = adj_values[res_idx]
            else:
                low = adj_values[res_idx]
                high = adj_values[res_idx + 1]
                if ((accept_mode is AcceptMode.BOTH and
                     np.abs(test_val) < np.abs(diff[res_idx])) or
                        (accept_mode is AcceptMode.POSITIVE and test_val > 0) or
                        (accept_mode is AcceptMode.NEGATIVE and test_val < 0)):
                    res_idx += 1

        return res_idx, accept, low, high

    def log_result(self, state: str, new_result: Mapping[str, Any]) -> None:
        fmt = '{:.5g}'
        msg_list = [f'state = {state}']
        for name, val in self._table.items():
            search = val[0]
            msg_list.append(f'{name} bnd = [{fmt.format(search.low)}, {fmt.format(search.high)})')
        msg_list.append('result:')
        msg_list.append(pprint.pformat(new_result, width=100))
        self.log('\n'.join(msg_list))


def _get(key: str, table1: Mapping[str, Any], table2: Mapping[str, Any], *args: Any) -> Any:
    if key in table1:
        return table1[key]
    if key in table2:
        return table2[key]
    if not args:
        raise ValueError(f'Cannot find key: {key}')
    return args[0]

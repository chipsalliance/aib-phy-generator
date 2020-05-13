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

from typing import Any, Tuple, Mapping, Optional, Union, cast

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_testbenches.measurement.digital.max_trf import MaxRiseFallTime

from ..util import get_digital_wrapper_params


class CapMaxRiseFallTime(MeasurementManager):
    """Measures maximum output capacitance given maximum rise/fall time.

    Assumes that no parameters/corners are swept.  Adds buffers to all input pins.

    Notes
    -----
    specification dictionary has the following entries:

    in_pin : str
        input pin name.
    out_pin : str
        output pin name.
    max_trf : float
        maximum rise/fall time, in seconds.
    fake : bool
        Defaults to False.  True to return fake data.
    buf_params : Mapping[str, Any]
        input buffer parameters.
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
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mm: Optional[MaxRiseFallTime] = None

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        specs = self.specs
        in_pin: str = specs['in_pin']
        out_pin: str = specs['out_pin']
        fake: str = specs.get('fake', False)

        if fake:
            return True, MeasInfo('done', dict(cap=100.0e-15, tr=20.0e-12, tf=20.0e-12))

        load_list = [dict(pin=out_pin, type='cap', value='c_load')]
        wrapper_params = get_digital_wrapper_params(specs, dut, [in_pin])
        mm_specs = {k: specs[k] for k in ['in_pin', 'out_pin', 'max_trf',
                                          'search_params', 'tbm_specs']}
        mm_specs['adj_name'] = 'c_load'
        mm_specs['adj_sign'] = True
        mm_specs['use_dut'] = True
        mm_specs['wrapper_params'] = wrapper_params
        mm_specs['load_list'] = load_list
        self._mm = sim_db.make_mm(MaxRiseFallTime, mm_specs)

        return False, MeasInfo('max_trf', {})

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        return self._mm, True

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        data = cast(MeasureResult, sim_results).data['c_load']
        new_result = dict(cap=data['value'], tr=data['tr'], tf=data['tf'])
        return True, MeasInfo('done', new_result)

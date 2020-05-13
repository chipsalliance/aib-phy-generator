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

from typing import Dict, Any, Tuple, Optional, Union, Mapping

from pathlib import Path

from bag.simulation.base import get_bit_list
from bag.simulation.core import TestbenchManager
from bag.simulation.cache import DesignInstance, SimulationDB, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM

from .util import get_digital_wrapper_params, get_in_buffer_pin_names


class BufferCombLogicTimingMM(MeasurementManager):
    """Measure combinational logic delay with input buffers.

    Same as CombLogicTimingMM, but add input buffers.

    Notes
    -----
    specification dictionary has the following entries in addition to CombLogicTimingMM:

    buf_params : Optional[Mapping[str, Any]]
        input buffer parameters.
    buf_config : Mapping[str, Any]
        Used only if buf_params is not specified.  input buffer configuration parameters.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        in_pin: str = specs['in_pin']
        buf_params: Optional[Mapping[str, Any]] = specs.get('buf_params', None)

        if buf_params is None:
            buf_config: Mapping[str, Any] = specs['buf_config']
            lch: int = buf_config['lch']
            w_p: int = buf_config['w_p']
            w_n: int = buf_config['w_n']
            th_p: str = buf_config['th_p']
            th_n: str = buf_config['th_n']
            cinv_unit: float = buf_config['cinv_unit']
            cin_guess: float = buf_config['cin_guess']
            fanout_in: float = buf_config['fanout_in']
            fanout_buf: float = buf_config.get('fanout_buf', 4)

            seg1 = int(round(max(cin_guess / fanout_in / cinv_unit, 1.0)))
            seg0 = int(round(max(seg1 / fanout_buf, 1.0)))
            buf_params = dict(
                export_pins=True,
                inv_params=[
                    dict(lch=lch, w_p=w_p, w_n=w_n, th_p=th_p, th_n=th_n, seg=seg0),
                    dict(lch=lch, w_p=w_p, w_n=w_n, th_p=th_p, th_n=th_n, seg=seg1),
                ],
            )

        in_pins = get_bit_list(in_pin)
        wrapper_params = get_digital_wrapper_params(specs, dut, in_pins, buf_params=buf_params)
        dut_in_pins = [get_in_buffer_pin_names(pin)[1] for pin in in_pins]

        mm_specs = self.specs.copy()
        mm_specs.pop('buf_config', None)
        mm_specs.pop('buf_params', None)
        mm_specs['in_pin'] = in_pin
        if 'start_pin' not in mm_specs:
            mm_specs['start_pin'] = dut_in_pins
        mm_specs['wrapper_params'] = wrapper_params
        mm = sim_db.make_mm(CombLogicTimingMM, mm_specs)

        return await mm.async_measure_performance(name, sim_dir, sim_db, dut)

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

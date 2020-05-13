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

import pprint

import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_liberty.enum import TimingType

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.digital.util import setup_digital_tran


class ClockDelayMM(MeasurementManager):
    """Measures measure/launch delay.

    Notes
    -----
    specification dictionary has the following entries:

    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

            t_rst :
                reset duration.
            t_rst_rf :
                reset rise/fall time.
            t_bit :
                bit value duration.
            c_load :
                load capacitance.
    fake : bool
        Defaults to False.  True to return fake data.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tbm_info: Optional[Tuple[DigitalTranTB, Mapping[str, Any]]] = None

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        specs = self.specs
        fake: bool = specs.get('fake', False)

        load_list = [dict(pin='ckout', type='cap', value='c_load')]
        pulse_list = [dict(pin='launch', tper='2*t_bit', tpw='t_bit', trf='t_rf',
                           td='t_bit', pos=True),
                      dict(pin='measure', tper='2*t_bit', tpw='t_bit', trf='t_rf',
                           td='2*t_bit', pos=True)]
        pin_values = dict(dcc_byp=0, clk_dcd=0)
        save_outputs = ['launch', 'measure', 'ckout']
        tbm_specs, tb_params = setup_digital_tran(specs, dut, pulse_list=pulse_list,
                                                  load_list=load_list, pin_values=pin_values,
                                                  save_outputs=save_outputs)
        tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, tbm_specs))

        if fake:
            td = np.full(tbm.sweep_shape[1:], 50.0e-12)
            trf = np.full(td.shape, 20.0e-12)
            result = _get_result_table(td, td, trf, trf)
            return True, MeasInfo('done', result)

        tbm.sim_params['t_sim'] = 't_rst+t_rst_rf+11*t_bit'
        self._tbm_info = tbm, tb_params
        return False, MeasInfo('sim', {})

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        return self._tbm_info, True

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        tbm: DigitalTranTB = self._tbm_info[0]

        sim_params = tbm.sim_params
        t_bit = sim_params['t_bit']

        data = cast(SimResults, sim_results).data
        t_rf = tbm.get_param_value('t_rf', data) / (tbm.thres_hi - tbm.thres_lo)

        t_launch = tbm.get_t_rst_end(data) + 9 * t_bit
        t0_launch = t_launch - t_rf / 2
        t_meas = t_launch + t_bit
        t0_meas = t0_launch + t_bit

        td_launch = tbm.calc_cross(data, 'ckout', EdgeType.RISE, t_start=t0_launch) - t_launch
        td_meas = tbm.calc_cross(data, 'ckout', EdgeType.FALL, t_start=t0_meas) - t_meas
        tr_launch = tbm.calc_trf(data, 'ckout', True, t_start=t0_launch)
        tf_meas = tbm.calc_trf(data, 'ckout', False, t_start=t0_meas)

        if td_launch.shape[0] == 1:
            td_launch = td_launch[0, ...]
            td_meas = td_meas[0, ...]
            tr_launch = tr_launch[0, ...]
            tf_meas = tf_meas[0, ...]
        result = _get_result_table(td_launch, td_meas, tr_launch, tf_meas)
        self.log(f'result:\n{pprint.pformat(result, width=100)}')

        return True, MeasInfo('done', result)


def _get_result_table(td_launch: np.ndarray, td_meas: np.ndarray,
                      tr_launch: np.ndarray, tf_meas: np.ndarray) -> Dict[str, Any]:
    return dict(
        ckout=[
            dict(
                related='launch',
                timing_type=TimingType.combinational_rise.name,
                sense='positive_unate',
                cond='rstb & !dcc_byp',
                data=dict(
                    cell_rise=td_launch,
                    rise_transition=tr_launch,
                )
            ),
            dict(
                related='measure',
                timing_type=TimingType.combinational_fall.name,
                sense='negative_unate',
                cond='rstb & !dcc_byp',
                data=dict(
                    cell_fall=td_meas,
                    fall_transition=tf_meas,
                )
            ),
        ],
    )

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

from typing import Any, Tuple, Mapping, Optional, Union, Sequence, cast

import pprint

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.digital.util import setup_digital_tran
from bag3_testbenches.measurement.digital.delay_match import DelayMatch

from ..util import get_digital_wrapper_params, get_in_buffer_pin_names


class CapDelayMatch(MeasurementManager):
    """Measures input capacitance by matching delay.

    Assumes that no parameters/corners are swept.

    Notes
    -----
    specification dictionary has the following entries:

    in_pin : str
        input pin name.
    fake : bool
        Defaults to False.  True to generate fake data.
    buf_params : Optional[Mapping[str, Any]]
        Optional.  Input buffer parameters.
    buf_config : Optional[Mapping[str, Any]]
        Optional.  Either buf_params or buf_config must be specified.
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
        self._tbm_info: Optional[Tuple[DigitalTranTB, Mapping[str, Any]]] = None
        self._mm: Optional[DelayMatch] = None
        self._wrapper_params: Mapping[str, Any] = {}

        # TODO: make cap measurement more accurate by determining buf_params automatically
        # TODO: add option to automatically adjust load cap to determine input cap accurately
        super().__init__(*args, **kwargs)

    @property
    def wrapper_params(self) -> Mapping[str, Any]:
        return self._wrapper_params

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        specs = self.specs
        in_pin: str = specs['in_pin']
        search_params = specs['search_params']
        buf_params: Optional[Mapping[str, Any]] = specs.get('buf_params', None)
        buf_config: Optional[Mapping[str, Any]] = specs.get('buf_config', None)
        fake: bool = specs.get('fake', False)
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])

        if fake:
            return True, MeasInfo('done', dict(cap_rise=1.0e-12, cap_fall=1.0e-12,
                                               tr_ref=50.0e-12, tf_ref=50.0e-12,
                                               tr_adj=50.0e-12, tf_adj=50.0e-12))

        if buf_params is None and buf_config is None:
            raise ValueError('one of buf_params or buf_config must be specified.')

        if buf_params is None:
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
            specs['buf_params'] = buf_params
            self.log(f'buf_params:\n{pprint.pformat(buf_params, width=100)}')
            search_params = dict(**search_params)
            search_params['guess'] = (cin_guess * 0.8, cin_guess * 1.2)

        # create testbench for measuring reference delay
        pulse_list = [dict(pin=in_pin, tper='2*t_bit', tpw='t_bit', trf='t_rf',
                           td='t_bit', pos=True)]
        self._wrapper_params = get_digital_wrapper_params(specs, dut, [in_pin])
        tbm_specs, tb_params = setup_digital_tran(specs, dut, wrapper_params=self._wrapper_params,
                                                  pulse_list=pulse_list, load_list=load_list)
        buf_mid, buf_out = get_in_buffer_pin_names(in_pin)
        tbm_specs['save_outputs'] = [buf_mid, buf_out]
        # remove input pin from reset list
        reset_list: Sequence[Tuple[str, bool]] = tbm_specs.get('reset_list', [])
        new_reset_list = [ele for ele in reset_list if ele[0] != in_pin]
        tbm_specs['reset_list'] = new_reset_list

        tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, tbm_specs))
        if tbm.swp_info:
            self.error('Parameter sweep is not supported.')
        if tbm.num_sim_envs != 1:
            self.error('Corner sweep is not supported.')
        tbm.sim_params['t_sim'] = f'{tbm.t_rst_end_expr}+3*t_bit'

        self._tbm_info = tbm, tb_params

        # create DelayMatch
        mm_tbm_specs = {k: v for k, v in tbm.specs.items()
                        if k not in {'pwr_domain', 'sup_values', 'dut_pins', 'pin_values',
                                     'pulse_load', 'reset_list', 'load_list', 'diff_list'}}
        gnd_name, pwr_name = DigitalTranTB.get_pin_supplies(in_pin, tbm_specs['pwr_domain'])
        sup_values: Mapping[str, Union[float, Mapping[str, float]]] = tbm_specs['sup_values']
        gnd_val = sup_values[gnd_name]
        pwr_val = sup_values[pwr_name]
        pwr_tup = ('VSS', 'VDD')
        mm_tbm_specs['pwr_domain'] = {}
        mm_tbm_specs['sup_values'] = dict(VSS=gnd_val, VDD=pwr_val)
        mm_tbm_specs['pin_values'] = {}

        thres_lo: float = mm_tbm_specs['thres_lo']
        thres_hi: float = mm_tbm_specs['thres_hi']
        t_start_expr = f't_rst+t_bit+(t_rst_rf-t_rf/2)/{thres_hi - thres_lo:.2f}'
        mm_specs = dict(
            adj_name='c_load',
            adj_sign=True,
            adj_params=dict(in_name='mid', out_name='out', t_start=t_start_expr),
            ref_delay=0,
            use_dut=False,
            search_params=search_params,
            tbm_specs=mm_tbm_specs,
            wrapper_params=dict(
                lib='bag3_digital',
                cell='inv_chain',
                params=buf_params,
                pins=['in', 'out', 'mid', 'VDD', 'VSS'],
                pwr_domain={'in': pwr_tup, 'out': pwr_tup, 'mid': pwr_tup},
            ),
            pulse_list=[dict(pin='in', tper='2*t_bit', tpw='t_bit', trf='t_rf',
                             td='t_bit', pos=True)],
            load_list=[dict(pin='out', type='cap', value='c_load')],
        )
        mm_specs.update(search_params)
        self._mm = sim_db.make_mm(DelayMatch, mm_specs)

        return False, MeasInfo('init', {})

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        state = cur_info.state
        if state == 'init':
            return self._tbm_info, True
        elif state == 'cap_rise':
            mm_specs = self._mm.specs
            adj_params = mm_specs['adj_params']
            adj_params['out_edge'] = EdgeType.RISE
            adj_params['in_edge'] = EdgeType.FALL
            mm_specs['ref_delay'] = cur_info.prev_results['tr_ref']

            self._mm.commit()
            return self._mm, False
        elif state == 'cap_fall':
            cap_rise = cur_info.prev_results['cap_rise']
            mm_specs = self._mm.specs
            adj_params = mm_specs['adj_params']
            adj_params['out_edge'] = EdgeType.FALL
            adj_params['in_edge'] = EdgeType.RISE
            mm_specs['ref_delay'] = cur_info.prev_results['tf_ref']

            search_params = mm_specs['search_params']
            search_params['guess'] = (cap_rise * 0.9, cap_rise * 1.1)
            self._mm.commit()
            return self._mm, False
        else:
            raise ValueError(f'Unknown state: {state}')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        state = cur_info.state
        if state == 'init':
            tbm = self._tbm_info[0]
            in_pin: str = self.specs['in_pin']

            sim_data = sim_results.data
            t0 = tbm.get_t_rst_end(sim_data)
            buf_mid, buf_out = get_in_buffer_pin_names(in_pin)
            tr = tbm.calc_delay(sim_data, buf_mid, buf_out, EdgeType.FALL,
                                EdgeType.RISE, t_start=t0)
            tf = tbm.calc_delay(sim_data, buf_mid, buf_out, EdgeType.RISE,
                                EdgeType.FALL, t_start=t0)
            return False, MeasInfo('cap_rise', dict(tr_ref=tr.item(), tf_ref=tf.item()))
        elif state == 'cap_rise':
            data = sim_results.data['c_load']
            new_result = cur_info.prev_results.copy()
            new_result['cap_rise'] = data['value']
            new_result['tr_adj'] = data['td_adj']
            return False, MeasInfo('cap_fall', new_result)
        elif state == 'cap_fall':
            data = sim_results.data['c_load']
            new_result = cur_info.prev_results.copy()
            new_result['cap_fall'] = data['value']
            new_result['tf_adj'] = data['td_adj']
            return True, MeasInfo('done', new_result)
        else:
            raise ValueError(f'Unknown state: {state}')

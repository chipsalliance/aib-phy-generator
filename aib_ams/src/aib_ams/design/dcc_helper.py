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

from typing import Mapping, Dict, Any, Tuple, Optional, Type

import numpy as np

from bag.layout.template import TemplateBase
from bag.env import get_tech_global_info
from bag.concurrent.util import GatherHelper

from bag.simulation.design import DesignerBase

from bag3_digital.layout.stdcells.util import STDCellWrapper

from ..layout.dcc_helper import DCCHelper
from ..measurement.dcc_helper.liberty import ClockDelayMM


class DCCHelperDesigner(DesignerBase):
    """
    This design script uses ClockDelayMM for sign-off.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)

        self.debug = True

        # local params
        self._thres_lo = 0.1
        self._thres_hi = 0.9

    @classmethod
    def get_dut_lay_class(cls) -> Optional[Type[TemplateBase]]:
        return DCCHelper
    
    @classmethod
    def _default_core_params(cls):
        return cls._get_scaled_core_params(1)

    @classmethod
    def _default_sync_params(cls):
        return cls._get_scaled_sync_params(1)

    @classmethod
    def _default_buf_params(cls):
        return cls._get_scaled_buf_params(1)
    
    async def async_design(self, nsync: int, cload: float, fmax: float, fmin: float, dcd_max: float,
                           rel_del_max: float, pinfo: Mapping[str, any], 
                           core_params: Optional[Dict[str, Any]] = None,
                           sync_params: Optional[Dict[str, Any]] = None, 
                           buf_params: Optional[Dict[str, Any]] = None) -> Mapping[str, Any]:
        """
        This design method is basically a logical-effort sizing and a sign-off in the end.
        If there are sizes passed in through optional parameters they will be used and this
        function will only execute the sign-off part.

        Parameters
        ----------
        nsync: int 
            Number of synchronizer flops.
        cload: float
            Loading cap.
        fmax: float
            Max. frequency for sign-off.
        fmin: float
            Min. frequency for sign-off.
        dcd_max: float
            Max. duty cycle distortion allowed.
        rel_del_max: float
            Max. relative delay requirement between launch and measure edges at output.
        pinfo: Mapping[str, any]:
            pinfo for layout.
        core_params: Dict[str, Any]
            Optional. If provided core design part will be skipped.
        sync_params: Dict[str, Any]
            Optional. If provided synchronizer design part will be skipped.
        buf_params: Dict[str, Any]
            Optional. If provided buffer design part will be skipped.

        Returns
        -------
        summary: Mapping[str, Any]
            Design summary.
        """
        tech_globals = get_tech_global_info('aib_ams')

        core_params = self._default_core_params() if core_params is None else core_params
        sync_params = self._default_sync_params() if sync_params is None else sync_params
        buf_params = self._default_buf_params() if buf_params is None else buf_params

        if 'width' not in pinfo['row_specs'][0] and 'width' not in pinfo['row_specs'][1]:
            pinfo['row_specs'][0]['width'] = 2 * tech_globals['w_minn']
            pwidth = int(np.round(tech_globals['inv_beta'] * 2 * tech_globals['w_minn']))
            pinfo['row_specs'][1]['width'] = pwidth

        gen_params = dict(
            cls_name=self.get_dut_lay_class().get_qualified_name(),
            draw_taps=True,
            params=dict(
                pinfo=pinfo,
                core_params=core_params,
                sync_params=sync_params,
                buf_params=buf_params,
                nsync=nsync,
            ),
        )

        dut = await self.async_new_dut("dcc_helper", STDCellWrapper, gen_params)

        helper = GatherHelper()
        for env_str in tech_globals['signoff_envs']['all_corners']['envs']:
            for freq in [fmin, fmax]:
                helper.append(self._sim_and_check_specs(dut, 
                                                        env_str, 
                                                        freq, 
                                                        cload, 
                                                        dcd_max, 
                                                        fmax, 
                                                        rel_del_max))

        results = await helper.gather_err()

        dcd_dict = {}
        del_rel_dict = {}
        idx = 0
        for env_str in tech_globals['signoff_envs']['all_corners']['envs']:
            for freq in ['fmin', 'fmax']:
                if freq == 'fmax':
                    dcd_dict[env_str] = results[idx][1]
                    del_rel_dict[env_str] = results[idx][2]
                idx += 1

        dcd_min = min(dcd_dict.values())
        dcd_max = max(dcd_dict.values())
        del_rel_min = min(del_rel_dict.values())
        del_rel_max = max(del_rel_dict.values())
        self.log(f'|DCD| ranges from {dcd_min*100:.4f}% to {dcd_max*100:.4f}%.')
        self.log(f'Relaive delay ranges from {del_rel_min*100:.4f}% to {del_rel_max*100:.4f}%')

        return gen_params

    async def _sim_and_check_specs(self, dut, env_str, freq, cload, dcd_max, fmax, rel_del_max):
        meas_params = self._build_meas_params(env_str, fmax, cload)
        mm = self.make_mm(ClockDelayMM, meas_params)
        sim_id = f'dcc_helper_signoff_{env_str}' + ('_fmax' if freq == fmax else '_fmin')
        sim_result = await self.async_simulate_mm_obj(sim_id, dut, mm)
        specs_met, dcd, rel_del = self._check_delay(sim_result, freq, dcd_max, rel_del_max)

        if not specs_met:
            raise ValueError(f'DCC_helper failed signoff @ {env_str}: '
                             f'dcd is  {dcd * 100:.2f} ({dcd_max * 100:.2f}) %,'
                             f' rel delay is {rel_del * 100:.2f} ({rel_del_max * 100:.2f}) %.')

        return specs_met, dcd, rel_del

    @classmethod
    def _build_meas_params(cls, sim_env: str, freq: float, cload: float) -> Dict[str, Any]:
        """
        Creates parameter dictionary for ClockDelayMM measurement manager class. ClockDelayMM 
        uses DigitalTranTB related test-bench manager parameters.
        
        Parameters
        ----------
        sim_env: str
            Corner-temperature environment.
        freq: float
            Frequency of clk in simulation.
        cload: float
            Loading cap.
            
        Returns
        -------
        meas_params: Dict[str, Any]
            Measurement dictionary params.
        """
        trf_nom = get_tech_global_info('aib_ams')['trf_nom']
        sim_env_info = get_tech_global_info('aib_ams')['signoff_envs']['all_corners']
        meas_params = dict(
            tbm_specs=dict(
                sim_envs=[sim_env],
                sim_params=dict(    # simulation parameters
                    t_rst=1.1 / freq,
                    t_rst_rf=trf_nom,
                    t_bit=1 / freq,
                ),
                thres_lo=0.1,
                thres_hi=0.9,
                rtol=1.0e-8,
                atol=1.0e-22,
                tran_options=dict(  
                    maxstep=1.0e-12,
                    errpreset='conservative',
                ),
                swp_info=[  # list of parameters to sweep
                    ['t_rf', {'type': 'LIST', 'values': [trf_nom]}],
                    ['c_load', {'type': 'LIST', 'values': [cload]}]
                ],
                pwr_domain=dict(    # pin's low/high pwr domains (defined later)
                    clk_dcd=('VSS', 'VDD'),
                    dcc_byp=('VSS', 'VDD'),
                    launch=('VSS', 'VDD'),
                    measure=('VSS', 'VDD'),
                    rstb=('VSS', 'VDD'),
                    ckout=('VSS', 'VDD')
                ),
                sup_values=dict(    # pwr domain definition with proper values
                    VDD=sim_env_info['vdd'][sim_env],
                    VSS=0.0,
                ),
                pin_values=dict(    # low(0) or high(1) value for pins with constant voltage
                    clk_dcd=0,
                    dcc_byp=0,
                    launch=0,
                    measure=0,
                ),
                reset_list=[['rstb', False]],   # reset rstb (active low)
            ),
            fake=False,    # make this true to get fake data (i.e. for debugging)
        )

        return meas_params

    def _check_delay(self, sim_result, freq: float, dcd_max: float,
                     rel_del_max: float) -> Tuple[bool, float, float]:
        met_specs = True
        t_launch = t_meas = -1
        for entry in sim_result.data['ckout']:
            if entry['related'] == 'launch':
                t_launch = entry['data']['cell_rise']
            elif entry['related'] == 'measure':
                t_meas = entry['data']['cell_fall']

        del_max = np.max(np.max([t_launch, t_meas]))
        rel_del = del_max * freq
        if rel_del > rel_del_max:
            self.log(f'Launch delay is {t_launch}, '
                     f'Meas delay is {t_meas}, '
                     f'max is {rel_del_max / freq}')
            met_specs = False
        dcd = np.subtract(t_launch, t_meas) * freq
        if np.max(np.abs(dcd)) > dcd_max:
            self.log(f'Duty cycle distortion is {dcd}, max absolute value is {dcd_max}.')
            met_specs = False

        return met_specs, np.max(np.abs(dcd)), rel_del

    @staticmethod
    def _get_scaled_core_params(scale):
        """ Logical effort (4) - based scaling, using some selected minimum sizes"""
        mux_params = dict(
            seg_dict={'tri': 4 * scale, 'buf': 4 * scale}
        )
        flop_params = dict(
            sa_params=dict(
                has_bridge=True,
                seg_dict={'in': 4 * scale, 'fb': 4 * scale,
                          'tail': 4 * scale, 'sw': 1 * scale},
            ),
            sr_params=dict(
                has_outbuf=True,
                seg_dict={'fb': 2 * scale, 'ps': 4 * scale, 'nr': 4 * scale,
                          'ibuf': 4 * scale, 'obuf': 4 * scale, 'rst': 4 * scale},
            )
        )
        inv_params = dict(seg=4 * scale)

        return dict(mux_params=mux_params, flop_params=flop_params, inv_params=inv_params)

    @staticmethod
    def _get_scaled_sync_params(scale):
        """Logical effort (4) - based scaling, using some selected minimum sizes"""
        flop_params = dict(
            sa_params=dict(
                has_bridge=True,
                seg_dict={'in': 2*scale, 'fb': 4*scale,
                          'tail': 4*scale, 'sw': scale}
            ),
            sr_params=dict(
                has_outbuf=True,
                seg_dict={'fb': 2*scale, 'ps': 2*scale,
                          'nr': 2*scale,
                          'ibuf': 2*scale, 'obuf': 2*scale,
                          'rst': 4*scale}
            )
        )

        return flop_params

    @staticmethod
    def _get_scaled_buf_params(scale):
        return dict(seg_list=[1*scale, 2*scale])

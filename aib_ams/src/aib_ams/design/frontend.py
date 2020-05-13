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

from typing import Dict, Any, Optional, Mapping, cast

from pybag.enum import LogLevel

from bag.io import read_yaml
from bag.env import get_tech_global_info
from xbase.layout.mos.base import TileInfoTable

from bag.simulation.design import DesignerBase
from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import EdgeType

from bag3_digital.layout.stdcells.util import STDCellWrapper

from aib_ams.layout.frontend import Frontend
from aib_ams.layout.top import FrontendESD
from aib_ams.design.txanlg import TXAnalogCoreDesigner
from aib_ams.design.rxanlg import RXAnalogDesigner


class FrontendDesigner(DesignerBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)
        self.dsn_tech_info = get_tech_global_info('aib_ams')
        self._txanlg_dsnr = None
        self._rxanlg_dsnr = None

    async def async_design(self,
                           num_units: int,
                           num_units_nom: int,
                           num_units_min: int,
                           r_targ: float,
                           r_min_weak: float,
                           c_ext: float,
                           freq: float,
                           trf_max: float,
                           trst: float,
                           rel_err: float,
                           del_err: float,
                           td_max: float,
                           stack_max: int,
                           dig_tbm_specs: Dict[str, Any],
                           dig_buf_params: Dict[str, Any],
                           cap_in_search_params: Dict[str, Any],
                           res_mm_specs: Dict[str, Any],
                           c_odat: float,
                           c_odat_async: float,
                           c_oclkp: float,
                           c_oclkn: float,
                           cin: float,
                           tp_targ: float,
                           tmismatch: float,
                           trf_in: float,
                           tile_specs: Mapping[str, Any],
                           k_ratio_ctrl: float,
                           k_ratio_data: float,
                           num_sup_clear: int = 0,
                           design_env_name: str = '',
                           tile_name: str = '',
                           ridx_p: int = 1,
                           ridx_n: int = 1,
                           tran_options: Optional[Mapping[str, Any]] = None,
                           inv_input_cap_meas_seg: Optional[int] = 16,
                           sim_options: Optional[Mapping[str, Any]] = None,
                           unit_inv_cin: float = 0,
                           cap_in_mm_specs: str = '',
                           tx_dsn_specs: Mapping[str, Any] = None,
                           rx_dsn_specs: Mapping[str, Any] = None,
                           frontend_dsn_specs: Dict[str, Any] = None,
                           with_esd: bool = False,
                           **kwargs: Any) -> Mapping[str, Any]:
        """Run sub-hierarchy design scripts and stitch together
        """
        # Get and set max widths of final drivers from tech defaults; set this in tile info
        tech_info = self.dsn_tech_info
        w_p = tech_info['w_maxp']
        w_n = tech_info['w_maxn']
        if 'lch' not in tile_specs['arr_info']:
            tile_specs['arr_info']['lch'] = tech_info['lch_min']
        tile_specs['place_info'][tile_name]['row_specs'][0]['width'] = w_n
        tile_specs['place_info'][tile_name]['row_specs'][1]['width'] = w_p

        # Make tile info
        tinfo_table = TileInfoTable.make_tiles(self.grid, tile_specs)
        tile_name_dict = dict(name=tile_name)
        pinfo = tinfo_table.make_place_info(tile_name_dict)

        if tx_dsn_specs is None and frontend_dsn_specs is None:
            tx_args = dict(
                num_units=num_units,
                num_units_nom=num_units_nom,
                num_units_min=num_units_min,
                r_targ=r_targ,
                r_min_weak=r_min_weak,
                c_ext=c_ext,
                freq=freq,
                trf_max=trf_max,
                trst=trst,
                trf_in=trf_in,
                k_ratio_ctrl=k_ratio_ctrl,
                k_ratio_data=k_ratio_data,
                rel_err=rel_err,
                del_err=del_err,
                td_max=td_max,
                stack_max=stack_max,
                tile_name=tile_name,
                tile_specs=tile_specs,
                dig_tbm_specs=dig_tbm_specs,
                dig_buf_params=dig_buf_params,
                cap_in_search_params=cap_in_search_params,
                res_mm_specs=res_mm_specs,
                ridx_p=ridx_p,
                ridx_n=ridx_n,
                tran_options=tran_options,
                inv_input_cap_meas_seg=inv_input_cap_meas_seg,
                sim_options=sim_options,
            )
            self._txanlg_dsnr = TXAnalogCoreDesigner(self._root_dir, self._sim_db, self._dsn_specs)
            tx_params = await self._txanlg_dsnr.async_design(**tx_args)
        else:
            tx_params = tx_dsn_specs

        if rx_dsn_specs is None and frontend_dsn_specs is None:
            # Calculate POR cap for RX POR level shifter sizing
            tx_ctrl_lv_params = tx_params['ctrl_lvshift']['dut_params']['params']['lv_params']
            tx_data_lv_params = tx_params['data_lvshift']['dut_params']['params']['lv_params']
            w_ctrl = tx_ctrl_lv_params['seg_dict']['rst'] * tx_ctrl_lv_params['w_dict']['rst']
            w_data = tx_data_lv_params['seg_dict']['rst'] * tx_data_lv_params['w_dict']['rst']
            cin_tx_por_total_w = 7 * w_ctrl + w_data
            cin_tx_por_vccl = cin_tx_por_total_w / tech_info['cin_inv']['w_per_seg'] * \
                              tech_info['cin_inv']['cin_per_seg']

            rx_args = read_yaml(kwargs['rx_specs_file'])
            rx_dsn_params = rx_args['dsn_params']
            rx_dsn_params['c_por_vccl_tx'] = cin_tx_por_vccl
            self._rxanlg_dsnr = RXAnalogDesigner(self._root_dir, self._sim_db, rx_dsn_params)
            rx_params = (await self._rxanlg_dsnr.async_design(**rx_args))['rx_params']
        else:
            rx_params = rx_dsn_specs

        if frontend_dsn_specs is None:
            dut_params = dict(
                pinfo=pinfo,
                buf_ctrl_lv_params=tx_params['ctrl_lvshift']['dut_params']['params'][
                    'in_buf_params'],
                ctrl_lv_params=tx_params['ctrl_lvshift']['dut_params']['params']['lv_params'],
                buf_por_lv_params=rx_params['buf_por_lv_params'],
                por_lv_params=rx_params['por_lv_params'],
                rx_lv_params=rx_params['data_lv_params'],
                inv_params=rx_params['inv_params'],
                se_params=rx_params['se_params'],
                match_params=rx_params['match_params'],
                buf_data_lv_params=tx_params['data_lvshift']['dut_params']['params'][
                    'in_buf_params'],
                tx_lv_params=tx_params['data_lvshift']['dut_params']['params']['lv_params'],
                drv_params=tx_params['driver']['dut_params']['params'],
            )
        else:
            dut_params = frontend_dsn_specs
            if not with_esd:
                dut_params['pinfo'] = pinfo

        print("=" * 80)
        print("Frontend: Running Signoff...")
        await self.verify_design(dut_params, tp_targ, tmismatch, c_ext, freq, trst, td_max,
                                 trf_max, c_odat, c_odat_async, c_oclkp, c_oclkn, trf_in, with_esd)

        return dut_params

    async def verify_design(self, dut_params, tp_targ, tmismatch, c_ext, freq, trst, td_max,
                            trf_max, c_odat, c_odat_async, c_oclkp, c_oclkn, trf_in, with_esd):
        """Sanity check that we are meeting spec
        """
        gen_params = dict(
            cls_name=Frontend.get_qualified_name(),
            draw_taps=False,
            params=dut_params,
        )
        if with_esd:
            dut = await self.async_new_dut('frontendESD_top', FrontendESD, dut_params,
                                           export_lay=True)
        else:
            dut = await self.async_new_dut('frontend_top', STDCellWrapper, gen_params)

        tx_ok = await self._signoff_tx(dut, c_ext, freq, trf_in, trst, td_max, trf_max, with_esd)
        if not tx_ok:
            self.log("TX failed top-level sign off", level=LogLevel.WARN)

        rx_ok = await self._signoff_rx(dut, tp_targ, tmismatch, trf_in, c_odat,
                                       c_odat_async, c_oclkp, c_oclkn, c_ext, freq, with_esd)
        if not rx_ok:
            self.log("RX failed top-level sign off", level=LogLevel.WARN)

        return tx_ok and rx_ok

    async def _signoff_tx(self, dut, c_ext, freq, trf_in, trst, td_max, trf_max, with_esd: bool,
                          spec_margin: float = 0.1, sim_options=None) -> bool:
        tech_info = self.dsn_tech_info
        all_corners = tech_info['signoff_envs']['all_corners']

        # Run all corners
        envs = all_corners['envs']
        worst_td = -float('inf')
        worst_trf = -float('inf')
        worst_env = ''
        worst_trf_env = ''
        for env in envs:
            vdd_io = all_corners['vddio'][env]
            vdd_core = all_corners['vdd'][env]
            tbm_specs = self.get_tx_tbm_specs(c_ext, freq, trf_in, trst, vdd_core, vdd_io, [env],
                                              with_esd, sim_options)

            tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'signoff_txanlg_{env}', dut,
                                                            tbm, tbm_specs)
            # get POR fall time to know start time
            # t_start = tbm.calc_cross(sim_results.data, 'XDUT.por_vccl', out_edge=EdgeType.FALL)

            td_in_hh = tbm.calc_delay(sim_results.data, 'din', 'iopad' if with_esd else 'txpadout',
                                      in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                      t_start=1 / freq)
            td_in_ll = tbm.calc_delay(sim_results.data, 'din', 'iopad' if with_esd else 'txpadout',
                                      in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                      t_start=1 / freq)
            td = max(td_in_hh[0], td_in_ll[0])
            if td > worst_td:
                worst_td = td
                worst_env = env
                data_td = sim_results.data

            trf_r = tbm.calc_trf(sim_results.data, 'iopad' if with_esd else 'txpadout',
                                 True, t_start=1 / freq)
            trf_f = tbm.calc_trf(sim_results.data, 'iopad' if with_esd else 'txpadout',
                                 False, t_start=1 / freq)
            trf = max(trf_r[0], trf_f[0])
            if trf > worst_trf:
                worst_trf = trf
                worst_trf_env = env
                data_trf = sim_results.data

        self.log(f'TX Data td_target = {td_max}, td_worst = {worst_td}, worst_env = {worst_env}')
        self.log(f'TX Data trf_target = {trf_max}, trf_worst = {worst_trf}, '
                 f'worst_env = {worst_trf_env}')

        return worst_td <= td_max * (1 + spec_margin) and worst_trf <= (1 + spec_margin) * trf_max

    def get_tx_tbm_specs(self, c_ext, freq, trf, trst, vdd_core, vdd_io, sim_envs, with_esd,
                         sim_options=None):
        load_pin, pins, pwr_domain = self.make_shared_tbm_params(with_esd)
        tbm_specs = dict(
            dut_pins=pins,
            pulse_list=[dict(pin='din',
                             tper='tbit',
                             tpw='tbit/2',
                             trf='trf',
                             pos=True
                             ),
                        ],
            load_list=[dict(pin=load_pin,
                            type='cap',
                            value=c_ext)],
            sup_values=dict(VSS=0,
                            VDDCore=vdd_core,
                            VDDIO=vdd_io,
                            ),
            pwr_domain=pwr_domain,
            pin_values={'indrv_buf<1:0>': 3,
                        'ipdrv_buf<1:0>': 3,
                        'itx_en_buf': 1,
                        'weak_pulldownen': 0,
                        'weak_pullupenb': 1,
                        'iclkn': 0,
                        'data_en': 0,
                        'clk_en': 0,
                        },
            reset_list=[('por', True)],
            sim_envs=sim_envs,
            sim_params=dict(
                freq=freq,
                tbit=1 / freq,
                trf=trf,
                t_rst=trst,
                t_rst_rf=trf,
                t_sim=2 / freq,
            ),
            save_outputs=['iopad', 'din'] if with_esd else ['txpadout', 'din'],
        )
        if sim_options:
            tbm_specs['sim_options'] = sim_options

        return tbm_specs

    @staticmethod
    def make_shared_tbm_params(with_esd: bool):
        pins = ['VDDCore', 'VDDIO', 'VSS', 'clk_en', 'data_en', 'din', 'iclkn',
                'indrv_buf<1:0>', 'ipdrv_buf<1:0>', 'itx_en_buf', 'oclkn', 'oclkp',
                'odat', 'odat_async', 'por', 'weak_pulldownen', 'weak_pullupenb']
        pins.extend(['iopad', 'iopad_out'] if with_esd else ['rxpadin', 'txpadout'])
        load_pin = 'iopad' if with_esd else 'txpadout'
        pwr_domain = {'din': ('VSS', 'VDDCore'),
                      'indrv_buf': ('VSS', 'VDDCore'),
                      'ipdrv_buf': ('VSS', 'VDDCore'),
                      'itx_en_buf': ('VSS', 'VDDCore'),
                      'por': ('VSS', 'VDDCore'),
                      'weak_pulldownen': ('VSS', 'VDDCore'),
                      'weak_pullupenb': ('VSS', 'VDDCore'),
                      'iclkn': ('VSS', 'VDDIO'),
                      'odat': ('VSS', 'VDDCore'),
                      'odat_async': ('VSS', 'VDDCore'),
                      'oclkp': ('VSS', 'VDDCore'),
                      'oclkn': ('VSS', 'VDDCore'),
                      'data_en': ('VSS', 'VDDCore'),
                      'clk_en': ('VSS', 'VDDCore'),
                      }
        if with_esd:
            pwr_domain['iopad'] = ('VSS', 'VDDIO')
            pwr_domain['iopad_out'] = ('VSS', 'VDDIO')
        else:
            pwr_domain['txpadout'] = ('VSS', 'VDDIO')
            pwr_domain['rxpadin'] = ('VSS', 'VDDIO')

        return load_pin, pins, pwr_domain

    async def _signoff_rx(self, dut, tp_targ, tmismatch, trf_in, c_odat,
                          c_odat_async, c_oclkp, c_oclkn, c_ext, freq, with_esd,
                          spec_margin: float = 10e-2):
        tech_info = self.dsn_tech_info
        all_corners = tech_info['signoff_envs']['all_corners']

        # Run level shifter signoff corner
        sim_id = "frontend_verify_rx_por_lvl_extreme"
        env = tech_info['signoff_envs']['lvl_func']['env']
        vdd_io = tech_info['signoff_envs']['lvl_func']['vddo']
        vdd_core = tech_info['signoff_envs']['lvl_func']['vddi']

        tbm_specs = self.get_rx_tbm_specs(c_ext, c_odat, c_odat_async, c_oclkp, c_oclkn, freq,
                                          trf_in, 5 * trf_in, vdd_core, vdd_io, env, with_esd)
        tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
        sim_results = await self.async_simulate_tbm_obj(f'{sim_id}', dut, tbm, tbm_specs)
        sim_data = sim_results.data

        td_list = self.get_rx_delays(sim_data, tbm, freq, with_esd)
        for del_arr in td_list:
            if del_arr[0] == float('inf') or del_arr[0] < 0:
                msg = f'RX failed extreme level shifer signoff - got infinite or ' + \
                      f'negative delay.\n' + \
                      f'Odat rise: {td_list[0][0]}, Odat fall: {td_list[1][0]}\n' + \
                      f'Odat_async rise: {td_list[2][0]}, Odat_async fall: {td_list[3][0]}\n' + \
                      f'Oclkp rise: {td_list[4][0]}, Oclkp fall: {td_list[5][0]}\n' + \
                      f'Oclkn rise: {td_list[6][0]}, Oclkn fall: {td_list[7][0]}'
                self.log(msg, level=LogLevel.WARN)

        # Run all corners
        envs = all_corners['envs']
        worst_td = -float('inf')
        worst_td_clk = -float('inf')
        worst_env = ''
        sim_id = "frontend_verify_rx"
        passed = True
        msg = ''
        for env in envs:
            vdd_io = all_corners['vddio'][env]
            vdd_core = all_corners['vdd'][env]
            tbm_specs = self.get_rx_tbm_specs(c_ext, c_odat, c_odat_async, c_oclkp, c_oclkn, freq,
                                              trf_in, 5 * trf_in, vdd_core, vdd_io, [env], with_esd)

            tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'{sim_id}_{env}', dut, tbm, tbm_specs)
            sim_data = sim_results.data
            td_list = self.get_rx_delays(sim_data, tbm, freq, with_esd)
            [td_odat_hh, td_odat_ll, td_odat_async_hh, td_odat_async_ll,
             td_oclkp_hh, td_oclkp_ll, td_oclkn_hh, td_oclkn_ll] = td_list

            # Verify clk, data meeting timing
            if td_oclkp_hh[0] > (1 + spec_margin) * tp_targ or td_oclkp_ll[0] > (1 + spec_margin) * tp_targ:
                msg = msg + \
                      f"\nFAIL: oclkp path delay too slow. Spec: {tp_targ}, " + \
                      f"actual rise: {td_oclkp_hh}, actual fall: {td_oclkp_ll} in corner {env}\n"
                passed = False
            if td_oclkn_hh[0] > (1 + spec_margin) * tp_targ or td_oclkn_ll[0] > (1 + spec_margin) * tp_targ:
                msg = msg + \
                      f"\nFAIL: clkn path delay too slow. Spec: {tp_targ}, " + \
                      f"actual rise: {td_oclkn_hh}, actual fall: {td_oclkn_ll}  in corner {env}\n"
                passed = False
            if td_odat_hh[0] > (1 + spec_margin) * tp_targ or td_odat_ll[0] > (1 + spec_margin) * tp_targ:
                msg = msg + \
                      f"\nFAIL: Data path delay too slow. Spec: {tp_targ}, " + \
                      f"\nactual rise: {td_odat_hh}, actual fall: {td_odat_ll}  in corner {env}\n"
                passed = False

            td = max(td_odat_hh[0], td_odat_ll[0])
            if td > worst_td:
                worst_td = td
                worst_env = env

            td_clk = max(td_oclkp_hh[0], td_oclkp_ll[0], td_oclkn_hh[0], td_oclkn_ll[0])
            if td_clk > worst_td_clk:
                worst_td_clk = td_clk

            # Verify clk - data alignment
            if abs(td_odat_hh[0] - td_oclkp_hh[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Skew between odat and oclkp rise delays too large.\n" + \
                      f"Spec: {tmismatch}, actual difference: {td_odat_hh[0] - td_oclkp_hh[0]}" + \
                      f"in corner {env}\n"
                passed = False
            if abs(td_odat_ll[0] - td_oclkp_ll[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Skew between odat and oclkp fall delays too large\n" + \
                      f"Spec: {tmismatch}, actual difference: {td_odat_ll[0] - td_oclkp_ll[0]}" + \
                      f" in corner {env}\n"
                passed = False
            if abs(td_odat_hh[0] - td_oclkn_hh[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Skew between odat and oclkn rise delays too large\n" + \
                      f"Spec: {tmismatch}, actual difference: {td_odat_hh[0] - td_oclkn_hh[0]}" + \
                      f" in corner {env}\n"
                passed = False
            if abs(td_odat_ll[0] - td_oclkn_ll[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Skew between odat and oclkn fall delays too large\n" + \
                      f"Spec: {tmismatch}, actual difference: {td_odat_ll[0] - td_oclkn_ll[0]}" + \
                      f" in corner {env}\n"
                passed = False

            # Verify clk - clkb alignment
            if abs(td_oclkp_hh[0] - td_oclkn_ll[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Skew between oclkp rise and oclkn fall too large.\n" + \
                      f"Spec: {tmismatch}, actual difference: {td_oclkp_hh[0] - td_oclkn_ll[0]}" + \
                      f" in corner {env}\n"
                passed = False
            if abs(td_oclkp_ll[0] - td_oclkn_hh[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Skew between oclkp fall and oclkn rise too large.\n" + \
                      f"Spec: {tmismatch}, actual difference: {td_oclkp_ll[0] - td_oclkn_hh[0]}" + \
                      f" in corner {env}\n"
                passed = False

            # Verify duty cycle
            if abs(td_oclkp_hh[0] - td_oclkp_ll[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Duty cycle distortion on oclkp too large.\n" + \
                      f"|Spec|: {tmismatch}, actual error: {td_oclkp_hh[0] - td_oclkp_ll[0]}" + \
                      f" in corner {env}\n"
                passed = False
            if abs(td_oclkn_hh[0] - td_oclkn_ll[0]) > (1 + spec_margin) * tmismatch:
                msg = msg + \
                      f"\nFAIL: Duty cycle distortion on oclkn too large.\n" + \
                      f"|Spec|: {tmismatch}, actual error: {td_oclkn_hh[0] - td_oclkn_ll[0]}" + \
                      f" in corner {env}\n"
                passed = False

        if not passed:
            self.log(msg, level=LogLevel.WARN)

        self.log(f'----------\nFrontend PASSED RX signoff.\n'
                 f'Delay target = {tp_targ}, Mismatch target = {tmismatch}\n'
                 f'Data worst delay = {worst_td}, '
                 f'worst_env = {worst_env}\n'
                 f'Clock worst delay = {worst_td_clk}')

        return passed

    def get_rx_tbm_specs(self, c_ext, c_odat, c_odat_async, c_oclkp, c_oclkn, freq, trf, trst,
                         vdd_core, vdd_io, sim_envs, with_esd, sim_options=None):
        load_pin, pins, pwr_domain = self.make_shared_tbm_params(with_esd)
        save_outputs = ['iopad', 'iclkn', 'oclkp', 'oclkn', 'odat', 'odat_async'] if with_esd else \
                       ['rxpadin', 'iclkn', 'oclkp', 'oclkn', 'odat', 'odat_async']
        tbm_specs = dict(
            dut_pins=pins,
            pulse_list=[dict(pin='iopad' if with_esd else 'rxpadin',
                             tper='tbit',
                             tpw='tbit/2',
                             trf='trf',
                             pos=True
                             ),
                        ],
            diff_list=[(['iopad'] if with_esd else ['rxpadin'], ['iclkn'])],
            load_list=[dict(pin=load_pin,
                            type='cap',
                            value=c_ext),
                       dict(pin='odat',
                            type='cap',
                            value=c_odat),
                       dict(pin='odat_async',
                            type='cap',
                            value=c_odat_async),
                       dict(pin='oclkp',
                            type='cap',
                            value=c_oclkp),
                       dict(pin='oclkn',
                            type='cap',
                            value=c_oclkn)],
            sup_values=dict(VSS=0,
                            VDDCore=vdd_core,
                            VDDIO=vdd_io,
                            ),
            pwr_domain=pwr_domain,
            pin_values={'indrv_buf<1:0>': 0,
                        'ipdrv_buf<1:0>': 0,
                        'itx_en_buf': 0,
                        'weak_pulldownen': 0,
                        'weak_pullupenb': 1,
                        'data_en': 1,
                        'clk_en': 1,
                        },
            reset_list=[('por', True)],
            sim_envs=sim_envs,
            sim_params=dict(
                freq=freq,
                tbit=1 / freq,
                trf=trf,
                t_rst=trst,
                t_rst_rf=trf,
                t_sim=2 / freq,
            ),
            save_outputs=save_outputs,
        )
        if sim_options:
            tbm_specs['sim_options'] = sim_options

        return tbm_specs

    @staticmethod
    def get_rx_delays(sim_data, tbm, freq, with_esd):
        in_pad = 'iopad' if with_esd else 'rxpadin'
        td_odat_hh = tbm.calc_delay(sim_data, in_pad, 'odat',
                                    in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                    t_start=1 / freq)
        td_odat_ll = tbm.calc_delay(sim_data, in_pad, 'odat',
                                    in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                    t_start=1 / freq)
        td_odat_async_hh = tbm.calc_delay(sim_data, in_pad, 'odat_async',
                                          in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                          t_start=1 / freq)
        td_odat_async_ll = tbm.calc_delay(sim_data, in_pad, 'odat_async',
                                          in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                          t_start=1 / freq)
        td_oclkp_hh = tbm.calc_delay(sim_data, in_pad, 'oclkp',
                                     in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                     t_start=1 / freq)
        td_oclkp_ll = tbm.calc_delay(sim_data, in_pad, 'oclkp',
                                     in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                     t_start=1 / freq)
        td_oclkn_hh = tbm.calc_delay(sim_data, 'iclkn', 'oclkn',
                                     in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                     t_start=1 / freq)
        td_oclkn_ll = tbm.calc_delay(sim_data, 'iclkn', 'oclkn',
                                     in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                     t_start=1 / freq)

        return [td_odat_hh, td_odat_ll, td_odat_async_hh, td_odat_async_ll,
                td_oclkp_hh, td_oclkp_ll, td_oclkn_hh, td_oclkn_ll]

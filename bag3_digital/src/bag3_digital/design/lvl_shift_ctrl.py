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
import math
from typing import Mapping, Dict, Any, Tuple, List, Sequence, cast

import numpy as np
import matplotlib.pyplot as plt

from bag.util.search import BinaryIterator
from bag.simulation.cache import DesignInstance

from xbase.layout.mos.placement.data import TileInfoTable

from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB

from bag3_digital.layout.stdcells.util import STDCellWrapper
from bag3_digital.layout.stdcells.levelshifter import LevelShifter, LevelShifterCore
from bag3_digital.measurement.cap.delay_match import CapDelayMatch
from bag3_digital.design.base import DigitalDesigner

from bag.env import get_tech_global_info


class LvlShiftCtrlDesigner(DigitalDesigner):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cin_specs: Dict[str, Any] = {}

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.dsn_specs
        c_load: float = specs['cload']
        buf_config: Mapping[str, Any] = specs['buf_config']
        search_params: Mapping[str, Any] = specs['search_params']

        pwr_tup = ('VSS', 'VDD')
        pwr_tup_in = ('VSS', 'VDDI')
        pwr_domain = {'in': pwr_tup_in, 'inb': pwr_tup_in}
        for name in ['rst_out', 'rst_outb', 'rst_casc', 'out', 'outb']:
            pwr_domain[name] = pwr_tup
        supply_map = dict(VDDI='VDDI', VDD='VDD', VSS='VSS')
        pin_values = dict(rst_outb=0)
        reset_list = [('rst_out', True)]
        diff_list = [(['rst_out'], ['rst_casc']), (['in'], ['inb']), (['out'], ['outb'])]
        tbm_specs = self.get_dig_tran_specs(pwr_domain, supply_map, pin_values=pin_values,
                                            reset_list=reset_list, diff_list=diff_list)
        tbm_specs['sim_params'] = sim_params = dict(**tbm_specs['sim_params'])
        sim_params['c_load'] = c_load

        self._cin_specs = dict(
            in_pin='in',
            buf_config=dict(**buf_config),
            search_params=search_params,
            tbm_specs=tbm_specs,
            load_list=[dict(pin='out', type='cap', value='c_load')],
        )

    async def async_design(self, cload: float, dmax: float, trf_in: float,
                           tile_specs: Mapping[str, Any],
                           k_ratio: float, tile_name: str, inv_input_cap: float,
                           inv_input_cap_per_fin: float,
                           fanout: float, vin: str, vout: str,
                           w_p: int = 0, w_n: int = 0,
                           ridx_p: int = -1, ridx_n: int = 0, has_rst: bool = False,
                           is_ctrl: bool = False, dual_output: bool = False,
                           exception_on_dmax: bool = True,
                           del_scale: float = 1, **kwargs: Any) -> Mapping[str, Any]:
        """ Design a Level Shifter
        This will try to design a level shifter to meet a maximum nominal delay, given the load cap
        """
        tech_info = get_tech_global_info('bag3_digital')
        w_p = tech_info['w_maxp'] if w_p == 0 else w_p
        w_n = tech_info['w_maxn'] if w_n == 0 else w_n
        if not 'lch' in tile_specs['arr_info']:
            tile_specs['arr_info']['lch'] = tech_info['lch_min']
        tile_specs['place_info'][tile_name]['row_specs'][0]['width'] = w_n
        tile_specs['place_info'][tile_name]['row_specs'][1]['width'] = w_p

        tinfo_table = TileInfoTable.make_tiles(self.grid, tile_specs)
        pinfo = tinfo_table[tile_name]

        # Design the output inverter, and the level shift core
        design_sim_env, vdd_in, vdd_out = self._build_env_vars('center', vin, vout)
        tbm_specs = self._get_tbm_params(design_sim_env, vdd_in, vdd_out, trf_in, cload, 10 * dmax)
        tbm_specs['save_outputs'] = ['in', 'inbar', 'out', 'outb', 'inb_buf', 'in_buf', 'midn',
                                     'midp']
        out_inv_m, pseg, nseg = self._design_lvl_shift_core_size(cload, k_ratio, inv_input_cap,
                                                                 fanout, is_ctrl)

        # Design the inverter creating the inverted input to the leveler
        inv_pseg, inv_nseg = await self._design_lvl_shift_internal_inv(pseg, nseg, out_inv_m,
                                                                       fanout, pinfo,
                                                                       tbm_specs, is_ctrl, has_rst,
                                                                       dual_output,
                                                                       vin, vout)

        # Design input inverter
        inv_in_nseg, inv_in_pseg = self._size_input_inv_for_fanout(inv_pseg, inv_nseg, pseg, nseg,
                                                                   fanout, has_rst)

        # Adjust the output inverter beta ratio to further reduce duty cycle distortion
        if not is_ctrl:
            pseg_off = await self._design_output_inverter(inv_in_pseg, inv_in_nseg, pseg, nseg,
                                                          inv_nseg, inv_pseg,
                                                          out_inv_m, fanout, pinfo, tbm_specs,
                                                          has_rst,
                                                          vin, vout)
        else:
            pseg_off, worst_env = 0, ''

        # Final Simulation
        dut_params = self._get_lvl_shift_params_dict(pinfo, pseg, nseg, inv_pseg, inv_nseg,
                                                     inv_in_pseg, inv_in_nseg, out_inv_m,
                                                     has_rst, dual_output, is_ctrl,
                                                     skew_out=not is_ctrl,
                                                     out_pseg_off=pseg_off)

        dut = await self.async_new_dut('lvshift', STDCellWrapper, dut_params)
        tdr, tdf, worst_env, worst_var, worst_var_env = await self.signoff_dut(dut, cload, vin,
                                                                               vout, dmax, trf_in,
                                                                               is_ctrl, has_rst,
                                                                               exception_on_dmax)

        if not is_ctrl and max(tdr, tdf) > dmax:
            # Find intrinsic delay based on stage-by-stage characterization
            tgate_dict, tint_dict, tint_tot = await self._find_tgate_and_tint(inv_in_pseg,
                                                                              inv_in_nseg, pseg,
                                                                              nseg,
                                                                              inv_nseg, inv_pseg,
                                                                              out_inv_m, pseg_off,
                                                                              inv_input_cap, cload,
                                                                              k_ratio, pinfo,
                                                                              tbm_specs, is_ctrl,
                                                                              has_rst, dual_output,
                                                                              vin, vout, worst_env)
        else:
            tint_tot = 0

        dut_params = dut_params['params'].copy()
        dut_params.pop('pinfo', None)
        dut_params.pop('export_pins', None)
        c_in = await self.get_cap(dut, 'in', inv_input_cap)

        ans = dict(dut_params=dut_params, tdr=tdr, tdf=tdf, tint=tint_tot, worst_var=worst_var,
                   c_in=c_in)
        if has_rst:
            c_rst_out = await self.get_cap(dut, 'rst_out', inv_input_cap)
            c_rst_casc = await self.get_cap(dut, 'rst_casc', inv_input_cap)
            ans['c_rst_out'] = c_rst_out
            ans['c_rst_casc'] = c_rst_casc

        return ans

    async def _find_tgate_and_tint(self, inv_in_pseg, inv_in_nseg, pseg, nseg, inv_nseg, inv_pseg,
                                   out_inv_m, pseg_off, inv_input_cap, cload, k_ratio, pinfo,
                                   tbm_specs,
                                   is_ctrl, has_rst, dual_output, vin, vout, worst_env) -> Tuple[
        dict, dict, float]:

        # Setup nominal parameters and delay
        lv_params = dict(
            pinfo=pinfo,
            seg_p=pseg,
            seg_n=nseg,
            seg_inv_p=inv_pseg,
            seg_inv_n=inv_nseg,
            seg_in_inv_n=inv_in_nseg,
            seg_in_inv_p=inv_in_pseg,
            out_inv_m=out_inv_m,
            out_pseg_off=pseg_off,
        )
        tb_params = self._get_full_tb_params()
        dut_params = self._get_lvl_shift_params_dict(**lv_params, has_rst=has_rst,
                                                     dual_output=False, skew_out=True)
        dut = await self.async_new_dut('lvshift', STDCellWrapper, dut_params)
        all_corners = get_tech_global_info('bag3_digital')['signoff_envs']['all_corners']
        tbm_specs['sim_envs'] = [worst_env]
        tbm_specs['sim_params']['vdd_in'] = all_corners[vin][worst_env]
        tbm_specs['sim_params']['vdd'] = all_corners[vout][worst_env]
        tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
        sim_results_orig = await self.async_simulate_tbm_obj(f'sim_output_inv_pseg_{pseg_off}', dut,
                                                             tbm,
                                                             tb_params)
        tdr_nom, tdf_nom = CombLogicTimingTB.get_output_delay(sim_results_orig.data, tbm.specs,
                                                              'in',
                                                              'out', False, in_pwr='vdd_in',
                                                              out_pwr='vdd')

        slope_dict = dict()
        tot_sense_dict = dict()
        tint_dict = dict()
        segs_out = cload / inv_input_cap
        tweak_vars = [('seg_in_inv_p', 'in', 'inb_buf', 'fall', True, 'vdd_in', 'vdd_in',
                       inv_in_pseg + inv_in_nseg,
                       nseg + inv_nseg + inv_pseg + (pseg if has_rst else 0), 0),
                      ('seg_n', 'inb_buf', 'midp', 'rise', True, 'vdd_in', 'vdd',
                       nseg, pseg, 1 / k_ratio),
                      ('seg_p', 'midp', 'midn', 'fall', True, 'vdd', 'vdd',
                       pseg, 2 * out_inv_m - pseg_off, 1),
                      ('out_inv_m', 'midn', 'out', 'rise', True, 'vdd', 'vdd',
                       2 * out_inv_m - pseg_off, segs_out, 0)]
        tint_tot = 0
        for var_tuple in tweak_vars:
            var, node_in, node_out, in_edge, invert, in_sup, out_sup, seg_in, seg_load, fan_min = var_tuple
            tdr_stg_nom, tdf_stg_nom = CombLogicTimingTB.get_output_delay(sim_results_orig.data,
                                                                          tbm.specs, node_in,
                                                                          node_out, invert,
                                                                          in_pwr=in_sup,
                                                                          out_pwr=out_sup)

            nom_var = lv_params[var]
            lv_params[var] = nom_var + 1

            dut_params = self._get_lvl_shift_params_dict(**lv_params, has_rst=has_rst,
                                                         dual_output=False, skew_out=True)
            dut = await self.async_new_dut('lvshift', STDCellWrapper, dut_params)
            sim_results = await self.async_simulate_tbm_obj(f'sim_check_tgate_tint', dut, tbm,
                                                            tb_params)
            tdr_new, tdf_new = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs, 'in',
                                                                  'out', False, in_pwr='vdd_in',
                                                                  out_pwr='vdd')
            tdr_stg_new, tdf_stg_new = CombLogicTimingTB.get_output_delay(sim_results.data,
                                                                          tbm.specs, node_in,
                                                                          node_out, invert,
                                                                          in_pwr=in_sup,
                                                                          out_pwr=out_sup)
            td_new, td_nom = (tdr_stg_new, tdr_stg_nom) if in_edge == 'rise' else (
            tdf_stg_new, tdf_stg_nom)

            slope_dict[var] = (td_nom[0] - td_new[0]) / (
                        seg_load / seg_in - seg_load / (seg_in + 1))
            tot_sense_dict[var] = tdf_nom[0] - tdf_new[0]
            tint_dict[var] = td_nom[0] - slope_dict[var] * seg_load / seg_in
            lv_params[var] = nom_var
            tint_tot += fan_min * slope_dict[var] + tint_dict[var]

        return slope_dict, tint_dict, tint_tot

    async def signoff_dut(self, dut, cload, vin, vout, dmax, trf_in, is_ctrl, has_rst,
                          exception_on_dmax: bool = True) -> Tuple[float, float, str, float, str]:

        tech_info = get_tech_global_info('bag3_digital')
        all_corners = tech_info['signoff_envs']['all_corners']

        # Run level shifter extreme corner signoff
        envs = tech_info['signoff_envs']['lvl_func']['env']
        vdd_out = tech_info['signoff_envs']['lvl_func']['vddo']
        vdd_in = tech_info['signoff_envs']['lvl_func']['vddi']

        tbm_specs = self._get_tbm_params(envs, vdd_in, vdd_out, trf_in,
                                         cload, 10 * dmax)
        tbm_specs['save_outputs'] = ['in', 'inbar', 'inb_buf', 'in_buf', 'midn', 'midp', 'out',
                                     'outb']
        tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))

        # sign off signal path
        tb_params = self._get_full_tb_params()
        sim_results = await self.async_simulate_tbm_obj(f'signoff_lvlshift_extreme', dut, tbm,
                                                        tb_params)
        tdr, tdf = CombLogicTimingTB.get_output_delay(sim_results.data, tbm_specs, 'in', 'out',
                                                      False, in_pwr='vdd_in', out_pwr='vdd')

        td = max(tdr, tdf)
        if td < float('inf'):
            self.log('Level shifter signal path passed extreme corner signoff.')
        else:
            plt.plot(sim_results.data['time'].flatten(), sim_results.data['in'].flatten(), 'b')
            plt.plot(sim_results.data['time'].flatten(), sim_results.data['out'].flatten(), 'g')
            plt.show(block=False)
            raise ValueError('Level shifter design failed extreme corner signoff.')

        # sign off reset
        if has_rst:
            tbm_specs['stimuli_pwr'] = 'vdd'
            tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
            rst_tb_params = self._get_rst_tb_params()
            sim_results = await self.async_simulate_tbm_obj(f'signoff_lvlshift_rst_extreme', dut,
                                                            tbm, rst_tb_params)
            tdr, tdf = CombLogicTimingTB.get_output_delay(sim_results.data, tbm_specs, 'in',
                                                          'out',
                                                          False, in_pwr='vdd_in', out_pwr='vdd')
            self.log(f"Reset Delay Overall: tdr: {tdr}, tdf: {tdf} ")
            td = max(tdr, tdf)
            if td < float('inf'):
                self.log('Level shifter reset path passed extreme corner signoff.')
            else:
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['in'].flatten(), 'b')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['out'].flatten(), 'g')
                plt.show(block=False)
                raise ValueError('Level shifter design failed reset extreme corner signoff.')

        envs = all_corners['envs']
        worst_trst = -float('inf')
        worst_td = -float('inf')
        worst_tdf = -float('inf')
        worst_tdr = -float('inf')
        worst_var = 0
        worst_env = ''
        worst_var_env = ''

        for env in envs:
            vdd_in = all_corners[vin][env]
            vdd_out = all_corners[vout][env]
            tbm_specs = self._get_tbm_params([env], vdd_in, vdd_out, trf_in,
                                             cload, 10 * dmax)
            tbm_specs['stimuli_pwr'] = 'vdd_in'
            tbm_specs['save_outputs'] = ['in', 'inb_buf', 'in_buf', 'midn', 'midp', 'out']
            tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))

            # sign off signal path
            tb_params = self._get_full_tb_params()
            sim_results = await self.async_simulate_tbm_obj(f'signoff_lvlshift_{env}', dut, tbm,
                                                            tb_params)
            tdr, tdf = CombLogicTimingTB.get_output_delay(sim_results.data, tbm_specs, 'in', 'out',
                                                          False, in_pwr='vdd_in', out_pwr='vdd')
            self.log(f"Delay Overall: tdr: {tdr}, tdf: {tdf} ")

            td = max(tdr, tdf)
            if td > worst_td:
                worst_td = td
                worst_tdf = tdf
                worst_tdr = tdr
                worst_env = env

            if not is_ctrl:
                delay_var = (tdr - tdf)
                if np.abs(delay_var) > np.abs(worst_var):
                    worst_var = delay_var
                    worst_var_env = env

            '''
            # Debug
            # -----
            td_iinv_r, td_iinv_f = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs, 'in',
                                                          'inb_buf', True, in_pwr='vdd_in',
                                                          out_pwr='vdd_in')
            td_minv_r, td_minv_f = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs, 'inb_buf',
                                                          'in_buf', True, in_pwr='vdd_in',
                                                          out_pwr='vdd_in')
            td_pdn_r, td_pdn_f = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs, 'in_buf',
                                                          'midn', True, in_pwr='vdd_in',
                                                          out_pwr='vdd')
            td_oinv_r, td_oinv_f = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs, 'midn',
                                                          'out', True, in_pwr='vdd_in',
                                                          out_pwr='vdd')
            td_pdp_r, td_pdp_f = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs, 'inb_buf',
                                                          'midp', True, in_pwr='vdd_in',
                                                          out_pwr='vdd')
            td_pun_r, td_pun_f = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs, 'midp',
                                                          'midn', True, in_pwr='vdd',
                                                          out_pwr='vdd')

            if env == 'ss_125':
                plt.figure()
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['in'].flatten(), 'b')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['inb_buf'].flatten(), 'g')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['in_buf'].flatten(), 'r')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['midn'].flatten(), 'k')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['midp'].flatten(), 'c')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['out'].flatten(), 'm')
                plt.legend(['in', 'inb_buf', 'in_buf', 'midn', 'midp', 'out'])
                plt.title(f'{env}')
                plt.show(block=False)

                print(f'Path rise out: {td_iinv_r} + {td_minv_f} + {td_pdn_r} + {td_oinv_f}')
                print(f'Path fall out: {td_iinv_f} + {td_pdp_r} + {td_pun_f} + {td_oinv_r}')
                breakpoint()
            # ----
            '''

            # sign off reset
            if has_rst:
                tbm_specs['stimuli_pwr'] = 'vdd'
                tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
                rst_tb_params = self._get_rst_tb_params()
                sim_results = await self.async_simulate_tbm_obj(f'signoff_lvlshift_rst_{env}', dut,
                                                                tbm, rst_tb_params)
                tdr, tdf = CombLogicTimingTB.get_output_delay(sim_results.data, tbm_specs, 'in',
                                                              'out',
                                                              False, in_pwr='vdd_in', out_pwr='vdd')
                self.log(f"Reset Delay Overall: tdr: {tdr}, tdf: {tdf} ")
                td = max(tdr, tdf)
                if td > worst_trst:
                    worst_trst = td
                    worst_trst_env = env

        td_target = 20 * trf_in if is_ctrl else dmax
        self.log(f'td_target = {td_target}, worst_tdr = {worst_tdr}, worst_tdf = {worst_tdf}, '
                 f'worst_env = {worst_env}')
        # breakpoint()

        if worst_tdr > td_target or worst_tdf > td_target:
            msg = 'Level shifter delay did not meet target.'
            if exception_on_dmax:
                raise RuntimeError(msg)
            else:
                self.log(msg)
        if has_rst:
            self.log(f'worst_trst = {worst_trst}, worst_trst_env = {worst_trst_env}')

        if worst_tdr > 20 * trf_in or worst_tdf > 20 * trf_in:
            raise RuntimeError("Level shifter reset delay exceeded simulation period.")
        return worst_tdr, worst_tdf, worst_env, worst_var, worst_var_env

    @staticmethod
    def _build_env_vars(env_str: str, vin: str, vout: str) -> Tuple[List[str], float, float]:
        dsn_env_info = get_tech_global_info('bag3_digital')['dsn_envs'][env_str]
        design_sim_env = dsn_env_info['env']
        vdd_in = dsn_env_info[vin]
        vdd_out = dsn_env_info[vout]

        return design_sim_env, vdd_in, vdd_out

    @staticmethod
    def _size_input_inv_for_fanout(inv_pseg: int, inv_nseg: int, pseg: int, nseg: int,
                                   fanout: float, has_rst: bool) -> Tuple[int, int]:
        beta = get_tech_global_info('bag3_digital')['inv_beta']
        seg_load = inv_pseg + inv_nseg + nseg + (pseg if has_rst else 0)
        iinv_nseg = int(np.round(seg_load / (1 + beta) / fanout))
        iinv_nseg = 1 if iinv_nseg < get_tech_global_info('bag3_digital')['seg_min'] else iinv_nseg
        iinv_pseg = int(np.round(seg_load * beta / (1 + beta) / fanout))
        iinv_pseg = 1 if iinv_pseg < get_tech_global_info('bag3_digital')['seg_min'] else iinv_pseg

        return iinv_nseg, iinv_pseg

    @staticmethod
    def _design_lvl_shift_core_size(cload: float, k_ratio: float, inv_input_cap: float,
                                    fanout: float, is_ctrl: bool) -> Tuple[int, int, int]:
        """ Size the core of the LVL Shifter given K_ratio, the ratio of the NMOS to PMOS
        """
        out_inv_input_cap = cload / fanout
        print(f'cload = {cload}')
        inv_m = int(round(out_inv_input_cap / inv_input_cap))
        inv_m = max(1, inv_m)
        pseg = int(round(2 * inv_m / fanout))
        pseg = max(1, pseg)
        if pseg == 1 and not is_ctrl:
            print("=" * 80)
            print(
                "WARNING: LvShift Designer: pseg has been set to 1; might want to remove output inverter.")
            print("=" * 80)

        '''
        # TODO: Find k_ratio based on functionality automatically rather than have it come from input params.
        all_corners = get_tech_global_info('bag3_digital')['signoff_envs']['all_corners']
        iterator = FloatBinaryIterator(low=1.0, high=10.0, tol=0.1)

        while iterator.has_next():
            k_cur = iterator.get_next()
            nseg = int(np.round(pseg*k_cur))

            dut_params = self._get_lvl_shift_core_params_dict(pinfo, pseg, nseg, has_rst, is_ctrl)
            dut = await self.async_new_dut('lvshift_core', STDCellWrapper, dut_params)
            functional = False

            for 
        '''

        nseg = int(np.round(pseg * k_ratio))

        return inv_m, pseg, nseg

    async def _design_lvl_shift_internal_inv(self, pseg: int, nseg: int, out_inv_m: int,
                                             fanout: float,
                                             pinfo: Any, tbm_specs: Dict[str, Any], is_ctrl: bool,
                                             has_rst: bool, dual_output: bool,
                                             vin: str, vout: str) -> Tuple[int, int]:
        """
        Given the NMOS segments and the PMOS segements ratio for the core, this function designs
        the internal inverter.
        For control level shifter, we don't care about matching rise / fall delay, so we just size
        for fanout.
        """
        if is_ctrl:  # size with fanout
            inv_nseg = int(np.round(nseg / fanout))
            inv_nseg = 1 if inv_nseg == 0 else inv_nseg
            inv_pseg = int(np.round(pseg / fanout))
            inv_pseg = 1 if inv_pseg == 0 else inv_pseg
            self.log(f"Calculated inv to need nseg : {inv_nseg}")
            self.log(f"Calculated inv to need pseg : {inv_pseg}")
            return inv_pseg, inv_nseg

        # First size the NMOS in the inverter assuming a reasonably sized PMOS
        inv_nseg = await self._design_lvl_shift_inv_pdn(pseg, nseg, out_inv_m, fanout, pinfo,
                                                        tbm_specs, has_rst, dual_output, vin, vout)
        self.log(f"Calculated inv to need at least nseg: {inv_nseg}")

        # Now using the inverter pull down size, we size the inverter pull up PMOS
        inv_pseg, inv_nseg = await self._design_lvl_shift_inv_pun(pseg, nseg, inv_nseg, out_inv_m,
                                                                  fanout, pinfo,
                                                                  tbm_specs, has_rst, dual_output,
                                                                  vin, vout)
        self.log(f"Calculated inv to need pseg: {inv_pseg} and nseg: {inv_nseg}")
        return inv_pseg, inv_nseg

    async def _design_lvl_shift_inv_pdn(self, pseg: int, nseg: int, out_inv_m: int,
                                        fanout: float, pinfo: Any, tbm_specs: Dict[str, Any],
                                        has_rst, dual_output, vin, vout) -> int:
        """
        This function figures out the NMOS nseg for the inverter given the target delay
        TODO: Make this use digitaldB instead
        """
        min_fanout: float = get_tech_global_info('bag3_digital')['min_fanout']
        inv_beta: float = get_tech_global_info('bag3_digital')['inv_beta']
        tb_params = self._get_full_tb_params()

        # Use a binary iterator to find the NMOS size
        max_nseg = int(np.round(nseg / min_fanout))
        iterator = BinaryIterator(1, max_nseg)
        load_seg = nseg + (pseg if has_rst else 0)
        inv_pseg = int(np.round(inv_beta * load_seg / ((1 + inv_beta) * fanout)))
        inv_pseg = 1 if inv_pseg == 0 else inv_pseg

        all_corners = get_tech_global_info('bag3_digital')['signoff_envs']['all_corners']
        while iterator.has_next():
            inv_nseg = iterator.get_next()
            inv_in_nseg, inv_in_pseg = self._size_input_inv_for_fanout(inv_pseg, inv_nseg, pseg,
                                                                       nseg, fanout, has_rst)

            dut_params = self._get_lvl_shift_params_dict(pinfo, pseg, nseg, inv_pseg, inv_nseg,
                                                         inv_in_pseg, inv_in_nseg, out_inv_m,
                                                         has_rst, dual_output)
            dut = await self.async_new_dut('lvshift', STDCellWrapper, dut_params)
            err_worst = -1 * float('Inf')
            for env in all_corners['envs']:
                tbm_specs['sim_envs'] = [env]
                tbm_specs['sim_params']['vdd_in'] = all_corners[vin][env]
                tbm_specs['sim_params']['vdd'] = all_corners[vout][env]
                tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))

                sim_results = await self.async_simulate_tbm_obj(f'sim_inv_nseg_{inv_nseg}_{env}',
                                                                dut, tbm,
                                                                tb_params)
                tdr_cur, tdf_cur = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs,
                                                                      'inb_buf', 'in_buf', True,
                                                                      in_pwr='vdd_in',
                                                                      out_pwr='vdd_in')
                target_cur, _ = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs,
                                                                   'inb_buf', 'midp', True,
                                                                   in_pwr='vdd_in', out_pwr='vdd')

                # Check for error conditions
                if math.isinf(np.max(tdr_cur)) or math.isinf(np.max(tdf_cur)) or math.isinf(
                        np.max(target_cur)):
                    raise ValueError(
                        "Got infinite delay in level shifter design script (sizing inverter NMOS).")
                if np.min(tdr_cur) < 0 or np.min(target_cur) < 0:
                    raise ValueError(
                        "Got negative delay in level shifter design script (sizing inverter NMOS). ")

                err_cur = tdr_cur[0] - target_cur[0]
                if err_cur > err_worst:
                    err_worst = err_cur
                    worst_env = env
                    tdr = tdr_cur[0]
                    target = target_cur[0]

            '''
            print(f'iter: {inv_nseg}')
            print(f'env: {worst_env}, tdr: {tdr}, target: {target}')
            '''

            if tdr < target:
                iterator.down(target - tdr)
                iterator.save_info(inv_nseg)
            else:
                iterator.up(target - tdr)

        tmp_inv_nseg = iterator.get_last_save_info()
        if tmp_inv_nseg is None:
            tmp_inv_nseg = max_nseg
            self.warn("Could not size pull down of inverter to meet required delay, picked the "
                      "max inv_nseg based on min_fanout.")

        return tmp_inv_nseg

    async def _design_lvl_shift_inv_pun(self, pseg: int, nseg: int, inv_nseg: int, out_inv_m: int,
                                        fanout: float,
                                        pinfo: Any, tbm_specs: Dict[str, Any], has_rst, dual_output,
                                        vin, vout) -> Tuple[int, int]:
        """
        Given the NMOS pull down size, this function will design the PMOS pull up so that the delay
        mismatch is minimized.
        # TODO: Need to double check on how this handles corners
        """
        inv_beta = get_tech_global_info('bag3_digital')['inv_beta']
        tb_params = self._get_full_tb_params()
        # Use a binary iterator to find the PMOS size
        load_seg = nseg + (pseg if has_rst else 0)
        inv_pseg_nom = int(np.round(inv_beta * load_seg / ((1 + inv_beta) * fanout)))
        inv_pseg_nom = 1 if inv_pseg_nom == 0 else inv_pseg_nom
        iterator = BinaryIterator(-inv_pseg_nom + 1, 0)
        err_best = float('inf')
        inv_in_nseg, inv_in_pseg = self._size_input_inv_for_fanout(inv_pseg_nom, inv_nseg, pseg,
                                                                   nseg, fanout, has_rst)
        all_corners = get_tech_global_info('bag3_digital')['signoff_envs']['all_corners']

        while iterator.has_next():
            pseg_off = iterator.get_next()
            inv_pseg = inv_pseg_nom + pseg_off
            dut_params = self._get_lvl_shift_params_dict(pinfo, pseg, nseg, inv_pseg, inv_nseg,
                                                         inv_in_nseg, inv_in_pseg, out_inv_m,
                                                         has_rst, dual_output)
            dut = await self.async_new_dut('lvshift', STDCellWrapper, dut_params)

            err_worst = -1 * float('Inf')
            for env in all_corners['envs']:
                tbm_specs['sim_envs'] = [env]
                tbm_specs['sim_params']['vdd_in'] = all_corners[vin][env]
                tbm_specs['sim_params']['vdd'] = all_corners[vout][env]
                tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
                sim_results = await self.async_simulate_tbm_obj(f'sim_inv_pseg_{inv_pseg}_{env}',
                                                                dut, tbm, tb_params)
                tdr_cur, tdf_cur = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs,
                                                                      'in',
                                                                      'out', False, in_pwr='vdd_in',
                                                                      out_pwr='vdd')

                '''
                plt.figure()
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['in'].flatten(), 'b')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['inb_buf'].flatten(), 'g')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['in_buf'].flatten(), 'r')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['midn'].flatten(), 'k')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['midp'].flatten(), 'c')
                plt.plot(sim_results.data['time'].flatten(), sim_results.data['out'].flatten(), 'm')
                plt.legend(['in', 'inb_buf', 'in_buf', 'midn', 'midp', 'out'])
                plt.title(f'pseg_off: {pseg_off}, pseg: {inv_pseg}, nseg: {inv_nseg-pseg_off}, fanout: {fanout}')
                plt.show(block=False)
                '''

                # Error checking
                if math.isinf(np.max(tdr_cur)) or math.isinf(np.max(tdf_cur)):
                    raise ValueError("Got infinite delay!")
                if np.min(tdr_cur) < 0 or np.min(tdf_cur) < 0:
                    raise ValueError("Got negative delay.")

                err_cur = np.abs(tdr_cur[0] - tdf_cur[0])
                if err_cur > err_worst:
                    err_worst = err_cur
                    worst_env = env
                    tdr = tdr_cur[0]
                    tdf = tdf_cur[0]

            '''
            print(f'iter: {inv_pseg}')
            print(f'env: {worst_env}, tdr: {tdr}, tdf: {tdf}')
            breakpoint()
            '''

            if tdr < tdf:
                iterator.down(tdr - tdf, False)
            else:
                iterator.up(tdr - tdf, False)

            err_abs = np.abs(tdr - tdf)
            if err_abs < err_best:
                err_best = err_abs
                iterator.save_info(pseg_off)

        pseg_off = iterator.get_last_save_info()
        pseg_off = 0 if pseg_off is None else pseg_off  # Should only hit this case if inv_pseg_nom = 1
        inv_pseg = inv_pseg_nom + pseg_off

        return inv_pseg, inv_nseg - 0 * pseg_off

    async def _design_output_inverter(self, inv_in_pseg: int, inv_in_nseg: int, pseg: int,
                                      nseg: int, inv_nseg: int,
                                      inv_pseg: int, out_inv_m: int, fanout: float, pinfo: Any,
                                      tbm_specs: Dict[str, Any], has_rst, vin, vout) -> int:
        """
        Given all other sizes and total output inverter segments, this function will optimize the output inverter
        to minimize rise/fall mismatch.
        """
        tb_params = self._get_full_tb_params()
        # Use a binary iterator to find the PMOS size
        iterator = BinaryIterator(-out_inv_m + 1, out_inv_m - 1)
        err_best = float('inf')
        all_corners = get_tech_global_info('bag3_digital')['signoff_envs']['all_corners']

        while iterator.has_next():
            pseg_off = iterator.get_next()
            dut_params = self._get_lvl_shift_params_dict(pinfo, pseg, nseg, inv_pseg, inv_nseg,
                                                         inv_in_nseg, inv_in_pseg, out_inv_m,
                                                         has_rst, dual_output=False, skew_out=True,
                                                         out_pseg_off=pseg_off)
            dut = await self.async_new_dut('lvshift', STDCellWrapper, dut_params)

            err_worst = -1 * float('Inf')
            worst_env = ''
            sim_worst = None
            for env in all_corners['envs']:
                tbm_specs['sim_envs'] = [env]
                tbm_specs['sim_params']['vdd_in'] = all_corners[vin][env]
                tbm_specs['sim_params']['vdd'] = all_corners[vout][env]
                tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
                sim_results = await self.async_simulate_tbm_obj(
                    f'sim_output_inv_pseg_{pseg_off}_{env}', dut, tbm,
                    tb_params)
                tdr_cur, tdf_cur = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs,
                                                                      'in',
                                                                      'out', False, in_pwr='vdd_in',
                                                                      out_pwr='vdd')

                if math.isinf(np.max(tdr_cur)) or math.isinf(np.max(tdf_cur)):
                    raise ValueError("Got infinite delay!")
                if tdr_cur[0] < 0 or tdf_cur[0] < 0:
                    raise ValueError("Got negative delay.")

                err_cur = np.abs(tdr_cur[0] - tdf_cur[0])
                if err_cur > err_worst:
                    err_worst = err_cur
                    worst_env = env
                    tdr = tdr_cur[0]
                    tdf = tdf_cur[0]
                    sim_worst = sim_results

            '''
            print(f'iter: {pseg_off}')
            print(f'env: {worst_env}, tdr: {tdr}, tdf: {tdf}')
            breakpoint()
            '''

            if tdr < tdf:
                iterator.down(tdr - tdf)
            else:
                iterator.up(tdr - tdf)

            err_abs = np.abs(tdr - tdf)
            if err_abs < err_best:
                err_best = err_abs
                iterator.save_info(pseg_off)

        pseg_off = iterator.get_last_save_info()
        if pseg_off is None:
            raise ValueError("Could not find PMOS size to match target delay")

        self.log(f'Calculated output inverter to skew PMOS by {pseg_off}.')

        return pseg_off

    @staticmethod
    def _get_lvl_shift_core_params_dict(pinfo: Any, seg_p: int, seg_n: int,
                                        has_rst: bool, is_ctrl: bool = False) -> Dict[str, Any]:
        """
        Creates a dictionary of parameters for the layout class LevelShifterCore
        seg_n : nmos Pull down nseg
        seg_p : pmos Pull up nseg
        pinfo : pinfo
        Note: This will let the width be passed through the pinfo, currently no rst
        """
        global_info = get_tech_global_info('bag3_digital')
        wn = global_info['w_minn'] if is_ctrl else 2 * global_info['w_minn']
        wp = global_info['w_minp'] if is_ctrl else 2 * global_info['w_minp']

        if has_rst:
            seg_dict = dict(pd=seg_n, pu=seg_p, rst=int(np.ceil(seg_n / 2)), prst=seg_p)
            w_dict = dict(pd=wn, pu=wp, rst=wn)
        else:
            seg_dict = dict(pd=seg_n, pu=seg_p)
            w_dict = dict(pd=wn, pu=wp)
        lv_params = dict(
            cls_name=LevelShifterCore.get_qualified_name(),
            draw_taps=True,
            params=dict(
                pinfo=pinfo,
                seg_dict=seg_dict,
                w_dict=w_dict,
                has_rst=has_rst,
                in_upper=has_rst,
            )
        )

        if has_rst:
            lv_params['params']['lv_params']['stack_p'] = 2

        return lv_params

    @staticmethod
    def _get_lvl_shift_params_dict(pinfo: Any, seg_p: int, seg_n: int, seg_inv_p: int,
                                   seg_inv_n: int, seg_in_inv_p: int, seg_in_inv_n: int,
                                   out_inv_m: int, has_rst: bool, dual_output: bool,
                                   is_ctrl: bool = False,
                                   skew_out: bool = False, out_pseg_off: int = 0) -> Dict[str, Any]:
        """
        Creates a dictionary of parameters for the layout class LevelShifter
        seg_n : nmos Pull down nseg
        seg_p : pmos Pull up nseg
        seg_inv : Inb_buf to In_buf inverter segments
        seg_in_inv : In to Inb_buf inverter segments
        pinfo : pinfo
        # TODO: UPDATE THIS DOCUMENTATION
        Note: This will let the width be passed through the pinfo, currently no rst
        """
        tech_info = get_tech_global_info('bag3_digital')
        wn = tech_info['w_minn'] if is_ctrl else 2 * tech_info['w_minn']
        wp = tech_info['w_minp'] if is_ctrl else 2 * tech_info['w_minp']

        if has_rst:
            seg_dict = dict(pd=seg_n, pu=seg_p, rst=int(np.ceil(seg_n / 2)), prst=seg_p)
            w_dict = dict(pd=wn, pu=wp, rst=wn)
        else:
            seg_dict = dict(pd=seg_n, pu=seg_p)
            w_dict = dict(pd=wn, pu=wp)

        lv_params = dict(
            cls_name=LevelShifter.get_qualified_name(),
            draw_taps=True,
            pwr_gnd_list=[('VDD_in', 'VSS'), ('VDD', 'VSS')],
            params=dict(
                pinfo=pinfo,
                lv_params=dict(
                    seg_dict=seg_dict,
                    w_dict=w_dict,
                    has_rst=has_rst,
                    in_upper=has_rst,
                    dual_output=dual_output,
                ),
                in_buf_params=dict(segp_list=[seg_in_inv_p, seg_inv_p],
                                   segn_list=[seg_in_inv_n, seg_inv_n],
                                   w_p=wp, w_n=wn),
                export_pins=True,
            )
        )

        # Note that setting stack_p = 2 actually changes the topology of the level shifter to include PMOS devices
        # tied to the input and in series with the cross-coupled PMOS pull-ups.
        if has_rst:
            lv_params['params']['lv_params']['stack_p'] = 2

        if skew_out:
            lv_params['params']['lv_params']['buf_segn_list'] = [out_inv_m]
            lv_params['params']['lv_params']['buf_segp_list'] = [out_inv_m + out_pseg_off]
        else:
            lv_params['params']['lv_params']['buf_seg_list'] = [out_inv_m]

        return lv_params

    @staticmethod
    def _get_full_tb_params() -> Dict[str, Any]:
        return dict(
            load_list=[('out', 'cload'), ('outb', 'cload')],
            vbias_list=[('VDD', 'vdd'), ('VDD_in', 'vdd_in'), ('rst', 'vrst'), ('rstb', 'vrst_b')],
            dut_conns={'in': 'in', 'rst_out': 'rst', 'rst_outb': 'rst', 'VDD': 'VDD',
                       'VDD_in': 'VDD_in', 'VSS': 'VSS', 'rst_casc': 'rstb', 'out': 'out',
                       'outb': 'outb', 'inb_buf': 'inb_buf', 'in_buf': 'in_buf',
                       'midn': 'midn', 'midp': 'midp'},

        )

    @staticmethod
    def _get_rst_tb_params() -> Dict[str, Any]:
        return dict(
            load_list=[('out', 'cload'), ('outb', 'cload')],
            vbias_list=[('VDD', 'vdd'), ('VDD_in', 'vdd_in')],
            dut_conns={'in': 'VDD', 'rst_out': 'inbar', 'rst_outb': 'in', 'VDD': 'VDD',
                       'VDD_in': 'VDD_in', 'VSS': 'VSS', 'rst_casc': 'VSS', 'out': 'out',
                       'outb': 'outb', 'inb_buf': 'inb_buf', 'in_buf': 'in_buf',
                       'midn': 'midn', 'midp': 'midp'},

        )

    @staticmethod
    def _get_core_tb_params() -> Dict[str, Any]:
        return dict(
            load_list=[('out', 'cload'), ('outb', 'cload')],
            vbias_list=[('VDD', 'vdd'), ('VDD_in', 'vdd_in'), ('rst', 'vrst'), ('rstb', 'vrst_b')],
            dut_conns={'inp': 'in', 'inn': 'VDD_in', 'rst_outp': 'rst', 'rst_outn': 'rst',
                       'VDD': 'VDD', 'VSS': 'VSS', 'rst_casc': 'rstb', 'outp': 'out',
                       'outn': 'outb'}
        )

    @staticmethod
    def _get_tbm_params(sim_envs: Sequence[str], vdd_in: float, vdd_out: float, trf: float,
                        cload: float, tbit: float) -> Dict[str, Any]:
        return dict(
            sim_envs=sim_envs,
            thres_lo=0.1,
            thres_hi=0.9,
            stimuli_pwr='vdd_in',
            tstep=None,
            gen_invert=True,
            sim_params=dict(
                vdd=vdd_out,
                vdd_in=vdd_in,
                vrst=0.0,
                vrst_b=vdd_out,
                cload=cload,
                tbit=tbit,
                trf=trf,
            ),
            rtol=1e-8,
            atol=1e-22,
        )

    async def get_cap(self, dut: DesignInstance, pin_name: str, inv_input_cap: float) -> float:
        cin_specs = self._cin_specs

        params = dut.lay_master.params['params']
        if pin_name == 'in':
            in_buf_params = params['in_buf_params']
            w_n = in_buf_params['w_n']
            w_p = in_buf_params['w_p']
            seg_p = in_buf_params['segp_list'][0]
            seg_n = in_buf_params['segn_list'][0]
        elif pin_name == 'rst_out':
            seg_dict = params['lv_params']['seg_dict']
            w_dict = params['lv_params']['w_dict']
            seg_p = 0
            seg_n = seg_dict['rst']
            w_p = 0
            w_n = w_dict['rst']
        else:
            seg_dict = params['lv_params']['seg_dict']
            w_dict = params['lv_params']['w_dict']
            seg_p = 0
            seg_n = seg_dict['pd']
            w_p = 0
            w_n = w_dict['pd']

        cin_guess = inv_input_cap * (seg_p * w_p + seg_n * w_n) / 8
        cin_specs['in_pin'] = pin_name
        cin_specs['buf_config']['cin_guess'] = cin_guess

        mm = self.make_mm(CapDelayMatch, cin_specs)
        data = (await self.async_simulate_mm_obj(f'c_{pin_name}_{dut.cache_name}', dut, mm)).data
        cap_fall = data['cap_fall']
        cap_rise = data['cap_rise']
        cap_avg = (cap_fall + cap_rise) / 2
        self.log(f'{pin_name} cap_fall={cap_fall:.4g}, cap_rise={cap_rise:.4g}, '
                 f'cap_avg={cap_avg:.4g}')
        return cap_avg

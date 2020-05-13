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

from typing import Mapping, Dict, Any, Optional, cast, Type

import numpy as np

from bag.layout.template import TemplateBase
from bag.simulation.design import DesignerBase
from bag.env import get_tech_global_info

from xbase.layout.mos.placement.data import TileInfoTable

from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB

from bag3_digital.layout.stdcells.util import STDCellWrapper

from .unit_cell import DriverUnitCellDesigner, DriverPullUpDownDesigner
from ..layout.driver import AIBOutputDriver


class OutputDriverDesigner(DesignerBase):
    """ Design the output Driver """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)
        self.pinfo = None
        self._tb_params = dict(
            load_list=[('out', 'cload')],
            dut_conns={'out': 'out', 'in': 'in', 'nand_pu': 'nand_pu', 'nor_pd': 'nor_pd',
                       'enb': 'VSS', 'en': 'VDD', 'VDD': 'VDD', 'VSS': 'VSS'},
        )

    @classmethod
    def get_dut_lay_class(cls) -> Optional[Type[TemplateBase]]:
        return AIBOutputDriver

    # TODO: refactor to use helper functions
    # TODO: deprecate num_units_nom
    async def async_design(self, num_units: int, num_units_nom: int, num_units_min: int,
                           trf_max: float, r_targ: float, r_min_weak: float, c_ext: float,
                           freq: float, trf_in: float, rel_err: float, del_err: float,
                           tile_name: str, tile_specs: Mapping[str, Any],
                           res_mm_specs: Dict[str, Any], ridx_p: int = 1, ridx_n: int = 1,
                           stack_max: int = 10, tran_options: Optional[Mapping[str, Any]] = None,
                           max_iter: int = 10,
                           **kwargs: Any) -> Mapping[str, Any]:
        """ Design the Output Driver Cell

        1) Calls functions to get initial design for main driver unit cell and weak driver
        2) Characterize design
        3) Iterate on main driver design to account for additional loading from other unit cells

        Parameters
        ----------
        num_units: int
            Number of unit cells. Must be between 3 and 6
        num_units_nom: int
            ???
        num_units_min: int
            Min. number of unit cells
        trf_max: float
            Max. output rise / fall time
        r_targ: float
            Target output resistance
        r_min_weak: float
            Target output resistance for weak driver
        c_ext: float
            Load capacitance
        freq: float
            Operating switching frequency
        trf_in: float
            Input rise / fall time
        rel_err: float
            Output resistance error tolerance, used in DriverPullUpDownDesigner
        del_err: float
            Delay mismatch tolerance, used in DriverUnitCellDesigner for sizing NAND + NOR
        tile_name: str
            Tile name for layout.
        tile_specs: Mapping[str, Any]
            Tile specifications for layout.
        res_mm_specs: Mapping[str, Any]
            MeasurementManager specs for DriverPullUpDownMM, used in DriverPullUpDownDesigner
        ridx_n: int
            NMOS transistor row
        ridx_p: int
            PMOS transistor row
        stack_max: int
            Max. number of stacks possible in pull up / pull down driver
        tran_options: Optional[Mapping[str, Any]]
            Additional transient simulation options dictionary, used in DriverPullUpDownDesigner
        max_iter: int
            Max. number of iterations for final main driver resizing
        kwargs: Any
            Additional keyword arguments. Unused here

        Returns
        -------
        ans: Mapping[str, Any]
            Design summary, including generator parameters and performance summary

        """
        tech_info = get_tech_global_info('aib_ams')
        w_p = tech_info['w_maxp']
        w_n = tech_info['w_maxn']
        tile_specs['place_info'][tile_name]['row_specs'][0]['width'] = w_n
        tile_specs['place_info'][tile_name]['row_specs'][1]['width'] = w_p
        self._dsn_specs['tile_specs']['place_info'][tile_name]['row_specs'][0]['width'] = w_n
        self._dsn_specs['tile_specs']['place_info'][tile_name]['row_specs'][1]['width'] = w_p
        core_params = dict(r_targ=r_targ * num_units,
                           stack_max=stack_max,
                           trf_max=trf_max,
                           c_min=c_ext / num_units,
                           c_max=c_ext / num_units_min,
                           w_p=w_p, w_n=w_n,
                           freq=freq,
                           vdd=tech_info['dsn_envs']['slow_io']['vddio'],
                           vdd_max=tech_info['signoff_envs']['vmax']['vddio'],
                           trf_in=trf_in,
                           rel_err=rel_err, del_err=del_err,
                           tile_specs=tile_specs, tile_name=tile_name,
                           tran_options=tran_options,
                           res_mm_specs=res_mm_specs,
                           )
        weak_params = core_params.copy()
        weak_params['w_p'] = tech_info['w_minp']
        weak_params['w_n'] = tech_info['w_minn']
        weak_params['r_targ'] = r_min_weak
        weak_params['is_weak'] = True

        # Design weak pull-up/pull-down
        weak_designer = DriverPullUpDownDesigner(self._root_dir, self._sim_db, self._dsn_specs)
        weak_designer.set_dsn_specs(weak_params)
        weak_results = await weak_designer.async_design(**weak_params)

        # Design main output driver
        core_designer = DriverUnitCellDesigner(self._root_dir, self._sim_db, self._dsn_specs)
        core_designer.set_dsn_specs(core_params)
        summary = await core_designer.async_design(**core_params)

        # Characterize the initial design
        result_params = summary['dut_params']['params']
        tinfo_table = TileInfoTable.make_tiles(self.grid, tile_specs)
        pinfo = tinfo_table[tile_name]
        dut_params = dict(
            cls_name='aib_ams.layout.driver.AIBOutputDriver',
            draw_taps=True,
            params=dict(
                pinfo=pinfo,
                pupd_params=dict(
                    seg_p=weak_results['seg_p'],
                    seg_n=weak_results['seg_n'],
                    stack=weak_results['stack'],
                    w_p=weak_results['w_p'],
                    w_n=weak_results['w_n'],
                ),
                unit_params=dict(
                    seg_p=result_params['seg_p'],
                    seg_n=result_params['seg_n'],
                    seg_nand=result_params['seg_nand'],
                    seg_nor=result_params['seg_nor'],
                    w_p=result_params['w_p'],
                    w_n=result_params['w_n'],
                    w_p_nand=result_params['w_p_nand'],
                    w_n_nand=result_params['w_n_nand'],
                    w_p_nor=result_params['w_p_nor'],
                    w_n_nor=result_params['w_n_nor']
                ),
                export_pins=True,
            )
        )
        tbm_specs: Dict[str, Any] = dict(
            thres_lo=0.1,
            thres_hi=0.9,
            tstep=None,
            sim_params=dict(
                cload=c_ext,
                tbit=20*r_targ*c_ext,
                trf=trf_in,
            ),
            rtol=1e-8,
            atol=1e-22,
            tran_options=tran_options,
            save_outputs=['out', 'in']
        )

        if num_units == 3:
            n_enb_str = 'VDD'
            p_en_str = 'VSS'
        elif num_units == 4:
            n_enb_str = 'VDD,VSS'
            p_en_str = 'VSS,VDD'
        elif num_units == 5:
            n_enb_str = 'VSS,VDD'
            p_en_str = 'VDD,VSS'
        elif num_units == 6:
            n_enb_str = 'VSS,VSS'
            p_en_str = 'VDD,VDD'
        else:
            raise ValueError('Number of units must be between 3 and 6.')

        tb_params = dict(
            load_list=[('out', 'cload')],
            dut_conns={'txpadout': 'out', 'din': 'in',
                       'n_enb_drv<1:0>': n_enb_str, 'p_en_drv<1:0>': p_en_str,
                       'tristate': 'VSS', 'tristateb': 'VDD',
                       'weak_pden': 'VSS', 'weak_puenb': 'VDD'}
        )

        all_corners = tech_info['signoff_envs']['all_corners']
        trf_worst = -float('inf')
        tdr_worst = -float('inf')
        tdf_worst = -float('inf')
        tf_worst = -float('inf')
        tr_worst = -float('inf')
        worst_env = ''
        duty_err_worst = 0
        dut = await self.async_new_dut('output_driver', STDCellWrapper, dut_params)
        for env in all_corners['envs']:
            tbm_specs['sim_envs'] = [env]
            tbm_specs['sim_params']['vdd'] = all_corners['vddio'][env]
            tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'sim_final_{env}', dut, tbm, tb_params)
            tdr, tdf = CombLogicTimingTB.get_output_delay(sim_results.data, tbm_specs, 'in', 'out',
                                                          False)

            if tdr[0] > tdr_worst:
                tdr_worst = tdr[0]
            if tdf[0] > tdf_worst:
                tdf_worst = tdf[0]

            duty_err = tdr[0] - tdf[0]
            if np.abs(duty_err) > np.abs(duty_err_worst):
                duty_err_worst = duty_err

            tr, tf = CombLogicTimingTB.get_output_trf(sim_results.data, tbm_specs, 'out')
            trf = max(np.max(tr), np.max(tf))

            if trf > trf_worst:
                trf_worst = trf
                tr_worst = tr[0]
                tf_worst = tf[0]
                worst_env = env

        self.logger.info('---')
        self.logger.info("Completed initial design.")
        self.logger.info(f'Target R per segment was: {core_params["r_targ"]}')
        self.logger.info(f'Worst trf was: {trf_worst} / rise: {tr_worst}, fall: {tf_worst}' +
                         f'in corner {worst_env}')
        self.logger.info(f'Worst delay was: {max(tdr_worst, tdf_worst)}')
        self.logger.info(f'Worst duty cycle error was: {duty_err_worst}')
        self.logger.info('---')

        # Loop on trf to take into account loading from unit cells that are nominally off
        # but not included in single unit-cell design script.
        for i in range(1, max_iter + 1):
            scale_error = trf_worst / trf_max

            if scale_error < 1:
                break

            core_params['r_targ'] = core_params['r_targ'] / scale_error * (1 - 1/(2*max_iter))
            self.logger.info(f'Scale error set to {scale_error}.')
            self.logger.info(f'New target R per segment set to {core_params["r_targ"]}')

            core_designer.set_dsn_specs(core_params)
            # TODO: Allow designer to start from previous design point and/or guess at scaled
            #       design (improve runtime)
            summary = await core_designer.async_design(**core_params)

            result_params = summary['dut_params']['params']

            dut_params['params']['unit_params'] = dict(
                seg_p=result_params['seg_p'],
                seg_n=result_params['seg_n'],
                seg_nand=result_params['seg_nand'],
                seg_nor=result_params['seg_nor'],
                w_p=result_params['w_p'],
                w_n=result_params['w_n'],
                w_p_nand=result_params['w_p_nand'],
                w_n_nand=result_params['w_n_nand'],
                w_p_nor=result_params['w_p_nor'],
                w_n_nor=result_params['w_n_nor'],
            )

            tbm_specs['sim_params']['tbit'] = 1/freq
            # tbm_specs['sim_params']['tbit'] = 20 * core_params['r_targ'] * c_ext

            trf_worst = -float('inf')
            tdr_worst = -float('inf')
            tdf_worst = -float('inf')
            duty_err_worst = 0
            worst_env = ''
            duty_env = ''
            dut = await self.async_new_dut('output_driver', STDCellWrapper, dut_params)
            for env in all_corners['envs']:
                tbm_specs['sim_envs'] = [env]
                tbm_specs['sim_params']['vdd'] = all_corners['vddio'][env]
                tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))

                sim_results = await self.async_simulate_tbm_obj(f'sim_final_{i}_{env}', dut,
                                                                tbm, tb_params)
                tdr, tdf = CombLogicTimingTB.get_output_delay(sim_results.data, tbm_specs,
                                                              'in', 'out',
                                                              False)

                if tdr[0] > tdr_worst:
                    tdr_worst = tdr[0]
                if tdf[0] > tdf_worst:
                    tdf_worst = tdf[0]

                duty_err = tdr[0] - tdf[0]
                if np.abs(duty_err) > np.abs(duty_err_worst):
                    duty_err_worst = duty_err
                    duty_env = env

                tr, tf = CombLogicTimingTB.get_output_trf(sim_results.data, tbm_specs, 'out')
                trf = max(np.max(tr), np.max(tf))

                if trf > trf_worst:
                    trf_worst = trf
                    tr_worst = tr[0]
                    tf_worst = tf[0]
                    worst_env = env

            self.logger.info('---')
            self.logger.info(f'Completed design iteration {i}.')
            self.logger.info(f'Worst trf was: {trf_worst} / rise: {tr_worst}, fall: {tf_worst}' +
                             f'in corner {worst_env}')
            self.logger.info(f'Worst delay was: {max(tdr_worst, tdf_worst)} in corner {worst_env}')
            self.logger.info(f'Worst duty cycle error was: {duty_err_worst} in corner {duty_env}')
            self.logger.info('')

        return dict(dut_params=dut_params, tdr=tdr_worst, tdf=tdf_worst, trf_worst=trf_worst,
                    duty_err=duty_err_worst)

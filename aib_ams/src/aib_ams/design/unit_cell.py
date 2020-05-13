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

"""This package contains designer classes for DriverUnitCellDesigner and DriverPullUpDownDesigner"""

from typing import Mapping, Dict, Any, Tuple, Optional, cast, Type
from pathlib import Path
import math
import numpy as np
from copy import deepcopy

from pybag.enum import DesignOutput

from bag.design.netlist import add_internal_sources
from bag.simulation.design import DesignerBase
from bag.simulation.cache import DesignInstance
from bag.layout.template import TemplateBase
from bag.util.search import BinaryIterator
from bag.env import get_tech_global_info
from bag.concurrent.util import GatherHelper

from xbase.layout.mos.placement.data import TileInfoTable

from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB

from bag3_digital.layout.stdcells.util import STDCellWrapper

from ..layout.driver import PullUpDown, OutputDriverCore
from ..measurement.driver_pu_pd import DriverPullUpDownMM


class DriverUnitCellDesigner(DesignerBase):
    """ Design the output driver unit cell"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)
        self.out_w_p = None
        self.out_w_n = None
        self.pinfo = None
        self._tb_params = dict(
            load_list=[('out', 'cload')],
            dut_conns={'out': 'out', 'in': 'in', 'nand_pu': 'nand_pu', 'nor_pd': 'nor_pd',
                       'enb': 'VSS', 'en': 'VDD', 'VDD': 'VDD', 'VSS': 'VSS'},
        )

    @classmethod
    def get_dut_lay_class(cls) -> Optional[Type[TemplateBase]]:
        return OutputDriverCore

    async def async_design(self,
                           c_max: float,
                           trf_in: float,
                           w_p: int,
                           w_n: int,
                           r_targ: float,
                           tile_specs: Mapping[str, Any],
                           del_err: float,
                           tile_name: str,
                           seg_even: Optional[bool] = False,
                           **kwargs: Any
                           ) -> Mapping[str, Any]:
        """This function designs the main output driver unit cell
        1) Calls DriverPullUpDownDesigner to design output pull up / pull down
        2) Design input NAND and NOR
        3) Characterize and sign off

        Parameters
        ----------
        c_max: float
            Target load capacitance
        trf_in: float
            Input rise / fall time in simulation
        w_p: int
            Initial output PMOS width
        w_n: int
            Initial output NMOS width
        r_targ: float:
            Target nominal (strong) output resistance.
        tile_name: str
            Tile name for layout.
        del_err: float
            Delay mismatch tolerance, for sizing NAND + NOR
        tile_specs: Mapping[str, Any]
            Tile specifications for layout.
        seg_even: Optional[bool]
            True to force segments to be even
        kwargs: Any
            Additional keyword arguments. Unused here

        Returns
        -------
        ans: Mapping[str, Any]
            Design summary, including performance specs and generator parameters
        """
        tinfo_table = TileInfoTable.make_tiles(self.grid, tile_specs)
        self.pinfo = tinfo_table[tile_name]

        self.out_w_p = w_p
        self.out_w_n = w_n

        self.dsn_specs['w_p'] = self.out_w_p
        self.dsn_specs['w_n'] = self.out_w_n

        driver_designer = DriverPullUpDownDesigner(self._root_dir, self._sim_db, self.dsn_specs)
        summary = await driver_designer.async_design(**driver_designer.dsn_specs)

        seg_p: int = summary['seg_p']
        seg_n: int = summary['seg_n']
        self.out_w_p = summary['w_p']
        self.out_w_n = summary['w_n']

        tbm_specs: dict = dict(
                         sim_envs=get_tech_global_info('aib_ams')['dsn_envs']['slow_io']['env'],
                         thres_lo=0.1,
                         thres_hi=0.9,
                         tstep=None,
                         sim_params=dict(
                             vdd=get_tech_global_info('aib_ams')['dsn_envs']['slow_io']['vddio'],
                             cload=c_max,
                             tbit=20 * r_targ * c_max,
                             trf=trf_in,
                         ),
                         rtol=1e-8,
                         atol=1e-22,
                         save_outputs=['out', 'nand_pu', 'nor_pd', 'in'])
        gate_sizes = await self._size_nand_nor(seg_p, seg_n, trf_in, tbm_specs, del_err, seg_even)
        nand_seg, nor_seg, nand_p_w_del, nor_n_w_del, w_nom = gate_sizes

        # Characterize Final design
        dut_params = self._get_unit_cell_params(self.pinfo, seg_p, seg_n, nand_seg, nor_seg,
                                                nand_p_w_del, nor_n_w_del, w_min=w_nom)
        dut = await self.async_new_dut('unit_cell', STDCellWrapper, dut_params)

        all_corners = get_tech_global_info('aib_ams')['signoff_envs']['all_corners']
        tdr_worst = -float('inf')
        tdf_worst = -float('inf')
        pu_tr_worst = -float('inf')
        pu_tf_worst = -float('inf')
        pd_tr_worst = -float('inf')
        pd_tf_worst = -float('inf')
        duty_err_worst = 0
        for env in all_corners['envs']:
            tbm_specs['sim_envs'] = [env]
            tbm_specs['sim_params']['vdd'] = all_corners['vddio'][env]
            tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'sim_final_{env}', dut, tbm,
                                                            self._tb_params)

            tdr, tdf = CombLogicTimingTB.get_output_delay(sim_results.data, tbm_specs, 'in', 'out',
                                                          False)
            pu_tr, pu_tf = CombLogicTimingTB.get_output_trf(sim_results.data, tbm_specs, 'nand_pu')
            pd_tr, pd_tf = CombLogicTimingTB.get_output_trf(sim_results.data, tbm_specs, 'nor_pd')

            tdr_worst = self.set_worst_spec(tdr, tdr_worst)
            tdf_worst = self.set_worst_spec(tdf, tdf_worst)
            pu_tr_worst = self.set_worst_spec(pu_tr, pu_tr_worst)
            pu_tf_worst = self.set_worst_spec(pu_tf, pu_tf_worst)
            pd_tr_worst = self.set_worst_spec(pd_tr, pd_tr_worst)
            pd_tf_worst = self.set_worst_spec(pd_tf, pd_tf_worst)

            duty_err = tdr - tdf
            if np.abs(duty_err) > np.abs(duty_err_worst):
                duty_err_worst = duty_err

        return dict(tdr=tdr_worst, tdf=tdf_worst, pu_tr=pu_tr_worst, pu_tf=pu_tf_worst,
                    pd_tr=pd_tr_worst, pd_tf=pd_tf_worst, duty_err=duty_err_worst,
                    dut_params=dut_params)

    @staticmethod
    def set_worst_spec(spec_in, spec_cur):
        if spec_in > spec_cur:
            spec_cur = spec_in

        return spec_cur

    async def _size_nand_nor(self, seg_p: int, seg_n: int, trf: float, tbm_specs: dict,
                             del_err: float,
                             seg_even: Optional[bool] = False,
                             nand_p_w_del: Optional[int] = None,
                             nor_n_w_del: Optional[int] = None,
                             w_nom: Optional[int] = -1) -> Tuple[int, int, int, int, int]:

        # Check for defaults and set them if needed.
        tech_globals = get_tech_global_info('aib_ams')
        if nand_p_w_del is None:
            nand_p_w_del = 0
        if nor_n_w_del is None:
            nor_n_w_del = 0

        if w_nom == -1:
            w_nom = max(tech_globals['w_nomp'], tech_globals['w_nomn'])

        # binary search: up size both nand and nor to meet the slope rate trf
        nand_seg = nor_seg = 1
        max_nand_seg = int(np.round(seg_p/tech_globals['min_fanout']))
        max_nor_seg = int(np.round(seg_n/tech_globals['min_fanout']))
        dut_params = self._get_unit_cell_params(self.pinfo, seg_p, seg_n, nand_seg, nor_seg,
                                                nand_p_w_del, nor_n_w_del, w_min=w_nom)

        tbm_specs['sim_envs'] = tech_globals['dsn_envs']['slow_io']['env']
        tbm_specs['sim_params']['vdd'] = tech_globals['dsn_envs']['slow_io']['vddio']
        tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
        nand_seg, _, tf = await self._upsize_gate_for_trf(dut_params, trf, nand_seg, True, tbm,
                                                          seg_even, max_nand_seg)
        nor_seg, tr, _ = await self._upsize_gate_for_trf(dut_params, trf, nor_seg, False, tbm,
                                                         seg_even, max_nor_seg)

        # Change tbm_spec to design for delay matching in tt corner
        dut_params_new = self._get_unit_cell_params(self.pinfo, seg_p, seg_n, nand_seg, nor_seg,
                                                    nand_p_w_del, nor_n_w_del, w_min=w_nom)
        tbm_specs['sim_envs'] = tech_globals['dsn_envs']['center']['env']
        tbm_specs['sim_params']['vdd'] = tech_globals['dsn_envs']['center']['vddio']
        tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
        dut = await self.async_new_dut('nand_nor_check_delay', STDCellWrapper,
                                       dut_params_new)
        sim_results = await self.async_simulate_tbm_obj('check_delay', dut, tbm,
                                                        self._tb_params)

        nand_tdf, nand_tdr = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs,
                                                                'in', 'nand_pu',
                                                                out_invert=True, out_pwr='vdd')
        nor_tdf, nor_tdr = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs,
                                                              'in', 'nor_pd',
                                                              out_invert=True, out_pwr='vdd')

        err_cur = np.abs(nand_tdf - nor_tdr)
        if np.max(err_cur) > del_err:
            # up_size the slower side to make it faster
            if np.max(nor_tdr) < np.max(nand_tdf):
                nand_seg, nand_tdr, nand_tdf = await self._upsize_gate_for_delay(dut_params,
                                                                                 np.max(nor_tdr),
                                                                                 nand_seg,
                                                                                 True, tbm,
                                                                                 seg_even,
                                                                                 max_nand_seg)
            else:
                nor_seg, nor_tdr, nor_tdf = await self._upsize_gate_for_delay(dut_params,
                                                                              np.max(nand_tdf),
                                                                              nor_seg,
                                                                              False, tbm,
                                                                              seg_even,
                                                                              max_nor_seg)

        # check if t_rise_nand < t_rise_nor and t_fall_nor < t_fall_nand
        rerun = False
        if not np.all(nand_tdr < nor_tdr):
            rerun = nand_p_w_del + 1 + w_nom <= tech_globals['w_maxp']
            new_nand_p_w_del = nand_p_w_del + 1 if rerun else nand_p_w_del
        else:
            new_nand_p_w_del = nand_p_w_del

        if not np.all(nor_tdf < nand_tdf):
            rerun = nor_n_w_del + 1 + w_nom <= tech_globals['w_maxn']
            new_nor_n_w_del = nor_n_w_del + 1 if rerun else nor_n_w_del
        else:
            new_nor_n_w_del = nor_n_w_del

        if rerun:
            await self._size_nand_nor(seg_p, seg_n, trf, tbm_specs, del_err, seg_even,
                                      new_nand_p_w_del, new_nor_n_w_del, w_nom)

        await self.run_nand_nor_signoff(dut, del_err, tbm_specs, tech_globals)

        return nand_seg, nor_seg, nand_p_w_del, nor_n_w_del, w_nom

    async def run_nand_nor_signoff(self, dut, del_err, tbm_specs, tech_globals):
        tbm_specs['env_params'] = dict(vdd=dict())
        tbm_specs['env_params']['vdd'] = tech_globals['signoff_envs']['all_corners']['vddio']
        nand_tdf = []
        nand_tdr = []
        nor_tdf = []
        nor_tdr = []
        for env in tech_globals['signoff_envs']['all_corners']['envs']:
            tbm_specs['sim_envs'] = [env]
            tbm_specs['sim_params']['vdd'] = tbm_specs['env_params']['vdd'][env]
            tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'signoff_delay_match_{env}', dut, tbm,
                                                            self._tb_params)
            pu_tdf, pu_tdr = CombLogicTimingTB.get_output_delay(sim_results.data,
                                                                tbm.specs,
                                                                'in', 'nand_pu',
                                                                out_invert=True,
                                                                out_pwr='vdd')
            pd_tdf, pd_tdr = CombLogicTimingTB.get_output_delay(sim_results.data,
                                                                tbm.specs,
                                                                'in', 'nor_pd',
                                                                out_invert=True,
                                                                out_pwr='vdd')
            nand_tdf.append(pu_tdf)
            nand_tdr.append(pu_tdr)
            nor_tdf.append(pd_tdf)
            nor_tdr.append(pd_tdr)

        err_cur = np.abs(np.subtract(nand_tdf, nor_tdr))
        msg = f'del_err: {np.max(err_cur)} [wanted: {del_err}]'
        if np.any(err_cur > del_err):
            self.error(f'Unable to match NAND/NOR gate delays to within target, {msg}')
        self.log(msg)

    async def _upsize_gate_for_delay(self, dut_params: Dict[str, Any], td_targ: float,
                                     seg_cur: int, is_nand: bool,
                                     tbm: CombLogicTimingTB,
                                     seg_even: bool,
                                     seg_max: Optional[int] = None,
                                     ) -> Tuple[int, np.ndarray, np.ndarray]:
        return await self._upsize_gate_for_del_spec(dut_params, td_targ, seg_cur, is_nand,
                                                    tbm, seg_even, 'delay', seg_max)

    async def _upsize_gate_for_trf(self, dut_params: Dict[str, Any], trf: float,
                                   seg_cur: int, is_nand: bool,
                                   tbm: CombLogicTimingTB,
                                   seg_even: bool,
                                   seg_max: Optional[int] = None,
                                   ) -> Tuple[int, np.ndarray, np.ndarray]:
        return await self._upsize_gate_for_del_spec(dut_params, trf, seg_cur, is_nand, tbm,
                                                    seg_even, 'slope', seg_max)

    async def _upsize_gate_for_del_spec(self, dut_params: Dict[str, Any], tspec: float,
                                        seg_cur: int, is_nand: bool,
                                        tbm: CombLogicTimingTB,
                                        seg_even: bool,
                                        spec_type: str,
                                        seg_max: Optional[int] = None,
                                        ) -> Tuple[int, np.ndarray, np.ndarray]:
        if spec_type != 'delay' and spec_type != 'slope':
            raise ValueError("spec_type must be either 'delay' or 'slope'.")

        bin_iter = BinaryIterator(seg_cur, seg_max, step=1 << seg_even)
        while bin_iter.has_next():
            new_seg = bin_iter.get_next()
            dut_params['params']['seg_nand' if is_nand else 'seg_nor'] = new_seg
            dut = await self.async_new_dut('nand_nor_upsize', STDCellWrapper, dut_params)
            sim_results = await self.async_simulate_tbm_obj('nand_nor_upsize_sim', dut, tbm,
                                                            self._tb_params)
            if spec_type == 'slope':
                ans = CombLogicTimingTB.get_output_trf(sim_results.data, tbm.specs,
                                                       'nand_pu' if is_nand else 'nor_pd')
                gate_tr, gate_tf = ans
            else:
                ans = CombLogicTimingTB.get_output_delay(sim_results.data, tbm.specs,
                                                         'in',
                                                         'nand_pu' if is_nand else 'nor_pd',
                                                         out_invert=True)
                gate_tf, gate_tr = ans

            trf_metric = gate_tf if is_nand else gate_tr
            if np.max(trf_metric) > tspec:
                bin_iter.up(np.max(trf_metric)-tspec)
            else:
                bin_iter.down(np.max(trf_metric)-tspec)
                bin_iter.save_info((new_seg, gate_tr, gate_tf))

        info = bin_iter.get_last_save_info()
        if info is None:
            gate_str = "nand" if is_nand else "nor"
            err_str = f'Could not find a size for {gate_str} to meet the target spec of {tspec}.'
            self.error(err_str)
        seg, tr, tf = info

        return seg, tr, tf

    def _get_unit_cell_params(self, pinfo: Any, seg_p: int, seg_n: int,
                              seg_nand: int, seg_nor: int,
                              nand_p_w_del: int, nor_n_w_del: int,
                              beta_ratio: Optional[float] = None,
                              w_min: Optional[int] = None) -> Dict[str, Any]:
        if w_min is None:
            w_n = get_tech_global_info('aib_ams')['w_minn']
        else:
            w_n = w_min

        if beta_ratio is None:
            beta_ratio = get_tech_global_info('aib_ams')['inv_beta']

        return dict(
            cls_name='aib_ams.layout.driver.OutputDriverCore',
            draw_taps=True,
            params=dict(
                pinfo=pinfo,
                seg_p=seg_p,
                seg_n=seg_n,
                seg_nand=seg_nand,
                seg_nor=seg_nor,
                w_p=self.out_w_p,
                w_n=self.out_w_n,
                w_p_nand=int(beta_ratio * w_n) + nand_p_w_del,
                w_n_nand=w_n,
                w_p_nor=int(beta_ratio * w_n),
                w_n_nor=w_n + nor_n_w_del,
                export_pins=True,
            )
        )


class DriverPullUpDownDesigner(DesignerBase):
    """Design the output driver pull up/ pull down

    NOTE: Assumes output cap is much larger than device.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)
        self._tb_params = dict(
            load_list=[('out', 'cload')],
            dut_conns=dict(
                out='out',
                pden='in',
                puenb='in',
                VDD='VDD',
                VSS='VSS',
            ),
        )

    @classmethod
    def get_dut_lay_class(cls) -> Optional[Type[TemplateBase]]:
        return PullUpDown

    async def async_design(self, r_targ: float, c_max: float, freq: float, trf_in: float,
                           rel_err: float, tile_specs: Mapping[str, Any], tile_name: str,
                           w_p: int, w_n: int, res_mm_specs: Dict[str, Any],
                           is_weak: Optional[bool] = False, stack_max: Optional[int] = 10,
                           seg_even: bool = True, em_options: Optional[Mapping[str, Any]] = None,
                           ridx_p: int = -1, ridx_n: int = 0,
                           max_iter: int = 10, **kwargs: Any) -> Mapping[str, Any]:
        """Design a driver unit cell.
        will try to achieve the given maximum trf and meet EM specs at c_max.
        The main driver increases the segments and width to achieve small output resistance
        The weak driver uses stacking to achieve large output resistance.

        1) Determines minimum transistor segments for output current for given load cap and
        frequency
        2) For weak driver, determines transistor stacking to meet output resistance spec
        3) For main driver, determines transistor segments and width to meet output resistance spec

        Parameters
        ----------
        r_targ: float
            Target unit cell output resistance
        c_max: float
            Load capacitance
        freq: float
            Operating switching frequency
        trf_in: float
            Input rise / fall time
        rel_err: float
            Output resistance error tolerance
        tile_name: str
            Tile name for layout.
        tile_specs: Mapping[str, Any]
            Tile specifications for layout.
        w_p: int
            Initial output PMOS width
        w_n: int
            Initial output NMOS width
        res_mm_specs: Dict[str, Any]
            Specs for DriverPullUpDownMM
        is_weak: Optional[bool]
            True if designing weak driver
        stack_max: int
            Maximum allowed transistor stack size
        seg_even: bool
            True to force number of segments to be even
        em_options: Optional[Mapping[str, Any]]
            Additional arguments for calculating minimum segments based on EM specs
        ridx_n: int
            NMOS transistor row
        ridx_p: int
            PMOS transistor row
        max_iter: int
            Maximum allowed number of iteration
        kwargs: Any
            Additional keyword arguments. Unused here

        Returns
        -------
        ans: Mapping[str, Any]
            Design summary, including generator parameters and performance summary
        """
        if em_options is None:
            em_options = {}

        tinfo_table = TileInfoTable.make_tiles(self.grid, tile_specs)
        arr_info = tinfo_table.arr_info
        pinfo = tinfo_table[tile_name]
        mos_tech = arr_info.tech_cls

        # get transistor parameters
        vdd_max = get_tech_global_info('aib_ams')['signoff_envs']['vmax']['vddio']
        iac_peak = c_max * vdd_max * freq
        iac_rms = iac_peak / math.sqrt(2)
        seg_min = mos_tech.get_segments_from_em(arr_info.conn_layer, 0, iac_rms, iac_peak,
                                                even=True, **em_options) if not is_weak else 1
        self.log(f'seg_min = {seg_min}')

        dut_params = dict(
            pinfo=pinfo,
            seg_p=seg_min,
            seg_n=seg_min,
            w_p=w_p,
            w_n=w_n,
            stack=1,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            export_pins=True,
        )

        # get number of stacks
        # Modified to set effectively skip stacking for main driver in order to improve run time
        r_pu, r_pd = await self._get_stack(dut_params, res_mm_specs, r_targ,
                                           stack_max if is_weak else None)
        self.log(f'stack = {dut_params["stack"]}')

        if is_weak:
            return dict(
                stack=dut_params['stack'],
                r_targ=r_targ,
                r_pu=r_pu,
                r_pd=r_pd,
                seg_p=seg_min,
                seg_n=seg_min,
                w_p=w_p,
                w_n=w_n,
            )

        # get widths/number of segments
        ans = await self._resize_transistors(dut_params, res_mm_specs, r_targ, r_pu, r_pd,
                                             seg_min, seg_even, rel_err, max_iter=6)

        return ans

    # TODO: deprecate is_weak
    # TODO: Code seems to confuse max_iter with seg_max
    async def _resize_transistors(self,
                                  dut_params: Dict[str, Any],
                                  mm_specs: Dict[str, Any],
                                  r_targ: float,
                                  r_pu: np.ndarray,
                                  r_pd: np.ndarray,
                                  seg_min: int,
                                  seg_even: bool,
                                  rel_err: float,
                                  max_iter: int = 4,
                                  is_weak: Optional[bool] = False,
                                  seg_p: int = 0,
                                  seg_n: int = 0,
                                  seg_max: int = 20) -> Dict[str, Any]:
        """Iteratively searches transistor segments and width to hit target r_targ output
        resistance. If no result is found up to seg_max, width is adjusted, and then this
        function is recursively called.

        Parameters
        ----------
        dut_params: Mapping[str, Any]
            Driver generator parameters
        mm_specs: Dict[str, Any]
            Specs for DriverPullUpDownMM
        r_targ: float
            Target output resistance
        r_pu: np.ndarray
            Measured pull-up output resistance across given corners, from previous search
        r_pd: np.ndarray
            Measured pull-down output resistance across given corners, from previous search
        seg_min: int
            Min. allowed segments
        seg_even: bool
            True to force number of segments to be even
        rel_err: float
            Output resistance error tolerance
        max_iter: int
            Maximum allowed number of iteration
        is_weak: Optional[bool]
            Deprecated: True if sizing weak driver
        seg_p: int
            If given, used as initial number of PMOS segments instead of seg_min
        seg_n: int
            If given, used as initial number of NMOS segments instead of seg_min
        seg_max: int
            Max. allowed segments

        Returns
        -------
        ans: Mapping[str, Any]
            Design summary, including generator parameters and performance summary
        """
        seg_p = seg_min if seg_p == 0 else seg_p
        seg_n = seg_min if seg_n == 0 else seg_n
        seg_p_best = seg_p
        seg_n_best = seg_n
        err_best_p = float('inf')
        err_best_n = float('inf')
        visited = set()
        cnt = 0
        r_pu_best = float('inf')
        r_pd_best = float('inf')

        for cnt in range(seg_max):
            if np.max(r_pu) > (1+rel_err)*r_targ:
                seg_p += 1
            elif np.max(r_pu) < (1-rel_err)*r_targ and seg_p >= 2:
                seg_p -= 1

            if np.max(r_pd) > (1+rel_err) * r_targ:
                seg_n += 1
            elif np.max(r_pd) < (1-rel_err) * r_targ and seg_n >= 2:
                seg_n -= 1

            new_tuple = (seg_p, seg_n)
            if new_tuple in visited:
                break
            else:
                visited.add(new_tuple)

            dut_params['seg_p'] = seg_p
            dut_params['seg_n'] = seg_n
            r_pu, r_pd = await self._get_resistance(f'resize_{cnt}', dut_params, mm_specs)

            err_p = np.abs(np.max(r_pu) - r_targ) / r_targ
            err_n = np.abs(np.max(r_pd) - r_targ) / r_targ

            self.log(f'Iter = {cnt}, err_p={err_p:.4g}, err_n={err_n:.4g}.')
            self.log(f'Iter = {cnt}, seg_p={seg_p:.4g}, seg_n={seg_n:.4g}.')

            if err_p < err_best_p:
                err_best_p = err_p
                seg_p_best = seg_p
                r_pu_best = np.max(r_pu)

            if err_n < err_best_n:
                err_best_n = err_n
                seg_n_best = seg_n
                r_pd_best = np.max(r_pd)

            if err_p <= rel_err and err_n <= rel_err:
                break

        if cnt == max_iter-1 or err_best_n > rel_err or err_best_p > rel_err:
            tech_globals = get_tech_global_info('aib_ams')
            dut_params['seg_p'] = seg_p_best
            dut_params['seg_n'] = seg_n_best
            if err_best_p > rel_err:
                dut_params['w_p'] = dut_params['w_p'] - 1
            if err_best_n > rel_err:
                dut_params['w_n'] = dut_params['w_n'] - 1
            if dut_params['w_p'] >= tech_globals['w_minp'] and \
               dut_params['w_n'] >= tech_globals['w_minn']:
                msg = f'Recursing through resize transistors with w_p: {dut_params["w_p"]} and ' \
                      f'w_n: {dut_params["w_n"]}'
                self.log(msg)
                r_pu, r_pd = await self._get_resistance(f'get_res_for_new_w', dut_params, mm_specs)
                return await self._resize_transistors(dut_params, mm_specs, r_targ, r_pu, r_pd,
                                                      seg_min, seg_even, rel_err, max_iter, is_weak,
                                                      seg_p_best, seg_n_best)
            else:
                msg = f'Unable to find sizing for output driver ' + \
                      f'that meets {rel_err*100:.2f}% error.'
                raise RuntimeError(msg)

        return dict(
            stack=dut_params['stack'],
            r_targ=r_targ,
            r_pu=r_pu_best,
            r_pd=r_pd_best,
            w_p=dut_params['w_p'],
            w_n=dut_params['w_n'],
            seg_p=seg_p_best,
            seg_n=seg_n_best,
            err_p=err_best_p,
            err_n=err_best_n,
        )

    async def _get_stack(self,
                         dut_params: Dict[str, Any],
                         mm_specs: Dict[str, Any],
                         r_targ: float,
                         stack_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Binary searches the stack size to hit target r_targ output resistance.
        If stack_max is None, we skip sizing. This is set when sizing the main driver.

        NOTE: this function modifies dut_params and tbm_specs.

        Parameters
        ----------
        dut_params: Dict[str, Any]
            Driver generator parameters
        mm_specs: Dict[str, Any]
            Specs for DriverPullUpDownMM
        r_targ:
            Target output resistance
        stack_max:
            Maximum allowed transistor stack size

        Returns
        -------
        r_pu, r_pd: Tuple[np.ndarray, np.ndarray]
            Measured pull-up / pull-down output resistance across given corners
        """

        if not stack_max:
            dut_params['stack'] = 1
            sim_id = f'stack_1'
            r_pu, r_pd = await self._get_resistance(sim_id, dut_params, mm_specs)
            return r_pu, r_pd

        r_best = 0.0
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_stack = bin_iter.get_next()
            while bin_iter.has_next() and cur_stack > stack_max:
                bin_iter.down(float('inf'))
                cur_stack = bin_iter.get_next()
            if cur_stack > stack_max:
                break
            dut_params['stack'] = cur_stack
            sim_id = f'stack_{cur_stack}'
            r_pu, r_pd = await self._get_resistance(sim_id, dut_params, mm_specs)

            r_test = min(np.min(r_pu), np.min(r_pd))
            r_best = max(r_test, r_best)
            if r_targ > min(np.min(r_pu), np.min(r_pd)):
                bin_iter.up(r_targ - min(np.min(r_pu), np.min(r_pd)))
            else:
                bin_iter.save_info((cur_stack, r_pu, r_pd))
                bin_iter.down(r_targ - min(np.min(r_pu), np.min(r_pd)))

        save_info = bin_iter.get_last_save_info()
        if save_info is None:
            self.error(f'Cannot meet spec with stack_max = {stack_max}, '
                       f'r_best = {r_best:.4g}')
        stack, r_pu, r_pd = bin_iter.get_last_save_info()

        dut_params['stack'] = stack
        return r_pu, r_pd

    async def _get_resistance(self, sim_id: str, dut_params: Mapping[str, Any],
                              mm_specs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute resistance from DC I-V measurements.
        """
        gen_params = dict(
            cls_name=PullUpDown.get_qualified_name(),
            draw_taps=True,
            params=dut_params,
        )
        dut = await self.async_new_dut('unit_cell', STDCellWrapper, gen_params, flat=True)

        extract = self._sim_db.extract
        netlist_type = DesignOutput.CDL if extract else self._sim_db.prj.sim_access.netlist_type

        # Create a new netlist that allows for the insertion of dc sources
        offset_netlist = Path(*dut.netlist_path.parts[:-1],
                              f'netlist_with_sources.{netlist_type.extension}')
        v_offset_map = add_internal_sources(dut.netlist_path, offset_netlist, netlist_type, ['d'])

        mm_specs['v_offset_map'] = v_offset_map
        mm_specs['extract'] = extract

        # Create a DesignInstance with the newly created netlist
        new_dut = DesignInstance(dut.cell_name, dut.sch_master, dut.lay_master,
                                 offset_netlist, dut.cv_info_list)

        all_corners = get_tech_global_info('aib_ams')['signoff_envs']['all_corners']
        res_pd = np.array([])
        res_pu = np.array([])

        helper = GatherHelper()
        for env in all_corners['envs']:
            helper.append(self._run_get_resistance(env, sim_id, new_dut, mm_specs, all_corners))

        results = await helper.gather_err()
        for idx in range(len(results)):
            res_pu = np.append(res_pu, results[idx][0])
            res_pd = np.append(res_pd, results[idx][1])

        return res_pu, res_pd

    async def _run_get_resistance(self, env: str, sim_id: str, dut, mm_specs: Dict[str, Any],
                                  all_corners: Dict[str, Any]) -> Tuple[float, float]:
        new_specs = deepcopy(mm_specs)
        new_specs.update(dict(
            tbm_specs=dict(
                sim_envs=[env],
            ),
            vdd=all_corners['vddio'][env]
        ))
        mm = cast(DriverPullUpDownMM, self.make_mm(DriverPullUpDownMM, new_specs))

        sim_results = await self.async_simulate_mm_obj(f'{sim_id}_{env}', dut, mm)
        sim_data = sim_results.data

        return sim_data['res_pu'][0], sim_data['res_pd'][0]

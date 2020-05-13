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

from typing import Any, Mapping, Optional, Tuple, Dict, Sequence, Callable

import math
import pprint
from dataclasses import dataclass

import numpy as np

from bag.concurrent.util import GatherHelper

from xbase.layout.mos.placement.data import MOSBasePlaceInfo

from bag3_digital.measurement.cap.delay_match import CapDelayMatch
from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM
from bag3_testbenches.measurement.digital.delay import RCDelayCharMM

from ...layout.stdcells.gates import InvCore, PassGateCore
from ...layout.stdcells.se_to_diff import SingleToDiff
from ...measurement.stdcells.passgate.delay import PassGateRCDelayCharMM
from ..base import DigitalDesigner, BinSearchSegWidth


class RCData:
    def __init__(self, rc_inv: Mapping[str, Any], rc_pg: Mapping[str, Any]
                 ) -> None:
        self._rc_inv = rc_inv
        self._rc_pg = rc_pg

    def get_inv_rc(self, out_rise: Optional[bool], size: float = 1
                   ) -> Tuple[float, float, float, float]:
        r_in = self._rc_inv['r_in']
        c_in = self._rc_inv['c_in']
        r_out = self._rc_inv['r_out']
        c_out = self._rc_inv['c_out']
        if out_rise is None:
            return ((r_in[0] + r_in[1]) / 2 / size,
                    (c_in[0] + c_in[1]) / 2 * size,
                    (r_out[0] + r_out[1]) / 2 / size,
                    (c_out[0] + c_out[1]) / 2 * size,)

        out_riseb = not out_rise
        return (r_in[out_riseb] / size, c_in[out_riseb] * size,
                r_out[out_rise] / size, c_out[out_rise] * size,)

    def get_pg_rc(self, out_rise: Optional[bool], size: float = 1
                  ) -> Tuple[float, float, float]:
        r_p = self._rc_pg['r_p']
        c_s = self._rc_pg['c_s']
        c_d = self._rc_pg['c_d']
        if out_rise is None:
            return ((r_p[0] + r_p[1]) / 2 / size, (c_s[0] + c_s[1]) / 2 * size,
                    (c_d[0] + c_d[1]) / 2 * size,)

        return r_p[out_rise] / size, c_s[out_rise] * size, c_d[out_rise] * size


@dataclass(frozen=True)
class DelayData:
    outp: Tuple[float, float]
    outn: Tuple[float, float]
    inv2: Tuple[float, float]
    inv4: Tuple[float, float]

    def get_delay_err(self, inv2_inv4: bool, out_rise: bool) -> float:
        if inv2_inv4:
            return _get_delay_err(self.inv2[out_rise], self.inv4[out_rise])
        else:
            return _get_delay_err(self.outp[out_rise], self.outn[not out_rise])

    def change_pos(self, inv2_inv4: bool, out_rise: bool) -> bool:
        return not inv2_inv4 or self.inv4[out_rise] > self.inv2[out_rise]

    def get_diff_up(self, inv2_inv4: bool, out_rise: bool, diff_inc: bool) -> Tuple[float, bool]:
        if inv2_inv4:
            diff = self.inv4[out_rise] - self.inv2[out_rise]
        else:
            diff = self.outp[out_rise] - self.outn[not out_rise]
        return diff, (diff >= 0) ^ diff_inc


class InvSizeSearch(BinSearchSegWidth):
    def __init__(self, dsn: SingleToDiffDesigner, dut_params: Dict[str, Any],
                 out_rise: bool, inv2_inv4: bool, diff_inc: bool,
                 size_fun: Callable[[Dict[str, Any], int, int, bool], None],
                 w_list: Sequence[int], err_targ: float, search_step: int = 1) -> None:
        super().__init__(w_list, err_targ, search_step=search_step)

        self._dsn = dsn
        self._params = dut_params
        self._out_rise = out_rise
        self._inv2_inv4 = inv2_inv4
        self._diff_inc = diff_inc
        self._size_fun = size_fun

    def get_bin_search_info(self, data: DelayData) -> Tuple[float, bool]:
        return data.get_diff_up(self._inv2_inv4, self._out_rise, self._diff_inc)

    def get_error(self, data: DelayData) -> float:
        return data.get_delay_err(self._inv2_inv4, self._out_rise)

    def set_size(self, seg: int, w: int) -> None:
        self._size_fun(self._params, seg, w, self._out_rise)

    async def get_data(self, seg: int, w: int) -> DelayData:
        self.set_size(seg, w)
        return await self._dsn.get_delays(self._params)


class SingleToDiffDesigner(DigitalDesigner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pinfo: Optional[MOSBasePlaceInfo] = None

        self._w_arr: np.ndarray = np.array([])
        self._beta: Tuple[float, float] = (1, 1)
        self._rc_inv_specs: Dict[str, Any] = {}
        self._rc_pg_specs: Dict[str, Any] = {}
        self._td_specs: Dict[str, Any] = {}
        self._cin_specs: Dict[str, Any] = {}

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.dsn_specs
        tile_name: str = specs['tile_name']
        ridx_n: int = specs['ridx_n']
        w_min: int = specs['w_min']
        w_res: int = specs['w_res']
        inv_char_params: Mapping[str, Any] = specs['inv_char_params']
        pg_char_params: Mapping[str, Any] = specs['pg_char_params']
        buf_config: Mapping[str, Any] = specs['buf_config']
        search_params: Mapping[str, Any] = specs['search_params']
        t_rf: float = specs['t_rf']
        c_load: float = specs['c_load']
        t_step_min: float = specs['t_step_min']

        self._pinfo = self.get_tile(tile_name)

        pwr_tup = ('VSS', 'VDD')
        pwr_domain = {'in': pwr_tup, 'out': pwr_tup}
        supply_map = dict(VDD='VDD', VSS='VSS')
        inv_tbm_specs = self.get_dig_tran_specs(pwr_domain, supply_map)

        self._rc_inv_specs = dict(
            in_pin='in',
            out_pin='out',
            out_invert=True,
            tbm_specs=inv_tbm_specs,
            r_src=inv_char_params['r_src_delay'] / math.log(2),
            c_in=inv_char_params['c_in'],
            c_load=inv_char_params['c_load'],
            scale_min=inv_char_params['scale_min'],
            scale_max=inv_char_params['scale_max'],
            num_samples=inv_char_params['num_samples'],
            t_step_min=t_step_min,
        )

        self._rc_pg_specs = dict(
            tbm_specs=inv_tbm_specs,
            r_src=pg_char_params['r_src_delay'] / math.log(2),
            c_in=pg_char_params['c_in'],
            c_load=pg_char_params['c_load'],
            scale_min=pg_char_params['scale_min'],
            scale_max=pg_char_params['scale_max'],
            num_samples=pg_char_params['num_samples'],
            t_step_min=t_step_min,
        )

        se_tbm_specs = dict(**inv_tbm_specs)
        se_tbm_specs['pwr_domain'] = {key: pwr_tup for key in ['in', 'outp', 'outn', 'midn_inv',
                                                               'midp', 'midn_pass0', 'midn_pass1']}
        se_tbm_specs['diff_list'] = [(['outp'], ['outn'])]
        se_tbm_specs['sim_params'] = sim_params = dict(**se_tbm_specs['sim_params'])
        sim_params['t_rf'] = t_rf
        sim_params['c_load'] = c_load
        self._td_specs = dict(
            in_pin='in',
            out_pin='outp',
            tbm_specs=se_tbm_specs,
            start_pin=['in', 'in', 'midn_inv', 'in'],
            stop_pin=['outp', 'midn_inv', 'midp', 'midn_pass1'],
            out_invert=[False, True, True, True],
            add_src_res=False,
        )

        w_max = self._pinfo.get_row_place_info(ridx_n).row_info.width
        self._w_arr = np.arange(w_min, w_max + 1, w_res)
        self._beta = (inv_char_params['w_p'] / inv_char_params['w_n'],
                      pg_char_params['w_p'] / pg_char_params['w_n'])

        self._cin_specs = dict(
            in_pin='in',
            buf_config=dict(**buf_config),
            search_params=search_params,
            tbm_specs=se_tbm_specs,
            load_list=[dict(pin='outp', type='cap', value='c_load')],
        )

    async def async_design(self, **kwargs: Any) -> Mapping[str, Any]:
        specs = self.dsn_specs
        max_iter: int = specs['max_iter']
        err_targ_inv2_inv4: float = specs['err_targ_inv2_inv4']
        err_targ_total: float = specs['err_targ_total']

        # get initial sizing
        rc_data, se_params = await self.get_init_sizing()
        self.log(f'\ninitial sizing:\n{pprint.pformat(se_params, width=100)}')

        # get current delay
        td = await self.get_delays(se_params)

        err_inv2_inv4 = err_total = float('inf')
        for iter_idx in range(max_iter):
            # match inv2 and inv4 delay
            td = await self.match_delay(se_params, td, True)
            # match total delay
            td = await self.match_delay(se_params, td, False)

            # get error
            err_rise_inv2_inv4 = td.get_delay_err(True, True)
            err_fall_inv2_inv4 = td.get_delay_err(True, False)
            err_rise_total = td.get_delay_err(False, True)
            err_fall_total = td.get_delay_err(False, False)
            err_inv2_inv4 = max(err_rise_inv2_inv4, err_fall_inv2_inv4)
            err_total = max(err_rise_total, err_fall_total)
            self.log(f'iter={iter_idx}:\n'
                     f'td_outp={td.outp}\ntd_outn={td.outn}\n'
                     f'td_inv2={td.inv2}\ntd_inv4={td.inv4}\n'
                     f'err_inv2_inv4={err_inv2_inv4}, err_total={err_total}')
            if err_inv2_inv4 <= err_targ_inv2_inv4 and err_total <= err_targ_total:
                break

        self.log(f'\nse_params:\n{pprint.pformat(se_params, width=100)}')

        if err_inv2_inv4 > err_targ_inv2_inv4 or err_total > err_targ_total:
            raise ValueError('Cannot meet error spec.')

        c_in = await self.get_cin(se_params, rc_data)
        td_table = await self.sign_off(se_params)

        return dict(
            se_params=se_params,
            delay=td_table,
            c_in=c_in,
        )

    async def get_init_sizing(self) -> Tuple[RCData, Dict[str, Any]]:
        specs = self.dsn_specs
        c_load: float = specs['c_load']
        fanout_max_user: float = specs['fanout_max']
        area_min_inv1: float = specs['area_min_inv1']
        area_min_inv2: int = specs['area_min_inv2']
        area_min_pg: float = specs['area_min_pg']

        # get RC characterization
        rc_data = await self._get_init_rc()
        r_i0, c_i0, r_o0, c_o0 = rc_data.get_inv_rc(None)
        r_p0, c_s0, c_d0 = rc_data.get_pg_rc(None)

        # update maximum fanout
        fanout_max = min(fanout_max_user, (c_load / (area_min_inv1 * c_i0)) ** (1 / 3),
                         (c_load / (area_min_inv2 * c_i0)) ** (1 / 2))

        # get inverter sizes
        a_3 = c_load / (c_i0 * fanout_max)
        a_2 = a_3 / fanout_max
        # NOTE: use fanout_max_user to get stage 1 size to avoid having large input inverter
        a_1 = max(area_min_inv1, a_2 / fanout_max_user)

        # quantize negative path sizes
        seg_n_1, w_n_1, seg_p_1, w_p_1 = self._get_dimension(a_1, False)
        seg_n_2, w_n_2, seg_p_2, w_p_2 = self._get_dimension(a_2, False)
        seg_n_3, w_n_3, seg_p_3, w_p_3 = self._get_dimension(a_3, False)
        a_1 = seg_n_1 * w_n_1
        a_2 = seg_n_2 * w_n_2
        a_3 = seg_n_3 * w_n_3

        # get passgate size
        # set a_0 = a_2, so that we have good rise/fall time at input of inv4
        a_0 = a_2
        t_i0 = r_i0 * c_i0
        t_o0 = r_o0 * c_o0
        td_n = t_i0 * 2 + t_o0 * 3 + r_o0 * c_i0 * (a_2 / a_1 + a_3 / a_2) + r_o0 * c_load / a_3
        coe_c = t_o0 * 2 + t_i0 + r_o0 * c_i0 * a_3 / a_0 + r_p0 * c_d0 + r_o0 * c_load / a_3
        coe_p = r_o0 * (c_s0 + c_d0) / a_0
        coe_i = r_p0 * c_i0 * a_3
        # td_p = coe_c + coe_p * a_p + coe_i / a_p
        # AM-GM to get minimum possible delay
        td_p_min = coe_c + 2 * math.sqrt(coe_p * coe_i)
        if td_p_min > td_n:
            raise ValueError('No valid passgate size.')

        b = coe_c - td_n
        a_p = (-b + math.sqrt(b ** 2 - 4 * coe_p * coe_i)) / (2 * coe_p)
        if a_p < area_min_pg:
            raise ValueError('passgate solution < area_min_pg')

        # set the passgate size that gets us min delay to get good rise/fall time at input of inv4
        a_p = max(area_min_pg, math.sqrt(coe_i / coe_p))
        seg_n_p, w_n_p, seg_p_p, w_p_p = self._get_dimension(a_p, True)

        # return initial sizing dictionary
        # NOTE: make inv4_params and inv3_params point to same dictionary to enforce
        # equality
        inv3_params = dict(seg_n=seg_n_3, seg_p=seg_p_3, w_n=w_n_3, w_p=w_p_3)
        se_params = dict(
            pinfo=self._pinfo,
            invp_params_list=[
                dict(seg_n=seg_n_2, seg_p=seg_p_2, w_n=w_n_2, w_p=w_p_2),
                inv3_params,
            ],
            invn_params_list=[
                dict(seg_n=seg_n_1, seg_p=seg_p_1, w_n=w_n_1, w_p=w_p_1),
                dict(seg_n=seg_n_2, seg_p=seg_p_2, w_n=w_n_2, w_p=w_p_2),
                inv3_params,
            ],
            pg_params=dict(seg_n=seg_n_p, seg_p=seg_p_p, w_n=w_n_p, w_p=w_p_p),
            export_pins=True,
        )
        return rc_data, se_params

    async def _get_init_rc(self) -> RCData:
        specs = self.dsn_specs
        inv_char_params: Mapping[str, Any] = specs['inv_char_params']
        pg_char_params: Mapping[str, Any] = specs['pg_char_params']

        inv_params = dict(
            pinfo=self._pinfo,
            seg=inv_char_params['seg'],
            w_p=inv_char_params['w_p'],
            w_n=inv_char_params['w_n'],
        )
        pg_params = dict(
            pinfo=self._pinfo,
            seg=pg_char_params['seg'],
            w_p=pg_char_params['w_p'],
            w_n=pg_char_params['w_n'],
        )

        # get initial inverter and passgate DUT
        gatherer = GatherHelper()
        gatherer.append(self.async_wrapper_dut('INV_CHAR', InvCore, inv_params))
        gatherer.append(self.async_wrapper_dut('PG_CHAR', PassGateCore, pg_params))
        dut_inv, dut_pg = await gatherer.gather_err()

        # get inverter and passgate RC
        inv_mm = self.make_mm(RCDelayCharMM, self._rc_inv_specs)
        pg_mm = self.make_mm(PassGateRCDelayCharMM, self._rc_pg_specs)
        gatherer.clear()
        gatherer.append(self.async_simulate_mm_obj('rc_inv_char', dut_inv, inv_mm))
        gatherer.append(self.async_simulate_mm_obj('rc_pg_char', dut_pg, pg_mm))
        inv_result, pg_result = await gatherer.gather_err()

        w_inv = inv_params['w_n'] * inv_params['seg']
        w_pg = pg_params['w_n'] * pg_params['seg']
        rc_inv = _format_inv_rc(inv_result.data, w_inv)
        rc_pg = _format_pg_rc(pg_result.data, w_pg)
        return RCData(rc_inv, rc_pg)

    async def match_delay(self, se_params: Dict[str, Any], td: DelayData, inv2_inv4: bool
                          ) -> DelayData:
        specs = self.dsn_specs
        max_iter: int = specs['max_iter']

        if inv2_inv4:
            err_targ: float = specs['err_targ_inv2_inv4']
        else:
            err_targ: float = specs['err_targ_total']

        iter_idx = 0
        modified = True
        while iter_idx < max_iter and modified:
            modified = False
            err_rise = td.get_delay_err(inv2_inv4, True)
            if err_rise > err_targ:
                td = await self._match_delay_helper(se_params, td, inv2_inv4, True)
                modified = True
            err_fall = td.get_delay_err(inv2_inv4, False)
            if err_fall > err_targ:
                td = await self._match_delay_helper(se_params, td, inv2_inv4, False)
                modified = True

            err_rise = td.get_delay_err(inv2_inv4, True)
            err_fall = td.get_delay_err(inv2_inv4, False)
            self.log(f'iter={iter_idx}, inv2_inv4={inv2_inv4}:\n'
                     f'td_outp={td.outp}\ntd_outn={td.outn}\n'
                     f'td_inv2={td.inv2}\ntd_inv4={td.inv4}\n'
                     f'err_fall={err_fall}, err_rise={err_rise}')

            iter_idx += 1

        if modified:
            raise ValueError(f'max iter reached when trying to match delay, inv2_inv4={inv2_inv4}')

        return td

    async def _match_delay_helper(self, se_params: Dict[str, Any], td: DelayData,
                                  inv2_inv4: bool, out_rise: bool) -> DelayData:
        specs = self.dsn_specs
        search_step: int = specs['search_step']
        if inv2_inv4:
            err_targ: float = specs['err_targ_inv2_inv4']
        else:
            err_targ: float = specs['err_targ_total']

        invn_params_list: Sequence[Dict[str, int]] = se_params['invn_params_list']

        change_pos = td.change_pos(inv2_inv4, out_rise)
        self.log(f'change_pos={change_pos}, inv2_inv4={inv2_inv4}')
        if change_pos:
            # change_pos positive path sizing to match delay
            if inv2_inv4:
                suffix = '_p' if out_rise else '_n'
                inv_params = invn_params_list[2]
                seg = inv_params['seg' + suffix]
                w = inv_params['w' + suffix]

                search = InvSizeSearch(self, se_params, out_rise, inv2_inv4, False,
                                       self._set_inv4_size, self._w_arr, err_targ, search_step)
                data, seg, w = await search.get_seg_width(w, seg, None, td, None)
                return data
            else:
                suffix = '_n' if out_rise else '_p'
                inv_params = se_params['invp_params_list'][0]
                seg = inv_params['seg' + suffix]
                w = inv_params['w' + suffix]
                up = td.get_diff_up(inv2_inv4, out_rise, False)[1]
                if up:
                    seg_min = seg
                    td_min = td
                    seg_max = td_max = None
                else:
                    seg_min = 1
                    seg_max = seg
                    td_max = td
                    td_min = None

                search = InvSizeSearch(self, se_params, out_rise, inv2_inv4, False,
                                       self._set_inv0_size, self._w_arr, err_targ, search_step)
                data, seg, w = await search.get_seg_width(w, seg_min, seg_max, td_min, td_max)
                return data
        else:
            # update inv2 to match delay
            suffix = '_p' if out_rise else '_n'
            inv_params = invn_params_list[1]
            seg = inv_params['seg' + suffix]
            w = inv_params['w' + suffix]

            search = InvSizeSearch(self, se_params, out_rise, inv2_inv4, True,
                                   self._set_inv2_size, self._w_arr, err_targ, search_step)
            data, seg, w = await search.get_seg_width(w, seg, None, td, None)
            return data

    async def sign_off(self, se_params: Dict[str, Any]) -> Mapping[str, Any]:
        sign_off_envs: Sequence[str] = self.dsn_specs['sign_off_envs']

        dut = await self.async_wrapper_dut('SE_TO_DIFF', SingleToDiff, se_params)

        gatherer = GatherHelper()
        for sim_env in sign_off_envs:
            mm_specs = self._td_specs.copy()
            mm_specs['tbm_specs'] = tbm_specs = mm_specs['tbm_specs'].copy()
            tbm_specs['sim_envs'] = [sim_env]

            mm = self.make_mm(CombLogicTimingMM, mm_specs)
            gatherer.append(self.async_simulate_mm_obj(f'sign_off_{sim_env}_{dut.cache_name}',
                                                       dut, mm))

        result_list = await gatherer.gather_err()

        ans = {}
        pin_list = ['outp', 'outn']
        for sim_env, meas_result in zip(sign_off_envs, result_list):
            timing_data = meas_result.data['timing_data']

            cur_results = {}
            for pin_name in pin_list:
                data = timing_data[pin_name]
                cur_results[f'td_{pin_name}'] = (data['cell_fall'], data['cell_rise'])
                cur_results[f'trf_{pin_name}'] = (data['fall_transition'], data['rise_transition'])

            td_outp = cur_results['td_outp']
            td_outn = cur_results['td_outn']
            td_avg_rise = (td_outp[1] + td_outn[0]) / 2
            td_avg_fall = (td_outp[0] + td_outn[1]) / 2
            err_rise = abs(td_outp[1] - td_outn[0]) / 2 / td_avg_rise
            err_fall = abs(td_outp[0] - td_outn[1]) / 2 / td_avg_fall
            cur_results['td_out_avg'] = (td_avg_fall, td_avg_rise)
            cur_results['td_err'] = (err_fall, err_rise)
            ans[sim_env] = cur_results

        return ans

    def _set_inv0_size(self, se_params: Dict[str, Any], seg: int, w: int, out_rise: bool) -> None:
        inv0_params: Dict[str, int] = se_params['invp_params_list'][0]

        if out_rise:
            inv0_params['seg_n'] = seg
            inv0_params['w_n'] = w
            self.log(f'setting seg_n0={seg}, w_n0={w}')
        else:
            inv0_params['seg_p'] = seg
            inv0_params['w_p'] = w
            self.log(f'setting seg_p0={seg}, w_p0={w}')

    def _set_inv2_size(self, se_params: Dict[str, Any], seg: int, w: int, out_rise: bool) -> None:
        invn_params_list: Sequence[Dict[str, int]] = se_params['invn_params_list']
        inv1_params = invn_params_list[0]
        inv2_params = invn_params_list[1]

        if out_rise:
            scale = seg * w / (inv2_params['seg_p'] * inv2_params['w_p'])
            area_inv1 = scale * inv1_params['w_n'] * inv1_params['seg_n']
            seg_1, w_1 = self._get_seg_w(area_inv1)
            inv1_params['seg_n'] = seg_1
            inv1_params['w_n'] = w_1
            inv2_params['seg_p'] = seg
            inv2_params['w_p'] = w
            self.log(f'setting seg_n1={seg_1}, w_n1={w_1}, seg_p2={seg}, w_p2={w}')
        else:
            scale = seg * w / (inv2_params['seg_n'] * inv2_params['w_n'])
            area_inv1 = scale * inv1_params['w_p'] * inv1_params['seg_p']
            seg_1, w_1 = self._get_seg_w(area_inv1)
            inv1_params['seg_p'] = seg_1
            inv1_params['w_p'] = w_1
            inv2_params['seg_n'] = seg
            inv2_params['w_n'] = w
            self.log(f'setting seg_p1={seg_1}, w_p1={w_1}, seg_n2={seg}, w_n2={w}')

    def _set_inv4_size(self, se_params: Dict[str, Any], seg: int, w: int, out_rise: bool) -> None:
        invp_params_list: Sequence[Dict[str, int]] = se_params['invp_params_list']
        pg_params: Dict[str, int] = se_params['pg_params']
        inv0_params = invp_params_list[0]
        inv4_params = invp_params_list[1]
        if out_rise:
            scale = seg * w / (inv4_params['seg_p'] * inv4_params['w_p'])
            area_inv0 = scale * inv0_params['w_n'] * inv0_params['seg_n']
            area_pg = scale * pg_params['w_n'] * pg_params['seg_n']
            seg_0, w_0 = self._get_seg_w(area_inv0)
            seg_p, w_p = self._get_seg_w(area_pg)
            inv0_params['seg_n'] = seg_0
            inv0_params['w_n'] = w_0
            pg_params['seg_n'] = seg_p
            pg_params['w_n'] = w_p
            inv4_params['seg_p'] = seg
            inv4_params['w_p'] = w
            self.log(f'setting seg_n0={seg_0}, w_n0={w_0}, seg_np={seg_p}, w_np={w_p}, '
                     f'seg_p4={seg}, w_p4={w}')
        else:
            scale = seg * w / (inv4_params['seg_n'] * inv4_params['w_n'])
            area_inv0 = scale * inv0_params['w_p'] * inv0_params['seg_p']
            area_pg = scale * pg_params['w_p'] * pg_params['seg_p']
            seg_0, w_0 = self._get_seg_w(area_inv0)
            seg_p, w_p = self._get_seg_w(area_pg)
            inv0_params['seg_p'] = seg_0
            inv0_params['w_p'] = w_0
            pg_params['seg_p'] = seg_p
            pg_params['w_p'] = w_p
            inv4_params['seg_n'] = seg
            inv4_params['w_n'] = w
            self.log(f'setting seg_p0={seg_0}, w_p0={w_0}, seg_pp={seg_p}, w_pp={w_p}, '
                     f'seg_n4={seg}, w_n4={w}')

    async def get_cin(self, dut_params: Mapping[str, Any], rc_data: RCData) -> float:
        c_i0 = rc_data.get_inv_rc(None)[1]

        invp0 = dut_params['invp_params_list'][0]
        invn0 = dut_params['invn_params_list'][0]
        cin_guess = c_i0 * (invp0['w_n'] * invp0['seg_n'] + invn0['w_n'] * invn0['seg_n'])
        self._cin_specs['buf_config']['cin_guess'] = cin_guess

        dut = await self.async_wrapper_dut('SE_TO_DIFF', SingleToDiff, dut_params)
        mm = self.make_mm(CapDelayMatch, self._cin_specs)
        data = (await self.async_simulate_mm_obj(f'cin_{dut.cache_name}', dut, mm)).data
        cap_fall = data['cap_fall']
        cap_rise = data['cap_rise']
        cap_avg = (cap_fall + cap_rise) / 2

        self.log(f'cap_fall={cap_fall:.4g}, cap_rise={cap_rise:.4g}, cap_avg={cap_avg:.4g}')
        return cap_avg

    async def get_delays(self, se_params: Dict[str, Any]) -> DelayData:
        dut = await self.async_wrapper_dut('SE_TO_DIFF', SingleToDiff, se_params)

        self.log(f'se_params:\n{pprint.pformat(se_params, width=100)}')

        mm = self.make_mm(CombLogicTimingMM, self._td_specs)
        result = await self.async_simulate_mm_obj(f'td_{dut.cache_name}', dut, mm)
        timing_data = result.data['timing_data']

        outp_data = timing_data['outp']
        outn_data = timing_data['outn']
        inv2_data = timing_data['midp']
        inv4_data = timing_data['midn_pass1']

        td_outp = (outp_data['cell_fall'].item(), outp_data['cell_rise'].item())
        td_outn = (outn_data['cell_fall'].item(), outn_data['cell_rise'].item())
        td_inv2 = (inv2_data['cell_fall'].item(), inv2_data['cell_rise'].item())
        td_inv4 = (td_outp[0] - inv4_data['cell_rise'].item(),
                   td_outp[1] - inv4_data['cell_fall'].item())

        return DelayData(td_outp, td_outn, td_inv2, td_inv4)

    def _get_dim_list(self, a_min: float, a_max: float, is_pg: bool
                      ) -> Sequence[Tuple[int, int, int, int]]:
        a_min = max(a_min, self._w_arr[0])
        table = {}
        beta = self._beta[is_pg]
        for w_n in reversed(self._w_arr):
            seg_min = int(math.ceil(a_min / w_n))
            seg_max = int(math.floor(a_max / w_n))
            w_p = int(round(w_n * beta))
            for seg in range(seg_min, seg_max + 1):
                a_cur = seg * w_n
                if a_cur not in table:
                    table[a_cur] = (a_cur, seg, w_n, w_p)
        return sorted(table.values())

    def _get_dimension(self, area_n: float, is_pg: bool) -> Tuple[int, int, int, int]:
        area_p = area_n * self._beta[is_pg]
        seg_n, w_n = self._get_seg_w(area_n)
        seg_p, w_p = self._get_seg_w(area_p)
        return seg_n, w_n, seg_p, w_p

    def _get_seg_w(self, area: float) -> Tuple[int, int]:
        seg_arr = np.round(area / self._w_arr).astype(int)
        err_arr = np.abs(area - (seg_arr * self._w_arr))
        rev_idx = np.argmin(err_arr[::-1])
        idx = err_arr.size - rev_idx - 1
        return seg_arr[idx].item(), self._w_arr[idx].item()


def _get_delay_err(td1: float, td2: float):
    return abs(td1 - td2) / 2 / ((td1 + td2) / 2)


def _get_dim_str(params: Mapping[str, int]) -> str:
    return f'ns{params["seg_n"]}w{params["w_n"]}_ps{params["seg_p"]}w{params["w_p"]}'


def _format_inv_rc(rc_dict: Mapping[str, Any], w_norm: int) -> Mapping[str, Any]:
    r_in = rc_dict['r_in']
    c_in = rc_dict['c_in']
    r_out = rc_dict['r_out']
    c_out = rc_dict['c_out']
    return dict(
        r_in=(r_in[0].item() * w_norm, r_in[1].item() * w_norm),
        r_out=(r_out[0].item() * w_norm, r_out[1].item() * w_norm),
        c_in=(c_in[0].item() / w_norm, c_in[1].item() / w_norm),
        c_out=(c_out[0].item() / w_norm, c_out[1].item() / w_norm),
    )


def _format_pg_rc(rc_dict: Mapping[str, Any], w_norm: int) -> Mapping[str, Any]:
    r_p = rc_dict['r_p']
    c_s = rc_dict['c_s']
    c_d = rc_dict['c_d']
    return dict(
        r_p=(r_p[0].item() * w_norm, r_p[1].item() * w_norm),
        c_s=(c_s[0].item() / w_norm, c_s[1].item() / w_norm),
        c_d=(c_d[0].item() / w_norm, c_d[1].item() / w_norm),
    )

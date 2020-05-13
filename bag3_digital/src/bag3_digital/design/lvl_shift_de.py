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

from typing import Mapping, Dict, Any, Tuple, Sequence, Optional

import math
import pprint

from bag.simulation.cache import DesignInstance

from xbase.layout.mos.placement.data import MOSBasePlaceInfo

from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM

from ..layout.stdcells.levelshifter import LevelShifterCoreOutBuffer
from ..measurement.cap.delay_match import CapDelayMatch
from .base import DigitalDesigner, BinSearchSegWidth


class InvDelayMatch(BinSearchSegWidth):
    def __init__(self, dsn: LvlShiftDEDesigner, dut_params: Dict[str, Any],
                 w_list: Sequence[int], size_p: bool, err_targ: float,
                 search_step: int = 1) -> None:
        super().__init__(w_list, err_targ, search_step=search_step)

        self._dsn = dsn
        self._params = dut_params
        self._size_p = size_p

    @classmethod
    def get_bin_val(cls, data: Tuple[float, float]) -> float:
        return data[1] - data[0]

    def get_bin_search_info(self, data: Tuple[float, float]) -> Tuple[float, bool]:
        diff = self.get_bin_val(data)
        return diff, (diff > 0) == self._size_p

    def get_error(self, data: Tuple[float, float]) -> float:
        diff = self.get_bin_val(data)
        return abs(diff) / (data[0] + data[1])

    def set_size(self, seg: int, w: int) -> None:
        if self._size_p:
            self._params['buf_segp_list'][0] = seg
            self._params['w_dict']['invp'] = w
        else:
            self._params['buf_segn_list'][0] = seg
            self._params['w_dict']['invn'] = w

    async def get_data(self, seg: int, w: int) -> Tuple[float, float]:
        self._dsn.log(f'size_p={self._size_p}, set seg={seg}, w={w}')
        self.set_size(seg, w)
        return await self._dsn.get_delays(self._params)


class LvlShiftDEDesigner(DigitalDesigner):
    """Designer class for Level Shifter for differential signals in the RX
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pinfo: Optional[MOSBasePlaceInfo] = None
        self._w_p_list: Sequence[int] = []
        self._w_n_list: Sequence[int] = []
        self._td_specs: Dict[str, Any] = {}
        self._cin_specs: Dict[str, Any] = {}

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.dsn_specs
        c_load: float = specs['c_load']
        tile_name: str = specs['tile_name']
        ridx_n: int = specs['ridx_n']
        ridx_p: int = specs['ridx_p']
        w_min: int = specs['w_min']
        w_res: int = specs['w_res']
        buf_config: Mapping[str, Any] = specs['buf_config']
        search_params: Mapping[str, Any] = specs['search_params']

        self._pinfo = self.get_tile(tile_name)
        w_n_max = self._pinfo.get_row_place_info(ridx_n).row_info.width
        w_p_max = self._pinfo.get_row_place_info(ridx_p).row_info.width
        self._w_n_list = list(range(w_min, w_n_max + 1, w_res))
        self._w_p_list = list(range(w_min, w_p_max + 1, w_res))

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

        self._td_specs = dict(
            in_pin='in',
            out_pin='out',
            tbm_specs=tbm_specs,
            out_invert=False,
            add_src_res=False,
            load_list=[],
        )

        self._cin_specs = dict(
            in_pin='in',
            buf_config=dict(**buf_config),
            search_params=search_params,
            tbm_specs=tbm_specs,
            load_list=[dict(pin='out', type='cap', value='c_load')],
        )

    async def async_design(self, **kwargs: Any) -> Mapping[str, Any]:
        lv_params = self.get_init_lv_params()
        dut, td, err = await self.resize_inv(lv_params)
        c_in = await self.get_cap(dut, 'in')
        ans = dict(lv_params=lv_params, td=td, err=err, c_in=c_in)

        if lv_params['has_rst']:
            c_rst_out = await self.get_cap(dut, 'rst_out')
            c_rst_casc = await self.get_cap(dut, 'rst_casc')
            ans['c_rst_out'] = c_rst_out
            ans['c_rst_casc'] = c_rst_casc

        return ans

    def get_init_lv_params(self) -> Dict[str, Any]:
        """Get nominal level shifter size based on fanout"""
        specs = self.dsn_specs
        fanout_inv: float = specs['fanout_inv']
        fanout_core: float = specs['fanout_core']
        c_load: float = specs['c_load']
        k_ratio_core: float = specs['k_ratio_core']
        lv_params: Mapping[str, Any] = specs['lv_params']
        ridx_n: int = specs['ridx_n']
        ridx_p: int = specs['ridx_p']
        w_n = specs.get('w_n_inv', self._w_n_list[-1])
        w_p = specs.get('w_p_inv', self._w_p_list[-1])

        stack_p: int = lv_params.get('stack_p', 1)
        has_rst: bool = lv_params.get('has_rst', False)
        in_upper: bool = lv_params.get('in_upper', True)
        dual_output: bool = lv_params.get('dual_output', True)
        seg_prst: int = lv_params.get('seg_prst', 0)

        c_unit_n_seg = self._get_c_in_guess(0, 1, w_p, w_n)
        c_unit_p_seg = self._get_c_in_guess(1, 0, w_p, w_n)
        c_unit_inv = c_unit_n_seg + c_unit_p_seg
        seg_inv = int(math.ceil(c_load / fanout_inv / c_unit_inv))
        c_inv = seg_inv * c_unit_inv

        p_scale = 2 if stack_p == 2 else 1
        seg_p = int(math.ceil(c_inv / fanout_core / c_unit_p_seg * p_scale))
        seg_n = int(math.ceil(seg_p * w_p * k_ratio_core / w_n))
        c_in_guess = seg_n * c_unit_n_seg

        seg_dict = dict(pd=seg_n, pu=seg_p)
        if has_rst:
            rst_ratio: float = specs['rst_ratio']
            seg_dict['rst'] = int(math.ceil(seg_n * rst_ratio))
            if stack_p == 2 and seg_prst > 0:
                seg_dict['prst'] = seg_prst

        lv_shift_params = dict(
            pinfo=self._pinfo,
            seg_dict=seg_dict,
            w_dict=dict(pd=w_n, pu=w_p, rst=w_n, invn=w_n, invp=w_p),
            stack_p=stack_p,
            buf_segn_list=[seg_inv],
            buf_segp_list=[seg_inv],
            has_rst=has_rst,
            dual_output=dual_output,
            in_upper=in_upper,
            ridx_n=ridx_n,
            ridx_p=ridx_p,
        )
        self.log(f'init c_in={c_in_guess:.4g}, lv_params:\n'
                 f'{pprint.pformat(lv_shift_params, width=100)}')
        return lv_shift_params

    async def resize_inv(self, dut_params: Dict[str, Any]
                         ) -> Tuple[DesignInstance, Tuple[float, float], float]:
        specs = self.dsn_specs
        err_targ: float = specs['err_targ']
        search_step: int = specs.get('search_step', 1)

        td = await self.get_delays(dut_params)

        # equalize rise/fall delays by slowing down fast edge
        if td[1] < td[0]:
            search = InvDelayMatch(self, dut_params, self._w_p_list, True, err_targ,
                                   search_step=search_step)
            seg = dut_params['buf_segp_list'][0]
            w = dut_params['w_dict']['invp']
        else:
            search = InvDelayMatch(self, dut_params, self._w_n_list, False, err_targ,
                                   search_step=search_step)
            seg = dut_params['buf_segn_list'][0]
            w = dut_params['w_dict']['invn']

        low_bnd = search.get_bin_search_info(td)[1]

        if low_bnd:
            seg_max = None
            seg_min = seg
            td_max = None
            td_min = td
        else:
            seg_max = seg
            seg_min = 1
            td_max = td
            td_min = None

        td, seg, w = await search.get_seg_width(w, seg_min, seg_max, td_min, td_max)
        err = search.get_error(td)
        self.log(f'final result:\ntd_fall={td[0]:.4g}, td_rise={td[1]:.4g}, err={err:.4g}')

        dut = await self.async_wrapper_dut('LV_SHIFT_DIFF', LevelShifterCoreOutBuffer, dut_params)
        return dut, td, err

    async def get_cap(self, dut: DesignInstance, pin_name: str) -> float:
        cin_specs = self._cin_specs

        params = dut.lay_master.params['params']
        seg_dict = params['seg_dict']
        w_dict = params['w_dict']
        if pin_name == 'in':
            seg_p = seg_dict['pu'] if params['stack_p'] == 2 else 0
            seg_n = seg_dict['pd']
            w_p = w_dict['pu']
            w_n = w_dict['pd']
        elif pin_name == 'rst_out':
            seg_p = 0
            seg_n = seg_dict['rst']
            w_p = 0
            w_n = w_dict['rst']
        else:
            seg_p = 0
            seg_n = seg_dict['pd']
            w_p = 0
            w_n = w_dict['pd']

        cin_specs['in_pin'] = pin_name
        cin_specs['buf_config']['cin_guess'] = self._get_c_in_guess(seg_p, seg_n, w_p, w_n)

        mm = self.make_mm(CapDelayMatch, cin_specs)
        data = (await self.async_simulate_mm_obj(f'c_{pin_name}_{dut.cache_name}', dut, mm)).data
        cap_fall = data['cap_fall']
        cap_rise = data['cap_rise']
        cap_avg = (cap_fall + cap_rise) / 2
        self.log(f'{pin_name} cap_fall={cap_fall:.4g}, cap_rise={cap_rise:.4g}, '
                 f'cap_avg={cap_avg:.4g}')
        return cap_avg

    async def get_delays(self, dut_params: Dict[str, Any]) -> Tuple[float, float]:
        dut = await self.async_wrapper_dut('LV_SHIFT_DIFF', LevelShifterCoreOutBuffer, dut_params)

        self.log(f'dut params:\n{pprint.pformat(dut_params, width=100)}')

        mm = self.make_mm(CombLogicTimingMM, self._td_specs)
        result = await self.async_simulate_mm_obj(f'td_{dut.cache_name}', dut, mm)
        timing_data = result.data['timing_data']

        out_data = timing_data['out']
        td_fall = out_data['cell_fall'].item()
        td_rise = out_data['cell_rise'].item()
        self.log(f'delays:\ntd_fall={td_fall:.4g}, td_rise={td_rise:.4g}')
        return td_fall, td_rise

    def _get_c_in_guess(self, seg_p: int, seg_n: int, w_p: int, w_n: int) -> float:
        specs = self.dsn_specs
        c_unit_p: float = specs['c_unit_p']
        c_unit_n: float = specs['c_unit_n']

        return seg_p * w_p * c_unit_p + seg_n * w_n * c_unit_n

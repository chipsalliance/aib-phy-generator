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

from typing import Any, Mapping, Tuple, Dict, Sequence, Optional

import pprint

from ....layout.stdcells.gates import InvCore
from ....measurement.util import get_in_buffer_pin_names
from ....measurement.comb import BufferCombLogicTimingMM
from ...base import DigitalDesigner, BinSearchSegWidth


class InvSizeSearch(BinSearchSegWidth):
    def __init__(self, dsn: InvCapInMatchDesigner, dut_params: Dict[str, Any],
                 w_list: Sequence[int], td_targ: float, rf_idx: Optional[int],
                 err_targ: float, search_step: int = 1) -> None:
        super().__init__(w_list, err_targ, search_step=search_step)

        self._dsn = dsn
        self._params = dut_params
        self._td_targ = td_targ
        self._rf_idx = rf_idx

    def get_bin_val(self, data: Tuple[float, float]) -> float:
        if self._rf_idx is None:
            td = (data[0] + data[1]) / 2
        else:
            td = data[self._rf_idx]
        return self._td_targ - td

    def get_bin_search_info(self, data: Tuple[float, float]) -> Tuple[float, bool]:
        diff = self.get_bin_val(data)
        return diff, diff > 0

    def get_error(self, data: Tuple[float, float]) -> float:
        diff = self.get_bin_val(data)
        return abs(diff) / self._td_targ

    def set_size(self, seg: int, w: int) -> None:
        self._params['seg'] = seg
        self._params['w_p'] = self._params['w_n'] = w

    async def get_data(self, seg: int, w: int) -> Tuple[float, float]:
        self.set_size(seg, w)
        return await self._dsn.get_delays(self._params)


class InvCapInMatchDesigner(DigitalDesigner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._td_specs: Dict[str, Any] = {}
        self._w_n_list = []
        self._w_p_list = []
        self._stop_pin: str = ''

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.dsn_specs
        tile_name: str = specs['tile_name']
        buf_params: Mapping[str, Any] = specs['buf_params']
        ridx_n: int = specs['ridx_n']
        ridx_p: int = specs['ridx_p']
        w_min: int = specs['w_min']
        w_res: int = specs['w_res']
        c_load: float = specs.get('c_load', 0)

        pinfo = self.get_tile(tile_name)
        w_n_max = pinfo.get_row_place_info(ridx_n).row_info.width
        w_p_max = pinfo.get_row_place_info(ridx_p).row_info.width

        self._w_n_list = list(range(w_min, w_n_max + 1, w_res))
        self._w_p_list = list(range(w_min, w_p_max + 1, w_res))

        pwr_tup = ('VSS', 'VDD')
        start_pin, self._stop_pin = get_in_buffer_pin_names('in')
        pwr_domain = {p_: pwr_tup for p_ in ['in', 'out']}
        supply_map = dict(VDD='VDD', VSS='VSS')
        tbm_specs = self.get_dig_tran_specs(pwr_domain, supply_map)

        if c_load != 0:
            tbm_specs['sim_params'] = sim_params = dict(**tbm_specs['sim_params'])
            sim_params['c_load'] = c_load
            out_pin = 'out'
        else:
            out_pin = ''

        self._td_specs = dict(
            tbm_specs=tbm_specs,
            in_pin='in',
            out_pin=out_pin,
            start_pin=start_pin,
            stop_pin=self._stop_pin,
            out_invert=True,
            buf_params=buf_params,
        )

    async def async_design(self, **kwargs: Any) -> Mapping[str, Any]:
        specs = self.dsn_specs
        tile_name: str = specs['tile_name']
        ridx_n: int = specs['ridx_n']
        ridx_p: int = specs['ridx_p']
        err_targ: float = specs['err_targ']
        inv_params_init: Mapping[str, Any] = specs['inv_params_init']
        rise_targ: Optional[float] = specs.get('td_rise', None)
        fall_targ: Optional[float] = specs.get('td_fall', None)

        if rise_targ is None:
            if fall_targ is None:
                raise ValueError('Must specify either td_rise or td_fall.')
            td_targ = fall_targ
            rf_idx = 0
        elif fall_targ is None:
            td_targ = rise_targ
            rf_idx = 1
        else:
            td_targ = (fall_targ + rise_targ) / 2
            rf_idx = None

        dut_params = dict(pinfo=self.get_tile(tile_name), ridx_p=ridx_p, ridx_n=ridx_n,
                          **inv_params_init)
        td = await self.get_delays(dut_params)

        td, err = await self._match_delay(dut_params, td, td_targ, rf_idx, err_targ)
        self.log(f'td_fall={td[0]:.4g}, td_rise={td[1]:.4g}, err={err:.4g}')
        return dict(inv_params=dut_params, td=td, err=err)

    async def _match_delay(self, dut_params: Dict[str, Any], td: Tuple[float, float],
                           td_targ: float, rf_idx: Optional[int], err_targ: float
                           ) -> Tuple[Tuple[float, float], float]:
        search_step: int = self.dsn_specs['search_step']

        w_list = self._w_n_list
        w = dut_params['w_n']
        seg = dut_params['seg']
        search = InvSizeSearch(self, dut_params, w_list, td_targ, rf_idx, err_targ, search_step)
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
        return td, err

    async def get_delays(self, dut_params: Dict[str, Any]) -> Tuple[float, float]:
        dut = await self.async_wrapper_dut('INV', InvCore, dut_params)

        self.log(f'dut_params:\n{pprint.pformat(dut_params, width=100)}')

        mm = self.make_mm(BufferCombLogicTimingMM, self._td_specs)
        result = await self.async_simulate_mm_obj(f'td_{dut.cache_name}', dut, mm)
        timing_data = result.data['timing_data'][self._stop_pin]

        return timing_data['cell_fall'].item(), timing_data['cell_rise'].item()

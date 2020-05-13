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

from typing import Dict, Any, Mapping, Optional, Tuple, Sequence

import math
import pprint
from dataclasses import dataclass

from bag.concurrent.util import GatherHelper
from bag.simulation.cache import DesignInstance

from xbase.layout.mos.base import MOSBasePlaceInfo

from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM

from bag3_digital.measurement.cap.delay_match import CapDelayMatch
from bag3_digital.design.base import DigitalDesigner, BinSearchSegWidth
from bag3_digital.design.stdcells.se_to_diff import SingleToDiffDesigner

from ..layout.se_to_diff import SingleToDiffEnable, DiffBufferEnable


@dataclass(frozen=True, eq=True)
class DelayData:
    outp: Tuple[float, float]
    outn: Tuple[float, float]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(outp=({self.outp[0]:.4g}, {self.outp[1]:.4g}), '
                f'outn=({self.outn[0]:.4g}, {self.outn[1]:.4g}), '
                f'td_fall={self.td_fall_avg}, td_rise={self.td_rise_avg})')

    @property
    def td_rise_avg(self) -> float:
        return (self.outp[1] + self.outn[0]) / 2

    @property
    def td_fall_avg(self) -> float:
        return (self.outp[0] + self.outn[1]) / 2


class DelayBalanceSearch(BinSearchSegWidth):
    def __init__(self, dsn: SingleToDiffEnableDesigner, dut_params: Dict[str, Any], size_p: bool,
                 w_list: Sequence[int], err_targ: float, search_step: int = 1) -> None:
        super().__init__(w_list, err_targ, search_step=search_step)

        self._dsn = dsn
        self._params = dut_params
        self._size_p = size_p

    def get_bin_search_info(self, data: DelayData) -> Tuple[float, bool]:
        val = data.td_rise_avg - data.td_fall_avg
        return val, (val > 0) ^ self._size_p

    def get_error(self, data: DelayData) -> float:
        td_rise_avg = data.td_rise_avg
        td_fall_avg = data.td_fall_avg
        td_avg = (td_rise_avg + td_fall_avg) / 2
        return abs(td_rise_avg - td_fall_avg) / td_avg

    def set_size(self, seg: int, w: int) -> None:
        nand_params = self._params['nand_params']
        if self._size_p:
            nand_params['seg_p'] = seg
            nand_params['w_p'] = w
        else:
            nand_params['seg_n'] = seg
            nand_params['w_n'] = w

    async def get_data(self, seg: int, w: int) -> DelayData:
        self.set_size(seg, w)
        return await self._dsn.get_delays(self._params, False)


class DelayMatchSearch(BinSearchSegWidth):
    def __init__(self, dsn: SingleToDiffEnableDesigner, dut_params: Dict[str, Any],
                 td_targ: float, size_p: bool, size_nand: bool, w_list: Sequence[int],
                 err_targ: float, search_step: int = 1) -> None:
        super().__init__(w_list, err_targ, search_step=search_step)

        self._dsn = dsn
        self._params = dut_params
        self._size_p = size_p
        self._size_nand = size_nand
        self._td_targ = td_targ

    def get_bin_search_val(self, data: DelayData) -> float:
        td_data = data.outp if self._size_nand else data.outn
        td_cur = td_data[0] if self._size_p else td_data[1]
        return self._td_targ - td_cur

    def get_bin_search_info(self, data: DelayData) -> Tuple[float, bool]:
        val = self.get_bin_search_val(data)
        return val, val < 0

    def get_error(self, data: DelayData) -> float:
        val = self.get_bin_search_val(data)
        return abs(val) / self._td_targ

    def set_size(self, seg: int, w: int) -> None:
        table = self._params['nand_params' if self._size_nand else 'nor_params']
        if self._size_p:
            table['seg_p'] = seg
            table['w_p'] = w
        else:
            table['seg_n'] = seg
            table['w_n'] = w

    async def get_data(self, seg: int, w: int) -> DelayData:
        self.set_size(seg, w)
        return await self._dsn.get_delays(self._params, True)

    async def get_seg_width_wrapper(self, data: DelayData) -> DelayData:
        table = self._params['nand_params' if self._size_nand else 'nor_params']
        if self._size_p:
            seg = table['seg_p']
            w = table['w_p']
        else:
            seg = table['seg_n']
            w = table['w_n']

        up = self.get_bin_search_info(data)[1]
        if up:
            seg_min = seg
            seg_max = None
            data_min = data
            data_max = None
        else:
            seg_min = 1
            seg_max = seg
            data_min = None
            data_max = data

        return (await self.get_seg_width(w, seg_min, seg_max, data_min, data_max))[0]


class SingleToDiffEnableDesigner(DigitalDesigner):
    """Designs se_to_diff_en and se_to_diff_match.

    1) Size the se_to_diff_en NAND for fanout. Adjust P/N ratio for equal rise/fall delay.
    2) copy inverter chain parameters to se_to_diff_match, size input NAND for se_to_diff_match
       for equal rise/fall delay to se_to_diff.
    3) size input NOR for se_to_diff_match for equal rise/fall delay.  Copy to se_to_diff_en.
    4) Sign off
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pinfo: Optional[MOSBasePlaceInfo] = None
        self._se_dsn_specs: Optional[Dict[str, Any]] = None
        self._td_se_en_specs: Dict[str, Any] = {}
        self._td_match_specs: Dict[str, Any] = {}
        self._cin_se_en_specs: Dict[str, Any] = {}
        self._cin_match_specs: Dict[str, Any] = {}
        self._w_p_list: Sequence[int] = []
        self._w_n_list: Sequence[int] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.dsn_specs
        ridx_n: int = specs['ridx_n']
        ridx_p: int = specs['ridx_p']
        w_min: int = specs['w_min']
        w_res: int = specs['w_res']
        tile_name: str = specs['tile_name']
        c_load: float = specs['c_load']
        buf_config: Mapping[str, Any] = specs['buf_config']
        search_params: Mapping[str, Any] = specs['search_params']
        se_to_diff_results: Optional[Mapping[str, Any]] = specs.get('se_to_diff_results', None)

        if se_to_diff_results is None:
            self._se_dsn_specs = dict(**specs['se_to_diff_specs'])
            for key in ['c_load', 'max_iter', 'search_step', 'w_min', 'w_res', 'ridx_n',
                        'buf_config', 'search_params', 'sign_off_envs', 'tile_name', 'tile_specs',
                        'dig_tran_specs', 'sup_values']:
                self._se_dsn_specs[key] = specs[key]
        else:
            self._se_dsn_specs = None

        self._pinfo = self.get_tile(tile_name)
        w_p_max = self._pinfo.get_row_place_info(ridx_p).row_info.width
        w_n_max = self._pinfo.get_row_place_info(ridx_n).row_info.width
        self._w_p_list = list(range(w_min, w_p_max + 1, w_res))
        self._w_n_list = list(range(w_min, w_n_max + 1, w_res))

        pwr_tup = ('VSS', 'VDD')
        supply_map = dict(VDD='VDD', VSS='VSS')
        se_en_pwr_domain = {p_: pwr_tup for p_ in ['in', 'outp', 'outn', 'en', 'enb']}
        se_en_diff_list = [(['outp'], ['outn']), (['en'], ['enb'])]
        pin_values = dict(en=1)
        se_en_tbm_specs = self.get_dig_tran_specs(se_en_pwr_domain, supply_map,
                                                  diff_list=se_en_diff_list,
                                                  pin_values=pin_values)
        se_en_tbm_specs['sim_params'] = se_en_sim_params = dict(**se_en_tbm_specs['sim_params'])
        se_en_sim_params['c_load'] = c_load
        self._td_se_en_specs = dict(
            in_pin='in',
            out_pin='outp',
            tbm_specs=se_en_tbm_specs,
            out_invert=False,
            add_src_res=False,
        )

        self._cin_se_en_specs = dict(
            in_pin='en',
            buf_config=dict(**buf_config),
            search_params=search_params,
            tbm_specs=se_en_tbm_specs,
            load_list=[dict(pin='outp', type='cap', value='c_load')],
        )

        match_pwr_domain = {p_: pwr_tup for p_ in ['inp', 'inn', 'outp', 'outn', 'en', 'enb']}
        match_diff_list = [(['outp'], ['outn']), (['en'], ['enb']), (['inp'], ['inn'])]
        pin_values = dict(en=1)
        match_tbm_specs = self.get_dig_tran_specs(match_pwr_domain, supply_map,
                                                  diff_list=match_diff_list,
                                                  pin_values=pin_values)
        match_tbm_specs['sim_params'] = match_sim_params = dict(**match_tbm_specs['sim_params'])
        match_sim_params['c_load'] = c_load

        self._td_match_specs = dict(
            in_pin='inp',
            out_pin='outp',
            tbm_specs=match_tbm_specs,
            out_invert=False,
            add_src_res=False,
        )

        self._cin_match_specs = dict(
            in_pin='en',
            buf_config=dict(**buf_config),
            search_params=search_params,
            tbm_specs=match_tbm_specs,
            load_list=[dict(pin='outp', type='cap', value='c_load')],
        )

    async def async_design(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Passed in kwargs are the same as self.dsn_specs.

        Parameters
        ----------
        kwargs: Any
            se_to_diff_specs: Mapping[str, Any]
                Single Ended to Differential design parameters
            nand_init_params: Mapping[str, Any]
                Initial NAND generator parameters
            c_load: float
                Target load capacitance
            err_targ: float
                Target error tolerance, for matching NAND rise and fall times
            se_to_diff_results: Optional[Mapping[str, Any]]
                If provided, return these generator parameters instead of running the design
                procedures
            Below are global specs shared and passed to each of the designers
            w_min: Union[int, float]
                Minimum width
            w_res: Union[int, float]
                Width resolution
            c_unit_n: float
                Unit NMOS transistor capacitance for w=1, seg=1
            c_unit_p: float
                Unit PMOS transistor capacitance for w=1, seg=1
            dig_tran_specs: Mapping[str, Any]
                DigitalTranTB testbench specs
            search_params: Mapping[str, Any]
                Parameters used for capacitor size binary search
            search_step: int
                Binary search step size
            max_iter: int
                Maximum allowed iterations to search for converge in binary search
            buf_config: Mapping[str, Any]
                Buffer parameters, used in DigitalTranTB and capacitor size search
            sign_off_envs: Sequence[str]
                Corners used for sign off
            sup_values: Mapping[str, Any]
                Per-corner supply values
            tile_name: str
                Name of the layout tile to use
            tile_specs: Mapping[str, Any]
                Tile Info Specs
            ridx_n: int
                NMOS transistor row
            ridx_p: int
                PMOS transistor Row

        Returns
        -------
        ans: Mapping[str, Any]
            Design summary
        """
        se_en_params = await self.get_init_se_en_params()
        match_params, td_se_en = await self.size_se_en_nand(se_en_params)
        await self.size_match(match_params, td_se_en, True)
        await self.size_match(match_params, td_se_en, False)

        se_en_params['nor_params'] = match_params['nor_params']

        td_data = await self.sign_off(se_en_params, match_params)
        c_en_se = await self.get_enable_cin(se_en_params, False)
        c_en_match = await self.get_enable_cin(match_params, True)

        return dict(se_params=se_en_params, match_params=match_params, td_data=td_data,
                    c_en_se=c_en_se, c_en_match=c_en_match)

    async def get_init_se_en_params(self) -> Dict[str, Any]:
        specs = self.dsn_specs
        nand_init_params: Mapping[str, Any] = specs['nand_init_params']
        ridx_p: int = specs['ridx_p']
        ridx_n: int = specs['ridx_n']

        w_p: int = nand_init_params['w_p']
        w_n: int = nand_init_params['w_n']
        fanout: float = nand_init_params['fanout']
        seg_min: int = nand_init_params['seg_min']

        if self._se_dsn_specs is None:
            se_results = specs['se_to_diff_results']
        else:
            se_dsn = self.new_designer(SingleToDiffDesigner, self._se_dsn_specs)
            se_results = await se_dsn.async_design()

        c_load = se_results['c_in']
        se_params = se_results['se_params']
        se_params.pop('pinfo', None)
        se_params.pop('export_pins', None)

        c_unit = self._get_c_in_guess(1, 1, w_p, w_n)
        seg = max(seg_min, int(math.ceil(c_load / fanout / c_unit)))

        nand_params = dict(seg_p=seg, seg_n=seg, w_p=w_p, w_n=w_n)
        dut_params = dict(
            pinfo=self._pinfo,
            nand_params=nand_params,
            nor_params=nand_params.copy(),
            core_params=se_params,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
        )

        return dut_params

    async def get_enable_cin(self, dut_params: Mapping[str, Any], match: bool) -> float:
        if match:
            mm_specs = self._cin_match_specs
        else:
            mm_specs = self._cin_se_en_specs

        table = dut_params['nand_params']
        cin_guess = self._get_c_in_guess(table['seg_p'], table['seg_n'], table['w_p'], table['w_n'])
        mm_specs['in_pin'] = 'en'
        mm_specs['buf_config']['cin_guess'] = cin_guess
        mm = self.make_mm(CapDelayMatch, mm_specs)
        dut = await self.get_dut(dut_params, match)
        en_data = (await self.async_simulate_mm_obj(f'c_en_{dut.cache_name}', dut, mm)).data
        en_cap_fall = en_data['cap_fall']
        en_cap_rise = en_data['cap_rise']
        en_cap_avg = (en_cap_fall + en_cap_rise) / 2

        table = dut_params['nor_params']
        cin_guess = self._get_c_in_guess(table['seg_p'], table['seg_n'], table['w_p'], table['w_n'])
        mm.specs['in_pin'] = 'enb'
        mm_specs['buf_config']['cin_guess'] = cin_guess
        mm.commit()
        enb_data = (await self.async_simulate_mm_obj(f'c_enb_{dut.cache_name}', dut, mm)).data
        enb_cap_fall = enb_data['cap_fall']
        enb_cap_rise = enb_data['cap_rise']
        enb_cap_avg = (enb_cap_fall + enb_cap_rise) / 2

        self.log(f'en_cap_fall={en_cap_fall:.4g}, en_cap_rise={en_cap_rise:.4g}, '
                 f'en_cap_avg={en_cap_avg:.4g}')
        self.log(f'enb_cap_fall={enb_cap_fall:.4g}, enb_cap_rise={enb_cap_rise:.4g}, '
                 f'enb_cap_avg={enb_cap_avg:.4g}')

        ans = max(en_cap_avg, enb_cap_avg)
        return ans

    async def size_se_en_nand(self, dut_params: Dict[str, Any]) -> Tuple[Dict[str, Any], DelayData]:
        specs = self.dsn_specs
        err_targ: float = specs['err_targ']
        search_step: int = specs['search_step']

        td = await self.get_delays(dut_params, False)

        td_rise = td.td_rise_avg
        td_fall = td.td_fall_avg
        nand_params = dut_params['nand_params']
        if td_rise < td_fall:
            size_p = True
            w_list = self._w_p_list
            w = nand_params['w_p']
            seg = nand_params['seg_p']
        else:
            size_p = False
            w_list = self._w_n_list
            w = nand_params['w_n']
            seg = nand_params['seg_n']
        search = DelayBalanceSearch(self, dut_params, size_p, w_list, err_targ, search_step)

        td = (await search.get_seg_width(w, seg, None, td, None))[0]
        self.log('se_en nand design done.  se_en params:\n'
                 f'{pprint.pformat(dut_params, width=100)}\n'
                 f'td_outp={td.outp}\ntd_outn={td.outn}')

        # create se_to_diff_match parameters
        dut = await self.get_dut(dut_params, False)
        se_en_master = dut.lay_master.core
        seg_p_list = []
        seg_n_list = []
        w_p_list = []
        w_n_list = []
        for table in dut_params['core_params']['invn_params_list']:
            seg_p_list.append(table['seg_p'])
            seg_n_list.append(table['seg_n'])
            w_p_list.append(table['w_p'])
            w_n_list.append(table['w_n'])
        match_params = dict(
            pinfo=self._pinfo,
            nand_params=dut_params['nand_params'].copy(),
            nor_params=dut_params['nor_params'].copy(),
            core_params=dict(
                segp_list=seg_p_list,
                segn_list=seg_n_list,
                w_p=w_p_list,
                w_n=w_n_list,
            ),
            ridx_p=dut_params['ridx_p'],
            ridx_n=dut_params['ridx_n'],
            en_ncol_min=se_en_master.en_ncol,
            buf_col_list=se_en_master.buf_col_list,
        )
        return match_params, td

    async def size_match(self, dut_params: Dict[str, Any], td_targ: DelayData, size_nand: bool
                         ) -> None:
        specs = self.dsn_specs
        err_targ: float = specs['err_targ']
        search_step: int = specs['search_step']
        max_iter: int = specs['max_iter']

        td = await self.get_delays(dut_params, True)

        modified = False
        rise_search = DelayMatchSearch(self, dut_params, td_targ.td_rise_avg, False, size_nand,
                                       self._w_n_list, err_targ, search_step)
        fall_search = DelayMatchSearch(self, dut_params, td_targ.td_fall_avg, True, size_nand,
                                       self._w_p_list, err_targ, search_step)
        for iter_idx in range(max_iter):
            modified = False
            # match rise delay
            if rise_search.get_error(td) > err_targ:
                td = await rise_search.get_seg_width_wrapper(td)
                modified = True

            # match fall delay
            if fall_search.get_error(td) > err_targ:
                td = await fall_search.get_seg_width_wrapper(td)
                modified = True

            if not modified:
                break

        if modified:
            rise_err = rise_search.get_error(td)
            fall_err = fall_search.get_error(td)
            raise ValueError('Cannot match both rise and fall delay, '
                             f'rise_err={rise_err:.4g}, fall_err={fall_err:.4g}')

        self.log(f'match sizing done, size_nand={size_nand} design done.  match params:\n'
                 f'{pprint.pformat(dut_params, width=100)}\n'
                 f'td_outp={td.outp}\ntd_outn={td.outn}')

    async def sign_off(self, se_params: Mapping[str, Any], match_params: Mapping[str, Any]
                       ) -> Mapping[str, Any]:
        sign_off_envs: Sequence[str] = self.dsn_specs['sign_off_envs']

        gatherer = GatherHelper()
        dut_se = await self.get_dut(se_params, False)
        dut_match = await self.get_dut(match_params, True)
        for sim_env in sign_off_envs:
            gatherer.append(self.get_delays_dut(dut_se, False, sim_env))
            gatherer.append(self.get_delays_dut(dut_match, True, sim_env))

        result_list = await gatherer.gather_err()

        ans = {}
        for idx, sim_env in enumerate(sign_off_envs):
            se_data = result_list[2 * idx]
            match_data = result_list[2 * idx + 1]
            ans[sim_env] = cur_result = {}
            for name, td_data in [('se_en', se_data), ('match', match_data)]:
                td_outp = td_data.outp
                td_outn = td_data.outn
                td_rise_avg = td_data.td_rise_avg
                td_fall_avg = td_data.td_fall_avg
                err_rise = abs(td_outp[1] - td_outn[0]) / 2 / td_rise_avg
                err_fall = abs(td_outp[0] - td_outn[1]) / 2 / td_fall_avg
                cur_result[name] = dict(
                    td_outp=td_outp,
                    td_outn=td_outn,
                    td_avg=(td_fall_avg, td_rise_avg),
                    td_err=(err_fall, err_rise),
                )

        return ans

    async def get_dut(self, dut_params: Mapping[str, Any], match: bool) -> DesignInstance:
        if match:
            dut_cls = DiffBufferEnable
            log_name = 'match'
            impl_cell = 'SE_TO_DIFF_MATCH'
        else:
            dut_cls = SingleToDiffEnable
            log_name = 'se_en'
            impl_cell = 'SE_TO_DIFF_EN'

        self.log(f'{log_name} params:\n{pprint.pformat(dut_params, width=100)}')
        return await self.async_wrapper_dut(impl_cell, dut_cls, dut_params)

    async def get_delays(self, dut_params: Mapping[str, Any], match: bool, sim_env: str = ''
                         ) -> DelayData:
        dut = await self.get_dut(dut_params, match)
        return await self.get_delays_dut(dut, match, sim_env=sim_env)

    async def get_delays_dut(self, dut: DesignInstance, match: bool, sim_env: str = ''
                             ) -> DelayData:
        if match:
            log_name = 'match'
            mm_specs = self._td_match_specs
        else:
            log_name = 'se_en'
            mm_specs = self._td_se_en_specs

        if sim_env:
            mm_specs = mm_specs.copy()
            mm_specs['tbm_specs'] = tbm_specs = mm_specs['tbm_specs'].copy()
            tbm_specs['sim_envs'] = [sim_env]

        mm = self.make_mm(CombLogicTimingMM, mm_specs)
        sim_id = f'td_{dut.cache_name}'
        if sim_env:
            sim_id = sim_id + f'_{sim_env}'
        data = (await self.async_simulate_mm_obj(sim_id, dut, mm)).data
        outp_data = data['timing_data']['outp']
        outn_data = data['timing_data']['outn']
        ans = DelayData((outp_data['cell_fall'].item(), outp_data['cell_rise'].item()),
                        (outn_data['cell_fall'].item(), outn_data['cell_rise'].item()))
        self.log(f'{log_name} delays:\ntd_outp={ans.outp}\ntd_outp={ans.outn}\n'
                 f'td_fall_avg={ans.td_fall_avg:.4g}, td_rise_avg={ans.td_rise_avg:.4g}')

        return ans

    def _get_c_in_guess(self, seg_p: int, seg_n: int, w_p: int, w_n: int) -> float:
        specs = self.dsn_specs
        c_unit_p: float = specs['c_unit_p']
        c_unit_n: float = specs['c_unit_n']

        return seg_p * w_p * c_unit_p + seg_n * w_n * c_unit_n

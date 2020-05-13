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

from typing import Any, Mapping, Optional, Sequence, Tuple, Type, Union, Iterable, Dict

import abc
import math

from bag.util.search import BinaryIterator
from bag.simulation.cache import DesignInstance

from xbase.layout.mos.placement.data import (
    TileInfoTable, MOSArrayPlaceInfo, MOSBasePlaceInfo, TilePattern
)
from xbase.layout.mos.base import MOSBase

from bag.simulation.design import DesignerBase

from ..layout.stdcells.util import STDCellWrapper


class DigitalDesigner(DesignerBase, abc.ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._tinfo_table: Optional[TileInfoTable] = None
        self._dig_tran_specs: Mapping[str, Any] = {}
        self._sup_values: Mapping[str, Union[float, Mapping[str, float]]] = {}

        super().__init__(*args, **kwargs)

    @property
    def arr_info(self) -> MOSArrayPlaceInfo:
        return self._tinfo_table.arr_info

    def commit(self) -> None:
        super().commit()

        specs = self.dsn_specs
        tile_specs: Mapping[str, Any] = specs['tile_specs']
        dig_tran_specs: Mapping[str, Any] = specs['dig_tran_specs']
        sup_values: Mapping[str, Union[float, Mapping[str, float]]] = specs['sup_values']

        self._tinfo_table: TileInfoTable = TileInfoTable.make_tiles(self.grid, tile_specs)
        self._dig_tran_specs = dig_tran_specs
        self._sup_values = sup_values

    def get_tile(self, name: str) -> MOSBasePlaceInfo:
        return self._tinfo_table[name]

    def make_tile_pattern(self, tiles: Iterable[Mapping[str, Any]]
                          ) -> Tuple[TilePattern, TileInfoTable]:
        return self._tinfo_table.make_tile_pattern(tiles), self._tinfo_table

    def get_dig_tran_specs(self, pwr_domain: Mapping[str, Tuple[str, str]],
                           supply_map: Mapping[str, str],
                           pin_values: Optional[Mapping[str, int]] = None,
                           reset_list: Optional[Sequence[Tuple[str, bool]]] = None,
                           diff_list: Optional[Sequence[Tuple[Sequence[str], Sequence[str]]]] = None
                           ) -> Dict[str, Any]:
        sup_values = {k: self._sup_values[v] for k, v in supply_map.items()}
        ans = dict(pwr_domain=pwr_domain, sup_values=sup_values, **self._dig_tran_specs)
        if pin_values:
            ans['pin_values'] = pin_values
        else:
            ans['pin_values'] = {}
        if reset_list:
            ans['reset_list'] = reset_list
        if diff_list:
            ans['diff_list'] = diff_list
        return ans

    async def async_wrapper_dut(self, impl_cell: str, dut_cls: Type[MOSBase],
                                dut_params: Mapping[str, Any], draw_taps: bool = True,
                                pwr_gnd_list: Optional[Sequence[Tuple[str, str]]] = None,
                                extract: Optional[bool] = None, name_prefix: str = '',
                                name_suffix: str = '', flat: bool = False
                                ) -> DesignInstance:
        wrap_params = dict(cls_name=dut_cls.get_qualified_name(), draw_taps=draw_taps,
                           pwr_gnd_list=pwr_gnd_list, params=dut_params)
        return await self.async_new_dut(impl_cell, STDCellWrapper, wrap_params,
                                        extract=extract, name_prefix=name_prefix,
                                        name_suffix=name_suffix, flat=flat)

    async def async_batch_wrapper_dut(self, dut_specs: Sequence[Mapping[str, Any]],
                                      ) -> Sequence[DesignInstance]:
        wrap_specs = []
        for info in dut_specs:
            cls_name = info['dut_cls'].get_qualified_name()
            draw_taps = info.get('draw_taps', True)
            pwr_gnd_list = info.get('pwr_gnd_list', None)
            wrap_params = dict(cls_name=cls_name, draw_taps=draw_taps,
                               pwr_gnd_list=pwr_gnd_list, params=info['dut_params'])
            wrap_info = dict(**info)
            wrap_info['dut_cls'] = STDCellWrapper
            wrap_info['dut_params'] = wrap_params
            wrap_specs.append(wrap_info)

        return await self.async_batch_dut(wrap_specs)


class BinSearchSegWidth(abc.ABC):
    def __init__(self, w_list: Sequence[int], err_targ: float, search_step: int = 1) -> None:
        self._w_list = w_list
        self._err_targ = err_targ
        self._search_step = search_step

    @abc.abstractmethod
    def get_bin_search_info(self, data: Any) -> Tuple[float, bool]:
        pass

    @abc.abstractmethod
    def get_error(self, data: Any) -> float:
        pass

    @abc.abstractmethod
    def set_size(self, seg: int, w: int) -> None:
        pass

    @abc.abstractmethod
    async def get_data(self, seg: int, w: int) -> Any:
        pass

    async def get_seg_width(self, w: int, seg_min: int, seg_max: Optional[int],
                            data_min: Optional[Any], data_max: Optional[Any],
                            no_throw: bool = False) -> Tuple[Any, int, int]:
        data, seg, a_min, a_max = await self._search_helper(w, seg_min, seg_max, data_min, data_max)
        err = self.get_error(data)
        if err <= self._err_targ:
            self.set_size(seg, w)
            return data, seg, w

        # tweak width to reduce error
        best_err = [err, seg, w, data]
        for w_new in reversed(self._w_list):
            if w_new == w:
                # skip to new width
                continue

            # try to find seg_min lower bound
            seg_min = max(1, int(math.floor(a_min / w_new)))
            seg_max = None
            data_min = await self.get_data(seg_min, w_new)
            low_bnd = self.get_bin_search_info(data_min)[1]
            data_max = None
            while not low_bnd:
                seg_max = seg_min
                data_max = data_min
                next_seg_min = max(seg_min // 2, 1)
                if next_seg_min == seg_min:
                    # we're stuck, break
                    seg_min = None
                    break

                data_min = await self.get_data(next_seg_min, w_new)
                low_bnd = self.get_bin_search_info(data_min)[1]
                seg_min = next_seg_min

            if seg_min is None:
                # weird, no minimum solution found, ignore this width
                continue

            if seg_max is None:
                # try to see if we can get upper bound from a_max
                seg_test = int(math.ceil(a_max / w_new))
                data_test = await self.get_data(seg_test, w_new)
                low_bnd = self.get_bin_search_info(data_test)[1]
                if low_bnd:
                    # seg_test is a lower bound, not a upper bound
                    seg_min = seg_test
                    data_min = data_test
                else:
                    # seg_test is a upper bound
                    seg_max = seg_test
                    data_max = data_test

            # do binary search at this width
            data, seg, a_min_new, a_max_new = await self._search_helper(
                w_new, seg_min, seg_max, data_min, data_max)
            err = self.get_error(data)
            if err <= self._err_targ:
                self.set_size(seg, w)
                return data, seg, w_new
            elif err < best_err[0]:
                best_err[0] = err
                best_err[1] = seg
                best_err[2] = w_new
                best_err[3] = data

            a_min = min(a_min, a_min_new)
            a_max = max(a_max, a_max_new)

        self.set_size(best_err[1], best_err[2])
        if no_throw:
            return best_err[3], best_err[1], best_err[2]
        else:
            raise ValueError('Cannot meet error spec.  '
                             f'Best err = {best_err[0]:.4g} at seg={best_err[1]}, w={best_err[2]}')

    async def _search_helper(self, w: int, seg_min: int, seg_max: Optional[int],
                             data_min: Optional[Any], data_max: Optional[Any],
                             ) -> Tuple[Any, int, int, int]:
        # first, binary search on segments without changing width
        bin_iter = BinaryIterator(seg_min, seg_max, search_step=self._search_step)

        bval_min = bval_max = None
        if data_max is not None:
            bval_max = self.get_bin_search_info(data_max)[0]
        if data_min is not None:
            bval_min = self.get_bin_search_info(data_min)[0]
            bin_iter.set_current(seg_min)
            bin_iter.up(val=bval_min)
        elif seg_max is not None and data_max is not None:
            bin_iter.set_current(seg_max)
            bin_iter.down(val=bval_max)

        bounds = [[seg_min, bval_min, data_min], [seg_max, bval_max, data_max]]
        while bin_iter.has_next():
            cur_seg = bin_iter.get_next()
            cur_data = await self.get_data(cur_seg, w)
            cur_bval, up = self.get_bin_search_info(cur_data)
            if up:
                bounds[0][0] = cur_seg
                bounds[0][1] = cur_bval
                bounds[0][2] = cur_data
                bin_iter.up(val=cur_bval)
            else:
                bounds[1][0] = cur_seg
                bounds[1][1] = cur_bval
                bounds[1][2] = cur_data
                bin_iter.down(val=cur_bval)

        if bounds[1][1] is None:
            idx = 0
            seg_min = seg_max = bounds[0][0]
        elif bounds[0][1] is None:
            idx = 1
            seg_min = seg_max = bounds[1][0]
        else:
            idx = int(abs(bounds[1][1]) < abs(bounds[0][1]))
            seg_min = bounds[0][0]
            seg_max = bounds[1][0]

        opt_bnd = bounds[idx]
        opt_seg = opt_bnd[0]
        opt_data = opt_bnd[2]

        a_min = seg_min * w
        a_max = seg_max * w
        return opt_data, opt_seg, a_min, a_max

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

from typing import Any, Dict, cast, Type, Optional, List, Tuple

from bag.design.module import Module
from bag.layout.template import TemplateDB
from bag.util.immutable import Param
from bag.util.importlib import import_class

from xbase.layout.mos.base import MOSBase
from xbase.layout.mos.top import GenericWrapper


class STDCellWithTap(MOSBase):
    """A MOSArrayWrapper that works with any given generator class."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

        self._sch_cls: Optional[Type[Module]] = None
        self._core: Optional[MOSBase] = None

    @property
    def core(self) -> MOSBase:
        return self._core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='wrapped class name.',
            params='parameters for the wrapped class.',
            pwr_gnd_list='List of supply names for each tile.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(pwr_gnd_list=None)

    def get_schematic_class_inst(self) -> Optional[Type[Module]]:
        return self._sch_cls

    def get_layout_basename(self) -> str:
        cls_name: str = self.params['cls_name']
        cls_name = cls_name.split('.')[-1]
        return cls_name + 'Tap'

    def draw_layout(self) -> None:
        gen_cls = cast(Type[MOSBase], import_class(self.params['cls_name']))
        pwr_gnd_list: List[Tuple[str, str]] = self.params['pwr_gnd_list']

        master: MOSBase = self.new_template(gen_cls, params=self.params['params'])
        self._core = master
        self.draw_base(master.draw_base_info)

        num_tiles = master.num_tile_rows
        tap_ncol = max((self.get_tap_ncol(tile_idx=tile_idx) for tile_idx in range(num_tiles)))
        tap_sep_col = self.sub_sep_col
        num_cols = master.num_cols + 2 * (tap_sep_col + tap_ncol)
        self.set_mos_size(num_cols, num_tiles=num_tiles)

        if not pwr_gnd_list:
            pwr_gnd_list = [('VDD', 'VSS')] * num_tiles
        elif len(pwr_gnd_list) != num_tiles:
            raise ValueError('pwr_gnd_list length mismatch.')

        inst = self.add_tile(master, 0, tap_ncol + tap_sep_col)
        sup_names = set()
        for tidx in range(num_tiles):
            pwr_name, gnd_name = pwr_gnd_list[tidx]
            sup_names.add(pwr_name)
            sup_names.add(gnd_name)
            vdd_list = []
            vss_list = []
            self.add_tap(0, vdd_list, vss_list, tile_idx=tidx)
            self.add_tap(num_cols, vdd_list, vss_list, tile_idx=tidx, flip_lr=True)
            self.add_pin(pwr_name, vdd_list, connect=True)
            self.add_pin(gnd_name, vss_list, connect=True)

        for name in inst.port_names_iter():
            if name in sup_names:
                self.reexport(inst.get_port(name), connect=True)
            else:
                self.reexport(inst.get_port(name))

        self.sch_params = master.sch_params
        self._sch_cls = master.get_schematic_class_inst()


class STDCellWrapper(GenericWrapper):
    """A MOSArrayWrapper that works with any given generator class."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        GenericWrapper.__init__(self, temp_db, params, **kwargs)

    @property
    def core(self) -> MOSBase:
        real_core = super().core
        if self.params['draw_taps']:
            return cast(STDCellWrapper, real_core).core
        return real_core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='wrapped class name.',
            params='parameters for the wrapped class.',
            draw_taps='True to draw taps.',
            pwr_gnd_list='List of supply names for each tile.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(draw_taps=True, pwr_gnd_list=[])

    def draw_layout(self) -> None:
        cls_name: str = self.params['cls_name']
        params: Param = self.params['params']
        draw_taps: bool = self.params['draw_taps']
        pwr_gnd_list: List[Tuple[str, str]] = self.params['pwr_gnd_list']

        if draw_taps:
            master = self.new_template(STDCellWithTap,
                                       params=dict(cls_name=cls_name, params=params,
                                                   pwr_gnd_list=pwr_gnd_list))
        else:
            gen_cls = cast(Type[MOSBase], import_class(cls_name))
            master = self.new_template(gen_cls, params=params)

        self.wrap_mos_base(master, False)

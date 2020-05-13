# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

from typing import Any, Dict, Optional, Type, List, Sequence, cast

import abc

from pybag.enum import RoundMode, MinLenMode

from bag.util.immutable import Param
from bag.layout.routing.base import TrackID, WireArray
from bag.layout.template import TemplateDB
from bag.design.database import ModuleDB
from bag.design.module import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from bag3_digital.layout.stdcells.gates import NAND2Core, NOR2Core, InvChainCore
from bag3_digital.layout.stdcells.se_to_diff import SingleToDiff


class DiffOutputBufferEnableBase(MOSBase, abc.ABC):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

        self._en_ncol = 0
        self._core: Optional[MOSBase] = None

    @property
    def en_ncol(self) -> int:
        return self._en_ncol

    @property
    def core(self) -> MOSBase:
        return self._core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            nand_params='NAND parameters',
            nor_params='NOR parameters',
            core_params='two-three splitter parameters.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            is_guarded='True to not route anything on conn_layer to allow space for guard rings',
            swap_tiles='True to swap outp/outn tiles.',
            vertical_out='True to make the vertical connection of out happen at this level',
            en_ncol_min='Minimum number of columns for NAND/NOR.',
            buf_col_list='List of inverter column indices.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_p=-1,
            ridx_n=0,
            is_guarded=False,
            swap_tiles=False,
            vertical_out=True,
            en_ncol_min=0,
            buf_col_list=None,
        )

    @abc.abstractmethod
    def draw_buffers(self, master: MOSBase, tile_outp: int, tile_outn: int, col: int
                     ) -> Dict[str, List[WireArray]]:
        pass

    def draw_layout_helper(self, differential: bool) -> None:
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, params['pinfo'])
        self.draw_base(pinfo)

        nand_params: Param = params['nand_params']
        nor_params: Param = params['nor_params']
        core_params: Param = params['core_params']
        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']
        is_guarded: bool = params['is_guarded']
        swap_tiles: bool = params['swap_tiles']
        vertical_out: bool = params['vertical_out']
        en_ncol_min: int = params['en_ncol_min']
        buf_col_list: Optional[Sequence[int]] = params['buf_col_list']

        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-2)
        nd1_tidx = self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=1)
        ng0_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-2)
        ng1_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)

        append = dict(pinfo=pinfo, ridx_p=ridx_p, ridx_n=ridx_n, is_guarded=is_guarded,
                      vertical_out=False, sig_locs=dict(nout=nd1_tidx, pout=pd0_tidx))
        nand_params = nand_params.copy(append=append)
        nor_params = nor_params.copy(append=append)

        core_append = dict(
            pinfo=pinfo, ridx_p=ridx_p, ridx_n=ridx_n, is_guarded=is_guarded,
            vertical_out=vertical_out, swap_tiles=swap_tiles,
            sig_locs=dict(nout0=nd0_tidx, nout1=nd1_tidx, pout0=pd1_tidx, pout1=pd0_tidx,
                          nin0=ng1_tidx, nin1=ng0_tidx),
        )
        if differential:
            core_cls = InvChainCore
            core_append['sep_stages'] = True
            core_append['buf_col_list'] = buf_col_list
        else:
            core_cls = SingleToDiff
            core_append['swap_tiles'] = swap_tiles

        core_params = core_params.copy(append=core_append)
        nand_master = self.new_template(NAND2Core, params=nand_params)
        nor_master = self.new_template(NOR2Core, params=nor_params)
        core_master = self.new_template(core_cls, params=core_params)
        self._core = core_master

        # placement
        nand_ncols = nand_master.num_cols
        nor_ncols = nor_master.num_cols
        self._en_ncol = max(nand_ncols, nor_ncols, en_ncol_min)
        sep = self.min_sep_col
        tile_outp = int(swap_tiles)
        tile_outn = 1 - tile_outp
        # NOTE: right-align NAND/NOR to reduce wire resistance
        nand = self.add_tile(nand_master, tile_outp, self._en_ncol - nand_ncols)
        nor = self.add_tile(nor_master, tile_outn, self._en_ncol - nor_ncols)
        core_ports = self.draw_buffers(core_master, tile_outp, tile_outn, self._en_ncol + sep)
        self.set_mos_size()

        # routing
        # vdd/vss
        vdd_list = core_ports['VDD']
        vss_list = core_ports['VSS']
        for inst in [nand, nor]:
            vdd_list.extend(inst.port_pins_iter('VDD'))
            vss_list.extend(inst.port_pins_iter('VSS'))

        vdd = self.connect_wires(vdd_list)[0]
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', self.connect_wires(vss_list))

        # connect NAND/NOR ports
        nor_list = [nor.get_pin('pout')]
        nand_list = [nand.get_pin('nout')]
        if nor.has_port('nout'):
            nor_list.append(nor.get_pin('nout'))
            nand_list.append(nand.get_pin('pout'))

        grid = self.grid
        tr_manager = self.tr_manager
        vm_layer = self.conn_layer + 2
        vm_w = tr_manager.get_width(vm_layer, 'sig')

        in0_nand = nand.get_pin('nin<0>')
        in0_nor = nor.get_pin('nin<0>')
        in1_nand = nand.get_pin('nin<1>')
        in1_nor = nor.get_pin('nin<1>')
        in1_xl = max(in1_nand.lower, in1_nor.lower)
        vm_en_tidx = grid.coord_to_track(vm_layer, in1_xl, mode=RoundMode.GREATER_EQ)
        vm_en_tid = TrackID(vm_layer, vm_en_tidx, width=vm_w)
        in1_nand = self.connect_to_tracks(in1_nand, vm_en_tid, min_len_mode=MinLenMode.UPPER)
        in1_nor = self.connect_to_tracks(in1_nor, vm_en_tid, min_len_mode=MinLenMode.LOWER)
        self.add_pin('en', in1_nand)
        self.add_pin('enb', in1_nor)
        # TODO: hack: add extra spacing to avoid corner spacing
        vm_in_tid = TrackID(vm_layer, tr_manager.get_next_track(vm_layer, vm_en_tidx, 'sig', 'sig',
                                                                up=-2), width=vm_w)

        if differential:
            core_sch_params = core_master.sch_params.copy(remove=['dual_output'])
            self.add_pin('inp', self.connect_to_tracks(in0_nand, vm_in_tid,
                                                       min_len_mode=MinLenMode.UPPER))
            self.add_pin('inn', self.connect_to_tracks(in0_nor, vm_in_tid,
                                                       min_len_mode=MinLenMode.LOWER))

            core_inp = core_ports['inp'][0]
            core_inn = core_ports['inn'][0]
            if core_inp.layer_id == vm_layer:
                self.connect_to_track_wires(nor_list, core_inn)
                self.connect_to_track_wires(nand_list, core_inp)
            else:
                # core_inp and core_inn are on hm_layer
                tidx_r = grid.coord_to_track(vm_layer, min(core_inp.middle, core_inn.middle),
                                             mode=RoundMode.LESS_EQ)
                tidx = tr_manager.get_next_track(vm_layer, tidx_r, 'sig', 'sig', up=False)
                vm_tid = TrackID(vm_layer, tidx, width=vm_w)
                nor_list.append(core_inn)
                nand_list.append(core_inp)
                self.connect_to_tracks(nor_list, vm_tid)
                self.connect_to_tracks(nand_list, vm_tid)
        else:
            core_sch_params = core_master.sch_params
            self.add_pin('in', self.connect_to_tracks(in0_nand, vm_in_tid,
                                                      min_len_mode=MinLenMode.UPPER))
            self.connect_to_tracks([in0_nor, vdd], vm_in_tid)

            core_in = core_ports['in'][0]
            if len(nor_list) > 1:
                # NAND and NOR outputs are disconnected
                tidx = grid.coord_to_track(vm_layer, nor_list[0].middle, mode=RoundMode.LESS_EQ)
                # avoid shorts
                tidx = max(tr_manager.get_next_track(vm_layer, vm_en_tidx, 'sig', 'sig', up=True),
                           tidx)
                self.connect_to_tracks(nor_list, TrackID(vm_layer, tidx))

            self.connect_to_track_wires(nand_list, core_in)

        # because of the input nand there is an inversion of polarity
        outp = core_ports['outp']
        outn = core_ports['outn']
        connect = len(outp) > 1
        self.add_pin('outp', outn, connect=connect)
        self.add_pin('outn', outp, connect=connect)

        self.sch_params = dict(
            core_params=core_sch_params,
            nand_params=nand_master.sch_params,
            nor_params=nor_master.sch_params,
        )


class SingleToDiffEnable(DiffOutputBufferEnableBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @property
    def buf_col_list(self) -> Sequence[int]:
        return cast(SingleToDiff, self.core).buf_col_list

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_se2diff')

    def draw_layout(self):
        self.draw_layout_helper(False)

    def draw_buffers(self, master: MOSBase, tile_outp: int, tile_outn: int, col: int
                     ) -> Dict[str, List[WireArray]]:
        core = self.add_tile(master, 0, col)
        return {name: core.get_all_port_pins(name)
                for name in ['VDD', 'VSS', 'in', 'outp', 'outn']}


class DiffBufferEnable(DiffOutputBufferEnableBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_se2diff_match')

    def draw_layout(self):
        self.draw_layout_helper(True)

    def draw_buffers(self, master: MOSBase, tile_outp: int, tile_outn: int, col: int
                     ) -> Dict[str, List[WireArray]]:
        bufp = self.add_tile(master, tile_outp, col)
        bufn = self.add_tile(master, tile_outn, col)
        return {
            'VDD': [bufp.get_pin('VDD'), bufn.get_pin('VDD')],
            'VSS': [bufp.get_pin('VSS'), bufn.get_pin('VSS')],
            'outn': bufp.get_all_port_pins('outb'),
            'outp': bufn.get_all_port_pins('outb'),
            'inp': [bufp.get_pin('in')],
            'inn': [bufn.get_pin('in')],
        }

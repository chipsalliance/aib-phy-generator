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

from typing import Any, Dict, Optional, Type, List

from pybag.enum import RoundMode

from bag.math import lcm
from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.layout.routing.base import TrackID, WireArray
from bag.layout.template import TemplateDB
from bag.design.database import Module

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from bag3_digital.layout.stdcells.se_to_diff import SingleToDiff
from bag3_digital.layout.sampler.flop_strongarm import FlopStrongArm
from bag3_digital.layout.stdcells.gates import InvCore

from ..schematic.aib_phasedet import aib_ams__aib_phasedet


class PhaseDetectorHalf(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The placement information object.',
            se_params='two-three splitter parameters.',
            flop_params='strongarm flop parameters.',
            inv_params='dummy inverter parameters.',
            vm_pitch='vm_layer pin pitch.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            vm_pitch=0.5,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        se_params: Param = self.params['se_params']
        flop_params: Param = self.params['flop_params']
        inv_params: Param = self.params['inv_params']
        vm_pitch: HalfInt = HalfInt.convert(self.params['vm_pitch'])

        # create masters
        flop_pinfo = self.get_draw_base_sub_pattern(2, 4)
        flop_params = flop_params.copy(append=dict(pinfo=flop_pinfo, out_pitch=vm_pitch))
        se_params = se_params.copy(append=dict(pinfo=self.get_tile_pinfo(0), vertical_out=False,
                                               vertical_in=False))
        inv_params = inv_params.copy(append=dict(pinfo=self.get_tile_pinfo(2),
                                                 ridx_n=0, ridx_p=-1))

        flop_master = self.new_template(FlopStrongArm, params=flop_params)
        se_master = self.new_template(SingleToDiff, params=se_params)
        inv_master = self.new_template(InvCore, params=inv_params)

        # floorplanning
        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        vm_w_hs = tr_manager.get_width(vm_layer, 'sig_hs')

        # get flop column quantization
        sd_pitch = self.sd_pitch
        vm_coord_pitch = int(vm_pitch * self.grid.get_track_pitch(vm_layer))
        sep_half = max(-(-self.min_sep_col // 2), -(-vm_coord_pitch // sd_pitch))
        blk_ncol = lcm([sd_pitch, vm_coord_pitch]) // sd_pitch

        sep_ncol = self.min_sep_col
        flop_ncol2 = flop_master.num_cols // 2
        se_col = sep_half
        center_col = sep_half + inv_master.num_cols + sep_ncol + flop_ncol2
        center_col = -(-center_col // blk_ncol) * blk_ncol
        cur_col = center_col - flop_ncol2
        if (cur_col & 1) != (se_col & 1):
            se_col += 1
        se = self.add_tile(se_master, 0, se_col)
        inv = self.add_tile(inv_master, 2, cur_col - sep_ncol, flip_lr=True)
        flop = self.add_tile(flop_master, 2, cur_col)
        cur_col += flop_master.num_cols + self.sub_sep_col // 2

        lay_range = range(self.conn_layer, xm_layer + 1)
        vdd_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        vss_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        sup_info = self.get_supply_column_info(xm_layer)
        for tile_idx in range(self.num_tile_rows):
            self.add_supply_column(sup_info, cur_col, vdd_table, vss_table, ridx_p=-1,
                                   ridx_n=0, tile_idx=tile_idx, flip_lr=False)

        self.set_mos_size()

        # connections
        # supplies
        for lay in range(hm_layer, xm_layer + 1, 2):
            vss = vss_table[lay]
            vdd = vdd_table[lay]
            if lay == hm_layer:
                for inst in [inv, flop, se]:
                    vdd.extend(inst.get_all_port_pins('VDD'))
                    vss.extend(inst.get_all_port_pins('VSS'))
            vdd = self.connect_wires(vdd)
            vss = self.connect_wires(vss)
            self.add_pin(f'VDD_{lay}', vdd, hide=True)
            self.add_pin(f'VSS_{lay}', vss, hide=True)

        inp = flop.get_pin('inp')
        inn = flop.get_pin('inn')
        loc_list = tr_manager.place_wires(vm_layer, ['sig_hs'] * 2, center_coord=inp.middle)[1]
        inp, inn = self.connect_differential_tracks(inp, inn, vm_layer, loc_list[0], loc_list[1],
                                                    width=vm_w_hs)
        self.add_pin('sa_inp', inp)
        self.add_pin('sa_inn', inn)

        in_vm_ref = self.grid.coord_to_track(vm_layer, 0)
        in_vm_tidx = tr_manager.get_next_track(vm_layer, in_vm_ref, 'sig', 'sig', up=True)
        vm_phtr = vm_pitch.dbl_value
        in_vm_dhtr = -(-(in_vm_tidx - in_vm_ref).dbl_value // vm_phtr) * vm_phtr
        in_vm_tidx = in_vm_ref + HalfInt(in_vm_dhtr)
        in_warr = self.connect_to_tracks(se.get_all_port_pins('in'),
                                         TrackID(vm_layer, in_vm_tidx, width=vm_w),
                                         track_lower=0)
        self.add_pin('in', in_warr)

        out = flop.get_pin('outp')
        self.add_pin('out', self.extend_wires(out, upper=self.bound_box.yh))

        self.reexport(flop.get_port('clkl'), net_name='clk', hide=False)
        self.reexport(flop.get_port('rstlb'))
        self.reexport(se.get_port('outp'), net_name='midp')
        self.reexport(se.get_port('outn'), net_name='midn')
        self.reexport(inv.get_port('in'), net_name='dum')

        self.sch_params = dict(
            se_params=se_master.sch_params,
            flop_params=flop_master.sch_params,
            inv_params=inv_master.sch_params,
        )


class PhaseDetector(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_phasedet

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return PhaseDetectorHalf.get_params_info()

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return PhaseDetectorHalf.get_default_param_values()

    def draw_layout(self) -> None:
        core_master: PhaseDetectorHalf = self.new_template(PhaseDetectorHalf, params=self.params)
        self.draw_base(core_master.draw_base_info)

        core_ncol = core_master.num_cols
        clka = self.add_tile(core_master, 0, core_ncol, flip_lr=True)
        clkb = self.add_tile(core_master, 0, core_ncol)

        self.set_mos_size()

        # connections
        grid = self.grid
        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        xm_w_hs = tr_manager.get_width(xm_layer, 'sig_hs')

        # supplies
        sup_list = ['VDD', 'VSS']
        for lay in [hm_layer, xm_layer]:
            for name in sup_list:
                cur_name = f'{name}_{lay}'
                warrs = clka.get_all_port_pins(cur_name)
                warrs.extend(clkb.port_pins_iter(cur_name))
                warrs = self.connect_wires(warrs)
                if lay == xm_layer:
                    self.add_pin(name, warrs)

        # strongarm inputs
        clkap = clka.get_all_port_pins('midp')
        clkan = clka.get_all_port_pins('midn')
        clka_sa_inp = clka.get_pin('sa_inp')
        clkap, clkan = self.connect_differential_wires(clkap, clkan, clka_sa_inp,
                                                       clka.get_pin('sa_inn'))
        clkbp = clkb.get_all_port_pins('midp')
        clkbn = clkb.get_all_port_pins('midn')
        clkbp, clkbn = self.connect_differential_wires(clkbp, clkbn, clkb.get_pin('sa_inn'),
                                                       clkb.get_pin('sa_inp'))

        # reset
        rstlb = self.connect_wires([clka.get_pin('rstlb'), clkb.get_pin('rstlb')])[0]
        vm_tidx = grid.coord_to_track(vm_layer, rstlb.middle, mode=RoundMode.NEAREST)
        rstlb = self.connect_to_tracks(rstlb, TrackID(vm_layer, vm_tidx, width=vm_w),
                                       track_lower=0)
        self.add_pin('RSTb', rstlb)

        # strongarm clocks
        xm_top_tidx = grid.find_next_track(xm_layer, clka_sa_inp.upper, tr_width=xm_w_hs,
                                           mode=RoundMode.LESS)
        loc_list = tr_manager.place_wires(xm_layer, ['sig_hs'] * 4, align_track=xm_top_tidx,
                                          align_idx=-1)[1]
        clkap, clkan = self.connect_differential_tracks(clkap, clkan, xm_layer, loc_list[1],
                                                        loc_list[0], width=xm_w_hs)
        self.connect_differential_wires(clkb.get_pin('dum'), clkb.get_pin('clk'),
                                        clkap, clkan)
        clkbp, clkbn = self.connect_differential_tracks(clkbp, clkbn, xm_layer, loc_list[3],
                                                        loc_list[2], width=xm_w_hs)
        self.connect_differential_wires(clka.get_pin('clk'), clka.get_pin('dum'),
                                        clkbp, clkbn)

        # re-exports
        self.reexport(clka.get_port('in'), net_name='CLKA')
        self.reexport(clkb.get_port('in'), net_name='CLKB')
        self.reexport(clka.get_port('out'), net_name='t_up')
        self.reexport(clkb.get_port('out'), net_name='t_down')

        self.sch_params = core_master.sch_params

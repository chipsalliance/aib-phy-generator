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

from typing import Any, Dict, Optional, Type

from pybag.enum import MinLenMode

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.database import Module

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from bag3_digital.layout.stdcells.gates import InvCore
from bag3_digital.layout.sampler.flop_strongarm import FlopStrongArm
from bag3_digital.layout.stdcells.mux import Mux2to1Matched

from ..schematic.aib_dcc_helper_core import aib_ams__aib_dcc_helper_core


class DCCHelperCore(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_dcc_helper_core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            mux_params='clock mux parameters.',
            flop_params='divider flop parameters.',
            inv_params='inverter parameters.',
            vm_pitch='vm_layer pin pitch',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(vm_pitch=0.5)

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, flip_tile=True)

        mux_params: Param = self.params['mux_params']
        flop_params: Param = self.params['flop_params']
        inv_params: Param = self.params['inv_params']
        vm_pitch: HalfInt = HalfInt.convert(self.params['vm_pitch'])

        vm_pitch_htr = vm_pitch.dbl_value
        # create masters
        mux_params = mux_params.copy(append=dict(pinfo=pinfo, vertical_sel=False))
        flop_params = flop_params.copy(append=dict(pinfo=pinfo, has_rstlb=True,
                                                   swap_outbuf=True, out_pitch=vm_pitch))
        mux_master: Mux2to1Matched = self.new_template(Mux2to1Matched, params=mux_params)
        flop_master: FlopStrongArm = self.new_template(FlopStrongArm, params=flop_params)

        inv_in_tidx = mux_master.get_port('in<0>').get_pins()[0].track_id.base_index
        inv_params = inv_params.copy(append=dict(pinfo=pinfo, sig_locs={'in': inv_in_tidx}))
        inv_master: InvCore = self.new_template(InvCore, params=inv_params)

        # placement
        min_sep = self.min_sep_col
        min_sep += (min_sep & 1)
        buf_ncol = inv_master.num_cols
        mux_ncol = mux_master.num_cols
        flop_ncol = flop_master.num_cols
        ncol_tot = max(flop_ncol, mux_ncol + 2 * min_sep + 2 * buf_ncol)
        mux_off = (ncol_tot - mux_ncol) // 2
        flop_off = (ncol_tot - flop_ncol) // 2

        mux_in = self.add_tile(mux_master, 0, mux_off)
        flop = self.add_tile(flop_master, 1, flop_off)
        mux_out = self.add_tile(mux_master, 3, mux_off)
        invl = self.add_tile(inv_master, 3, 0)
        invr = self.add_tile(inv_master, 3, ncol_tot, flip_lr=True)
        self.set_mos_size()
        yh = self.bound_box.yh
        xm = self.bound_box.xm

        # routing
        vm_layer = self.conn_layer + 2
        grid = self.grid
        tr_manager = self.tr_manager
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        vm_w_ck = tr_manager.get_width(vm_layer, 'clk')

        # supplies
        vdd_list = []
        vss_list = []
        for inst in [mux_in, flop, mux_out, invl, invr]:
            vdd_list.extend(inst.port_pins_iter('VDD'))
            vss_list.extend(inst.port_pins_iter('VSS'))
        self.add_pin('VDD', self.connect_wires(vdd_list), connect=True)
        self.add_pin('VSS', self.connect_wires(vss_list), connect=True)

        # input mux
        flop_clk = mux_in.get_pin('out')
        self.connect_to_track_wires(flop_clk, flop.get_pin('clk'))
        in0 = mux_in.get_pin('in<0>')
        in1 = mux_in.get_pin('in<1>')
        clk_vm_tidx = flop_clk.track_id.base_index
        in_vm_test = tr_manager.get_next_track(vm_layer, clk_vm_tidx, 'sig', 'clk', up=True)
        in_dhtr = (-(in_vm_test - clk_vm_tidx).dbl_value // vm_pitch_htr) * vm_pitch_htr
        in_delta = HalfInt(in_dhtr)
        in0 = self.connect_to_tracks(in0, TrackID(vm_layer, clk_vm_tidx + in_delta, width=vm_w_ck),
                                     min_len_mode=MinLenMode.LOWER)
        in1 = self.connect_to_tracks(in1, TrackID(vm_layer, clk_vm_tidx - in_delta, width=vm_w_ck),
                                     min_len_mode=MinLenMode.LOWER)
        self.add_pin('launch', in0)
        self.add_pin('measure', in1)

        # divider flop
        clkp = flop.get_pin('outp')
        clkn = flop.get_pin('outn')
        self.connect_differential_wires(mux_in.get_pin('nsel'), mux_in.get_pin('nselb'),
                                        clkp, clkn)
        self.connect_to_track_wires(clkp, mux_in.get_pin('psel'))
        self.connect_to_track_wires(clkn, mux_in.get_pin('pselb'))
        self.connect_to_track_wires(flop.get_pin('inn'), clkp)
        self.connect_to_track_wires(flop.get_pin('inp'), clkn)
        self.connect_to_tracks(mux_out.get_pin('in<0>'), clkp.track_id, track_upper=yh,
                               track_lower=clkp.lower)
        self.reexport(flop.get_port('rstlb'))
        self.reexport(flop.get_port('rstlb_vm_l'))
        self.reexport(flop.get_port('rsthb'))

        # output mux
        clkn_tid = clkn.track_id
        vm_sp_le = grid.get_line_end_space(vm_layer, clkn_tid.width)
        clk_dcd = self.connect_to_tracks(mux_out.get_pin('in<1>'), clkn_tid,
                                         track_lower=clkn.upper + vm_sp_le, track_upper=yh)
        clk_out = self.extend_wires(mux_out.get_pin('out'), upper=yh)
        self.add_pin('clk_dcd', clk_dcd)
        self.add_pin('clk_out', clk_out)

        # output mux select signals
        out_sel = invl.get_pin('out')
        out_selb = invr.get_pin('out')
        dcc_byp_delta = tr_manager.get_next_track(vm_layer, out_selb.track_id.base_index,
                                                  'sig', 'sig', up=True) - clk_vm_tidx
        dcc_byp_dhtr = -(-dcc_byp_delta.dbl_value // vm_pitch_htr) * vm_pitch_htr
        dcc_byp_delta = HalfInt(dcc_byp_dhtr)
        dcc_byp_tidx = clk_vm_tidx + dcc_byp_delta
        out_selb_vm_tidx = clk_vm_tidx - dcc_byp_delta

        self.connect_to_track_wires(mux_out.get_pin('psel'), out_sel)
        self.connect_to_track_wires(mux_out.get_pin('pselb'), out_selb)
        nsel, nselb = self.connect_differential_wires(out_sel, out_selb,
                                                      mux_out.get_pin('nsel'),
                                                      mux_out.get_pin('nselb'))
        sel_hm_list = []
        selb_vm = self.connect_to_tracks([nselb, invl.get_pin('in')],
                                         TrackID(vm_layer, out_selb_vm_tidx, width=vm_w),
                                         track_upper=yh, ret_wire_list=sel_hm_list)
        # get differential matching
        xl = min((wire.lower for wire in sel_hm_list))
        xh = 2 * xm - xl
        self.extend_wires([nsel, nselb], lower=xl, upper=xh)

        dcc_byp = self.connect_to_tracks(invr.get_pin('in'),
                                         TrackID(vm_layer, dcc_byp_tidx, width=vm_w),
                                         track_lower=selb_vm.lower, track_upper=yh)
        self.add_pin('dcc_byp', dcc_byp)

        self.sch_params = dict(
            mux_params=mux_master.sch_params,
            flop_params=flop_master.sch_params,
            inv_params=inv_master.sch_params,
        )

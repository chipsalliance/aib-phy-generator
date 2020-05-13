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

from typing import Any, Dict, Optional, Type, List, Union

from pybag.enum import RoundMode, MinLenMode

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID, WireArray
from bag.design.database import Module

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from bag3_digital.layout.stdcells.gates import InvChainCore
from bag3_digital.layout.sampler.flop_strongarm import FlopStrongArm

from ..schematic.aib_dcc_helper import aib_ams__aib_dcc_helper
from .dcc_helper_core import DCCHelperCore


class SyncChain(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._buf_ncol = 0

    @property
    def buf_ncol(self) -> int:
        return self._buf_ncol

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            sync_params='synchronizer flop parameters.',
            buf_params='clock buffer parameters.',
            nsync='Number of synchronizer flops.',
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        sync_params: Param = self.params['sync_params']
        buf_params: Param = self.params['buf_params']
        nsync: int = self.params['nsync']

        if nsync < 1:
            raise ValueError('nsync must be positive.')
        if nsync & 1:
            raise ValueError('nsync must be even.')
        nsync2 = nsync // 2

        # create masters
        sync_params = sync_params.copy(append=dict(pinfo=pinfo, has_rstlb=True))
        buf_params = buf_params.copy(append=dict(pinfo=pinfo))

        sync_master: FlopStrongArm = self.new_template(FlopStrongArm, params=sync_params)
        buf_master: InvChainCore = self.new_template(InvChainCore, params=buf_params)

        # placement
        conn_layer = self.conn_layer
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        grid = self.grid
        tr_manager = self.tr_manager
        xm_w = tr_manager.get_width(xm_layer, 'sig')
        xm_w_clk = tr_manager.get_width(xm_layer, 'clk')

        min_sep = self.min_sep_col
        min_sep += (min_sep & 1)
        sub_sep = self.sub_sep_col
        sub_sep2 = sub_sep // 2
        sync_ncol = sync_master.num_cols
        self._buf_ncol = buf_master.num_cols
        sup_info = self.get_supply_column_info(xm_layer)
        tap_ncol = sup_info.ncol
        tap_off = self._buf_ncol + sub_sep2
        sync_off = tap_off + tap_ncol + sub_sep2 + sync_ncol
        sync_delta = sync_ncol + min_sep
        ncol_tot = sync_off + (nsync2 - 1) * sync_delta

        # add buffer and set sized
        buf = self.add_tile(buf_master, 0, 0)
        self.set_mos_size(num_cols=ncol_tot, num_tiles=5)

        # add flops
        bot_pins = self._draw_sync_row(sync_master, 0, nsync2, sync_off, sync_delta, True)
        top_pins = self._draw_sync_row(sync_master, 2, nsync2, ncol_tot, -sync_delta, False)

        # add supply connections
        lay_range = range(conn_layer, xm_layer + 1)
        vdd_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        vss_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        for tile_idx in range(self.num_tile_rows):
            self.add_supply_column(sup_info, tap_off, vdd_table, vss_table, ridx_p=-1,
                                   ridx_n=0, tile_idx=tile_idx)

        # routing
        # supplies
        vdd_hm_list = bot_pins['VDD']
        vdd_hm_list.extend(top_pins['VDD'])
        vss_hm_list = bot_pins['VSS']
        vss_hm_list.extend(top_pins['VSS'])
        vdd_hm_list.extend(buf.port_pins_iter('VDD'))
        vss_hm_list.extend(buf.port_pins_iter('VSS'))
        vdd_hm_list = self.connect_wires(vdd_hm_list)
        vss_hm_list = self.connect_wires(vss_hm_list)
        for lay in range(hm_layer, xm_layer + 1, 2):
            vss = vss_table[lay]
            vdd = vdd_table[lay]
            if lay == hm_layer:
                vss.extend(vss_hm_list)
                vdd.extend(vdd_hm_list)
            vdd = self.connect_wires(vdd)
            vss = self.connect_wires(vss)
            self.add_pin(f'VDD_{lay}', vdd)
            self.add_pin(f'VSS_{lay}', vss)

        # datapath
        self.connect_to_track_wires(bot_pins['outp'], top_pins['inp'])
        self.connect_to_track_wires(bot_pins['outn'], top_pins['inn'])
        self.connect_to_track_wires(bot_pins['inp'], vdd_hm_list[0])
        self.connect_to_track_wires(bot_pins['inn'], vss_hm_list[0])
        self.add_pin('VSS_bot', vss_hm_list[0])
        outp = top_pins['outp']
        xm_tidx = grid.coord_to_track(xm_layer, outp.middle, mode=RoundMode.NEAREST)
        out = self.connect_to_tracks(outp, TrackID(xm_layer, xm_tidx, width=xm_w),
                                     min_len_mode=MinLenMode.UPPER)
        self.add_pin('out', out)

        # clk
        clk_list = bot_pins['clk']
        clk_list.extend(top_pins['clk'])
        clk = self.connect_wires(clk_list)
        buf_out = buf.get_pin('out')
        xm_tidx = grid.coord_to_track(xm_layer, buf_out.middle, mode=RoundMode.NEAREST)
        buf_out = self.connect_to_tracks(buf_out, TrackID(xm_layer, xm_tidx, width=xm_w_clk),
                                         min_len_mode=MinLenMode.UPPER)
        self.add_pin('clk_buf', buf_out)
        self.add_pin('clk_sync', clk)
        self.reexport(buf.get_port('in'), net_name='clk_in')

        # rstlb
        rstlb_list = [bot_pins['rstlb'], top_pins['rstlb']]
        self.add_pin('rstlb', rstlb_list)

        self.sch_params = dict(
            sync_params=sync_master.sch_params,
            buf_params=buf_master.sch_params.copy(remove=['dual_output']),
            nsync=nsync,
        )

    def _draw_sync_row(self, master: FlopStrongArm, tile: int, num: int, cur_col: int, delta: int,
                       in_vm: bool) -> Dict[str, Union[WireArray, List[WireArray]]]:
        vm_layer = self.conn_layer + 2
        xm_layer = vm_layer + 1
        grid = self.grid
        tr_manager = self.tr_manager
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        xm_w = tr_manager.get_width(xm_layer, 'sig')

        outp_prev = outn_prev = None
        ans = {}
        rstlb_list = []
        clk_list = []
        vdd_list = []
        vss_list = []
        xmt = xmb = 0
        for col in range(cur_col, cur_col + num * delta, delta):
            inst = self.add_tile(master, tile, col, flip_lr=True)
            rstlb_list.append(inst.get_pin('rstlb'))
            vdd_list.extend(inst.port_pins_iter('VDD'))
            vss_list.extend(inst.port_pins_iter('VSS'))
            clk = inst.get_pin('clk')
            vm_tidx = grid.coord_to_track(vm_layer, clk.middle, mode=RoundMode.NEAREST)
            clk_list.append(self.connect_to_tracks(clk, TrackID(vm_layer, vm_tidx, width=vm_w)))
            rstlbr = inst.get_pin('rstlb_vm_l')
            rstlbl = inst.get_pin('rsthb')
            if outp_prev is None:
                outp_prev = inst.get_pin('outp')
                outn_prev = inst.get_pin('outn')
                loc_list = tr_manager.place_wires(xm_layer, ['sig', 'sig', 'sig'],
                                                  center_coord=outp_prev.middle)[1]
                xmb = loc_list[0]
                xmt = loc_list[2]
                if in_vm:
                    vml = tr_manager.get_next_track(vm_layer, rstlbl.track_id.base_index,
                                                    'sig', 'sig', up=-2)
                    vmr = tr_manager.get_next_track(vm_layer, rstlbr.track_id.base_index,
                                                    'sig', 'sig', up=2)
                    inp = self.connect_to_tracks(inst.get_pin('inp'),
                                                 TrackID(vm_layer, vml, width=vm_w))
                    inn = self.connect_to_tracks(inst.get_pin('inn'),
                                                 TrackID(vm_layer, vmr, width=vm_w))
                    ans['inp'] = inp
                    ans['inn'] = inn
                else:
                    ans['inp'] = inst.get_pin('inp')
                    ans['inn'] = inst.get_pin('inn')
            else:
                inp = inst.get_pin('inp')
                inn = inst.get_pin('inn')
                vml = tr_manager.get_next_track(vm_layer, rstlbl.track_id.base_index,
                                                'sig', 'sig', up=False)
                vmr = tr_manager.get_next_track(vm_layer, rstlbr.track_id.base_index,
                                                'sig', 'sig', up=True)
                inp = self.connect_to_tracks(inp, TrackID(vm_layer, vml, width=vm_w))
                inn = self.connect_to_tracks(inn, TrackID(vm_layer, vmr, width=vm_w))
                inp, inn = self.connect_differential_tracks(inp, inn, xm_layer, xmt, xmb,
                                                            width=xm_w)
                self.connect_differential_wires(outp_prev, outn_prev, inp, inn)

                outp_prev = inst.get_pin('outp')
                outn_prev = inst.get_pin('outn')

        ans['outp'] = outp_prev
        ans['outn'] = outn_prev
        ans['clk'] = clk_list
        ans['rstlb'] = self.connect_wires(rstlb_list)[0]
        ans['VDD'] = self.connect_wires(vdd_list)
        ans['VSS'] = self.connect_wires(vss_list)
        return ans


class DCCHelper(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_dcc_helper

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            core_params='DCCHelperCore parameters.',
            sync_params='synchronizer flop parameters.',
            buf_params='clock buffer parameters.',
            nsync='Number of synchronizer flops.',
            vm_pitch='vm_layer pin pitch',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(vm_pitch=0.5)

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        core_params: Param = self.params['core_params']
        sync_params: Param = self.params['sync_params']
        buf_params: Param = self.params['buf_params']
        nsync: int = self.params['nsync']
        vm_pitch: HalfInt = HalfInt.convert(self.params['vm_pitch'])

        # create masters
        core_params = core_params.copy(append=dict(pinfo=pinfo, vm_pitch=vm_pitch))
        half_params = dict(pinfo=pinfo, sync_params=sync_params, buf_params=buf_params,
                           nsync=nsync)

        core_master: DCCHelperCore = self.new_template(DCCHelperCore, params=core_params)
        sync_master: SyncChain = self.new_template(SyncChain, params=half_params)
        buf_ncol = sync_master.buf_ncol

        # placement
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        vm_w = self.tr_manager.get_width(vm_layer, 'sig')

        min_sep = self.min_sep_col
        min_sep += (min_sep & 1)
        core_ncol = core_master.num_cols
        sync_ncol = sync_master.num_cols
        center_ncol = max(2 * buf_ncol + min_sep, core_ncol)
        half_sep = center_ncol - 2 * buf_ncol
        core_off = sync_ncol - buf_ncol + (center_ncol - core_ncol) // 2

        # add masters
        core = self.add_tile(core_master, 1, core_off)
        syncl = self.add_tile(sync_master, 0, sync_ncol, flip_lr=True)
        syncr = self.add_tile(sync_master, 0, sync_ncol + half_sep)
        self.set_mos_size(num_cols=2 * sync_ncol + half_sep)
        bbox = self.bound_box
        xl = bbox.xl
        xh = bbox.xh
        xm = (xl + xh) // 2

        # routing
        # supplies
        for lay in [hm_layer, xm_layer]:
            for name in ['VDD', 'VSS']:
                cur_name = f'{name}_{lay}'
                warrs = syncl.get_all_port_pins(cur_name)
                warrs.extend(syncr.port_pins_iter(cur_name))
                warrs = self.connect_wires(warrs, lower=xl, upper=xh)
                if lay == xm_layer:
                    self.add_pin(name, warrs)

        # rstb for synchronizers
        grid = self.grid
        vm_p_htr = vm_pitch.dbl_value
        vm_center = grid.coord_to_track(vm_layer, xm)
        vm_delta = grid.coord_to_track(vm_layer, xh, mode=RoundMode.LESS_EQ) - vm_center
        vm_dhtr = -(-vm_delta.dbl_value // vm_p_htr) * vm_p_htr
        vm_delta = HalfInt(vm_dhtr)
        rstlb_r_tidx = vm_center + vm_delta
        rstlb_l_tidx = vm_center - vm_delta
        rstlbr_list = syncr.get_all_port_pins('rstlb')
        vss_bot = syncr.get_pin('VSS_bot')
        rstlbr_list.append(vss_bot)
        rstb_ref = self.connect_to_tracks(rstlbr_list, TrackID(vm_layer, rstlb_r_tidx, width=vm_w))
        rstlbl_list = syncl.get_all_port_pins('rstlb')
        rstb = self.connect_to_tracks(rstlbl_list, TrackID(vm_layer, rstlb_l_tidx, width=vm_w),
                                      track_lower=rstb_ref.lower)
        self.add_pin('rstb', rstb)

        # clk for synchronizers
        clk_ref = self.connect_to_track_wires(vss_bot, syncr.get_pin('clk_sync'))
        clkl = self.extend_wires(syncl.get_pin('clk_sync'), lower=clk_ref.lower)
        clk_ref = self.connect_to_track_wires(clkl, syncl.get_pin('clk_buf'))
        self.extend_wires(syncr.get_pin('clk_buf'), upper=2 * xm - clk_ref.lower)

        # rstb_sync
        warr_list = []
        rstb_sync = syncl.get_pin('out')
        rstb_ref = self.connect_to_tracks(core.get_pin('rstlb_vm_l'), rstb_sync.track_id,
                                          ret_wire_list=warr_list)
        self.extend_wires(core.get_pin('rsthb'), upper=warr_list[0].upper)
        rstb_ref = self.connect_wires([rstb_ref, syncl.get_pin('out')])[0]
        self.extend_wires(syncr.get_pin('out'), lower=2 * xm - rstb_ref.upper)

        # input clocks
        launch = self.connect_to_track_wires(syncl.get_pin('clk_in'), core.get_pin('launch'))
        measure = self.connect_to_track_wires(syncr.get_pin('clk_in'), core.get_pin('measure'))

        self.add_pin('launch', self.extend_wires(launch, lower=0))
        self.add_pin('measure', self.extend_wires(measure, lower=0))

        # reexports
        self.reexport(core.get_port('clk_out'), net_name='ckout')
        self.reexport(core.get_port('clk_dcd'))
        self.reexport(core.get_port('dcc_byp'))

        self.sch_params = sync_master.sch_params.copy(append=dict(
            core_params=core_master.sch_params))

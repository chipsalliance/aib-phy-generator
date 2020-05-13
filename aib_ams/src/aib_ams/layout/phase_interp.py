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

from pybag.enum import RoundMode, MinLenMode

from bag.util.immutable import Param
from bag.design.module import Module
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.enum import MOSWireType

from bag3_analog.layout.phase.phase_interp import PhaseInterpolator
from bag3_digital.layout.stdcells.gates import InvCore, NAND2Core

from aib_ams.layout.delay_line import DelayCellCore

from ..schematic.aib_phase_interp import aib_ams__aib_phase_interp


class PhaseInterpolatorWithDelay(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_phase_interp

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The layout information object.',
            pi_params='Phase interpolator parameters.',
            dc_params='delay cell parameters.',
            inv_params='Inverter input buffer params',
            nand_params='Nand input buffer params',
            num_core='Number of delay cell cores',
            nbits='number of control bits.',
            export_dc_out='export delay cell output',
            export_dc_in='export the delay cell input',
            pi_tile='Phase interpolator tile index',
            dc_tile='Delay cell tile index',
            split_nand='Split the nand across two rows',
            split_inv='Split the inverter across two rows'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(export_dc_out=False, pi_tile=0, dc_tile=4, split_nand=True, split_inv=True)

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        pi_params: Param = self.params['pi_params']
        dc_params: Param = self.params['dc_params']
        inv_params: Param = self.params['inv_params']
        nand_params: Param = self.params['nand_params']
        ncores: int = self.params['num_core']
        nbits: int = self.params['nbits']
        split_nand: bool = self.params['split_nand']
        split_inv: bool = self.params['split_inv']
        pi_tile: int = self.params['pi_tile']
        dc_tile: int = self.params['dc_tile']
        inv_tile_l = dc_tile + 1
        inv_tile_h = dc_tile + 3

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        grid = self.grid
        tr_manager = self.tr_manager

        # Recompute sizes for the inverter and nand
        inv_seg: int = inv_params['seg']
        nand_seg: int = nand_params['seg']
        split_inv = split_inv and (inv_seg > 1)
        split_nand = split_nand and (nand_seg > 1)
        if split_inv:
            inv_seg_2 = inv_seg // 2
            inv_seg_1 = inv_seg - inv_seg_2
        else:
            inv_seg_1 = inv_seg
            inv_seg_2 = None
        if split_nand:
            nand_seg_2 = nand_seg // 2
            nand_seg_1 = nand_seg - nand_seg_2
        else:
            nand_seg_1 = nand_seg
            nand_seg_2 = None

        # create masters
        low_pinfo = self.get_tile_pinfo(inv_tile_l)
        high_pinfo = self.get_tile_pinfo(inv_tile_h)
        pi_params = pi_params.copy(append=dict(pinfo=pinfo, nbits=nbits + 1, flip_b_en=True))
        dc_params = dc_params.copy(append=dict(pinfo=pinfo, substrate_row=True, vertical_out=True,
                                               draw_substrate_row=False, tile0=3, tile1=1,
                                               flip_vm=True))
        pi_master: PhaseInterpolator = self.new_template(PhaseInterpolator, params=pi_params)
        dc_master: DelayCellCore = self.new_template(DelayCellCore, params=dc_params)
        dc0_params = dc_params.copy(append=dict(feedback=True))
        dc0_master: DelayCellCore = self.new_template(DelayCellCore, params=dc0_params)

        # Place all the cells
        ntr, vert_trs = tr_manager.place_wires(vm_layer, ['sig', 'sig', 'sup'])
        ncol = self.arr_info.get_column_span(vm_layer, ntr)
        pi = self.add_tile(pi_master, pi_tile, ncol)
        dc_arr = [self.add_tile(dc0_master, dc_tile, 0)]
        sep = max(self.min_sep_col, self.get_hm_sp_le_sep_col())
        curr_col = dc_master.num_cols + sep
        for _ in range(ncores - 1):
            dc_arr.append(self.add_tile(dc_master, dc_tile, curr_col))
            curr_col += dc_master.num_cols + sep

        nand_params_l = nand_params.copy(append=dict(pinfo=low_pinfo, vertical_sup=True,
                                                     vertical_out=False, seg=nand_seg_1))
        nand_master_l: NAND2Core = self.new_template(NAND2Core, params=nand_params_l)
        nand_l = self.add_tile(nand_master_l, inv_tile_l, curr_col)
        if split_nand:
            nand_params_h = nand_params.copy(append=dict(pinfo=high_pinfo, vertical_sup=True,
                                                         vertical_out=False, seg=nand_seg_2))
            nand_master_h: NAND2Core = self.new_template(NAND2Core, params=nand_params_h)
            inv_in_tid_h = nand_master_h.get_track_index(1, MOSWireType.G, wire_name='sig',
                                                         wire_idx=0)
            nand_h = self.add_tile(nand_master_h, inv_tile_h, curr_col)
            nand_sch_params = [nand_master_l.sch_params, nand_master_h.sch_params]
        else:
            inv_in_tid_h = None
            nand_h = None
            nand_sch_params = [nand_master_l.sch_params]
        inv_col = curr_col + nand_master_l.num_cols + sep
        inv_in_tid = nand_master_l.get_track_index(1, MOSWireType.G, wire_name='sig', wire_idx=0)
        inv_params_l = inv_params.copy(append=dict(pinfo=low_pinfo, vertical_sup=True,
                                                   seg=inv_seg_1, vertical_out=False,
                                                   sig_locs=dict(nin=inv_in_tid)))
        inv_master_l: InvCore = self.new_template(InvCore, params=inv_params_l)
        inv_l = self.add_tile(inv_master_l, inv_tile_l, inv_col)
        if split_inv:
            if inv_in_tid_h:
                inv_params_h = inv_params.copy(append=dict(pinfo=high_pinfo, vertical_sup=True,
                                                           seg=inv_seg_2, vertical_out=False,
                                                           sig_locs=dict(nin=inv_in_tid_h)))
            else:
                inv_params_h = inv_params.copy(append=dict(pinfo=high_pinfo, vertical_sup=True,
                                                           seg=inv_seg_2, vertical_out=False))
            inv_master_h: InvCore = self.new_template(InvCore, params=inv_params_h)
            inv_h = self.add_tile(inv_master_h, inv_tile_h, inv_col)
            inv_sch_params = [inv_master_l.sch_params, inv_master_h.sch_params]
        else:
            inv_h = None
            inv_sch_params = [inv_master_l.sch_params]

        # substrate taps
        vss_hm = [pi.get_pin('VSS0'), pi.get_pin('VSS1'),
                  self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=8)]
        vdd_hm = [pi.get_pin('VDD_hm'), self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=6)]
        ncol_tot = self.num_cols
        sub = self.add_substrate_contact(0, 0, tile_idx=0, seg=ncol_tot)
        self.connect_to_track_wires(sub, vss_hm[0])
        sub = self.add_substrate_contact(0, 0, tile_idx=2, seg=ncol_tot)
        self.connect_to_track_wires(sub, vdd_hm[0])
        sub = self.add_substrate_contact(0, 0, tile_idx=4, seg=ncol_tot)
        self.connect_to_track_wires(sub, vss_hm[1])
        sub = self.add_substrate_contact(0, 0, tile_idx=6, seg=ncol_tot)
        sub_vdd1 = self.connect_to_tracks(sub, vdd_hm[1])
        vdd_hm[1] = sub_vdd1
        sub = self.add_substrate_contact(0, 0, tile_idx=8, seg=ncol_tot)
        sub_vss2 = self.connect_to_tracks(sub, vss_hm[2])
        vss_hm[2] = sub_vss2

        self.set_mos_size()
        xh = self.bound_box.xh

        # routing
        xm_w = tr_manager.get_width(xm_layer, 'sig')
        xm_w_sup = tr_manager.get_width(xm_layer, 'sup')
        xm_w_hs = tr_manager.get_width(xm_layer, 'sig_hs')
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        vm_w_sup = tr_manager.get_width(vm_layer, 'sup')
        vm_sep_sup = tr_manager.get_sep(vm_layer, ('sup', 'sup'))

        xm_vss_tids, xm_vdd_tids = [], []
        for tid_arr, sup_hm_arr in [(xm_vss_tids, vss_hm), (xm_vdd_tids, vdd_hm)]:
            for sup_w in sup_hm_arr:
                y = grid.track_to_coord(hm_layer, sup_w.track_id.base_index)
                tid_arr.append(TrackID(xm_layer,
                                       grid.coord_to_track(xm_layer, y, mode=RoundMode.NEAREST),
                                       width=xm_w_sup))
        tr_list = ['sup'] + ['sig'] * (nbits + 1)
        pidx_list_l = tr_manager.place_wires(xm_layer, tr_list,
                                             align_track=xm_vss_tids[0].base_index, align_idx=0)[1]
        pidx_list_h = tr_manager.place_wires(xm_layer, tr_list,
                                             align_track=xm_vss_tids[1].base_index, align_idx=0)[1]
        tr_list.reverse()
        nidx_list_l = tr_manager.place_wires(xm_layer, tr_list,
                                             align_track=xm_vss_tids[1].base_index, align_idx=-1)[1]
        nidx_list_h = tr_manager.place_wires(xm_layer, tr_list,
                                             align_track=xm_vss_tids[2].base_index, align_idx=-1)[1]
        io_idx_list = tr_manager.place_wires(xm_layer, ['sig_hs', 'sup', 'sig_hs'],
                                             align_track=xm_vdd_tids[0].base_index, align_idx=1)[1]
        int_idx_list = tr_manager.place_wires(xm_layer, ['sig_hs', 'sup', 'sig_hs'],
                                              align_track=xm_vdd_tids[1].base_index, align_idx=1)[1]

        # PI signals
        vss_list = pi.get_all_port_pins('VSS')
        vdd_list = pi.get_all_port_pins('VDD')
        suf = f'<{nbits}>'
        en_vss = self.connect_to_track_wires(vss_hm[0:2], pi.get_pin('a_enb' + suf))
        en_l = en_vss.lower
        en_u = en_vss.upper
        self.connect_to_tracks(vdd_hm[0], pi.get_pin('a_en' + suf).track_id,
                               track_lower=en_l, track_upper=en_u)
        en_xm_upper = None
        for idx in range(nbits):
            suf = f'<{idx}>'
            enb = self.connect_wires([pi.get_pin('a_en' + suf), pi.get_pin('b_enb' + suf)],
                                     lower=en_l, upper=en_u)
            en = self.connect_wires([pi.get_pin('a_enb' + suf), pi.get_pin('b_en' + suf)],
                                    lower=en_l, upper=en_u)
            ptid = TrackID(xm_layer, pidx_list_l[1 + idx], width=xm_w)
            ntid = TrackID(xm_layer, nidx_list_l[-idx - 2], width=xm_w)
            if idx == 0:
                en = self.connect_to_tracks(en, ptid, track_lower=0)
                en_xm_upper = en.upper
            else:
                en = self.connect_to_tracks(en, ptid, track_lower=0, track_upper=en_xm_upper)
            enb = self.connect_to_tracks(enb, ntid, track_lower=0, track_upper=en_xm_upper)
            self.add_pin('sp' + suf, en)
            self.add_pin('sn' + suf, enb)

        intout = self.connect_to_tracks(pi.get_pin('out'),
                                        TrackID(xm_layer, io_idx_list[2], width=xm_w_hs),
                                        track_lower=0)
        self.add_pin('intout', intout)
        vdd0 = self.connect_to_tracks(vdd_list, xm_vdd_tids[0], track_lower=0, track_upper=xh)
        vss0 = self.connect_to_tracks(vss_list, xm_vss_tids[0], track_lower=0, track_upper=xh)
        vss1 = self.connect_to_tracks(vss_list, xm_vss_tids[1], track_lower=0, track_upper=xh)

        # delay cell signals
        vdd_list, vss0_list, vss1_list = [], [], []
        for dc in dc_arr:
            vdd_list += dc.get_all_port_pins('VDD')
            vss0_list += dc.get_all_port_pins('VSS1')
            vss1_list += dc.get_all_port_pins('VSS0')
        vdd_list.extend(nand_l.get_all_port_pins('VDD'))
        vss0_list.extend(nand_l.get_all_port_pins('VSS'))
        if split_nand:
            vdd_list.extend(nand_h.get_all_port_pins('VDD'))
            vss1_list.extend(nand_h.get_all_port_pins('VSS'))
        vss0_list.extend(inv_l.get_all_port_pins('VSS'))
        vdd_list.extend(inv_l.get_all_port_pins('VDD'))
        if split_inv:
            vdd_list.extend(inv_h.get_all_port_pins('VDD'))
            vss1_list.extend(inv_h.get_all_port_pins('VSS'))
        vdd_list.extend([dc.get_pin('bk1_vm') for dc in dc_arr])
        self.connect_to_track_wires(vdd_list, vdd_hm[1])
        self.connect_to_track_wires(vss0_list, vss_hm[1])
        self.connect_to_track_wires(vss1_list, vss_hm[2])
        if ncores > 1:
            for idx in range(ncores-1, 0, -1):
                cur_cop = dc_arr[idx].get_pin('co_p')
                next_in = dc_arr[idx-1].get_pin('in_p')
                ptid = TrackID(xm_layer, pidx_list_h[1 + (idx % 2)], width=xm_w)
                self.connect_to_tracks([cur_cop, next_in], ptid)

                cur_ci = dc_arr[idx].get_pin('ci_p')
                next_out = dc_arr[idx-1].get_pin('out_p')
                ntid = TrackID(xm_layer, nidx_list_h[-2-(idx % 2)], width=xm_w)
                self.connect_to_tracks([cur_ci, next_out], ntid)

        # NAND to delay cell and PI
        nand_out_vm_tr = self.grid.coord_to_track(vm_layer, nand_l.get_pin('pout').middle,
                                                  RoundMode.NEAREST)
        nand_out_pins = [nand_l.get_pin('pout'), nand_l.get_pin('nout')]
        if split_nand:
            nand_out_pins.extend([nand_h.get_pin('pout'), nand_h.get_pin('nout')])
        nand_out_vm_w = self.connect_to_tracks(nand_out_pins,
                                               TrackID(vm_layer, nand_out_vm_tr, vm_w))
        a_in_buf_xm = self.connect_to_tracks([nand_out_vm_w, dc_arr[-1].get_pin('in_p')],
                                             TrackID(xm_layer, int_idx_list[0], xm_w))
        a_in_buf_vm = self.connect_to_tracks([a_in_buf_xm, pi.get_pin('a_in')],
                                             TrackID(vm_layer, vert_trs[0], vm_w))

        # Delay Cell to PI
        b_in_xm = self.connect_to_tracks(dc_arr[-1].get_pin("out_p"),
                                         TrackID(xm_layer, int_idx_list[-1], xm_w))
        b_in_vm = self.connect_to_tracks([b_in_xm, pi.get_pin("b_in")],
                                         TrackID(vm_layer, vert_trs[1], vm_w))

        nand_in_hm = [nand_l.get_pin("nin<0>"), vdd_hm[1]]
        if split_nand:
            nand_in_hm.append(nand_h.get_pin("nin<0>"))
        nand_in_vm_tr = tr_manager.get_next_track_obj(nand_out_vm_w.track_id, 'sig', 'sig')
        self.connect_to_tracks(nand_in_hm, nand_in_vm_tr)

        # Inv out to nand in
        inv_out_tr = grid.coord_to_track(vm_layer, inv_l.get_pin("pout").middle, RoundMode.NEAREST)
        inv_out_hm_w = [inv_l.get_pin("pout"), inv_l.get_pin("nout"), nand_l.get_pin("nin<1>")]
        inv_in_tr = tr_manager.get_next_track(vm_layer, inv_out_tr, 'sig', 'sig')
        inv_in_hm_w = [inv_l.get_pin('in')]
        if split_inv:
            inv_out_hm_w.extend([inv_h.get_pin("nout"), inv_h.get_pin("pout")])
            inv_in_hm_w.append(inv_h.get_pin("in"))
        if split_nand:
            inv_out_hm_w.append(nand_h.get_pin("nin<1>"))
        inv_out_w = self.connect_to_tracks(inv_out_hm_w, TrackID(vm_layer, inv_out_tr, vm_w))
        a_in = self.connect_to_tracks(inv_in_hm_w, TrackID(vm_layer, inv_in_tr, vm_w))
        self.add_pin('a_in', a_in)

        # Supply routing over nand, dc, and inv
        vm_wires = [nand_out_vm_w, inv_out_w]
        for dc in dc_arr:
            for port_name in dc.port_names_iter():
                if port_name not in ['VDD', 'VSS0', 'VSS1']:
                    vm_wires.extend(dc.get_all_port_pins(port_name, vm_layer))
        # for idx, wire in enumerate(vm_wires):
        #     self.add_pin(f'vm_wire_{idx}', wire)
        vss_vm_arr = pi.get_all_port_pins('VSS')[0]
        vdd_vm_arr = pi.get_all_port_pins('VDD')[0]
        vss_vm, vdd_vm = [], []
        for wire_arr, target_wire, vm_list in [(vss_vm_arr, vss_hm[2], vss_vm),
                                               (vdd_vm_arr, vdd_hm[1], vdd_vm)]:
            for wire in wire_arr.warr_iter():
                tr_idx = wire.track_id.base_index
                min_idx = tr_manager.get_next_track(vm_layer, tr_idx, 'sup', 'sig', up=False)
                max_idx = tr_manager.get_next_track(vm_layer, tr_idx, 'sup', 'sig', up=True)
                will_intersect = False
                for sig_wire in vm_wires:
                    if min_idx < sig_wire.track_id.base_index <= max_idx:
                        will_intersect = True
                if not will_intersect:
                    self.connect_to_tracks(wire, target_wire.track_id)
                    vm_list.append(wire)
        vdd1 = self.connect_to_tracks(vdd_vm, xm_vdd_tids[1], track_lower=0, track_upper=xh)
        vss2 = self.connect_to_tracks(vss_vm, xm_vss_tids[2], track_lower=0, track_upper=xh)
        self.add_pin('VDD', [vdd0, vdd1], connect=True)
        self.add_pin('VSS', [vss0, vss1, vss2], connect=True)

        tidx_list = self.get_available_tracks(vm_layer, 0, inv_in_tr, 0, self.bound_box.yh,
                                              width=vm_w_sup, sep=vm_sep_sup)
        for tidx in tidx_list:
            tid = TrackID(vm_layer, tidx, vm_w_sup)
            for wire in vdd_hm + [vss_hm[1], vdd0, vdd1, vss1]:
                self.connect_to_tracks(wire, tid, min_len_mode=MinLenMode.MIDDLE)
            for wire in [vss_hm[0], vss0]:
                self.connect_to_tracks(wire, tid, min_len_mode=MinLenMode.UPPER)
            for wire in [vss_hm[2], vss2]:
                self.connect_to_tracks(wire, tid, min_len_mode=MinLenMode.LOWER)

        if self.params['export_dc_in']:
            self.add_pin('a_in_buf', a_in_buf_vm)
        if self.params['export_dc_out']:
            self.add_pin('b_in', b_in_vm)

        # Set the schematic parameters
        self.sch_params = dict(
            pi_params=pi_master.sch_params,
            dc_params=dc_master.sch_params,
            inv_params=inv_sch_params,
            nand_params=nand_sch_params,
            num_core=ncores,
            export_dc_out=self.params['export_dc_out'],
            export_dc_in=self.params['export_dc_in'],
        )

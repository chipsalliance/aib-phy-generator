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

from typing import Any, Dict, Sequence, Optional, Union, Type, Mapping

from pybag.enum import RoundMode, MinLenMode

from bag.util.immutable import Param, ImmutableSortedDict
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.database import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ...schematic.mux2to1_matched import bag3_digital__mux2to1_matched
from .gates import InvTristateCore, InvCore


class Mux2to1MatchedHalf(MOSBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='segments dictionary.',
            w_dict='width dictionary.',
            ridx_n='nmos row index.',
            ridx_p='pmos row index.',
            vertical_sel='True to connect select signals to vertical layer.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_n=0,
            ridx_p=-1,
            vertical_sel=True,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        vertical_sel: bool = self.params['vertical_sel']

        w_dict = self._get_w_dict(ridx_n, ridx_p)

        seg_tri = seg_dict['tri']
        seg_buf = seg_dict['buf']
        w_pbuf = w_dict['pbuf']
        w_nbuf = w_dict['nbuf']
        w_ptri = w_dict['ptri']
        w_ntri = w_dict['ntri']

        if (seg_tri & 1) or (seg_buf % 4 != 0):
            raise ValueError('segments of transistors must be even, buf must be multiple of 4.')
        seg_buf_half = seg_buf // 2

        # placement
        vm_layer = self.conn_layer + 2
        grid = self.grid
        tr_manager = self.tr_manager
        vm_w = tr_manager.get_width(vm_layer, 'sig')

        vss_tid = self.get_track_id(ridx_n, False, wire_name='sup')
        vdd_tid = self.get_track_id(ridx_p, True, wire_name='sup')
        nd_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        pd_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0)
        ng2_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        pg0_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-2)
        pg1_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)
        in_tid = ng2_tid
        pmid_tid = pg0_tid
        psel_tid = pg1_tid

        tri_ncol = 2 * seg_tri
        min_sep = self.min_sep_col
        min_sep += (min_sep & 1)
        ntri = self.add_mos(ridx_n, 0, seg_tri, w=w_ntri, stack=2, g_on_s=True, sep_g=True)
        ptri = self.add_mos(ridx_p, 0, seg_tri, w=w_ptri, stack=2, g_on_s=True, sep_g=True)
        cur_col = tri_ncol + min_sep
        nbuf = self.add_mos(ridx_n, cur_col, seg_buf_half, w=w_nbuf)
        pbuf = self.add_mos(ridx_p, cur_col, seg_buf_half, w=w_pbuf)
        self.set_mos_size()

        # routing
        # supplies
        vdd = self.connect_to_tracks([ptri.s, pbuf.s], vdd_tid)
        vss = self.connect_to_tracks([ntri.s, nbuf.s], vss_tid)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        # select signals
        in_warr = self.connect_to_tracks([ntri.g[0::2], ptri.g[0::2]], in_tid)
        nsel = ntri.g[1::2]
        psel = self.connect_to_tracks(ptri.g[1::2], psel_tid, min_len_mode=MinLenMode.MIDDLE)
        vm_tidx = grid.coord_to_track(vm_layer, psel.middle, mode=RoundMode.GREATER_EQ)
        if vertical_sel:
            psel_vm = self.connect_to_tracks(psel, TrackID(vm_layer, vm_tidx, width=vm_w),
                                             min_len_mode=MinLenMode.LOWER)
            self.add_pin('psel_vm', psel_vm)
        self.add_pin('in', in_warr)
        self.add_pin('nsel', nsel)
        self.add_pin('psel', psel)

        # mid
        pmid = self.connect_to_tracks(ptri.d, pd_tid, min_len_mode=MinLenMode.UPPER)
        nmid = self.connect_to_tracks(ntri.d, nd_tid, min_len_mode=MinLenMode.UPPER)
        vm_tidx = tr_manager.get_next_track(vm_layer, vm_tidx, 'sig', 'sig', up=True)
        mid = self.connect_to_tracks([pmid, nmid], TrackID(vm_layer, vm_tidx, width=vm_w))
        mid = self.connect_to_tracks([mid, nbuf.g, pbuf.g], pmid_tid)
        self.add_pin('mid', mid)

        # output
        pout = self.connect_to_tracks(pbuf.d, pd_tid, min_len_mode=MinLenMode.UPPER)
        nout = self.connect_to_tracks(nbuf.d, nd_tid, min_len_mode=MinLenMode.UPPER)
        vm_tidx = grid.coord_to_track(vm_layer, self.bound_box.xh)
        out = self.connect_to_tracks([pout, nout], TrackID(vm_layer, vm_tidx, width=vm_w))
        self.add_pin('out', out)

        lch = self.arr_info.lch
        th_p = self.get_row_info(ridx_p, 0).threshold
        th_n = self.get_row_info(ridx_n, 0).threshold
        self.sch_params = dict(
            inv_params=ImmutableSortedDict(dict(
                lch=lch,
                seg_p=seg_buf,
                seg_n=seg_buf,
                w_p=w_pbuf,
                w_n=w_nbuf,
                th_p=th_p,
                th_n=th_n,
            )),
            tri_params=ImmutableSortedDict(dict(
                lch=lch,
                seg=seg_tri,
                w_p=w_ptri,
                w_n=w_ntri,
                th_p=th_p,
                th_n=th_n,
            )),
        )

    def _get_w_dict(self, ridx_n: int, ridx_p: int) -> Mapping[str, int]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        for row_idx, name_list in [(ridx_n, ['ntri', 'nbuf']),
                                   (ridx_p, ['ptri', 'pbuf'])]:
            rinfo = self.get_row_info(row_idx, 0)
            w_default = rinfo.width
            for name in name_list:
                w_ans[name] = w_dict.get(name, w_default)

        return w_ans


class Mux2to1Matched(MOSBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__mux2to1_matched

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return Mux2to1MatchedHalf.get_params_info()

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return Mux2to1MatchedHalf.get_default_param_values()

    def draw_layout(self) -> None:
        master: Mux2to1MatchedHalf = self.new_template(Mux2to1MatchedHalf, params=self.params)
        self.draw_base(master.draw_base_info)

        ridx_n: int = self.params['ridx_n']
        vertical_sel: bool = self.params['vertical_sel']

        # placement
        nhalf = master.num_cols
        corel = self.add_tile(master, 0, 0)
        corer = self.add_tile(master, 0, 2 * nhalf, flip_lr=True)
        self.set_mos_size(num_cols=2 * nhalf)

        # routing
        hm_layer = self.conn_layer + 1
        tr_manager = self.tr_manager
        hm_w = tr_manager.get_width(hm_layer, 'sig')

        ng0_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        selb_tidx = ng0_tidx
        sel_tidx = ng1_tidx

        # simple re-exports
        for name in ['VDD', 'VSS']:
            self.add_pin(name, self.connect_wires([corel.get_pin(name), corer.get_pin(name)]))
        self.connect_wires([corel.get_pin('mid'), corer.get_pin('mid')])
        self.reexport(corel.get_port('in'), net_name='in<0>')
        self.reexport(corer.get_port('in'), net_name='in<1>')
        self.reexport(corel.get_port('out'), net_name='out')

        # select signals
        nselb = corel.get_pin('nsel')
        psel = corel.get_pin('psel')
        nsel = corer.get_pin('nsel')
        pselb = corer.get_pin('psel')
        nsel, nselb = self.connect_differential_tracks(nsel, nselb, hm_layer,
                                                       sel_tidx, selb_tidx, width=hm_w)
        self.add_pin('psel', psel, hide=True)
        self.add_pin('pselb', pselb, hide=True)
        self.add_pin('nsel', nsel, hide=True)
        self.add_pin('nselb', nselb, hide=True)

        if vertical_sel:
            sel, selb = self.connect_differential_wires(nsel, nselb, corel.get_pin('psel_vm'),
                                                        corer.get_pin('psel_vm'))
            self.add_pin('sel', sel)
            self.add_pin('selb', selb)
        else:
            self.add_pin('sel', [psel, nsel], connect=True)
            self.add_pin('selb', [pselb, nselb], connect=True)

        self.sch_params = master.sch_params


class Mux2to1Core(MOSBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments for the tristate inverter',
            sel_seg='number of segments used for side inverters (select inverters)',
            fout='fanout of the output inverter',
            w_p='pmos width, can be list or integer if all widths are the same.',
            w_n='pmos width, can be list or integer if all widths are the same.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            vertical_out='True to have output on vertical layer',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            vertical_out=True,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if pinfo.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        seg: int = self.params['seg']
        sel_seg: int = self.params['sel_seg']
        fout: int = self.params['fout']
        w_p: Union[int, Sequence[int]] = self.params['w_p']
        w_n: Union[int, Sequence[int]] = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        vertical_out: bool = self.params['vertical_out']

        if seg % 2 != 0:
            raise ValueError(f'Mux2to1: seg = {seg} is not even')
        if sel_seg % 2 != 0:
            raise ValueError(f'Mux2to1: dummy_seg = {sel_seg} is not even')
        if sig_locs is None:
            sig_locs = {}

        inv_seg = seg * fout

        en_tidx = sig_locs.get('nen', self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig',
                                                           wire_idx=0))
        in0_tidx = sig_locs.get('nin0', self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig',
                                                             wire_idx=1))
        in1_tidx = sig_locs.get('pin1', self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig',
                                                             wire_idx=1))
        enb_tidx = sig_locs.get('penb', self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig',
                                                             wire_idx=0))
        tristate0_params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                                ridx_n=ridx_n, vertical_out=False,
                                sig_locs={'nen': en_tidx, 'nin': in0_tidx, 'pen': enb_tidx})
        tristate0_master = self.new_template(InvTristateCore, params=tristate0_params)
        tristate1_params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                                ridx_n=ridx_n, vertical_out=False,
                                sig_locs={'nen': en_tidx, 'nin': in1_tidx, 'pen': enb_tidx})
        tristate1_master = self.new_template(InvTristateCore, params=tristate1_params)

        in_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        out_inv_sig_locs = {'pin': in_tidx}
        for key in ('pout', 'nout'):
            if key in sig_locs:
                out_inv_sig_locs[key] = sig_locs[key]
        out_inv_params = dict(pinfo=pinfo, seg=inv_seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                              ridx_n=ridx_n, sig_locs=out_inv_sig_locs, vertical_out=vertical_out)
        out_inv_master = self.new_template(InvCore, params=out_inv_params)

        in_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        sel_inv_sig_locs = {'nin': in_tidx}
        for key in ('pout', 'nout'):
            if f'sel_{key}' in sig_locs:
                sel_inv_sig_locs[key] = sig_locs[f'sel_{key}']
        sel_inv_params = dict(pinfo=pinfo, seg=sel_seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                              ridx_n=ridx_n, sig_locs=sel_inv_sig_locs)
        sel_inv_master = self.new_template(InvCore, params=sel_inv_params)

        sel_ncol = sel_inv_master.num_cols
        tristate_ncols = tristate0_master.num_cols
        out_inv_ncols = out_inv_master.num_cols
        sep = max(self.get_hm_sp_le_sep_col(), self.min_sep_col)

        # --- Placement --- #
        cur_col = 0
        sel = self.add_tile(sel_inv_master, 0, cur_col)
        cur_col += sel_ncol + sep
        t0 = self.add_tile(tristate0_master, 0, cur_col)
        cur_col += tristate_ncols + sep
        t1 = self.add_tile(tristate1_master, 0, cur_col)
        cur_col += tristate_ncols + sep
        out_inv = self.add_tile(out_inv_master, 0, cur_col)
        cur_col += out_inv_ncols

        self.set_mos_size()

        # --- Routing --- #
        tr_manager = pinfo.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        # vdd/vss
        vdd_list, vss_list = [], []
        inst_arr = [sel, t0, t1, out_inv, t1]
        for inst in inst_arr:
            vdd_list += inst.get_all_port_pins('VDD')
            vss_list += inst.get_all_port_pins('VSS')

        vdd_list = self.connect_wires(vdd_list)
        vss_list = self.connect_wires(vss_list)

        # connect sel and selb
        sel_warr = sel.get_pin('nin')
        selb_idx = sig_locs.get('pselb', self.get_track_index(ridx_p, MOSWireType.G,
                                                              wire_name='sig', wire_idx=0))
        selb_tid = TrackID(hm_layer, selb_idx)
        selb_warr = self.connect_to_tracks(sel.get_pin('out'), selb_tid)

        # connect right en and left enb differentially
        sel_vms = sel.get_all_port_pins('in', self.conn_layer)
        t0_en = t0.get_pin('en')
        t1_en = t1.get_pin('en')
        l_en_tidx = self.grid.coord_to_track(vm_layer, t0_en.lower, RoundMode.LESS_EQ)
        selb_vm = self.connect_to_tracks(t0_en, TrackID(vm_layer, l_en_tidx, width=tr_w_v))

        r_en_tidx = self.grid.coord_to_track(vm_layer, t1_en.upper, RoundMode.GREATER_EQ)
        sel_vm = self.connect_to_tracks(t1_en, TrackID(vm_layer, r_en_tidx, width=tr_w_v))
        self.connect_differential_wires(sel_vm, selb_vm, sel_warr, selb_warr)
        sel_vms.append(sel_vm)

        # connect right enb and left en differentially
        t0_enb = t0.get_pin('enb')
        t1_enb = t1.get_pin('enb')
        l_enb_tidx = self.grid.coord_to_track(vm_layer, t0_enb.upper, RoundMode.GREATER_EQ)
        sel_vm = self.connect_to_tracks(t0_enb, TrackID(vm_layer, l_enb_tidx, width=tr_w_v))
        sel_vms.append(sel_vm)

        r_en_tidx = self.grid.coord_to_track(vm_layer, t1_enb.lower, RoundMode.LESS_EQ)
        selb_vm = self.connect_to_tracks(t1_enb, TrackID(vm_layer, r_en_tidx, width=tr_w_v))
        self.connect_differential_wires(sel_vm, selb_vm, sel_warr, selb_warr)
        self.add_pin('nsel', sel_warr, hide=True)
        self.add_pin('psel', sel_warr, hide=True)

        # connect outb to out
        if vertical_out:
            out_idx = out_inv.get_pin('out').track_id.base_index
            mux_out_idx = tr_manager.get_next_track(vm_layer, out_idx, 'out', 'in', up=False)
        else:
            out_hm = out_inv.get_pin('nout')
            mux_out_idx = self.grid.coord_to_track(vm_layer, out_hm.middle,
                                                   mode=RoundMode.NEAREST)
        mux_out_warrs = [t0.get_pin('nout'), t0.get_pin('pout'), t1.get_pin('nout'),
                         t1.get_pin('pout'), out_inv.get_pin('nin')]

        self.connect_to_tracks(mux_out_warrs, TrackID(vm_layer, mux_out_idx, width=tr_w_v))

        # add pins
        self.add_pin('VDD', vdd_list)
        self.add_pin('VSS', vss_list)
        self.add_pin('sel', sel_warr)
        self.reexport(t0.get_port('nin'), net_name='in<0>', hide=False)
        self.reexport(t1.get_port('pin'), net_name='in<1>', hide=False)
        if vertical_out:
            self.reexport(out_inv.get_port('out'), net_name='out')
        self.reexport(out_inv.get_port('pout'), label='out:', hide=vertical_out)
        self.reexport(out_inv.get_port('nout'), label='out:', hide=vertical_out)

        self.sch_params = dict(sel_inv=sel_inv_master.sch_params,
                               out_inv=out_inv_master.sch_params,
                               tristate=tristate0_master.sch_params)

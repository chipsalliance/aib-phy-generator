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

"""This module contains layout generators for a classic StrongArm latch."""

from typing import Any, Dict, Mapping, Optional, Type, Tuple

from pybag.enum import RoundMode

from bag.util.immutable import Param, ImmutableSortedDict
from bag.design.database import Module
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.enum import MOSWireType

from ...schematic.strongarm_frontend import bag3_digital__strongarm_frontend


class SAFrontendHalf(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ridx_n='bottom nmos row index.',
            ridx_p='pmos row index.',
            has_rstb='True to add rstb functionality.',
            has_bridge='True to add bridge switch.',
            vertical_out='True to connect outputs to vm_layer.',
            vertical_rstb='True to connect rstb to vm_layer.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_n=0,
            ridx_p=-1,
            has_rstb=False,
            has_bridge=False,
            vertical_out=True,
            vertical_rstb=True,
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        has_rstb: bool = self.params['has_rstb']
        has_bridge: bool = self.params['has_bridge']
        vertical_out: bool = self.params['vertical_out']
        vertical_rstb: bool = self.params['vertical_rstb']

        w_dict, th_dict = self._get_w_th_dict(ridx_n, ridx_p, has_bridge)

        seg_in = seg_dict['in']
        seg_tail = seg_dict['tail']
        seg_nfb = seg_dict['nfb']
        seg_pfb = seg_dict['pfb']
        seg_swm = seg_dict['sw']
        w_in = w_dict['in']
        w_tail = w_dict['tail']
        w_nfb = w_dict['nfb']
        w_pfb = w_dict['pfb']

        if seg_in & 1 or (seg_tail % 4 != 0) or seg_nfb & 1 or seg_pfb & 1:
            raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        # NOTE: make seg_swo even so we can abut transistors
        seg_swo = seg_swm + (seg_swm & 1)
        seg_tail = seg_tail // 2

        # placement
        ridx_in = ridx_n + 1
        ridx_nfb = ridx_in + 1
        m_in = self.add_mos(ridx_in, 0, seg_in, w=w_in)
        m_nfb = self.add_mos(ridx_nfb, 0, seg_nfb, w=w_nfb)
        m_pfb = self.add_mos(ridx_p, 0, seg_pfb, w=w_pfb)

        ng_tid = self.get_track_id(ridx_nfb, MOSWireType.G, wire_name='sig', wire_idx=-1)
        mid_tid = self.get_track_id(ridx_nfb, MOSWireType.DS, wire_name='sig')
        pg_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')
        vdd_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sup')
        pclk_tid = pg_tid

        if has_rstb:
            vss_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sup')
            nclk_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
            nrst_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
            prst_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
            tail_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='tail')
            tail_in_tid = self.get_track_id(ridx_in, MOSWireType.DS, wire_name='sig')

            m_tail = self.add_mos(ridx_n, 0, seg_tail, w=w_tail, g_on_s=True, stack=2, sep_g=True)
            m_swo_rst = self.add_mos(ridx_p, seg_pfb, seg_swo, w=w_pfb)
            m_swo = self.add_mos(ridx_p, seg_pfb + seg_swo, seg_swo, w=w_pfb)
            m_swm = self.add_mos(ridx_p, seg_pfb + 2 * seg_swo, seg_swm, w=w_pfb)
            vss_conn = m_tail.s
            tail_conn = m_tail.d
            g_conn = m_tail.g
            rstb_conn = g_conn[0::2]
            clk_conn = g_conn[1::2]
        else:
            vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            nclk_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)
            nrst_tid = prst_tid = None
            tail_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig')
            tail_in_tid = tail_tid

            m_tail = self.add_mos(ridx_n, 0, seg_tail, w=w_tail)
            m_swo = self.add_mos(ridx_p, seg_pfb, seg_swo, w=w_pfb)
            m_swm = self.add_mos(ridx_p, seg_pfb + seg_swo, seg_swm, w=w_pfb)
            m_swo_rst = None
            vss_conn = m_tail.s
            tail_conn = m_tail.d
            rstb_conn = None
            clk_conn = m_tail.g

        # NOTE: force even number of columns to make sure VDD conn_layer wires are on even columns.
        ncol_tot = self.num_cols
        self.set_mos_size(num_cols=ncol_tot + (ncol_tot & 1))

        # routing
        conn_layer = self.conn_layer
        vm_layer = conn_layer + 2
        vm_w = self.tr_manager.get_width(vm_layer, 'sig')
        grid = self.grid

        if has_rstb:
            nrst = self.connect_to_tracks(rstb_conn, nrst_tid)
            prst = self.connect_to_tracks([m_swo_rst.g], prst_tid)
            self.add_pin('nrstb', nrst)
            self.add_pin('prstb', prst)
            if vertical_rstb:
                xrst = grid.track_to_coord(conn_layer, m_swo_rst.g.track_id.base_index)
                vm_tidx = grid.coord_to_track(vm_layer, xrst, mode=RoundMode.GREATER_EQ)
                self.connect_to_tracks([nrst, prst], TrackID(vm_layer, vm_tidx, width=vm_w))

            tail = self.connect_to_tracks(tail_conn, tail_tid)
            in_d = m_in.d
            tail_in = self.connect_to_tracks(in_d, tail_in_tid)
            self.add_pin('tail_in', tail_in)
            tail_list = [tail, tail_in]
            for warr in in_d.warr_iter():
                xwire = grid.track_to_coord(conn_layer, warr.track_id.base_index)
                vm_tidx = grid.coord_to_track(vm_layer, xwire, mode=RoundMode.GREATER_EQ)
                self.connect_to_tracks(tail_list, TrackID(vm_layer, vm_tidx, width=vm_w))

            out = self.connect_wires([m_nfb.d, m_pfb.d, m_swo.d, m_swo_rst.d])
            vdd = self.connect_to_tracks([m_pfb.s, m_swo.s, m_swm.s, m_swo_rst.s],
                                         vdd_tid)
            mid = self.connect_to_tracks([m_in.s, m_nfb.s, m_swm.d], mid_tid)
        else:
            tail = self.connect_to_tracks([tail_conn, m_in.d], tail_tid)
            out = self.connect_wires([m_nfb.d, m_pfb.d, m_swo.d])
            vdd = self.connect_to_tracks([m_pfb.s, m_swo.s, m_swm.s], vdd_tid)
            mid = self.connect_to_tracks([m_in.s, m_nfb.s, m_swm.d], mid_tid)

        vss = self.connect_to_tracks(vss_conn, vss_tid)
        nclk = self.connect_to_tracks(clk_conn, nclk_tid)
        nout = self.connect_to_tracks(m_nfb.g, ng_tid)
        pout = self.connect_to_tracks(m_pfb.g, pg_tid)
        pclk = self.connect_to_tracks([m_swo.g, m_swm.g], pclk_tid)

        xclk = grid.track_to_coord(conn_layer, m_swo.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xclk, mode=RoundMode.GREATER_EQ)
        clk_vm = self.connect_to_tracks([nclk, pclk], TrackID(vm_layer, vm_tidx, width=vm_w))
        self.add_pin('clk_vm', clk_vm)

        xout = grid.track_to_coord(conn_layer, m_pfb.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xout, mode=RoundMode.GREATER_EQ)

        if vertical_out:
            out_vm = self.connect_to_tracks([nout, pout], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('out_vm', out_vm)
        else:
            self.add_pin('pout', pout)
            self.add_pin('nout', nout)

        self.add_pin('VSS', vss)
        self.add_pin('VDD', vdd)
        self.add_pin('tail', tail)
        self.add_pin('clk', nclk)
        self.add_pin('in', m_in.g)
        self.add_pin('out', out)
        self.add_pin('mid', mid)

        append_dict = dict(swo=seg_swo, swm=seg_swm)
        if has_bridge:
            append_dict['br'] = 1
        sch_seg_dict = seg_dict.copy(append=append_dict, remove=['sw'])
        self.sch_params = dict(
            lch=self.arr_info.lch,
            seg_dict=sch_seg_dict,
            w_dict=w_dict,
            th_dict=th_dict,
            has_rstb=has_rstb,
            has_bridge=has_bridge,
        )

    def _get_w_th_dict(self, ridx_n: int, ridx_p: int, has_bridge: bool
                       ) -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        th_ans = {}
        for name, row_idx in [('tail', ridx_n), ('in', ridx_n + 1), ('nfb', ridx_n + 2),
                              ('pfb', ridx_p)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        w_ans['swm'] = w_ans['swo'] = w_ans['pfb']
        th_ans['swm'] = th_ans['swo'] = th_ans['pfb']
        if has_bridge:
            w_ans['br'] = w_ans['in']
            th_ans['br'] = th_ans['in']
        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class SAFrontend(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__strongarm_frontend

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = SAFrontendHalf.get_params_info()
        ans['even_center'] = 'True to force center column to be even.'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = SAFrontendHalf.get_default_param_values()
        ans['even_center'] = False
        return ans

    def draw_layout(self):
        master: SAFrontendHalf = self.new_template(SAFrontendHalf, params=self.params)
        self.draw_base(master.draw_base_info)

        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        has_bridge: bool = self.params['has_bridge']
        vertical_out: bool = self.params['vertical_out']
        vertical_rstb: bool = self.params['vertical_rstb']
        even_center: bool = self.params['even_center']

        # placement
        nsep = self.min_sep_col
        nsep += (nsep & 1)
        if even_center and nsep % 4 == 2:
            nsep += 2

        nhalf = master.num_cols
        corel = self.add_tile(master, 0, nhalf, flip_lr=True)
        corer = self.add_tile(master, 0, nhalf + nsep)
        self.set_mos_size(num_cols=nsep + 2 * nhalf)

        # routing
        ridx_in = ridx_n + 1
        ridx_nfb = ridx_in + 1
        inn_tidx, hm_w = self.get_track_info(ridx_in, MOSWireType.G, wire_name='sig', wire_idx=0)
        inp_tidx = self.get_track_index(ridx_in, MOSWireType.G, wire_name='sig', wire_idx=-1)
        outn_tidx = self.get_track_index(ridx_nfb, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        outp_tidx = self.get_track_index(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0)

        hm_layer = self.conn_layer + 1
        inp, inn = self.connect_differential_tracks(corel.get_pin('in'), corer.get_pin('in'),
                                                    hm_layer, inp_tidx, inn_tidx, width=hm_w)
        self.add_pin('inp', inp)
        self.add_pin('inn', inn)

        outp, outn = self.connect_differential_tracks(corer.get_all_port_pins('out'),
                                                      corel.get_all_port_pins('out'),
                                                      hm_layer, outp_tidx, outn_tidx, width=hm_w)
        if vertical_out:
            outp_vm = corel.get_pin('out_vm')
            outn_vm = corer.get_pin('out_vm')
            self.connect_to_track_wires(outp, outp_vm)
            self.connect_to_track_wires(outn, outn_vm)
            self.add_pin('outp', outp_vm)
            self.add_pin('outn', outn_vm)
        else:
            self.add_pin('outp', outp, connect=True)
            self.add_pin('outn', outn, connect=True)
            self.add_pin('outp', corel.get_pin('pout'), connect=True)
            self.add_pin('outp', corel.get_pin('nout'), connect=True)
            self.add_pin('outn', corer.get_pin('pout'), connect=True)
            self.add_pin('outn', corer.get_pin('nout'), connect=True)

        self.add_pin('outp_hm', outp, hide=True)
        self.add_pin('outn_hm', outn, hide=True)

        clk = self.connect_wires([corel.get_pin('clk'), corer.get_pin('clk')])
        vss = self.connect_wires([corel.get_pin('VSS'), corer.get_pin('VSS')])
        vdd = self.connect_wires([corel.get_pin('VDD'), corer.get_pin('VDD')])
        self.connect_wires([corel.get_pin('tail'), corer.get_pin('tail')])
        self.add_pin('clk', clk)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)
        self.reexport(corel.get_port('clk_vm'), net_name='clkl', hide=True)
        self.reexport(corer.get_port('clk_vm'), net_name='clkr', hide=True)

        # bridge_switch
        if has_bridge:
            m_br0 = self.add_mos(ridx_n + 1, nhalf, 1, w=master.sch_params['w_dict']['br'],
                                 stack=nsep)
            self.connect_to_track_wires(m_br0.g, clk)

        if corel.has_port('nrstb'):
            self.connect_wires([corel.get_pin('tail_in'), corer.get_pin('tail_in')])
            rstb = self.connect_wires([corel.get_pin('nrstb'), corer.get_pin('nrstb')])
            if vertical_rstb:
                self.add_pin('rstb', rstb)
            else:
                rstl = corel.get_pin('prstb')
                rstr = corer.get_pin('prstb')
                self.add_pin('rstb', rstb, connect=True)
                self.add_pin('rstb', rstl, connect=True)
                self.add_pin('rstb', rstr, connect=True)
                self.add_pin('nrstb', rstb, hide=True)
                self.add_pin('prstbl', rstl, hide=True)
                self.add_pin('prstbr', rstr, hide=True)

        if has_bridge:
            self.sch_params = master.sch_params.copy(append=dict(stack_br=nsep))
        else:
            self.sch_params = master.sch_params

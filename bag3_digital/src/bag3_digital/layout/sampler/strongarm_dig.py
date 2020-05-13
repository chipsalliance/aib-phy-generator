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

"""Digital style strong arm latch."""

from typing import Any, Dict, Mapping, Optional, Type, Tuple

from pybag.enum import RoundMode, MinLenMode

from bag.util.math import HalfInt
from bag.util.immutable import Param, ImmutableSortedDict
from bag.design.database import Module
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.enum import MOSWireType

from ...schematic.strongarm_frontend import bag3_digital__strongarm_frontend


class SAFrontendDigitalHalf(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        zero = HalfInt(0)
        self._out_tinfo = (1, zero, zero)

    @property
    def out_tinfo(self) -> Tuple[int, HalfInt, HalfInt]:
        return self._out_tinfo

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ridx_n='bottom nmos row index.',
            ridx_p='pmos row index.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_n=0,
            ridx_p=-1,
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']

        w_dict, th_dict = self._get_w_th_dict(ridx_n, ridx_p)

        seg_in = seg_dict['in']
        seg_tail = seg_dict['tail']
        seg_fb = seg_dict['fb']
        seg_swm = seg_dict['sw']
        w_in = w_dict['in']
        w_tail = w_dict['tail']
        w_pfb = w_dict['pfb']

        min_sep = self.min_sep_col
        min_sep += (min_sep & 1)
        if seg_in & 1 or seg_tail & 1 or seg_fb & 1:
            raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        # NOTE: make seg_swo even so we can abut transistors
        seg_swo = seg_swm + (seg_swm & 1)
        seg_tail = seg_tail // 2
        if 2 * seg_swo + seg_swm > seg_in + seg_tail + min_sep:
            raise ValueError('the reset switches are too large compared to input devices.')

        # placement
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        grid = self.grid
        tr_manager = self.tr_manager
        hm_w = tr_manager.get_width(hm_layer, 'sig')
        vm_w = tr_manager.get_width(vm_layer, 'sig')

        vss_tid = self.get_track_id(ridx_n, False, wire_name='sup')
        vdd_tid = self.get_track_id(ridx_p, True, wire_name='sup')
        nd0_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-3)
        nd1_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-2)
        nd2_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        pd0_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0)
        pd1_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=1)
        ng0_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg0_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-2)
        pg1_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)
        self._out_tinfo = (hm_w, pg0_tid.base_index, ng1_tid.base_index)

        m_tail = self.add_mos(ridx_n, 0, seg_tail, w=w_tail, g_on_s=True, stack=2, sep_g=True)
        cur_col = seg_tail * 2 + min_sep
        m_in = self.add_mos(ridx_n, cur_col, seg_in, w=w_in)
        cur_col += seg_in
        m_nfb = self.add_mos(ridx_n, cur_col, seg_fb, w=w_in)
        m_pfb = self.add_mos(ridx_p, cur_col, seg_fb, w=w_pfb)
        cur_col -= seg_swo
        m_swo_rst = self.add_mos(ridx_p, cur_col, seg_swo, w=w_pfb)
        cur_col -= seg_swo
        m_swo = self.add_mos(ridx_p, cur_col, seg_swo, w=w_pfb)
        # flip middle node switch so it works for both even and odd number of segments
        m_swm = self.add_mos(ridx_p, cur_col, seg_swm, w=w_pfb, flip_lr=True)
        self.set_mos_size()

        # routing
        # supplies
        vdd = self.connect_to_tracks([m_pfb.s, m_swo.s, m_swo_rst.s, m_swm.s], vdd_tid)
        vss = self.connect_to_tracks(m_tail.s, vss_tid)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        # drain/source connections
        tail = self.connect_to_tracks([m_tail.d, m_in.d], nd0_tid)
        nmid = self.connect_to_tracks([m_in.s, m_nfb.s], nd2_tid)
        nout = self.connect_to_tracks(m_nfb.d, nd1_tid, min_len_mode=MinLenMode.LOWER)
        pout = self.connect_to_tracks([m_pfb.d, m_swo.d, m_swo_rst.d], pd1_tid)
        pmid = self.connect_to_tracks(m_swm.d, pd0_tid, min_len_mode=MinLenMode.MIDDLE)
        self.add_pin('pout', pout)
        self.add_pin('nout', nout)
        self.add_pin('tail', tail)
        # mid node connection
        vm_tidx = grid.coord_to_track(vm_layer, pmid.middle, mode=RoundMode.NEAREST)
        self.connect_to_tracks([nmid, pmid], TrackID(vm_layer, vm_tidx, width=vm_w))

        # gate connections
        nclk = self.connect_to_tracks(m_tail.g[1::2], ng0_tid, min_len_mode=MinLenMode.MIDDLE)
        pclk = self.connect_to_tracks([m_swo.g, m_swm.g], pg0_tid)
        vm_tidx = tr_manager.get_next_track(vm_layer, vm_tidx, 'sig', 'sig', up=False)
        clk_vm = self.connect_to_tracks([nclk, pclk], TrackID(vm_layer, vm_tidx, width=vm_w))
        self.add_pin('clk_vm', clk_vm)
        self.add_pin('clk', nclk)

        prstb = self.connect_to_tracks([m_tail.g[0::2], m_swo_rst.g], pg1_tid,
                                       min_len_mode=MinLenMode.LOWER)
        self.add_pin('outb', self.connect_wires([m_nfb.g, m_pfb.g]))
        self.add_pin('in', self.connect_to_tracks(m_in.g, ng1_tid, min_len_mode=MinLenMode.LOWER))
        self.add_pin('rstb', prstb)

        sch_seg_dict = seg_dict.copy(append=dict(swo=seg_swo, swm=seg_swm, nfb=seg_fb, pfb=seg_fb,
                                                 br=1),
                                     remove=['sw', 'fb'])
        self.sch_params = dict(
            lch=self.arr_info.lch,
            seg_dict=sch_seg_dict,
            w_dict=w_dict,
            th_dict=th_dict,
            has_rstb=True,
            has_bridge=True,
        )

    def _get_w_th_dict(self, ridx_n: int, ridx_p: int
                       ) -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        th_ans = {}
        for name, row_idx in [('tail', ridx_n), ('in', ridx_n + 1), ('pfb', ridx_p)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        w_ans['br'] = w_ans['nfb'] = w_ans['in']
        w_ans['swm'] = w_ans['swo'] = w_ans['pfb']
        th_ans['br'] = th_ans['nfb'] = th_ans['in']
        th_ans['swm'] = th_ans['swo'] = th_ans['pfb']
        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class SAFrontendDigital(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__strongarm_frontend

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = SAFrontendDigitalHalf.get_params_info()
        ans['even_center'] = 'True to force center column to be even.'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = SAFrontendDigitalHalf.get_default_param_values()
        ans['even_center'] = False
        return ans

    def draw_layout(self):
        master: SAFrontendDigitalHalf = self.new_template(SAFrontendDigitalHalf, params=self.params)
        self.draw_base(master.draw_base_info)

        ridx_n: int = self.params['ridx_n']
        even_center: bool = self.params['even_center']

        # placement
        nsep = self.min_sep_col
        nsep += (nsep & 1)
        if even_center and nsep % 4 == 2:
            nsep += 2

        nhalf = master.num_cols
        corel = self.add_tile(master, 0, 0)
        corer = self.add_tile(master, 0, 2 * nhalf + nsep, flip_lr=True)
        self.set_mos_size(num_cols=2 * nhalf + nsep)

        # bridge switch
        clkl = corel.get_pin('clk')
        clkr = corer.get_pin('clk')
        m_br = self.add_mos(ridx_n, nhalf, 1, w=master.sch_params['w_dict']['br'], stack=nsep)
        clk = self.connect_wires([clkl, clkr])
        self.connect_to_track_wires(m_br.g, clk)
        self.add_pin('clk', clk)

        # routing
        for name in ['VDD', 'VSS', 'rstb']:
            self.add_pin(name, self.connect_wires([corel.get_pin(name), corer.get_pin(name)]))
        self.connect_wires([corel.get_pin('tail'), corer.get_pin('tail')])

        hm_layer = self.conn_layer + 1
        hm_w, outp_tidx, outn_tidx = master.out_tinfo
        outp, outn = self.connect_differential_tracks(corer.get_all_port_pins('outb'),
                                                      corel.get_all_port_pins('outb'),
                                                      hm_layer, outp_tidx, outn_tidx, width=hm_w)
        self.add_pin('outp', outp, connect=True)
        self.add_pin('outn', outn, connect=True)

        self.reexport(corel.get_port('in'), net_name='inn')
        self.reexport(corer.get_port('in'), net_name='inp')
        self.reexport(corel.get_port('clk_vm'), net_name='clkl', hide=-True)
        self.reexport(corer.get_port('clk_vm'), net_name='clkr', hide=True)
        self.add_pin('outn', corer.get_pin('pout'), connect=True)
        self.add_pin('outn', corer.get_pin('nout'), connect=True)
        self.add_pin('outp', corel.get_pin('pout'), connect=True)
        self.add_pin('outp', corel.get_pin('nout'), connect=True)

        self.sch_params = master.sch_params.copy(append=dict(stack_br=nsep))

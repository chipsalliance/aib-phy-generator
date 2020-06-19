# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Blue Cheetah Analog Design Inc.
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
"""This module contains layout generators for various latches."""

from typing import Any, Dict, Optional, Type, Tuple, Mapping

from pybag.enum import MinLenMode, RoundMode

from bag.util.math import HalfInt
from bag.util.immutable import Param, ImmutableSortedDict
from bag.design.module import Module
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ...schematic.sr_latch_symmetric import bag3_digital__sr_latch_symmetric


class SRLatchSymmetricHalf(MOSBase):
    """Half of symmetric SR latch
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        zero = HalfInt(0)
        self._q_tr_info = (0, zero, zero)
        self._sr_hm_tr_info = self._q_tr_info
        self._sr_vm_tr_info = self._q_tr_info

    @property
    def q_tr_info(self) -> Tuple[int, HalfInt, HalfInt]:
        return self._q_tr_info

    @property
    def sr_hm_tr_info(self) -> Tuple[int, HalfInt, HalfInt]:
        return self._sr_hm_tr_info

    @property
    def sr_vm_tr_info(self) -> Tuple[int, HalfInt, HalfInt]:
        return self._sr_vm_tr_info

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ridx_n='bottom nmos row index.',
            ridx_p='pmos row index.',
            has_rstb='True to add rstb functionality.',
            has_outbuf='True to add output buffers.',
            has_inbuf='True to add input buffers.',
            out_pitch='output wire pitch from center.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_n=0,
            ridx_p=-1,
            has_rstb=False,
            has_outbuf=True,
            has_inbuf=True,
            out_pitch=0.5,
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        has_rstb: bool = self.params['has_rstb']
        has_outbuf: bool = self.params['has_outbuf']
        has_inbuf: bool = self.params['has_inbuf']
        out_pitch: HalfInt = HalfInt.convert(self.params['out_pitch'])

        w_dict, th_dict = self._get_w_th_dict(ridx_n, ridx_p, has_rstb)

        seg_fb = seg_dict['fb']
        seg_ps = seg_dict['ps']
        seg_nr = seg_dict['nr']
        seg_obuf = seg_dict['obuf'] if has_outbuf else 0
        seg_ibuf = seg_dict['ibuf'] if has_inbuf else 0

        w_pfb = w_dict['pfb']
        w_nfb = w_dict['nfb']
        w_ps = w_dict['ps']
        w_nr = w_dict['nr']
        w_rst = w_dict.get('pr', 0)
        w_nbuf = w_nr
        w_pbuf = w_ps

        sch_seg_dict = dict(nfb=seg_fb, pfb=seg_fb, ps=seg_ps, nr=seg_nr)
        if has_rstb:
            sch_seg_dict['pr'] = seg_rst = seg_dict['rst']
        else:
            seg_rst = 0

        if seg_ps & 1 or seg_nr & 1 or seg_rst & 1 or seg_obuf & 1:
            raise ValueError('ps, nr, rst, and buf must have even number of segments')

        # placement
        min_sep = self.min_sep_col
        # use even step size to maintain supply conn_layer wires parity.
        min_sep += (min_sep & 1)

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        grid = self.grid
        arr_info = self.arr_info
        tr_manager = self.tr_manager
        hm_w = tr_manager.get_width(hm_layer, 'sig')
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        hm_sep_col = self.get_hm_sp_le_sep_col(ntr=hm_w)
        mid_sep = max(hm_sep_col - 2, 0)
        mid_sep = (mid_sep + 1) // 2

        if has_inbuf:
            m_nibuf = self.add_mos(ridx_n, mid_sep, seg_ibuf, w=w_nbuf)
            m_pibuf = self.add_mos(ridx_p, mid_sep, seg_ibuf, w=w_pbuf)
            cur_col = mid_sep + seg_ibuf
            psrb_list = [m_nibuf.g, m_pibuf.g]
        else:
            m_nibuf = m_pibuf = None
            cur_col = mid_sep
            psrb_list = []

        nr_col = cur_col
        m_nr = self.add_mos(ridx_n, cur_col, seg_nr, w=w_nr)
        m_ps = self.add_mos(ridx_p, cur_col, seg_ps, w=w_ps)
        psrb_list.append(m_ps.g)
        pcol = cur_col + seg_ps
        if has_rstb:
            m_rst = self.add_mos(ridx_p, pcol, seg_rst, w=w_rst)
            pcol += seg_rst
        else:
            m_rst = None

        cur_col = max(cur_col + seg_nr, pcol)
        if has_outbuf:
            m_pinv = self.add_mos(ridx_p, cur_col, seg_obuf, w=w_pbuf)
            m_ninv = self.add_mos(ridx_n, cur_col, seg_obuf, w=w_nbuf)
            cur_col += seg_obuf
        else:
            m_pinv = m_ninv = None

        cur_col += min_sep
        fb_col = cur_col
        m_pfb = self.add_mos(ridx_p, cur_col, seg_fb, w=w_pfb, g_on_s=True, stack=2, sep_g=True)
        m_nfb = self.add_mos(ridx_n, cur_col, seg_fb, w=w_nfb, g_on_s=True, stack=2, sep_g=True)
        self.set_mos_size()

        # track planning
        vss_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sup')
        nbuf_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-2)
        nq_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        psr_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)
        pq_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0)
        pbuf_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=1)
        vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sup')

        # try to spread out gate wires to lower parasitics on differential Q wires
        ng_lower = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng_upper = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)
        g_idx_list = tr_manager.spread_wires(hm_layer, ['sig', 'sig_hs', 'sig_hs', 'sig'],
                                             ng_lower, ng_upper, ('sig_hs', 'sig_hs'), alignment=0)
        self._q_tr_info = (hm_w, g_idx_list[2], g_idx_list[1])
        self._sr_hm_tr_info = (hm_w, g_idx_list[3], g_idx_list[0])

        if has_rstb:
            rst_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-2)
            pq_conn_list = [m_ps.d, m_rst.d, m_pfb.s]
            vdd_list = [m_ps.s, m_rst.s, m_pfb.d]
            vss_list = [m_nr.s, m_nfb.d]

            rstb = self.connect_to_tracks(m_rst.g, rst_tid, min_len_mode=MinLenMode.MIDDLE)
            rst_vm_tidx = grid.coord_to_track(vm_layer, rstb.middle, mode=RoundMode.GREATER_EQ)
            rstb_vm = self.connect_to_tracks(rstb, TrackID(vm_layer, rst_vm_tidx, width=vm_w),
                                             min_len_mode=MinLenMode.MIDDLE)
            self.add_pin('rstb', rstb_vm)
        else:
            pq_conn_list = [m_ps.d, m_pfb.s]
            vdd_list = [m_ps.s, m_pfb.d]
            vss_list = [m_nr.s, m_nfb.d]

        self.add_pin('nsr', m_nr.g)
        self.add_pin('nsrb', m_nfb.g[0::2])
        nq = self.connect_to_tracks([m_nr.d, m_nfb.s], nq_tid)
        pq = self.connect_to_tracks(pq_conn_list, pq_tid)
        psrb = self.connect_to_tracks(psrb_list, psr_tid, min_len_mode=MinLenMode.UPPER)
        psr = self.connect_to_tracks(m_pfb.g[0::2], psr_tid, min_len_mode=MinLenMode.LOWER)
        qb = self.connect_wires([m_nfb.g[1::2], m_pfb.g[1::2]])
        self.add_pin('qb', qb)

        if has_inbuf:
            vdd_list.append(m_pibuf.s)
            vss_list.append(m_nibuf.s)

            nbuf = self.connect_to_tracks(m_nibuf.d, nbuf_tid, min_len_mode=MinLenMode.UPPER)
            pbuf = self.connect_to_tracks(m_pibuf.d, pbuf_tid, min_len_mode=MinLenMode.UPPER)
            vm_tidx = grid.coord_to_track(vm_layer, nbuf.middle, mode=RoundMode.LESS_EQ)
            buf = self.connect_to_tracks([nbuf, pbuf], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('sr_buf', buf)

        out_p_htr = out_pitch.dbl_value
        vm_ref = grid.coord_to_track(vm_layer, 0)
        srb_vm_tidx = arr_info.col_to_track(vm_layer, nr_col + 1, mode=RoundMode.GREATER_EQ)
        if has_outbuf:
            vdd_list.append(m_pinv.s)
            vss_list.append(m_ninv.s)

            nbuf = self.connect_to_tracks(m_ninv.d, nbuf_tid, min_len_mode=MinLenMode.MIDDLE)
            pbuf = self.connect_to_tracks(m_pinv.d, pbuf_tid, min_len_mode=MinLenMode.MIDDLE)
            vm_delta = grid.coord_to_track(vm_layer, nbuf.middle, mode=RoundMode.LESS_EQ) - vm_ref
            vm_htr = -(-vm_delta.dbl_value // out_p_htr) * out_p_htr
            vm_tidx = vm_ref + HalfInt(vm_htr)
            buf = self.connect_to_tracks([nbuf, pbuf], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('buf_out', buf)
            buf_in = self.connect_wires([m_ninv.g, m_pinv.g])
            self.add_pin('buf_in', buf_in)

            q_vm_tidx = tr_manager.get_next_track(vm_layer, srb_vm_tidx, 'sig', 'sig')
        else:
            vm_delta = tr_manager.get_next_track(vm_layer, srb_vm_tidx, 'sig', 'sig') - vm_ref
            vm_htr = -(-vm_delta.dbl_value // out_p_htr) * out_p_htr
            q_vm_tidx = vm_ref + HalfInt(vm_htr)

        sr_vm_tidx = arr_info.col_to_track(vm_layer, fb_col, mode=RoundMode.LESS_EQ)
        self._sr_vm_tr_info = (vm_w, sr_vm_tidx, srb_vm_tidx)

        q_vm = self.connect_to_tracks([nq, pq], TrackID(vm_layer, q_vm_tidx, width=vm_w))
        self.add_pin('q_vm', q_vm)
        self.add_pin('psr', psr)
        self.add_pin('psrb', psrb)

        self.add_pin('VDD', self.connect_to_tracks(vdd_list, vdd_tid))
        self.add_pin('VSS', self.connect_to_tracks(vss_list, vss_tid))

        lch = arr_info.lch
        buf_params = ImmutableSortedDict(dict(
            lch=lch,
            w_p=w_pbuf,
            w_n=w_nbuf,
            th_p=th_dict['ps'],
            th_n=th_dict['nr'],
            seg=seg_obuf,
        ))
        obuf_params = buf_params if has_outbuf else None
        ibuf_params = buf_params.copy(append=dict(seg=seg_ibuf)) if has_inbuf else None
        self.sch_params = dict(
            core_params=ImmutableSortedDict(dict(
                lch=lch,
                seg_dict=ImmutableSortedDict(sch_seg_dict),
                w_dict=w_dict,
                th_dict=th_dict,
            )),
            outbuf_params=obuf_params,
            inbuf_params=ibuf_params,
            has_rstb=has_rstb,
        )

    def _get_w_th_dict(self, ridx_n: int, ridx_p: int, has_rstb: bool
                       ) -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        th_ans = {}
        for row_idx, name_list in [(ridx_n, ['nfb', 'nr']),
                                   (ridx_p, ['pfb', 'ps'])]:
            rinfo = self.get_row_info(row_idx, 0)
            for name in name_list:
                w = w_dict.get(name, 0)
                if w == 0:
                    w = rinfo.width
                w_ans[name] = w
                th_ans[name] = rinfo.threshold

        if has_rstb:
            w_ans['pr'] = w_ans['ps']
            th_ans['pr'] = th_ans['ps']

        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class SRLatchSymmetric(MOSBase):
    """Symmetric SR latch.  Mainly designed to be used with strongarm.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__sr_latch_symmetric

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = SRLatchSymmetricHalf.get_params_info()
        ans['swap_outbuf'] = 'True to swap output buffers, so outp is on opposite side of inp.'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = SRLatchSymmetricHalf.get_default_param_values()
        ans['swap_outbuf'] = False
        return ans

    def draw_layout(self) -> None:
        master: SRLatchSymmetricHalf = self.new_template(SRLatchSymmetricHalf, params=self.params)
        self.draw_base(master.draw_base_info)

        swap_outbuf: bool = self.params['swap_outbuf']

        hm_w, q_tidx, qb_tidx = master.q_tr_info
        _, sr_hm_top, sr_hm_bot = master.sr_hm_tr_info
        vm_w, sr_vm_tidx, srb_vm_tidx = master.sr_vm_tr_info

        # placement
        nhalf = master.num_cols
        corel = self.add_tile(master, 0, nhalf, flip_lr=True)
        corer = self.add_tile(master, 0, nhalf)
        self.set_mos_size(num_cols=2 * nhalf)

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        arr_info = self.arr_info
        vm0 = arr_info.col_to_track(vm_layer, 0)
        vmh = arr_info.col_to_track(vm_layer, nhalf)
        vmdr = vmh - vm0
        vmdl = vmh + vm0

        pr = corel.get_pin('psr')
        psb = corel.get_pin('psrb')
        nr = corel.get_pin('nsr')
        nsb = corel.get_pin('nsrb')
        ps = corer.get_pin('psr')
        prb = corer.get_pin('psrb')
        ns = corer.get_pin('nsr')
        nrb = corer.get_pin('nsrb')

        nr, nsb = self.connect_differential_tracks(nr, nsb, hm_layer, sr_hm_top, sr_hm_bot,
                                                   width=hm_w)
        ns, nrb = self.connect_differential_tracks(ns, nrb, hm_layer, sr_hm_bot, sr_hm_top,
                                                   width=hm_w)
        sb = self.connect_to_tracks([psb, nsb], TrackID(vm_layer, vmdl - srb_vm_tidx, width=vm_w))
        r = self.connect_to_tracks([pr, nr], TrackID(vm_layer, vmdl - sr_vm_tidx, width=vm_w),
                                   track_lower=sb.lower)
        s = self.connect_to_tracks([ps, ns], TrackID(vm_layer, vmdr + sr_vm_tidx, width=vm_w))
        rb = self.connect_to_tracks([prb, nrb], TrackID(vm_layer, vmdr + srb_vm_tidx, width=vm_w),
                                    track_lower=s.lower)

        self.add_pin('sb', sb)
        self.add_pin('rb', rb)
        if corel.has_port('sr_buf'):
            sbuf = corel.get_pin('sr_buf')
            rbuf = corer.get_pin('sr_buf')
            self.connect_to_tracks(sbuf, ns.track_id, track_upper=ns.upper)
            self.connect_to_tracks(rbuf, nr.track_id, track_lower=nr.lower)
        else:
            self.add_pin('s', s)
            self.add_pin('r', r)

        q_list = [corel.get_pin('q_vm'), corer.get_pin('qb')]
        qb_list = [corer.get_pin('q_vm'), corel.get_pin('qb')]
        if corel.has_port('buf_out'):
            if swap_outbuf:
                self.reexport(corel.get_port('buf_out'), net_name='qb')
                self.reexport(corer.get_port('buf_out'), net_name='q')
                q_list.append(corel.get_pin('buf_in'))
                qb_list.append(corer.get_pin('buf_in'))
            else:
                self.reexport(corel.get_port('buf_out'), net_name='q')
                self.reexport(corer.get_port('buf_out'), net_name='qb')
                q_list.append(corer.get_pin('buf_in'))
                qb_list.append(corel.get_pin('buf_in'))
        else:
            self.add_pin('q', q_list[0])
            self.add_pin('qb', qb_list[0])

        self.connect_differential_tracks(q_list, qb_list, self.conn_layer + 1,
                                         q_tidx, qb_tidx, width=hm_w)

        if corel.has_port('rstb'):
            self.reexport(corel.get_port('rstb'), net_name='rsthb')
            self.reexport(corer.get_port('rstb'), net_name='rstlb')

        self.add_pin('VDD', self.connect_wires([corel.get_pin('VDD'), corer.get_pin('VDD')]))
        self.add_pin('VSS', self.connect_wires([corel.get_pin('VSS'), corer.get_pin('VSS')]))

        self.sch_params = master.sch_params

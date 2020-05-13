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

"""This module contains layout generators for various memory elements."""

from typing import Any, Dict, Optional, Type

from pybag.enum import RoundMode, MinLenMode

from bag.util.immutable import Param
from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from .mux import Mux2to1Core
from .gates import InvTristateCore, NOR2Core, InvCore, PassGateCore

# noinspection PyUnresolvedReferences
from ._flop_scan_rst import FlopScanRstlbTwoTile


class LatchCore(MOSBase):
    """A transmission gate based latch."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @property
    def seg_in(self) -> int:
        return self.sch_params['seg_dict']['tin']

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output inverter.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='input stage fanout.',
            fanout_kp='keeper stage fanout.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        hm_layer = pinfo.conn_layer + 1
        vm_layer = hm_layer + 1
        if pinfo.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']

        # setup floorplan
        self.draw_base(pinfo)

        # compute track locations
        tr_manager = pinfo.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        if sig_locs is None:
            sig_locs = {}
        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        t0_in_tidx = sig_locs.get(key, self.get_track_index(ridx_p, MOSWireType.G,
                                                            wire_name='sig', wire_idx=0))
        t0_enb_tidx = sig_locs.get('pclkb', self.get_track_index(ridx_p, MOSWireType.G,
                                                                 wire_name='sig', wire_idx=1))
        t0_en_tidx = sig_locs.get('nclk', self.get_track_index(ridx_n, MOSWireType.G,
                                                               wire_name='sig', wire_idx=0))
        t1_en_tidx = sig_locs.get('nclkb', self.get_track_index(ridx_n, MOSWireType.G,
                                                                wire_name='sig', wire_idx=1))
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        nd1_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)

        seg_t1 = max(1, int(round(seg / (2 * fanout_kp))) * 2)
        seg_t0 = max(2 * seg_t1, max(2, int(round(seg / (2 * fanout_in))) * 2))
        params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      sig_locs={'nin': t0_en_tidx, 'pout': pd1_tidx, 'nout': nd1_tidx})
        inv_master = self.new_template(InvCore, params=params)

        params['seg'] = seg_t0
        params['vertical_out'] = False
        params['sig_locs'] = {'nin': t0_in_tidx, 'pout': pd0_tidx, 'nout': nd0_tidx,
                              'nen': t0_en_tidx, 'pen': t0_enb_tidx}
        t0_master = self.new_template(InvTristateCore, params=params)
        params['seg'] = seg_t1
        params['sig_locs'] = {'nin': t0_enb_tidx, 'pout': pd0_tidx, 'nout': nd0_tidx,
                              'nen': t1_en_tidx, 'pen': t0_in_tidx}
        t1_master = self.new_template(InvTristateCore, params=params)

        # set size
        blk_sp = self.min_sep_col
        t0_ncol = t0_master.num_cols
        t1_ncol = t1_master.num_cols
        inv_ncol = inv_master.num_cols
        num_cols = t0_ncol + t1_ncol + inv_ncol + blk_sp * 2
        self.set_mos_size(num_cols)

        # add instances
        t1_col = t0_ncol + blk_sp
        inv_col = num_cols - inv_ncol
        t0 = self.add_tile(t0_master, 0, 0)
        t1 = self.add_tile(t1_master, 0, t1_col)
        inv = self.add_tile(inv_master, 0, inv_col)

        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in (t0, t1, inv):
            vss_list.append(inst.get_pin('VSS'))
            vdd_list.append(inst.get_pin('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # export input
        self.reexport(t0.get_port('nin'), net_name='nin', hide=True)
        self.reexport(t0.get_port('pin'), net_name='pin', hide=True)
        self.reexport(t0.get_port('in'))

        # connect output
        out = inv.get_pin('out')
        in2 = t1.get_pin('nin')
        self.connect_to_track_wires(in2, out)
        self.add_pin('out', out)
        self.add_pin('nout', in2, hide=True)
        self.add_pin('pout', in2, hide=True)

        # connect middle node
        col = inv_col - max(1, blk_sp // 2)
        mid_tid = TrackID(vm_layer, pinfo.get_source_track(col), width=tr_w_v)
        warrs = [t0.get_pin('pout'), t0.get_pin('nout'), t1.get_pin('pout'), t1.get_pin('nout'),
                 inv.get_pin('nin')]
        self.connect_to_tracks(warrs, mid_tid)
        self.add_pin('outb', inv.get_pin('in'))
        self.add_pin('noutb', inv.get_pin('nin'), hide=True)
        self.add_pin('poutb', inv.get_pin('nin'), hide=True)

        # connect clocks
        clk_tidx = sig_locs.get('clk', pinfo.get_source_track(t1_col + 1))
        clkb_tidx = sig_locs.get('clkb', pinfo.get_source_track(t1_col - blk_sp - 1))
        clk_tid = TrackID(vm_layer, clk_tidx, width=tr_w_v)
        clkb_tid = TrackID(vm_layer, clkb_tidx, width=tr_w_v)
        t0_en = t0.get_pin('en')
        t1_en = t1.get_pin('en')
        t1_enb = t1.get_pin('enb')
        t0_enb = t0.get_pin('enb')
        clk = self.connect_to_tracks([t0_en, t1_enb], clk_tid)
        clkb = self.connect_to_tracks([t0_enb, t1_en], clkb_tid)
        self.add_pin('clk', clk)
        self.add_pin('clkb', clkb)
        self.add_pin('nclk', t0_en, hide=True)
        self.add_pin('pclk', t1_enb, hide=True)
        self.add_pin('nclkb', t1_en, hide=True)
        self.add_pin('pclkb', t0_enb, hide=True)

        # set properties
        self._sch_params = dict(
            lch=self.place_info.lch,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            seg_dict=dict(tin=seg_t0, tfb=seg_t1, buf=seg),
        )


class FlopCore(MOSBase):
    """A transmission gate based flip-flop."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._cntr_col_clk = None

    @property
    def seg_in(self):
        return self.sch_params['seg_m']['in']

    @property
    def cntr_col_clk(self):
        return self._cntr_col_clk

    def get_schematic_class_inst(self) -> Optional[Type[Module]]:
        rst: bool = self.params['resetable']
        scan: bool = self.params['scanable']
        if scan:
            raise ValueError('See Developer')
        if rst:
            # noinspection PyTypeChecker
            return ModuleDB.get_schematic_class('bag3_digital', 'rst_flop')
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'flop')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output inverter.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='latch input stage fanout.',
            fanout_kp='latch keeper stage fanout.',
            fanout_lat='fanout between latches.',
            fanout_mux='fanout of scan mux, if present.',
            seg_ck='number of segments for clock inverter.  0 to disable.',
            seg_mux='Dictionary of segments for scan mux, if scanable',
            resetable='True if flop is resetable, default is False',
            scanable='True if flop needs to have scanability',
            extra_sp='This parameter is added to the min value of one of the separations '
                     '(mostly used to make power vertical stripes aligned)'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
            fanout_lat=4,
            fanout_mux=4,
            seg_ck=0,
            seg_mux=None,
            resetable=False,
            scanable=False,
            extra_sp=0,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']
        fanout_lat: float = self.params['fanout_lat']
        fanout_mux: float = self.params['fanout_mux']
        seg_ck: int = self.params['seg_ck']
        seg_mux: Optional[Dict[str, int]] = self.params['seg_mux']
        rst: bool = self.params['resetable']
        scan: bool = self.params['scanable']
        extra_sp: int = self.params['extra_sp']

        # setup floorplan
        self.draw_base(pinfo)

        # compute track locations
        if sig_locs is None:
            sig_locs = {}
        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        in_tidx = sig_locs.get(key, self.get_track_index(ridx_p, MOSWireType.G,
                                                         wire_name='sig', wire_idx=0))
        pclkb_tidx = sig_locs.get('pclkb', self.get_track_index(ridx_p, MOSWireType.G,
                                                                wire_name='sig', wire_idx=0))
        nclk_idx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        nclkb_idx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        pclk_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        clk_idx = sig_locs.get('clk', None)
        clkb_idx = sig_locs.get('clkb', None)

        # make masters
        lat_params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                          sig_locs={'nclk': nclk_idx, 'nclkb': nclkb_idx, 'pclk': pclk_tidx,
                                    'nin': pclk_tidx, 'pclkb': pclkb_tidx,
                                    'pout': sig_locs.get('pout', pclkb_tidx)},
                          fanout_in=fanout_in, fanout_kp=fanout_kp)

        s_master = self.new_template(RstLatchCore if rst else LatchCore, params=lat_params)
        seg_m = max(2, int(round(s_master.seg_in / (2 * fanout_lat))) * 2)
        lat_params['seg'] = seg_m
        lat_params['sig_locs'] = lat_sig_locs = {'nclk': nclkb_idx, 'nclkb': nclk_idx,
                                                 'pclk': pclkb_tidx, 'nin': in_tidx,
                                                 'pclkb': pclk_tidx}
        if clk_idx is not None:
            lat_sig_locs['clkb'] = clk_idx
        if clkb_idx is not None:
            lat_sig_locs['clk'] = clkb_idx

        m_master = self.new_template(RstLatchCore if rst else LatchCore, params=lat_params)

        cur_col = 0
        blk_sp = self.min_sep_col
        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)

        inst_list = []
        mux_inst = None
        mux_master = None
        if scan:
            if seg_mux is None:
                raise ValueError('Please specify segments for scan mux.')
            mux_params = dict(pinfo=pinfo, seg=seg_mux['seg'], w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                              ridx_n=ridx_n, sel_seg=seg_mux['sel_seg'], fout=fanout_mux,
                              sig_locs={'pselb': pd1_tidx, 'pin1': pg0_tidx, 'penb': pg1_tidx},
                              vertical_out=False)
            mux_master = self.new_template(Mux2to1Core, params=mux_params)
            mux_inst = self.add_tile(mux_master, 0, cur_col)
            mux_ncol = mux_master.num_cols
            cur_col += mux_ncol + blk_sp
            inst_list.append(mux_inst)

        m_ncol = m_master.num_cols
        s_ncol = s_master.num_cols
        m_inv_sp = blk_sp if rst else 0
        inv_master = None
        if seg_ck > 0:
            params = dict(pinfo=pinfo, seg=seg_ck, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                          ridx_n=ridx_n, sig_locs={'nin': nclk_idx, 'pout': pd0_tidx,
                                                   'nout': nd0_tidx})

            inv_master = self.new_template(InvCore, params=params)
            ncol = cur_col + m_ncol + s_ncol + blk_sp + inv_master.num_cols + m_inv_sp
            scol = cur_col + m_ncol + inv_master.num_cols + blk_sp + m_inv_sp + extra_sp
            b_inst = self.add_tile(inv_master, 0, cur_col + m_ncol + m_inv_sp)
            self._cntr_col_clk = scol - (blk_sp + extra_sp) // 2
        else:
            ncol = cur_col + m_ncol + s_ncol + blk_sp
            scol = cur_col + m_ncol + blk_sp + extra_sp
            self._cntr_col_clk = scol - (blk_sp + extra_sp) // 2
            b_inst = None

        # set size
        self.set_mos_size(ncol)

        # add instances
        m_inst = self.add_tile(m_master, 0, cur_col)
        s_inst = self.add_tile(s_master, 0, scol)
        inst_list.append(m_inst)
        inst_list.append(s_inst)

        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in inst_list:
            vss_list.append(inst.get_pin('VSS'))
            vdd_list.append(inst.get_pin('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # connect intermediate node
        self.connect_wires([s_inst.get_pin('nin'), m_inst.get_pin('nout')])
        # connect clocks
        pclkb = self.connect_wires([s_inst.get_pin('pclkb'), m_inst.get_pin('pclk')])
        if b_inst is None:
            self.connect_wires([s_inst.get_pin('nclk'), m_inst.get_pin('nclkb')])
            self.reexport(m_inst.get_port('clk'), net_name='clkb')
            self.reexport(m_inst.get_port('nclk'), net_name='nclkb')
        else:
            self.connect_wires([s_inst.get_pin('nclk'), m_inst.get_pin('nclkb'),
                                b_inst.get_pin('nin')])
            self.connect_to_track_wires(pclkb, b_inst.get_pin('out'))

        # connect rst if rst is True
        if rst:
            rst_warr = self.connect_wires([s_inst.get_pin('nrst'), m_inst.get_pin('nrst')])
            self.add_pin('nrst', rst_warr, hide=True)
            self.add_pin('prst', rst_warr, hide=True)
            self.add_pin('rst', [s_inst.get_pin('rst'), m_inst.get_pin('rst')], label='rst:')

        # connect mux output to flop input if scan is true
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if scan:
            flop_in = m_inst.get_pin('nin')
            mux_out_vm_tidx = self.grid.coord_to_track(vm_layer, flop_in.lower,
                                                       mode=RoundMode.NEAREST)
            mux_out = [mux_inst.get_pin('pout'), mux_inst.get_pin('nout'), flop_in]
            flop_in_vm = self.connect_to_tracks(mux_out, TrackID(vm_layer, mux_out_vm_tidx))
            self.add_pin('flop_in', flop_in_vm, hide=True)

        # add pins
        # NOTE: use reexport so hidden pins propagate correctly
        if scan:
            self.reexport(mux_inst.get_port('in<0>'), net_name='in', hide=False)
            self.reexport(mux_inst.get_port('in<1>'), net_name='scan_in', hide=False)
            self.reexport(mux_inst.get_port('sel'), net_name='scan_en', hide=False)
        else:
            self.reexport(m_inst.get_port('in'))
        self.reexport(m_inst.get_port('nin'))
        self.reexport(m_inst.get_port('pin'))
        self.reexport(s_inst.get_port('out'))
        self.reexport(s_inst.get_port('nout'))
        self.reexport(s_inst.get_port('pout'))
        self.reexport(m_inst.get_port('clkb'), net_name='clk')
        self.reexport(m_inst.get_port('nclkb'), net_name='nclk')
        self.reexport(s_inst.get_port('outb'), net_name='outb')
        self.reexport(s_inst.get_port('noutb'), net_name='noutb')
        self.reexport(s_inst.get_port('poutb'), net_name='poutb')
        if rst:
            self.reexport(m_inst.get_port('mid_vm'), net_name='mid_vm_m', hide=True)
            self.reexport(s_inst.get_port('mid_vm'), net_name='mid_vm_s', hide=True)

        # set properties
        if rst:
            self.sch_params = dict(
                m_params=m_master.sch_params,
                s_params=s_master.sch_params,
                inv_params=inv_master.sch_params if inv_master else None
            )
        else:
            self.sch_params = dict(
                lch=self.place_info.lch,
                w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
                w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
                th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
                th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
                seg_m=m_master.sch_params['seg_dict'],
                seg_s=s_master.sch_params['seg_dict'],
                seg_ck=seg_ck,
            )

        if scan:
            self.sch_params = dict(
                rst_flop_params=self.sch_params,
                mux_params=mux_master.sch_params,
            )


class RstLatchCore(MOSBase):
    """A transmission gate based latch with reset pin."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._seg_in = None

    @property
    def seg_in(self) -> int:
        return self._seg_in

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'rst_latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output NOR.',
            seg_dict='Dictionary of number of segments',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='input stage fanout.',
            fanout_kp='keeper stage fanout.',
            vertical_clk='True to have vertical clk and clkb',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
            seg_dict=None,
            seg=1,
            vertical_clk=True,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        hm_layer = pinfo.conn_layer + 1
        vm_layer = hm_layer + 1
        if pinfo.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')
        # setup floorplan
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_dict: Optional[Dict[str, int]] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']
        vertical_clk: bool = self.params['vertical_clk']

        # compute track locations and create masters
        tr_manager = pinfo.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        if sig_locs is None:
            sig_locs = {}

        ng0 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg0 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg2 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=2)
        nd0 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        nd1 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd0 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        if seg_dict is None:
            seg_t1 = max(1, int(round(seg / (2 * fanout_kp))) * 2)
            seg_t0 = self._seg_in = max(2 * seg_t1, max(2, int(round(seg / (2 * fanout_in))) * 2))
        else:
            seg = seg_dict['nor']
            seg_t1 = seg_dict['keep']
            seg_t0 = seg_dict['in']

        nor_sig_locs = dict(
            nin0=sig_locs.get('rst', pg2),
            nin1=sig_locs.get('nclk', ng0),
            pout=pd1,
            nout=nd1,
        )
        params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      sig_locs=nor_sig_locs)
        nor_master = self.new_template(NOR2Core, params=params)

        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        t0_sig_locs = dict(
            nin=sig_locs.get(key, pg0),
            pout=pd0,
            nout=nd0,
            nen=sig_locs.get('nclk', ng0),
            pen=sig_locs.get('pclkb', pg1),
        )
        params = dict(pinfo=pinfo, seg=seg_t0, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=t0_sig_locs)
        t0_master = self.new_template(InvTristateCore, params=params)

        t1_sig_locs = dict(
            nin=sig_locs.get('pout', pg1),
            pout=pd0,
            nout=nd0,
            nen=sig_locs.get('nclkb', ng1),
            pen=sig_locs.get('pclk', pg0),
        )
        params = dict(pinfo=pinfo, seg=seg_t1, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=t1_sig_locs)
        t1_master = self.new_template(InvTristateCore, params=params)

        # set size
        blk_sp = max(self.min_sep_col, self.get_hm_sp_le_sep_col())
        t0_ncol = t0_master.num_cols
        t1_ncol = t1_master.num_cols
        nor_ncol = nor_master.num_cols
        num_cols = t0_ncol + t1_ncol + nor_ncol + blk_sp * 2
        self.set_mos_size(num_cols)

        # add instances
        t1_col = t0_ncol + blk_sp
        nor_col = num_cols - nor_ncol
        t0 = self.add_tile(t0_master, 0, 0)
        t1 = self.add_tile(t1_master, 0, t1_col)
        nor = self.add_tile(nor_master, 0, nor_col)

        # routing
        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in (t0, t1, nor):
            vss_list.append(inst.get_pin('VSS'))
            vdd_list.append(inst.get_pin('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # export input
        self.reexport(t0.get_port('nin'), net_name='nin', hide=True)
        self.reexport(t0.get_port('pin'), net_name='pin', hide=True)
        self.reexport(t0.get_port('in'))

        # connect output
        out = nor.get_pin('out')
        in2 = t1.get_pin('nin')
        self.connect_to_track_wires(in2, out)
        self.add_pin('out', out)
        self.add_pin('nout', in2, hide=True)
        self.add_pin('pout', in2, hide=True)

        # connect middle node
        mid_coord = (t1.bound_box.xh + nor.bound_box.xl) // 2
        mid_tidx = self.grid.coord_to_track(vm_layer, mid_coord, RoundMode.NEAREST)
        mid_tid = TrackID(vm_layer, mid_tidx, width=tr_w_v)
        if out.layer_id == vm_layer:
            next_tidx = tr_manager.get_next_track(vm_layer, mid_tidx, 'sig', 'sig', up=True)
            if next_tidx >= out.track_id.base_index:
                raise ValueError('oops!')

        warrs = [t0.get_pin('pout'), t0.get_pin('nout'), t1.get_pin('pout'), t1.get_pin('nout'),
                 nor.get_pin('nin<1>')]
        mid_vm_warr = self.connect_to_tracks(warrs, mid_tid)

        # connect clocks
        clk_tidx = sig_locs.get('clk', pinfo.get_source_track(t1_col + 1))
        clkb_tidx = sig_locs.get('clkb', pinfo.get_source_track(t1_col - blk_sp - 1))
        clk_tid = TrackID(vm_layer, clk_tidx, width=tr_w_v)
        clkb_tid = TrackID(vm_layer, clkb_tidx, width=tr_w_v)
        t0_en = t0.get_pin('en')
        t1_en = t1.get_pin('en')
        t1_enb = t1.get_pin('enb')
        t0_enb = t0.get_pin('enb')
        clk_hm = [t0_en, t1_enb]
        clkb_hm = [t0_enb, t1_en]
        if vertical_clk:
            clk = self.connect_to_tracks(clk_hm, clk_tid)
            clkb = self.connect_to_tracks(clkb_hm, clkb_tid)
            self.add_pin('clk', clk)
            self.add_pin('clkb', clkb)

        self.add_pin('outb', [nor.get_pin('in<1>'), mid_vm_warr])
        self.add_pin('noutb', nor.get_pin('nin<1>'), hide=True)
        self.add_pin('poutb', nor.get_pin('nin<1>'), hide=True)
        self.add_pin('rst', nor.get_pin('in<0>'))
        self.add_pin('nrst', nor.get_pin('nin<0>'), hide=True)
        self.add_pin('prst', nor.get_pin('nin<0>'), hide=True)
        self.add_pin('mid_vm', mid_vm_warr, hide=True)


        self.add_pin('nclk', t0_en, label='clk:', hide=vertical_clk)
        self.add_pin('pclk', t1_enb, label='clk:', hide=vertical_clk)
        self.add_pin('nclkb', t1_en, label='clkb:', hide=vertical_clk)
        self.add_pin('pclkb', t0_enb, label='clkb:', hide=vertical_clk)

        # set properties
        self.sch_params = dict(
            tin=t0_master.sch_params,
            tfb=t1_master.sch_params,
            nor=nor_master.sch_params,
        )


class RstLatchCore2Row(MOSBase):
    """A transmission gate based latch with reset pin and optional scanability, 2 row layout."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._seg_in = None

    @property
    def seg_in(self) -> int:
        return self._seg_in

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'scan_rst_latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output NOR.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='input stage fanout.',
            fanout_kp='keeper stage fanout.',
            scan='True to enable scanability.',
            dual_output='True to show out and outb',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
            scan=True,
            dual_output=True,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        hm_layer = pinfo.conn_layer + 1
        vm_layer = hm_layer + 1
        if pinfo.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # setup floorplan
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']
        scan: bool = self.params['scan']
        dual_output: bool = self.params['dual_output']

        # compute track locations and create masters
        tr_manager = pinfo.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        if sig_locs is None:
            sig_locs = {}

        ng0 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        ng2 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        pg0 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg2 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=2)
        nd0 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        nd1 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd0 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        seg_t1 = max(1, int(round(seg / (2 * fanout_kp))) * 2)
        seg_tg = max(2 * seg_t1, max(2, int(round(seg / (2 * fanout_in))) * 2))
        seg_t0 = self._seg_in = max(2, int(round(seg_tg / (2 * fanout_in))) * 2)
        seg_inv = seg_t0

        nor_sig_locs = dict(
            nin0=sig_locs.get('rst', ng2),
            nin1=sig_locs.get('nclk', pg0 if scan else ng0),
            pout=pd1,
            nout=nd1,
        )
        params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      sig_locs=nor_sig_locs)
        nor_master = self.new_template(NOR2Core, params=params)

        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        t0_sig_locs = dict(
            nin=sig_locs.get(key, pg2),
            pout=pd0,
            nout=nd0,
            nen=sig_locs.get('nclk_in', ng1),
            pen=sig_locs.get('pclkb_in', pg1),
        )
        params = dict(pinfo=pinfo, seg=seg_t0, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=t0_sig_locs)
        t0_in_master = self.new_template(InvTristateCore, params=params)

        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        t0_sig_locs = dict(
            nin=sig_locs.get(key, pg2),
            pout=pd0,
            nout=nd1,
            nen=sig_locs.get('nclk_scan', ng1),
            pen=sig_locs.get('pclkb_scan', pg1),
        )
        params = dict(pinfo=pinfo, seg=seg_t0, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=t0_sig_locs)
        t0_scan_master = self.new_template(InvTristateCore, params=params)

        t1_sig_locs = dict(
            nin=sig_locs.get('pout', pg1),
            pout=pd0,
            nout=nd1,
            nen=sig_locs.get('nclkb', ng2),
            pen=sig_locs.get('pclk', pg0),
        )
        params = dict(pinfo=pinfo, seg=seg_t1, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=t1_sig_locs)
        t1_master = self.new_template(InvTristateCore, params=params)

        tg_sig_locs = dict(
            ns=sig_locs.get('pout', nd0),
        )
        params = dict(pinfo=pinfo, seg=seg_tg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=True, sig_locs=tg_sig_locs)
        tg_in_master = self.new_template(PassGateCore, params=params)

        tg_sig_locs = dict(
            ns=sig_locs.get('pout', ng1),
        )
        params = dict(pinfo=pinfo, seg=seg_tg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=True, sig_locs=tg_sig_locs)
        tg_scan_master = self.new_template(PassGateCore, params=params)

        inv_sig_locs = dict()
        params = dict(pinfo=pinfo, seg=seg_inv, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=inv_sig_locs)
        inv_master = self.new_template(InvCore, params=params)

        # set size
        blk_sp = max(self.min_sep_col, self.get_hm_sp_le_sep_col())
        t0_ncol = t0_in_master.num_cols
        t1_ncol = t1_master.num_cols
        nor_ncol = nor_master.num_cols
        tg_ncol = tg_in_master.num_cols
        inv_ncol = inv_master.num_cols

        num_cols = t0_ncol + blk_sp + max(t1_ncol, nor_ncol)
        if scan:
            num_cols += inv_ncol + blk_sp + tg_ncol + blk_sp

        self.set_mos_size(num_cols, 2)

        # --- Placement --- #
        inv_col = 0
        t0_col = inv_col + inv_ncol + blk_sp if scan else 0
        tg_col = t0_col + t0_ncol + blk_sp
        nor_col = tg_col + tg_ncol + blk_sp if scan else tg_col

        # add instances
        inv = self.add_tile(inv_master, 0, inv_col) if scan else None
        t0_in = self.add_tile(t0_in_master, 1, t0_col)
        t0_scan = self.add_tile(t0_scan_master, 0, t0_col) if scan else None
        tg_in = self.add_tile(tg_in_master, 1, tg_col) if scan else None
        tg_scan = self.add_tile(tg_scan_master, 0, tg_col) if scan else None
        nor = self.add_tile(nor_master, 1, nor_col)
        t1 = self.add_tile(t1_master, 0, nor_col + t1_ncol, flip_lr=True)

        # --- Routing --- #
        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        inst_list = [t0_in, nor, t1]
        if scan:
            inst_list += [inv, t0_scan, tg_in, tg_scan]
        for inst in inst_list:
            vss_list.append(inst.get_pin('VSS'))
            vdd_list.append(inst.get_pin('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # export input
        self.reexport(t0_in.get_port('nin'), net_name='in', hide=False)

        # export rst
        self.reexport(nor.get_port('nin<0>'), net_name='rst', hide=False)

        # connect clk and clkb
        t0_in_nout = t0_in.get_pin('nout')
        mid_tidx = self.grid.coord_to_track(vm_layer, t0_in_nout.middle, mode=RoundMode.NEAREST)
        clk_vm_tidx = tr_manager.get_next_track(vm_layer, mid_tidx, 'sig', 'sig', up=False)
        clkb_vm_tidx = tr_manager.get_next_track(vm_layer, mid_tidx, 'sig', 'sig', up=True)
        inb_vm_tidx = tr_manager.get_next_track(vm_layer, clkb_vm_tidx, 'sig', 'sig', up=True)
        clk_in = t0_in.get_pin('en')
        clkb_in = t0_in.get_pin('enb')
        self.add_pin('clk', clk_in)
        self.add_pin('clkb', clkb_in)
        clk_hm_warrs = [clk_in, t1.get_pin('enb')]
        clkb_hm_warrs = [clkb_in, t1.get_pin('en')]

        # connect output of scan tristate inverter to input of scan pass gate
        inb_warrs = [t0_in.get_pin(name) for name in ['pout', 'nout']]
        inb_warrs += [tg_in.get_pin('s')] if scan else [nor.get_pin('nin<1>'), t1.get_pin('nout'),
                                                        t1.get_pin('pout')]
        self.connect_to_tracks(inb_warrs, TrackID(vm_layer, inb_vm_tidx, width=tr_w_v))

        # export scan input
        if scan:
            self.reexport(t0_scan.get_port('nin'), net_name='scan_in', hide=False)

            # connect scan_en inverter
            tg_out_tidx = tg_scan.get_pin('d').track_id.base_index
            scan_en_tidx = tr_manager.get_next_track(vm_layer, tg_out_tidx, 'sig', 'sig', up=False)
            scan_enb_tidx = tr_manager.get_next_track(vm_layer, tg_out_tidx, 'sig', 'sig', up=True)

            scan_en = inv.get_pin('in')
            self.add_pin('scan_en', scan_en)
            self.connect_to_tracks([scan_en, tg_scan.get_pin('en'), tg_in.get_pin('enb')],
                                   TrackID(vm_layer, scan_en_tidx, width=tr_w_v))

            self.connect_to_tracks([inv.get_pin('pout'), inv.get_pin('nout'),
                                    tg_scan.get_pin('enb'), tg_in.get_pin('en')],
                                   TrackID(vm_layer, scan_enb_tidx, width=tr_w_v))

            # clk and clkb of scan tri-state
            clk_hm_warrs.append(t0_scan.get_pin('en'))
            clkb_hm_warrs.append(t0_scan.get_pin('enb'))

            # connect output of scan tristate inverter to input of scan pass gate
            self.connect_to_tracks([t0_scan.get_pin('nout'), t0_scan.get_pin('pout'),
                                    tg_scan.get_pin('s')],
                                   TrackID(vm_layer, inb_vm_tidx, width=tr_w_v))

            # connect passgate outputs together to in1 of nor and out of feedback tristate
            self.connect_to_track_wires([nor.get_pin('nin<1>'), t1.get_pin('pout'),
                                         t1.get_pin('nout')],
                                        self.connect_wires([tg_scan.get_pin('d'),
                                                            tg_in.get_pin('d')]))

        # finish clk and clkb connection
        clk_vm = self.connect_to_tracks(clk_hm_warrs, TrackID(vm_layer, clk_vm_tidx, width=tr_w_v))
        clkb_vm = self.connect_to_tracks(clkb_hm_warrs, TrackID(vm_layer, clkb_vm_tidx,
                                                                width=tr_w_v))
        self.add_pin('clk_vm', clk_vm, hide=True)
        self.add_pin('clkb_vm', clkb_vm, hide=True)

        # connect nor output to feedback tristate input
        out = self.connect_to_track_wires(t1.get_pin('nin'), nor.get_pin('out'))
        self.add_pin('out', out)
        self.reexport(nor.get_port('pout'))
        self.reexport(nor.get_port('nout'))
        self.reexport(nor.get_port('nin<1>'), net_name='outb', hide=not dual_output)

        # set properties
        self.sch_params = dict(
            tin=t0_in_master.sch_params,
            tfb=t1_master.sch_params,
            nor=nor_master.sch_params,
            scan=scan,
            dual_output=dual_output,
        )
        if scan:
            self.sch_params.update(dict(
                pg=tg_in_master.sch_params,
                inv=inv_master.sch_params,
            ))


class FlopCore2Row(MOSBase):
    """A transmission gate based flip-flop in 2 rows."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._cntr_col_clk = None

    @property
    def seg_in(self):
        return self.sch_params['seg_m']['in']

    @property
    def cntr_col_clk(self):
        return self._cntr_col_clk

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'scan_rst_flop')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output inverter.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='latch input stage fanout.',
            fanout_kp='latch keeper stage fanout.',
            fanout_lat='fanout between latches.',
            seg_ck='number of segments for clock inverter.  0 to disable.',
            resetable='True if flop is resetable, default is False',
            scan='True if flop needs to have scanability',
            extra_sp='This parameter is added to the min value of one of the separations '
                     '(mostly used to make power vertical stripes aligned)'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
            fanout_lat=4,
            fanout_mux=4,
            seg_ck=0,
            seg_mux=None,
            resetable=False,
            scan=False,
            extra_sp=0,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']
        fanout_lat: float = self.params['fanout_lat']
        seg_ck: int = self.params['seg_ck']
        rst: bool = self.params['resetable']
        scan: bool = self.params['scan']
        extra_sp: int = self.params['extra_sp']

        if rst is False:
            raise NotImplementedError('2 row Latch not implemented yet')

        # setup floorplan
        self.draw_base(pinfo)

        # compute track locations
        if sig_locs is None:
            sig_locs = {}
        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        in_idx = sig_locs.get(key, self.get_track_index(ridx_p, MOSWireType.G,
                                                        wire_name='sig', wire_idx=0))
        pclkb_idx = sig_locs.get('pclkb', self.get_track_index(ridx_p, MOSWireType.G,
                                                               wire_name='sig', wire_idx=0))
        nclk_idx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        nclkb_idx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        pclk_idx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        clk_idx = sig_locs.get('clk', None)
        clkb_idx = sig_locs.get('clkb', None)

        # make masters
        s_sig_locs = {
            'pclkb_in': pclkb_idx,
        }
        lat_params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                          sig_locs=s_sig_locs,
                          fanout_in=fanout_in, fanout_kp=fanout_kp, scan=False)

        s_master = self.new_template(RstLatchCore2Row, params=lat_params)
        seg_m = max(2, int(round(s_master.seg_in / (2 * fanout_lat))) * 2)
        lat_params['seg'] = seg_m
        lat_params['scan'] = scan
        lat_params['dual_output'] = False
        lat_params['sig_locs'] = {}

        m_master = self.new_template(RstLatchCore2Row, params=lat_params)

        cur_col = 0
        blk_sp = self.min_sep_col
        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)

        inst_list = []
        m_ncol = m_master.num_cols
        s_ncol = s_master.num_cols
        m_inv_sp = blk_sp if rst else 0
        inv_master = None
        if seg_ck > 0:
            params = dict(pinfo=pinfo, seg=seg_ck, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                          ridx_n=ridx_n, sig_locs={'nin': pclk_idx, 'pout': pd0_tidx,
                                                   'nout': nd0_tidx})

            inv_master = self.new_template(InvCore, params=params)
            ncol = cur_col + m_ncol + s_ncol + blk_sp + inv_master.num_cols + m_inv_sp
            scol = cur_col + m_ncol + inv_master.num_cols + blk_sp + m_inv_sp + extra_sp
            b_inst = self.add_tile(inv_master, 1, cur_col + m_ncol + m_inv_sp)
            self._cntr_col_clk = scol - (blk_sp + extra_sp) // 2
        else:
            ncol = cur_col + m_ncol + s_ncol + blk_sp
            scol = cur_col + m_ncol + blk_sp + extra_sp
            self._cntr_col_clk = scol - (blk_sp + extra_sp) // 2
            b_inst = None

        # set size
        self.set_mos_size(ncol)

        # add instances
        m_inst = self.add_tile(m_master, 0, cur_col)
        s_inst = self.add_tile(s_master, 0, scol)
        inst_list.append(m_inst)
        inst_list.append(s_inst)

        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in inst_list:
            vss_list += inst.get_all_port_pins('VSS')
            vdd_list += inst.get_all_port_pins('VDD')
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # export input and output
        self.reexport(m_inst.get_port('in'))
        self.reexport(s_inst.get_port('out'))
        self.reexport(s_inst.get_port('outb'))

        # connect output of m_inst to input of s_inst
        self.connect_to_track_wires(s_inst.get_pin('in'), m_inst.get_pin('out'))

        # connect rst
        if rst:
            rst_sig = self.connect_wires([m_inst.get_pin('rst'), s_inst.get_pin('rst')])
            self.add_pin('rst', rst_sig)

        # export scan pins
        if scan:
            self.reexport(m_inst.get_port('scan_in'))
            self.reexport(m_inst.get_port('scan_en'))

        # connect clk to clkb of m_inst and input of inverter
        clk_hm = m_inst.get_pin('clkb')
        clk = self.connect_to_track_wires([clk_hm, b_inst.get_pin('in')],
                                          s_inst.get_pin('clk_vm'))
        self.add_pin('clk', clk_hm)

        # connect clkb to clk of m_inst and output of inverter
        self.connect_to_track_wires([m_inst.get_pin('clk'), s_inst.get_pin('clkb')],
                                    b_inst.get_pin('out'))

        # set properties
        self.sch_params = dict(
            m_params=m_master.sch_params,
            s_params=s_master.sch_params,
            inv_params=inv_master.sch_params if inv_master else None
        )


class ScanRstLatchCore(MOSBase):
    """A transmission gate based latch with reset pin and scanability, optimized."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._seg_in = None

    @property
    def seg_in(self) -> int:
        return self._seg_in

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'scan_rst_latch2')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of number of segments.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            dual_output='True to show out and outb',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            dual_output=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        hm_layer = pinfo.conn_layer + 1
        vm_layer = hm_layer + 1
        if pinfo.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # setup floorplan
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        dual_output: bool = self.params['dual_output']

        # compute track locations and create masters
        tr_manager = pinfo.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        if sig_locs is None:
            sig_locs = {}

        ng0 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg0 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-3)
        pg1 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-2)
        pg2 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)
        nd0 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        nd1 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd0 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd2 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=2)
        seg_kp = seg_dict['keep']
        seg_tin = seg_dict['in']
        seg_pg = seg_dict['pass']
        seg_nor = seg_dict['nor']
        for key, val in seg_dict.items():
            if val != 1:
                raise ValueError('Layout optimized for seg = 1 only')

        nor_sig_locs = dict(
            nin0=sig_locs.get('rst', pg0),
            nin1=sig_locs.get('nclk', pg1),
            pout=sig_locs.get('pout', pd2),
            nout=sig_locs.get('nout', nd1),
        )
        params = dict(pinfo=pinfo, seg=seg_nor, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      sig_locs=nor_sig_locs)
        nor_master = self.new_template(NOR2Core, params=params)

        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        tin_sig_locs = dict(
            # nin=sig_locs.get(key, pg2),
            # pout=pd0,
            # nout=nd0,
            # nen=sig_locs.get('nclk_in', ng2),
            # pen=sig_locs.get('pclkb_in', pg0),
        )
        params = dict(pinfo=pinfo, seg=seg_tin, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=tin_sig_locs)
        tin_master = self.new_template(InvTristateCore, params=params)

        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        tscan_sig_locs = dict(
            nin=sig_locs.get(key, pg1),
            # pout=pd0,
            # nout=nd1,
            # nen=sig_locs.get('nclk_scan', ng2),
            # pen=sig_locs.get('pclkb_scan', pg0),
        )
        params = dict(pinfo=pinfo, seg=seg_tin, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=tscan_sig_locs)
        tscan_master = self.new_template(InvTristateCore, params=params)

        kp_sig_locs = dict(
            # nin=sig_locs.get('pout', pg1),
            pout=pd1,
            nout=nd0,
            # nen=sig_locs.get('nclkb', ng2),
            # pen=sig_locs.get('pclk', pg0),
        )
        params = dict(pinfo=pinfo, seg=seg_kp, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=kp_sig_locs)
        kp_master = self.new_template(InvTristateCore, params=params)

        # set size
        blk_sp = max(self.min_sep_col, self.get_hm_sp_le_sep_col())
        tin_ncol = tin_master.num_cols
        kp_ncol = kp_master.num_cols
        nor_ncol = nor_master.num_cols

        num_cols = tin_ncol + blk_sp + 2 * seg_pg + blk_sp + tin_ncol + blk_sp + kp_ncol + blk_sp\
                   + nor_ncol

        self.set_mos_size(num_cols, 1)

        # --- Placement --- #

        # add instances
        cur_col = 0
        tin = self.add_tile(tin_master, 0, cur_col)
        cur_col += tin_ncol + blk_sp

        passg_n = self.add_mos(ridx_n, cur_col, 2 * seg_pg)
        passg_p = self.add_mos(ridx_p, cur_col, 2 * seg_pg)
        cur_col += 2 * seg_pg + blk_sp

        tscan = self.add_tile(tscan_master, 0, cur_col + tin_ncol, flip_lr=True)
        cur_col += tin_ncol + blk_sp

        kp = self.add_tile(kp_master, 0, cur_col + kp_ncol, flip_lr=True)
        cur_col += kp_ncol + blk_sp

        nor = self.add_tile(nor_master, 0, cur_col)

        # --- Routing --- #
        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        inst_list = [tin, tscan, kp, nor]
        for inst in inst_list:
            vss_list.append(inst.get_pin('VSS'))
            vdd_list.append(inst.get_pin('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # export input and scan_input
        self.reexport(tin.get_port('nin'), net_name='in', hide=False)
        self.reexport(tscan.get_port('nin'), net_name='scan_in', hide=False)

        # export rst
        self.reexport(nor.get_port('nin<0>'), net_name='rst', hide=False)

        # connect clk and clkb
        clk_hm_tid = TrackID(hm_layer, ng0)
        clkb_hm_tid = TrackID(hm_layer, pg2)
        clk_hm_l = self.connect_to_tracks(passg_n.g, clk_hm_tid, min_len_mode=MinLenMode.MIDDLE)
        clkb_hm_l = self.connect_to_tracks(passg_p.g, clkb_hm_tid, min_len_mode=MinLenMode.MIDDLE)
        self.add_pin('clk_pass', clk_hm_l, label='clk:')
        self.add_pin('clk_keep', kp.get_pin('enb'), label='clk:')
        self.add_pin('clkb_pass', clkb_hm_l, label='clkb:')
        self.add_pin('clkb_keep', kp.get_pin('en'), label='clkb:')

        # connect scan and scanb
        self.add_pin('scan_en', tscan.get_pin('en'), label='scan_en:')
        self.add_pin('in_en', tin.get_pin('enb'), label='scan_en:')
        self.add_pin('scan_enb', tscan.get_pin('enb'), label='scan_enb:')
        self.add_pin('in_enb', tin.get_pin('en'), label='scan_enb:')

        # connect in and scan tristate outputs to passgate
        self.connect_to_track_wires(passg_p.s[0], tin.get_pin('pout'))
        self.connect_to_track_wires(passg_p.s[1], tscan.get_pin('pout'))

        self.connect_to_track_wires(passg_n.s[0], tin.get_pin('nout'))
        self.connect_to_track_wires(passg_n.s[1], tscan.get_pin('nout'))

        self.connect_wires([passg_p.s[0], passg_n.s[0]])
        self.connect_wires([passg_p.s[1], passg_n.s[1]])

        # connect outb
        poutb = self.connect_to_track_wires(passg_p.d, kp.get_pin('pout'))
        noutb = self.connect_to_track_wires(passg_n.d, kp.get_pin('nout'))

        out_vm = nor.get_pin('out')
        outb_vm_tid = tr_manager.get_next_track_obj(out_vm, 'sig', 'sig')
        outb = self.connect_to_tracks([poutb, noutb, nor.get_pin('nin<1>')], outb_vm_tid)
        self.add_pin('out', out_vm)
        self.add_pin('outb', outb, hide=not dual_output)

        self.reexport(nor.get_port('pout'))
        self.reexport(nor.get_port('nout'))

        # connect out to input of keeper tristate
        self.connect_to_track_wires(kp.get_pin('nin'), out_vm)

        passg_sch_params = nor_master.sch_params.copy()
        passg_sch_params['seg'] = seg_pg

        # set properties
        self.sch_params = dict(
            tin=tin_master.sch_params,
            tfb=kp_master.sch_params,
            nor=nor_master.sch_params,
            passg=passg_sch_params,
            dual_output=dual_output,
        )


class ScanRstFlopCore(MOSBase):
    """A transmission gate based flip-flop in 2 rows with 1 row latches."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._cntr_col_clk = None

    @property
    def seg_in(self):
        return self.sch_params['seg_m']['in']

    @property
    def cntr_col_clk(self):
        return self._cntr_col_clk

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'scan_rst_flop')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of number of segments.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            dual_output='True to export out and outb',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            dual_output=False,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        dual_output: bool = self.params['dual_output']

        # setup floorplan
        self.draw_base(pinfo)

        # compute track locations
        if sig_locs is None:
            sig_locs = {}

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        ng0 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg0 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-3)
        pg1 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-2)
        pg2 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)
        nd0 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        nd1 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd0 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)

        # make masters
        s_sig_locs = {
            'rst': pg1,
            'pout': pg2,
            'in': pg2,
        }
        lat_params = dict(pinfo=pinfo, seg_dict=seg_dict, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                          ridx_n=ridx_n, sig_locs=s_sig_locs, vertical_clk=False)

        s_master = self.new_template(RstLatchCore, params=lat_params)

        lat_params['seg_dict'] = seg_dict
        lat_params['dual_output'] = False
        lat_params['sig_locs'] = {}

        m_master = self.new_template(ScanRstLatchCore, params=lat_params)

        clk_inv_sig_locs = {
            'pout': pd1,
            'nout': nd1,
        }
        params = dict(pinfo=pinfo, seg=seg_dict['inv'], w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                      ridx_n=ridx_n, sig_locs=clk_inv_sig_locs, vertical_out=False)

        clk_inv_master = self.new_template(InvCore, params=params)

        scan_inv_sig_locs = {
            'nin': ng1,
        }
        params = dict(pinfo=pinfo, seg=seg_dict['inv'], w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                      ridx_n=ridx_n, sig_locs=scan_inv_sig_locs, vertical_out=False)

        scan_inv_master = self.new_template(InvCore, params=params)

        blk_sp = max(self.min_sep_col, self.get_hm_sp_le_sep_col())

        m_ncol = m_master.num_cols
        s_ncol = s_master.num_cols
        inv_ncol = clk_inv_master.num_cols
        num_cols = max(m_ncol, inv_ncol + blk_sp + inv_ncol + blk_sp + s_ncol)

        # set size
        self.set_mos_size(num_cols)

        # add instances
        m_inst = self.add_tile(m_master, 0, num_cols - m_ncol)

        cur_col = num_cols - s_ncol
        s_inst = self.add_tile(s_master, 1, cur_col)
        cur_col -= (blk_sp + inv_ncol)

        clk_inv_inst = self.add_tile(clk_inv_master, 1, cur_col + inv_ncol, flip_lr=True)
        # flipped to maintain VSS connection polarity
        cur_col -= (blk_sp + inv_ncol)

        scan_inv_inst = self.add_tile(scan_inv_master, 1, cur_col)

        inst_list = [m_inst, s_inst, clk_inv_inst, scan_inv_inst]

        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in inst_list:
            vss_list += inst.get_all_port_pins('VSS')
            vdd_list += inst.get_all_port_pins('VDD')
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        out_int = m_inst.get_pin('out')
        self.reexport(s_inst.get_port('out'))
        self.reexport(s_inst.get_port('mid_vm'), net_name='outb', hide=not dual_output)

        # connect rst of m_inst and s_inst
        rst = m_inst.get_pin('rst')
        self.add_pin('rst', rst)
        rst_vm_tid = self.tr_manager.get_next_track_obj(out_int, 'sig', 'sig', count_rel_tracks=-1)
        self.connect_to_tracks([rst, s_inst.get_pin('nrst')], rst_vm_tid)

        # connect output of m_inst to input of s_inst
        s_in = s_inst.get_pin('nin')
        int_vm_idx = self.grid.coord_to_track(vm_layer, s_in.middle, mode=RoundMode.NEAREST)
        self.connect_to_tracks([s_in, m_inst.get_pin('pout')], TrackID(vm_layer, int_vm_idx))

        # connect clk and clkb
        clk_keep_m = m_inst.get_pin('clk_keep')
        clk_r_vm_idx = self.grid.coord_to_track(vm_layer, clk_keep_m.middle, mode=RoundMode.NEAREST)
        clkb_r_vm_idx = self.tr_manager.get_next_track(vm_layer, clk_r_vm_idx, 'sig', 'sig',
                                                       up=True)
        self.connect_to_tracks([clk_inv_inst.get_pin('nin'), s_inst.get_pin('nclk'),
                                s_inst.get_pin('pclk'), m_inst.get_pin('clkb_keep')],
                               TrackID(vm_layer, clk_r_vm_idx))
        self.connect_to_tracks([s_inst.get_pin('nclkb'), s_inst.get_pin('pclkb'),
                                clk_keep_m, clk_inv_inst.get_pin('pout'),
                                clk_inv_inst.get_pin('nout')],
                               TrackID(vm_layer, clkb_r_vm_idx))

        clk_pass_m = m_inst.get_pin('clk_pass')
        clk_l_vm_idx = self.grid.coord_to_track(vm_layer, clk_pass_m.middle, mode=RoundMode.NEAREST)
        clkb_l_vm_idx = self.tr_manager.get_next_track(vm_layer, clk_l_vm_idx, 'sig', 'sig',
                                                       up=True)
        self.connect_to_tracks([clk_inv_inst.get_pin('nin'), m_inst.get_pin('clkb_pass')],
                               TrackID(vm_layer, clk_l_vm_idx))
        self.connect_to_tracks([clk_inv_inst.get_pin('pout'), clk_inv_inst.get_pin('nout'),
                                clk_pass_m], TrackID(vm_layer, clkb_l_vm_idx))

        # connect scan_en and scan_enb
        scan_en_r_vm_idx = self.tr_manager.get_next_track(vm_layer, int_vm_idx, 'sig', 'sig',
                                                          up=True)
        scan_enb_r_vm_idx = self.tr_manager.get_next_track(vm_layer, int_vm_idx, 'sig', 'sig',
                                                           up=False)
        self.connect_to_tracks([scan_inv_inst.get_pin('nin'), m_inst.get_pin('scan_en')],
                               TrackID(vm_layer, scan_en_r_vm_idx))
        self.connect_to_tracks([m_inst.get_pin('scan_enb'), scan_inv_inst.get_pin('pout'),
                                scan_inv_inst.get_pin('nout')],
                               TrackID(vm_layer, scan_enb_r_vm_idx))

        in_en_m = m_inst.get_pin('in_en')
        scan_en_l_vm_idx = self.grid.coord_to_track(vm_layer, in_en_m.middle,
                                                    mode=RoundMode.NEAREST)
        scan_enb_l_vm_idx = self.tr_manager.get_next_track(vm_layer, scan_en_l_vm_idx, 'sig', 'sig',
                                                           up=False)
        self.connect_to_tracks([scan_inv_inst.get_pin('nin'), in_en_m],
                               TrackID(vm_layer, scan_en_l_vm_idx))
        self.connect_to_tracks([m_inst.get_pin('in_enb'), scan_inv_inst.get_pin('pout'),
                                scan_inv_inst.get_pin('nout')],
                               TrackID(vm_layer, scan_enb_l_vm_idx))

        # export pins
        self.reexport(m_inst.get_port('in'))
        self.reexport(m_inst.get_port('scan_in'))
        self.reexport(scan_inv_inst.get_port('nin'), net_name='scan_en', hide=False)
        self.reexport(clk_inv_inst.get_port('nin'), net_name='clk', hide=False)

        # set properties
        self.sch_params = dict(
            m_params=m_master.sch_params,
            s_params=s_master.sch_params,
            inv_params=clk_inv_master.sch_params,
            dual_output=dual_output,
        )

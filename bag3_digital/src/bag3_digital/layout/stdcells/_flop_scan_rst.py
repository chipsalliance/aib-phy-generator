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

"""Layout generator for a scannable, resettable flop."""

from typing import Any, Dict, Optional, Union, Mapping, Type, List, Tuple

from pybag.enum import MinLenMode, RoundMode

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB
from xbase.layout.enum import MOSWireType, MOSType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase


class FlopScanRstlbTwoTile(MOSBase):
    """A scannable, resettable flop"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._substrate_row_intvl: List[Tuple[int, int]] = []

    @property
    def substrate_row_intvl(self) -> List[Tuple[int, int]]:
        return self._substrate_row_intvl

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'flop_scan_rstlb')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='segments dictionary.',
            w_dict='width dictionary.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal location dictionary.',
            vertical_rst='True to have rst on vm_layer',
            substrate_row='True to have dedicated substrate row.',
            tile0='Tile index of logic tile 0',
            tile1='Tile index of logic tile 1',
            flip_tile='True to flip tiles.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg_dict={},
            w_dict={},
            ridx_p=-1,
            ridx_n=0,
            sig_locs={},
            vertical_rst=False,
            substrate_row=False,
            tile0=0,
            tile1=1,
            flip_tile=False,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        flip_tile: bool = self.params['flip_tile']
        self.draw_base(pinfo, flip_tile=flip_tile)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_dict: Dict[str, int] = self.params['w_dict']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        vertical_rst: bool = self.params['vertical_rst']
        substrate_row: bool = self.params['substrate_row']

        min_sep = self.min_sep_col

        seg_buf = seg_dict.get('buf', 1)
        seg_in = seg_dict.get('in', 1)
        seg_mux = seg_dict.get('mux', 1)
        seg_keep = seg_dict.get('keep', 1)
        seg_pass = seg_dict.get('pass', 1)
        seg_rst = seg_dict.get('rst', 1)
        seg_out = seg_dict.get('out', 1)

        tile0 = self.params['tile0']
        tile1 = self.params['tile1']
        if substrate_row:
            vss0_tid = None
            vdd_tid = None
            vss1_tid = None
        else:
            vss0_tid = self.get_track_id(ridx_n, MOSWireType.DS, 'sup', tile_idx=tile0)
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS, 'sup')
            vss1_tid = self.get_track_id(ridx_n, MOSWireType.DS, 'sup', tile_idx=tile1)

        place_info = self.get_tile_pinfo(tile0)
        default_wn = place_info.get_row_place_info(ridx_n).row_info.width
        default_wp = place_info.get_row_place_info(ridx_p).row_info.width
        wp_buf = w_dict.get('p_buf', default_wp)
        wn_buf = w_dict.get('n_buf', default_wn)
        wp_in = w_dict.get('p_in', default_wp)
        wn_in = w_dict.get('n_in', default_wn)
        wp_mux = w_dict.get('p_mux', default_wp)
        wn_mux = w_dict.get('n_mux', default_wn)
        wp_keep = w_dict.get('p_keep', default_wp)
        wn_keep = w_dict.get('n_keep', default_wn)
        wp_pass = w_dict.get('p_pass', default_wp)
        wn_pass = w_dict.get('n_pass', default_wn)
        wp_rst = w_dict.get('p_rst', default_wp)
        wn_rst = w_dict.get('n_rst', default_wn)
        wp_out = w_dict.get('p_out', default_wp)
        wn_out = w_dict.get('n_out', default_wn)

        # tracks
        t0_nd0_tid = self.get_track_id(ridx_n, MOSWireType.DS, 'sig', wire_idx=0, tile_idx=tile0)
        t0_nd1_tid = self.get_track_id(ridx_n, MOSWireType.DS, 'sig', wire_idx=1, tile_idx=tile0)
        t0_ng0_tid = self.get_track_id(ridx_n, MOSWireType.G, 'sig', wire_idx=0, tile_idx=tile0)
        t0_ng1_tid = self.get_track_id(ridx_n, MOSWireType.G, 'sig', wire_idx=1, tile_idx=tile0)
        t0_pg0_tid = self.get_track_id(ridx_p, MOSWireType.G, 'sig', wire_idx=0, tile_idx=tile0)
        t0_pg1_tid = self.get_track_id(ridx_p, MOSWireType.G, 'sig', wire_idx=1, tile_idx=tile0)
        t0_pd0_tid = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', wire_idx=0, tile_idx=tile0)
        t0_pd1_tid = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', wire_idx=1, tile_idx=tile0)
        t1_pd1_tid = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', wire_idx=1, tile_idx=tile1)
        t1_pd0_tid = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', wire_idx=0, tile_idx=tile1)
        t1_pg1_tid = self.get_track_id(ridx_p, MOSWireType.G, 'sig', wire_idx=1, tile_idx=tile1)
        t1_pg0_tid = self.get_track_id(ridx_p, MOSWireType.G, 'sig', wire_idx=0, tile_idx=tile1)
        t1_ng1_tid = self.get_track_id(ridx_n, MOSWireType.G, 'sig', wire_idx=1, tile_idx=tile1)
        t1_ng0_tid = self.get_track_id(ridx_n, MOSWireType.G, 'sig', wire_idx=0, tile_idx=tile1)
        t1_nd1_tid = self.get_track_id(ridx_n, MOSWireType.DS, 'sig', wire_idx=1, tile_idx=tile1)
        t1_nd0_tid = self.get_track_id(ridx_n, MOSWireType.DS, 'sig', wire_idx=0, tile_idx=tile1)

        # placement
        # tile 0
        cur_col = 0
        mos_in_n = self.add_mos(ridx_n, cur_col, seg_in, w=wn_in, g_on_s=True, stack=2,
                                sep_g=True, tile_idx=tile0)
        mos_in_p = self.add_mos(ridx_p, cur_col, seg_in, w=wp_in, g_on_s=True, stack=2,
                                sep_g=True, tile_idx=tile0)
        cur_col += 2 * seg_in + min_sep

        mos_keep_n = self.add_mos(ridx_n, cur_col, seg_keep, w=wn_keep, g_on_s=True, stack=2,
                                  sep_g=True, tile_idx=tile0)
        mos_keep_p = self.add_mos(ridx_p, cur_col, seg_keep, w=wp_keep, g_on_s=True, stack=2,
                                  sep_g=True, tile_idx=tile0)
        cur_col += 2 * seg_keep + min_sep

        mos_pass_n = self.add_mos(ridx_n, cur_col, seg_pass, w=wn_pass, g_on_s=False,
                                  tile_idx=tile0)
        mos_pass_p = self.add_mos(ridx_p, cur_col, seg_pass, w=wp_pass, g_on_s=False,
                                  tile_idx=tile0)
        cur_col += seg_pass + min_sep

        mos_buf_n = self.add_mos(ridx_n, cur_col, 2 * seg_buf, w=wn_buf, g_on_s=True,
                                 sep_g=True, tile_idx=tile0)
        mos_buf_p = self.add_mos(ridx_p, cur_col, 2 * seg_buf, w=wp_buf, g_on_s=True,
                                 sep_g=True, tile_idx=tile0)

        cur_col += 2 * seg_buf + min_sep
        mos_out_n = self.add_mos(ridx_n, cur_col, 2 * seg_out, w=wn_out, tile_idx=tile0)
        mos_out_p = self.add_mos(ridx_p, cur_col, 2 * seg_out, w=wp_out, tile_idx=tile0)
        col_tot = cur_col + 2 * seg_out

        # tile 1
        cur_col = 0
        mos_si_n = self.add_mos(ridx_n, cur_col, seg_in, w=wn_in, g_on_s=True, stack=2,
                                sep_g=True, tile_idx=tile1)
        mos_si_p = self.add_mos(ridx_p, cur_col, seg_in, w=wp_in, g_on_s=True, stack=2,
                                sep_g=True, tile_idx=tile1)
        cur_col += 2 * seg_in + min_sep

        mos_mux_n = self.add_mos(ridx_n, cur_col, seg_mux, w=wn_mux, g_on_s=True, stack=2,
                                 sep_g=True, tile_idx=tile1)
        mos_mux_p = self.add_mos(ridx_p, cur_col, seg_mux, w=wp_mux, g_on_s=True, stack=2,
                                 sep_g=True, tile_idx=tile1)

        cur_col = col_tot - seg_rst - min_sep - 4 * seg_rst
        sub_col = cur_col
        mos_rst_n = self.add_mos(ridx_n, cur_col, 2 * seg_rst, w=wn_rst, g_on_s=True, stack=2,
                                 sep_g=True, tile_idx=tile1)
        mos_rst_p = self.add_mos(ridx_p, cur_col, 4 * seg_rst, w=wp_rst, g_on_s=True,
                                 sep_g=True, tile_idx=tile1)

        cur_col = col_tot - seg_rst
        mos_rst2_n = self.add_mos(ridx_n, cur_col, seg_rst, w=wn_rst, g_on_s=True, tile_idx=tile1)
        mos_rst2_p = self.add_mos(ridx_p, cur_col, seg_rst, w=wp_rst, g_on_s=False, tile_idx=tile1)

        if substrate_row:
            self._substrate_row_intvl.append((sub_col, self.num_cols-sub_col))
            for _tidx in range(pinfo[0].num_tiles):
                _pinfo = self.get_tile_pinfo(_tidx)
                for _ridx in range(_pinfo.num_rows):
                    rtype = _pinfo.get_row_place_info(_ridx).row_info.row_type
                    if rtype.is_substrate:
                        warrs = self.add_substrate_contact(_ridx, sub_col, tile_idx=_tidx,
                                                           seg=self.num_cols-sub_col)
                        sup_name = 'VDD' if rtype is MOSType.ntap else 'VSS'
                        self.add_pin(f'{sup_name}_sub', warrs, label=f'{sup_name}:')

        self.set_mos_size()

        # vertical routing track planning
        grid = self.grid
        tr_manager = self.tr_manager
        conn_layer = self.conn_layer
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        vm_se_tidx = grid.coord_to_track(vm_layer, 0)
        vm_seb_tidx = tr_manager.get_next_track(vm_layer, vm_se_tidx, 'sig', 'sig')
        vm_o1_tidx = tr_manager.get_next_track(vm_layer, vm_seb_tidx, 'sig', 'sig')
        vm_ck1_tidx = tr_manager.get_next_track(vm_layer, vm_o1_tidx, 'sig', 'sig')
        x_ck1 = grid.track_to_coord(conn_layer, mos_mux_p.g0.track_id.base_index)
        vm_ck1_tidx = max(vm_ck1_tidx, grid.coord_to_track(vm_layer, x_ck1, mode=RoundMode.LESS_EQ))
        x_ck2 = grid.track_to_coord(conn_layer, mos_rst_p.g[-1].track_id.base_index)
        vm_ck2_tidx = grid.coord_to_track(vm_layer, x_ck2, mode=RoundMode.GREATER_EQ)
        vm_o5_tidx = tr_manager.get_next_track(vm_layer, vm_ck2_tidx, 'sig', 'sig')
        vm_o42_tidx = tr_manager.get_next_track(vm_layer, vm_o5_tidx, 'sig', 'sig')
        x_ckb = grid.track_to_coord(conn_layer, mos_buf_n.s[0].track_id.base_index)
        vm_ckb_tidx = grid.coord_to_track(vm_layer, x_ckb, mode=RoundMode.GREATER_EQ)
        vm_seb2_tidx = tr_manager.get_next_track(vm_layer, vm_ck2_tidx, 'sig', 'sig', up=False)
        x_o41 = grid.track_to_coord(conn_layer, mos_pass_n.d.track_id.base_index)
        vm_o41_tidx = grid.coord_to_track(vm_layer, x_o41, mode=RoundMode.LESS_EQ)
        x_o2 = x_o41 - self.place_info.sd_pitch * 2
        vm_o2_tidx = grid.coord_to_track(vm_layer, x_o2, mode=RoundMode.LESS_EQ)
        # routing
        # supplies
        vdd_list = [mos_in_p.s, mos_keep_p.d, mos_buf_p.d, mos_out_p.d,
                    mos_si_p.s, mos_mux_p.d, mos_rst_p.s[0:2], mos_rst2_p.s]
        vss0_list = [mos_in_n.s, mos_keep_n.d, mos_buf_n.d, mos_out_n.d]
        vss1_list = [mos_si_n.s, mos_mux_n.d, mos_rst_n.d]
        if substrate_row:
            if tile0 < tile1:
                self.add_pin('VSS1', vss1_list, label='VSS:')
                self.add_pin('VSS0', vss0_list, label='VSS:')
            else:
                self.add_pin('VSS1', vss0_list, label='VSS:')
                self.add_pin('VSS0', vss1_list, label='VSS:')
            self.add_pin('VDD', vdd_list, label='VDD:')
        else:
            self.add_pin('VDD', self.connect_to_tracks(vdd_list, vdd_tid))
            self.add_pin('VSS', self.connect_to_tracks(vss0_list, vss0_tid), connect=True)
            self.add_pin('VSS', self.connect_to_tracks(vss1_list, vss1_tid), connect=True)

        # input
        self.add_pin('in', self.connect_to_tracks([mos_in_n.g[0], mos_in_p.g[0]], t0_pg1_tid))
        # scan in
        self.add_pin('scan_in', self.connect_to_tracks([mos_si_n.g[0], mos_si_p.g[0]],
                                                       t1_ng1_tid))
        # scan enable
        se = self.connect_to_tracks([mos_in_p.g[1], mos_buf_p.g[1], mos_buf_n.g[1]], t0_pg0_tid)
        se_top = self.connect_to_tracks(mos_si_n.g[1], t1_ng0_tid, min_len_mode=MinLenMode.LOWER)
        se_vm = self.connect_to_tracks([se, se_top], TrackID(vm_layer, vm_se_tidx, width=vm_w))
        self.add_pin('scan_en', se)
        self.add_pin('scan_en_vm', se_vm, hide=True)

        # scan enable bar
        seb0 = self.connect_to_tracks(mos_buf_n.s[1], t0_nd0_tid, min_len_mode=MinLenMode.UPPER)
        seb1 = self.connect_to_tracks(mos_in_n.g[1], t0_ng1_tid, min_len_mode=MinLenMode.LOWER)
        seb2 = self.connect_to_tracks(mos_buf_p.s[1], t0_pd0_tid, min_len_mode=MinLenMode.LOWER)
        seb3 = self.connect_to_tracks(mos_si_p.g[1], t1_pg0_tid, min_len_mode=MinLenMode.LOWER)
        self.connect_to_tracks([seb1, seb2, seb3], TrackID(vm_layer, vm_seb_tidx, width=vm_w))
        self.connect_to_tracks([seb0, seb2], TrackID(vm_layer, vm_seb2_tidx, width=vm_w))

        # clk
        ck = self.connect_to_tracks([mos_keep_n.g[0], mos_pass_n.g, mos_buf_n.g[0], mos_buf_p.g[0]],
                                    t0_ng0_tid)
        ck1 = self.connect_to_tracks(mos_mux_p.g0, t1_pg0_tid, min_len_mode=MinLenMode.UPPER)
        ck_vm = self.connect_to_tracks([ck, ck1], TrackID(vm_layer, vm_ck1_tidx, width=vm_w))
        ck2 = self.connect_to_tracks(mos_rst_p.g[-1], t1_pg0_tid, min_len_mode=MinLenMode.UPPER)
        self.connect_to_tracks([ck, ck2], TrackID(vm_layer, vm_ck2_tidx, width=vm_w))
        self.add_pin('clk', ck)
        self.add_pin('clk_vm', ck_vm, hide=True)

        # clkb
        ckb0 = self.connect_to_tracks(mos_buf_n.s[0], t0_nd1_tid, min_len_mode=MinLenMode.UPPER)
        ckb1 = self.connect_to_tracks([mos_keep_p.g[0], mos_pass_p.g], t0_pg1_tid)
        ckb2 = self.connect_to_tracks(mos_buf_p.s[0], t0_pd1_tid, min_len_mode=MinLenMode.UPPER)
        ckb3 = self.connect_to_tracks([mos_mux_n.g[0], mos_rst2_n.g], t1_ng0_tid)
        self.connect_to_tracks([ckb0, ckb1, ckb2, ckb3], TrackID(vm_layer, vm_ckb_tidx, width=vm_w))

        # o1
        o10 = self.connect_to_tracks(mos_in_n.d, t0_nd1_tid, min_len_mode=MinLenMode.LOWER)
        o11 = self.connect_to_tracks([mos_in_p.d, mos_si_p.d], t1_pd0_tid,
                                     min_len_mode=MinLenMode.MIDDLE)
        o12 = self.connect_to_tracks([mos_mux_n.g[1], mos_mux_p.g[1]], t1_ng1_tid)
        o13 = self.connect_to_tracks(mos_si_n.d, t1_nd1_tid, min_len_mode=MinLenMode.MIDDLE)
        self.connect_to_tracks([o10, o11, o12, o13], TrackID(vm_layer, vm_o1_tidx, width=vm_w))

        # o2
        o20 = self.connect_to_tracks(mos_keep_n.s, t0_nd1_tid)
        o21 = self.connect_to_tracks([mos_keep_p.s, mos_mux_p.s], t1_pd1_tid)
        o22 = self.connect_to_tracks([mos_rst_n.g[0], mos_rst_p.g[0]], t1_pg0_tid)
        o23 = self.connect_to_tracks(mos_mux_n.s, t1_nd0_tid)
        self.connect_to_tracks([o20, o21, o22, o23], TrackID(vm_layer, vm_o2_tidx, width=vm_w))

        # o3
        o30 = self.connect_to_tracks([mos_keep_n.g[1], mos_keep_p.g[1]], t0_ng1_tid)
        o31 = self.connect_to_tracks(mos_rst_p.d[0], t1_pd0_tid)
        o32 = self.connect_to_tracks(mos_rst_n.s[0], t1_nd1_tid)
        o3_track = self.connect_wires([mos_pass_n.s, mos_pass_p.s])
        self.connect_to_track_wires([o30, o31, o32], o3_track)

        # o4
        o40 = self.connect_to_tracks(mos_pass_n.d, t0_nd0_tid, min_len_mode=MinLenMode.LOWER)
        o41 = self.connect_to_tracks([mos_out_n.g, mos_out_p.g], t0_pg1_tid)
        o42 = self.connect_to_tracks(mos_pass_p.d, t0_pd1_tid, min_len_mode=MinLenMode.LOWER)
        o43 = self.connect_to_tracks(mos_rst_p.d[1], t1_pd1_tid)
        o44 = self.connect_to_tracks(mos_rst2_n.s, t1_nd0_tid)
        self.connect_to_tracks([o40, o42, o43, o44], TrackID(vm_layer, vm_o41_tidx, width=vm_w))
        self.connect_to_tracks([o41, o43], TrackID(vm_layer, vm_o42_tidx, width=vm_w))

        # tp and tn
        self.connect_to_tracks([mos_rst_n.s[-1], mos_rst2_n.d], t1_nd1_tid)
        self.connect_to_tracks([mos_rst_p.s[-1], mos_rst2_p.d], t1_pd0_tid)

        # o5
        o50 = self.connect_to_tracks([mos_out_n.s[0], mos_out_p.s[0]], t0_pd1_tid,
                                     min_len_mode=MinLenMode.UPPER)
        o51 = self.connect_to_tracks([mos_rst_n.g[-1], mos_rst2_p.g], t1_ng1_tid)
        self.connect_to_tracks([o50, o51], TrackID(vm_layer, vm_o5_tidx, width=vm_w))

        # output
        out_idx = sig_locs.get('out', t0_ng1_tid.base_index)
        out = self.connect_to_tracks([mos_out_n.s[1], mos_out_p.s[1]], TrackID(hm_layer, out_idx))
        self.add_pin('out', out)

        # rstlb
        rstlb = self.connect_to_tracks([mos_rst_n.g[1], mos_rst_p.g[1]], t1_pg1_tid)
        self.add_pin('rstlb', rstlb)
        if vertical_rst:
            vm_idx = self.grid.coord_to_track(vm_layer, rstlb.middle, mode=RoundMode.NEAREST)
            avail = self.get_available_tracks(vm_layer, vm_ck1_tidx, vm_ck2_tidx, 0,
                                              self.bound_box.h)
            if vm_idx not in avail:
                raise ValueError(f'Recheck routing on layer {vm_layer}')
            rstlb_vm = self.connect_to_tracks(rstlb, TrackID(vm_layer, vm_idx, width=vm_w))
            self.add_pin('rstlb_vm', rstlb_vm)

        self.sch_params = dict(
            lch=place_info.lch,
            seg_dict={'buf': seg_buf, 'in': seg_in, 'mux': seg_mux, 'keep': seg_keep,
                      'pass': seg_pass, 'rst': seg_rst, 'out': seg_out},
            w_dict=dict(p_buf=wp_buf, n_buf=wn_buf, p_in=wp_in, n_in=wn_in,
                        p_mux=wp_mux, n_mux=wn_mux, p_keep=wp_keep, n_keep=wn_keep,
                        p_pass=wp_pass, n_pass=wn_pass, p_rst=wp_rst, n_rst=wn_rst,
                        p_out=wp_out, n_out=wn_out),
            th_p=place_info.get_row_place_info(ridx_p).row_info.threshold,
            th_n=place_info.get_row_place_info(ridx_n).row_info.threshold,
        )

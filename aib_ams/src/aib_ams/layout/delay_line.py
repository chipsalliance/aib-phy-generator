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

from typing import Any, Dict, Optional, Type, List, Union, Mapping, Tuple

from pybag.enum import RoundMode, MinLenMode, Direction
from ..enum import DrawTaps

from bag.layout.core import PyLayInstance
from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID, WireArray
from bag.design.module import Module
from bag.util.math import HalfInt

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.enum import MOSWireType, MOSType
from xbase.layout.mos.placement.data import TilePatternElement

from bag3_digital.layout.stdcells.gates import NAND2Core, InvCore
from bag3_digital.layout.stdcells.memory import FlopScanRstlbTwoTile

from ..schematic.aib_dlyline import aib_ams__aib_dlyline
from ..schematic.aib_dlycell import aib_ams__aib_dlycell
from ..schematic.aib_dlycell_core import aib_ams__aib_dlycell_core
from ..schematic.aib_dlycell_no_flop import aib_ams__aib_dlycell_no_flop


class DelayCellCore(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._left_vm_margin = 0
        self._right_vm_margin = 0

    @property
    def left_vm_margin(self) -> int:
        return self._left_vm_margin

    @property
    def right_vm_margin(self) -> int:
        return self._right_vm_margin

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_dlycell_core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of NAND segments',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            vertical_out='True to have output on vertical metal',
            flip_vm='Flip vm_layer tracks used for routing in_p and co_p, and ci_p and out_p',
            sig_locs='Dictionary of signal locations',
            substrate_row='True to have dedicated substrate row.',
            draw_substrate_row='If substrate row, draw taps at this level of hierarchy',
            tile0='Tile index of logic tile 0',
            tile1='Tile index of logic tile 1',
            stack_nand='Number of stacks in NAND gates of DelayCellCore',
            feedback='True to connect ci_p and co_p',
            output_sr_pins='True to output measurement pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_n=0,
            ridx_p=-1,
            draw_taps='NONE',
            vertical_out=True,
            flip_vm=False,
            sig_locs={},
            substrate_row=False,
            draw_substrate_row=True,
            tile0=0,
            tile1=1,
            stack_nand=1,
            feedback=False,
            output_sr_pins=False,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, Any] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        vertical_out: bool = self.params['vertical_out']
        flip_vm: bool = self.params['flip_vm']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        substrate_row: bool = self.params['substrate_row']
        draw_substrate_row: bool = self.params['draw_substrate_row']
        tile0: int = self.params['tile0']
        tile1: int = self.params['tile1']
        stack_nand: int = self.params['stack_nand']
        feedback: bool = self.params['feedback']
        output_sr_pins: bool = self.params['output_sr_pins']

        if feedback and not vertical_out:
            raise ValueError('cannot connect ci_pand co_p if vertical_out=False')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        _pinfo = self.get_tile_pinfo(tile0)
        nand_params = dict(
            pinfo=_pinfo,
            ridx_n=ridx_n,
            ridx_p=ridx_p,
            show_pins=False,
            vertical_sup=substrate_row,
            stack_p=stack_nand,
            stack_n=stack_nand,
        )

        nand_in_params = dict(
            seg=seg_dict['in'],
            vertical_out=False,
            **nand_params,
        )
        nand_in_master = self.new_template(NAND2Core, params=nand_in_params)

        nd0 = nand_in_master.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig',
                                             wire_idx=0)
        pd1 = nand_in_master.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig',
                                             wire_idx=-1)

        ng1 = nand_in_master.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg0 = nand_in_master.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)
        pg1 = nand_in_master.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-2)

        bk_idx = sig_locs.get('bk', pg1)
        sr1_in1_idx = ng1 if bk_idx == pg1 else pg1
        nand_in_params = dict(
            seg=seg_dict['in'],
            sig_locs={'nout': nd0, 'pout': pd1, 'nin0': pg0, 'nin1': bk_idx},
            vertical_out=False,
            **nand_params,
        )
        nand_in_master = self.new_template(NAND2Core, params=nand_in_params)

        nand_out_params = dict(
            seg=seg_dict['out'],
            sig_locs={'nout': nd0, 'pout': pd1},
            vertical_out=False,
            **nand_params,
        )
        nand_out_master = self.new_template(NAND2Core, params=nand_out_params)

        nand_sr0_params = dict(
            seg=seg_dict['sr'],
            sig_locs={'nin0': pg1, 'nin1': ng1},
            vertical_out=False,
            **nand_params,
        )
        nand_sr0_master = self.new_template(NAND2Core, params=nand_sr0_params)

        nand_sr1_params = dict(
            seg=seg_dict['sr'],
            sig_locs={'nin0': pg0, 'nin1': sr1_in1_idx},
            vertical_out=False,
            **nand_params,
        )
        nand_sr1_master = self.new_template(NAND2Core, params=nand_sr1_params)

        # --- Placement --- #
        # 1. Compute number of columns for taps
        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col
        l_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH else 0
        r_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH else 0

        tr_manager = _pinfo.tr_manager

        # 2. place output NAND on row 0 and input NAND above it on row 1
        cur_col = l_offset
        nand_out_inst = self.add_tile(nand_out_master, tile0, cur_col)
        nand_in_inst = self.add_tile(nand_in_master, tile1, cur_col)

        # 3. place sr NANDs
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        sep = max(self.min_sep_col, self.get_hm_sp_le_sep_col(tr_w_h))
        cur_col += max(nand_out_master.num_cols, nand_in_master.num_cols) + sep
        nand_sr0_inst = self.add_tile(nand_sr0_master, tile0, cur_col)
        nand_sr1_inst = self.add_tile(nand_sr1_master, tile1, cur_col)

        # 4. set size
        cur_col += max(nand_sr0_master.num_cols, nand_sr1_master.num_cols) + r_offset
        self.set_mos_size(cur_col)

        # 5. add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            for i in range(self.num_tile_rows):
                self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=i)
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            for i in range(self.num_tile_rows):
                self.add_tap(cur_col, tap_vdd_list, tap_vss_list, flip_lr=True, tile_idx=i)

        # --- Routing --- #
        # 1. vdd/vss
        vdd_list, vss0_list, vss1_list = [], [], []
        for inst in [nand_in_inst, nand_sr1_inst]:
            vdd_list += inst.get_all_port_pins('VDD')
            vss1_list += inst.get_all_port_pins('VSS')
        for inst in [nand_out_inst, nand_sr0_inst]:
            vdd_list += inst.get_all_port_pins('VDD')
            vss0_list += inst.get_all_port_pins('VSS')

        if substrate_row:
            if draw_substrate_row:
                for _tidx in range(self.num_tile_rows):
                    _pinfo = self.get_tile_pinfo(_tidx)
                    for _ridx in range(_pinfo.num_rows):
                        rtype = _pinfo.get_row_place_info(_ridx).row_info.row_type
                        if rtype.is_substrate:
                            warrs = self.add_substrate_contact(_ridx, 0, tile_idx=_tidx,
                                                               seg=self.num_cols)
                            sup_name = 'VDD' if rtype is MOSType.ntap else 'VSS'
                            self.add_pin(f'{sup_name}_sub', warrs, label=f'{sup_name}:')
            self.add_pin('VDD', vdd_list, label='VDD:')
            self.add_pin('VSS0', vss0_list, label='VSS:')
            self.add_pin('VSS1', vss1_list, label='VSS:')
        else:
            vdd = self.connect_wires(vdd_list)
            vss = self.connect_wires(vss0_list + vss1_list)

            self.connect_to_track_wires(tap_vdd_list, vdd)
            self.connect_to_track_wires(tap_vss_list, vss)

            self.add_pin('VDD', vdd, label='VDD:')
            self.add_pin('VSS', vss, label='VSS:')

        # 2. connect in_p to input of nand_in and nand_sr1
        in_p = self.connect_wires([nand_in_inst.get_pin('nin<0>'), nand_sr1_inst.get_pin('nin<0>')])

        # 3. Cross couple
        # 3a. output of sr_0 to input of sr_1
        sr0_o_idx = self.grid.coord_to_track(vm_layer, nand_sr0_inst.get_pin('nout').middle,
                                             mode=RoundMode.LESS_EQ)
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        sr0_o_tid = TrackID(vm_layer, sr0_o_idx, width=tr_w_v)
        if self.can_short_adj_tracks:
            right = self.connect_to_tracks([nand_sr0_inst.get_pin('nout'),
                                            nand_sr1_inst.get_pin('nin<1>')], sr0_o_tid)
        else:
            right = self.connect_to_tracks([nand_sr0_inst.get_pin('pout'),
                                            nand_sr0_inst.get_pin('nout'),
                                            nand_sr1_inst.get_pin('nin<1>')], sr0_o_tid)
        avail_idx = tr_manager.get_next_track(vm_layer, right.track_id.base_index, 'sig', 'sup',
                                              up=True)
        self.add_pin('vm_r', right, hide=True)
        self._right_vm_margin = self.bound_box.xh - self.grid.track_to_coord(vm_layer, avail_idx)

        # 3b. output of sr_1 to input of sr_0 and nand_out
        sr1_o_tid = tr_manager.get_next_track_obj(sr0_o_tid, 'sig', 'sig', count_rel_tracks=-1)
        if self.can_short_adj_tracks:
            left = self.connect_to_tracks(
                [nand_sr1_inst.get_pin('nout'), nand_sr0_inst.get_pin('nin<1>'),
                 nand_out_inst.get_pin('nin<1>')], sr1_o_tid)
        else:
            left = self.connect_to_tracks(
                [nand_sr1_inst.get_pin('pout'), nand_sr1_inst.get_pin('nout'),
                 nand_sr0_inst.get_pin('nin<1>'),
                 nand_out_inst.get_pin('nin<1>')], sr1_o_tid)
        self.add_pin('vm_l', left, hide=True)

        # 4. Connect bk1
        bk1_vm_idx_2 = tr_manager.get_next_track_obj(sr1_o_tid, 'sig', 'sig', count_rel_tracks=-1)
        bk1 = nand_in_inst.get_pin('nin<1>')
        bk1_2 = nand_sr0_inst.get_pin('nin<0>')
        bk1_vm_idx = self.grid.coord_to_track(vm_layer, bk1_2.lower, mode=RoundMode.LESS_EQ)
        bk1_vm_idx = min(bk1_vm_idx, bk1_vm_idx_2.base_index)
        bk1_tid = TrackID(vm_layer, bk1_vm_idx, width=tr_w_v)
        bk1_vm = self.connect_to_tracks([bk1, bk1_2], bk1_tid)
        self.add_pin('bk1_vm', bk1_vm, hide=True)

        avail_vm_idx = tr_manager.get_next_track(vm_layer, bk1_vm_idx, 'sig', 'sig', up=False)

        # 5. Get in_p, co_p, ci_p, out_p on vm_layer if vertical_out is True
        if self.can_short_adj_tracks:
            co_p_warrs = [nand_in_inst.get_pin('nout')]
            out_p_warrs = [nand_out_inst.get_pin('nout')]
        else:
            co_p_warrs = [nand_in_inst.get_pin('pout'), nand_in_inst.get_pin('nout')]
            out_p_warrs = [nand_out_inst.get_pin('pout'), nand_out_inst.get_pin('nout')]
        if vertical_out:
            # place 4 vm_layer tracks
            num_vm, vm_locs = self.tr_manager.place_wires(vm_layer, ['sig'] * 4)
            mid_idx = self.grid.coord_to_track(vm_layer, nand_in_inst.get_pin('nout').middle,
                                               mode=RoundMode.NEAREST)
            try:
                loc_mid = (vm_locs[0] + vm_locs[-1]) / 2
            except ValueError:
                loc_mid = (vm_locs[0] + vm_locs[-1]) // 2 + HalfInt(1)
            vm_offset = mid_idx - loc_mid

            # make sure vm tracks don't collide with bk1_vm_idx
            if vm_offset + vm_locs[-1] > avail_vm_idx:
                vm_offset = avail_vm_idx - vm_locs[-1]

            # output
            co_p_idx = vm_offset + (vm_locs[0] if flip_vm else vm_locs[2])
            in_p_idx = vm_offset + (vm_locs[2] if flip_vm else vm_locs[0])
            out_p_idx = vm_offset + (vm_locs[3] if flip_vm else vm_locs[1])
            ci_p_idx = vm_offset + (vm_locs[1] if flip_vm else vm_locs[3])

            if feedback:
                co_p_warrs.append(nand_out_inst.get_pin('nin<0>'))
                co_p_idx = ci_p_idx

            co_p = self.connect_to_tracks(co_p_warrs, TrackID(vm_layer, co_p_idx, width=tr_w_v),
                                          min_len_mode=MinLenMode.MIDDLE)
            self.add_pin('co_p', co_p, hide=feedback)

            self.add_pin('out_p', self.connect_to_tracks(out_p_warrs,
                                                         TrackID(vm_layer, out_p_idx,
                                                                 width=tr_w_v),
                                                         min_len_mode=MinLenMode.MIDDLE))
            # input
            in_p = self.connect_to_tracks(in_p, TrackID(vm_layer, in_p_idx, width=tr_w_v),
                                          min_len_mode=MinLenMode.LOWER)
            self._left_vm_margin = self.grid.track_to_coord(vm_layer, in_p_idx) - self.bound_box.xl
            self.add_pin('in_p', in_p)
            if not feedback:
                ci_p = self.connect_to_tracks(nand_out_inst.get_pin('nin<0>'),
                                              TrackID(vm_layer, ci_p_idx, width=tr_w_v),
                                              min_len_mode=MinLenMode.UPPER)
                # TODO: HACKS to fix corner spacing rule
                ci_p = self.extend_wires(ci_p, upper=bk1_vm.lower + 16)
                self.add_pin('ci_p', ci_p)
        else:
            # output
            if self.can_short_adj_tracks:
                self.add_pin('co_p', co_p_warrs, label='co_p:')
                self.add_pin('out_p', out_p_warrs, label='out_p:')
            else:
                self.add_pin('co_pp', co_p_warrs[0], label='co_p:')
                self.add_pin('co_pn', co_p_warrs[1], label='co_p:')
                self.add_pin('out_pp', out_p_warrs[0], label='out_p:')
                self.add_pin('out_pn', out_p_warrs[1], label='out_p:')
            # input
            self.add_pin('in_p', in_p)
            self.add_pin('ci_p', nand_out_inst.get_pin('nin<0>'))

        # --- Pins --- #
        # input pins
        self.add_pin('bk1_in', bk1, label='bk1:', hide=False)
        self.add_pin('bk1_out', bk1_2, label='bk1:', hide=False)

        self.sch_params = dict(
            in_params=nand_in_master.sch_params,
            sr0_params=nand_sr0_master.sch_params,
            sr1_params=nand_sr1_master.sch_params,
            out_params=nand_out_master.sch_params,
            feedback=feedback,
            output_sr_pins=output_sr_pins,
        )


class DelayCell(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._sp_vm_locs: List[Union[int, HalfInt]] = []
        self._substrate_row_intvl: List[Tuple[int, int]] = []

    @property
    def substrate_row_intvl(self) -> List[Tuple[int, int]]:
        return self._substrate_row_intvl

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_dlycell

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            sp_vm_tracks='Number of vm_tracks that should fit in middle space',
            row_idx='Row index of this delay cell instance in delay line',
            flip_nand_vm='flip_vm flag for DelayCellCore',
            substrate_row='True to have dedicated substrate row.',
            tile0='Tile index of logic tile 0',
            tile1='Tile index of logic tile 1',
            flop_char='True to add flop characterization pins.',
            stack_nand='Number of stacks in NAND gates of DelayCellCore',
            num_core='Number of delay cell cores in one delay cell',
            is_dum='True if this is a dummy cell in the delay line',
            output_sr_pins='True to output measurement pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_n=0,
            ridx_p=-1,
            draw_taps='NONE',
            sp_vm_tracks=0,
            row_idx=-1,
            flip_nand_vm=False,
            substrate_row=False,
            tile0=0,
            tile1=1,
            flop_char=False,
            stack_nand=1,
            num_core=1,
            is_dum=False,
            output_sr_pins=False,
        )

    def sp_vm_locs(self) -> List[Union[int, HalfInt]]:
        return self._sp_vm_locs

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, Any] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        sp_vm_tracks: int = self.params['sp_vm_tracks']
        row_idx: Optional[int] = self.params['row_idx']
        flip_nand_vm: bool = self.params['flip_nand_vm']
        substrate_row: bool = self.params['substrate_row']
        tile0: int = self.params['tile0']
        tile1: int = self.params['tile1']
        flop_char: bool = self.params['flop_char']
        stack_nand: bool = self.params['stack_nand']
        num_core: int = self.params['num_core']
        is_dum: bool = self.params['is_dum']
        output_sr_pins: bool = self.params['output_sr_pins']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        # make masters
        _pinfo = self.get_tile_pinfo(tile0)
        params = dict(
            ridx_n=ridx_n,
            ridx_p=ridx_p,
            show_pins=False,
            substrate_row=substrate_row,
        )

        inv_tp = TilePatternElement(_pinfo)
        bk_pin_idx = inv_tp.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        dc_core_params = dict(
            pinfo=pinfo,
            seg_dict=seg_dict['dc_core'],
            sig_locs={'bk': bk_pin_idx},
            flip_vm=flip_nand_vm,
            tile0=tile0,
            tile1=tile1,
            stack_nand=stack_nand,
            output_sr_pins=output_sr_pins,
            **params,
        )
        dc_core_master = self.new_template(DelayCellCore, params=dc_core_params)
        dc_ncol = dc_core_master.num_cols

        if isinstance(pinfo, tuple):
            place_info = pinfo[0]
        else:
            place_info = pinfo
        dc_core_tp = TilePatternElement(place_info)
        out_pin_idx = dc_core_tp.get_track_index(ridx_n, MOSWireType.G, wire_name='sig',
                                                 wire_idx=1, tile_idx=tile1)
        scan_rst_flop_params = dict(
            pinfo=pinfo,
            seg_dict=seg_dict['scan_rst_flop'],
            sig_locs={'out': out_pin_idx},
            vertical_rst=True,
            tile0=tile1,
            tile1=tile0,
            **params,
        )
        scan_rst_flop_master = self.new_template(FlopScanRstlbTwoTile, params=scan_rst_flop_params)

        so_inv_params = dict(
            pinfo=_pinfo,
            seg=seg_dict['so_inv'],
            sig_locs={'pin': bk_pin_idx},
            vertical_out=False,
            vertical_sup=substrate_row,
            **params,
        )
        so_inv_master = self.new_template(InvCore, params=so_inv_params)

        # --- Placement --- #
        # 1. Compute number of columns for taps
        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col
        l_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH else 0
        r_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH else 0

        sep = max(self.min_sep_col, self.get_hm_sp_le_sep_col())

        # 1.5 Leave space for vm_tracks
        if sp_vm_tracks == 0:
            cur_col = 0
        else:
            start_idx = self.grid.coord_to_track(vm_layer, - sep * self.sd_pitch -
                                                 dc_core_master.right_vm_margin,
                                                 mode=RoundMode.NEAREST)
            num_vm, vm_locs = self.tr_manager.place_wires(vm_layer,
                                                          ['sup'] + ['sig'] * sp_vm_tracks,
                                                          align_track=start_idx)
            # make sure first signal wire is not beyond leftmost edge of cell to avoid collision
            # on hm_layer
            if vm_locs[1] < HalfInt(1):
                offset = HalfInt(1) - vm_locs[1]
                vm_locs = [loc + offset for loc in vm_locs]
            self._sp_vm_locs = vm_locs
            next_idx = self.tr_manager.get_next_track(vm_layer, vm_locs[-1], 'sig', 'sig', up=True)
            new_cur_col = _pinfo.coord_to_col(self.grid.track_to_coord(vm_layer, next_idx),
                                              round_mode=RoundMode.NEAREST)
            cur_col = new_cur_col + new_cur_col % 2

        # 2. Place scan_rst_flop
        cur_col = max(l_offset, cur_col)
        flop_col = cur_col
        scan_rst_flop_inst = self.add_tile(scan_rst_flop_master, 0, cur_col)
        cur_col += scan_rst_flop_master.num_cols + sep
        cur_col += cur_col % 2

        # 2.5 Get bk and si on vm_layer
        used_vm_idx = scan_rst_flop_inst.get_pin('scan_en_vm').track_id.base_index
        si_vm_idx = self.tr_manager.get_next_track(vm_layer, used_vm_idx, 'sig', 'sig', up=False)
        si_vm = self.connect_to_tracks(scan_rst_flop_inst.get_pin('scan_in'),
                                       TrackID(vm_layer, si_vm_idx), min_len_mode=MinLenMode.LOWER)
        self.add_pin('si', si_vm)

        if sp_vm_tracks == 0:
            self.reexport(scan_rst_flop_inst.get_port('in'), net_name='bk')
        else:
            bk_vm = self.connect_to_tracks(scan_rst_flop_inst.get_pin('in'),
                                           TrackID(vm_layer, self._sp_vm_locs[row_idx + 1]),
                                           min_len_mode=MinLenMode.MIDDLE)
            self.add_pin('bk', bk_vm)

        # 3. Place so buffer
        inv_col = cur_col
        so_inv0_inst = self.add_tile(so_inv_master, tile0, cur_col)
        so_inv1_inst = self.add_tile(so_inv_master, tile1, cur_col)
        so_ncol = so_inv_master.num_cols
        cur_col += so_ncol + sep
        cur_col += cur_col % 2

        # 4. Place delay_cell_core
        dc_core_list = []
        dc_col_list = []
        for core_idx in range(num_core):
            dc_col_list.append(cur_col)
            dc_core_list.append(self.add_tile(dc_core_master, 0, cur_col))
            cur_col += dc_ncol + sep
        cur_col -= sep

        # 5. set size
        num_cols = cur_col + r_offset
        self.set_mos_size(num_cols)

        # 6. add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            for i in range(self.num_tile_rows):
                self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=i)
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            for i in range(self.num_tile_rows):
                self.add_tap(num_cols, tap_vdd_list, tap_vss_list, flip_lr=True, tile_idx=i)

        # --- Routing --- #
        # 1. vdd/vss
        vdd_list, vss0_list, vss1_list = [], [], []
        for inst in [scan_rst_flop_inst] + dc_core_list:
            vdd_list += inst.get_all_port_pins('VDD')
            if substrate_row:
                vss0_list += inst.get_all_port_pins('VSS0')
                vss1_list += inst.get_all_port_pins('VSS1')

                self.reexport(inst.get_port('VDD_sub'))
                self.reexport(inst.get_port('VSS_sub'))
            else:
                vss0_list += inst.get_all_port_pins('VSS')

        vdd_list += so_inv0_inst.get_all_port_pins('VDD')
        vdd_list += so_inv1_inst.get_all_port_pins('VDD')
        if substrate_row:
            vss0_list += so_inv0_inst.get_all_port_pins('VSS')
            vss1_list += so_inv1_inst.get_all_port_pins('VSS')
        else:
            vss0_list += so_inv0_inst.get_all_port_pins('VSS')
            vss0_list += so_inv1_inst.get_all_port_pins('VSS')

        if substrate_row:
            flop_sub_intvl = scan_rst_flop_master.substrate_row_intvl
            for start, num_sub in flop_sub_intvl:
                self._substrate_row_intvl.append((start + flop_col, num_sub))
            self._substrate_row_intvl.append((inv_col, so_ncol))
            for dc_col in dc_col_list:
                self._substrate_row_intvl.append((dc_col, dc_ncol))
            for _tidx in range(self.num_tile_rows):
                _pinfo = self.get_tile_pinfo(_tidx)
                for _ridx in range(_pinfo.num_rows):
                    rtype = _pinfo.get_row_place_info(_ridx).row_info.row_type
                    if rtype.is_substrate:
                        warrs = self.add_substrate_contact(_ridx, inv_col, tile_idx=_tidx,
                                                           seg=so_ncol)
                        sup_name = 'VDD' if rtype is MOSType.ntap else 'VSS'
                        self.add_pin(f'{sup_name}_sub', warrs, label=f'{sup_name}:')
            self.add_pin('VDD', vdd_list, label='VDD:')
            self.add_pin('VSS0', vss0_list, label='VSS:')
            self.add_pin('VSS1', vss1_list, label='VSS:')
        else:
            vdd = self.connect_wires(vdd_list)
            vss = self.connect_wires(vss0_list)

            if tap_vdd_list:
                self.connect_to_track_wires(tap_vdd_list, vdd)
            if tap_vss_list:
                self.connect_to_track_wires(tap_vss_list, vss)

            self.add_pin('VDD', vdd, label='VDD:')
            self.add_pin('VSS', vss, label='VSS:')

        # 2. Export flop pins
        self.reexport(scan_rst_flop_inst.get_port('scan_en'), net_name='SE')
        self.reexport(scan_rst_flop_inst.get_port('scan_en_vm'), net_name='SE_vm', hide=True)
        self.reexport(scan_rst_flop_inst.get_port('clk'), net_name='ck')
        self.reexport(scan_rst_flop_inst.get_port('clk_vm'), net_name='ck_vm')
        self.reexport(scan_rst_flop_inst.get_port('rstlb'), net_name='RSTb')
        self.reexport(scan_rst_flop_inst.get_port('rstlb_vm'), net_name='RSTb_vm', hide=True)

        # 3. Export delay_cell_core pins
        if flip_nand_vm:
            start_idx = num_core - 1
            stop_idx = 0
            step = -1
        else:
            start_idx = 0
            stop_idx = num_core - 1
            step = 1
        self.reexport(dc_core_list[start_idx].get_port('in_p'))
        self.reexport(dc_core_list[stop_idx].get_port('co_p'))
        self.reexport(dc_core_list[stop_idx].get_port('ci_p'))
        self.reexport(dc_core_list[start_idx].get_port('out_p'))

        # 3.5. If num_core > 1, connect between delay cell cores on xm_layer
        if num_core > 1 and not is_dum:
            in_vm = dc_core_list[0].get_pin('in_p')
            in_xm = self.grid.coord_to_track(xm_layer, in_vm.middle, mode=RoundMode.NEAREST)

            si_xm = self.grid.coord_to_track(xm_layer, si_vm.middle, mode=RoundMode.NEAREST)
            out_xm = self.tr_manager.get_next_track(xm_layer, si_xm, 'sig', 'sig', up=True)

            for idx in range(start_idx, stop_idx, step):
                cur_co = dc_core_list[idx].get_pin('co_p')
                next_in = dc_core_list[idx + step].get_pin('in_p')
                self.connect_to_tracks([cur_co, next_in], TrackID(xm_layer, in_xm))

                cur_ci = dc_core_list[idx].get_pin('ci_p')
                next_out = dc_core_list[idx + step].get_pin('out_p')
                self.connect_to_tracks([cur_ci, next_out], TrackID(xm_layer, out_xm))

        # 4. Connect bk1 from flop to delay_cell_core
        inv_bk1 = so_inv1_inst.get_pin('in')
        flop_bk1 = scan_rst_flop_inst.get_pin('out')
        bk1_top_list = [dc_core_list[idx].get_pin('bk1_in') for idx in range(num_core)]
        self.connect_wires([inv_bk1, flop_bk1] + bk1_top_list)
        if flop_char:
            self.add_pin('bk1', flop_bk1)

        bk1_bot_list = [dc_core_list[idx].get_pin('bk1_out') for idx in range(num_core)]
        self.connect_wires(bk1_bot_list)

        # 5. Connect output of so_inv1 to input of so_inv0, and export output of so_inv0
        vm_idx0 = self.grid.coord_to_track(vm_layer, inv_bk1.middle, mode=RoundMode.NEAREST)
        vm_idx1 = self.tr_manager.get_next_track(vm_layer, vm_idx0, 'sig', 'sig', up=True)

        self.connect_to_tracks([so_inv1_inst.get_pin('pout'), so_inv1_inst.get_pin('nout'),
                                so_inv0_inst.get_pin('in')], TrackID(vm_layer, vm_idx0))

        so_out = self.connect_to_tracks([so_inv0_inst.get_pin('pout'),
                                         so_inv0_inst.get_pin('nout')], TrackID(vm_layer, vm_idx1))
        self.add_pin('so', so_out)

        # set properties
        self.sch_params = dict(
            so_inv_params=so_inv_master.sch_params,
            scan_rst_flop_params=scan_rst_flop_master.sch_params,
            dc_core_params=dc_core_master.sch_params,
            flop_char=flop_char,
            num_core=num_core,
            is_dum=is_dum,
            output_sr_pins=output_sr_pins,
        )


class DelayCellNoFlop(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._sp_vm_locs: List[Union[int, HalfInt]] = []
        self._substrate_row_intvl: List[Tuple[int, int]] = []

    @property
    def substrate_row_intvl(self) -> List[Tuple[int, int]]:
        return self._substrate_row_intvl

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_dlycell_no_flop

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            sp_vm_tracks='Number of vm_tracks that should fit in middle space',
            row_idx='Row index of this delay cell instance in delay line',
            flip_nand_vm='flip_vm flag for DelayCellCore',
            substrate_row='True to have dedicated substrate row.',
            tile0='Tile index of logic tile 0',
            tile1='Tile index of logic tile 1',
            stack_nand='Number of stacks in NAND gates of DelayCellCore',
            num_core='Number of delay cell cores in one delay cell',
            is_dum='True if this is a dummy cell in the delay line',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_n=0,
            ridx_p=-1,
            draw_taps='NONE',
            sp_vm_tracks=0,
            row_idx=-1,
            flip_nand_vm=False,
            substrate_row=False,
            tile0=0,
            tile1=1,
            stack_nand=1,
            num_core=1,
            is_dum=False,
        )

    def sp_vm_locs(self) -> List[Union[int, HalfInt]]:
        return self._sp_vm_locs

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, mirror=False)

        seg_dict: Dict[str, Any] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        sp_vm_tracks: int = self.params['sp_vm_tracks']
        row_idx: Optional[int] = self.params['row_idx']
        flip_nand_vm: bool = self.params['flip_nand_vm']
        substrate_row: bool = self.params['substrate_row']
        tile0: int = self.params['tile0']
        tile1: int = self.params['tile1']
        stack_nand: int = self.params['stack_nand']
        num_core: int = self.params['num_core']
        is_dum: bool = self.params['is_dum']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        # make masters
        _pinfo = self.get_tile_pinfo(tile0)
        params = dict(
            ridx_n=ridx_n,
            ridx_p=ridx_p,
            show_pins=False,
            substrate_row=substrate_row,
        )

        dc_core_params = dict(
            pinfo=pinfo,
            seg_dict=seg_dict['dc_core'],
            flip_vm=flip_nand_vm,
            tile0=tile0,
            tile1=tile1,
            stack_nand=stack_nand,
            **params,
        )
        dc_core_master = self.new_template(DelayCellCore, params=dc_core_params)
        if flip_nand_vm:
            dc_core_noflip = dc_core_master.new_template_with(flip_vm=False)
        else:
            dc_core_noflip = dc_core_master
        dc_ncol = dc_core_master.num_cols

        bk_inv_params = dict(
            pinfo=_pinfo,
            seg=seg_dict['bk_inv'],
            vertical_out=False,
            vertical_sup=substrate_row,
            **params,
        )
        bk_inv_master = self.new_template(InvCore, params=bk_inv_params)

        # --- Placement --- #
        # 1. Compute number of columns for taps
        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col
        l_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH else 0
        r_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH else 0

        sep = max(self.min_sep_col, self.get_hm_sp_le_sep_col())

        # 1.5 Leave space for vm_tracks
        if sp_vm_tracks == 0:
            cur_col = 0
            vm_inv_l_idx = self.grid.coord_to_track(vm_layer, 0, mode=RoundMode.GREATER_EQ)
        else:
            start_idx = self.grid.coord_to_track(vm_layer, - sep * self.sd_pitch -
                                                 dc_core_master.right_vm_margin,
                                                 mode=RoundMode.GREATER_EQ)
            wire_list = ['sup']
            wire_list.extend(('sig' for _ in range(sp_vm_tracks)))
            _, vm_locs = self.tr_manager.place_wires(vm_layer, wire_list, align_track=start_idx)
            # make sure first signal wire is not beyond leftmost edge of cell to avoid collision
            # on hm_layer
            if vm_locs[1] < HalfInt(1):
                offset = HalfInt(1) - vm_locs[1]
                vm_locs = [loc + offset for loc in vm_locs]
            self._sp_vm_locs = vm_locs
            vm_inv_l_idx = self.tr_manager.get_next_track(vm_layer, vm_locs[-1], 'sig', 'sig',
                                                          up=True)
            new_cur_col = _pinfo.coord_to_col(self.grid.track_to_coord(vm_layer, vm_inv_l_idx),
                                              round_mode=RoundMode.GREATER_EQ)
            cur_col = new_cur_col + new_cur_col % 2

        # Get current column
        cur_col = max(l_offset, cur_col)

        # 2. Place bk buffer
        inv_col = cur_col
        bk_inv0_inst = self.add_tile(bk_inv_master, tile0, cur_col)
        bk_inv1_inst = self.add_tile(bk_inv_master, tile1, cur_col)
        inv_bk1 = bk_inv1_inst.get_pin('in')
        vm_inv_r_idx = self.tr_manager.get_next_track(vm_layer, vm_inv_l_idx, 'sig', 'sig',
                                                      up=True)
        bk_ncol = bk_inv_master.num_cols
        cur_col += bk_ncol

        # 2.5 Get bk and si on vm_layer
        if sp_vm_tracks == 0:
            self.reexport(bk_inv0_inst.get_port('nin'), net_name='bk', hide=False)
        else:
            bk_vm = self.connect_to_tracks(bk_inv0_inst.get_pin('nin'),
                                           TrackID(vm_layer, self._sp_vm_locs[row_idx + 1]),
                                           min_len_mode=MinLenMode.MIDDLE)
            self.add_pin('bk', bk_vm)

        # 3. Place delay_cell_core
        core_inp_idx = self.tr_manager.get_next_track(vm_layer, vm_inv_r_idx, 'sig', 'sig',
                                                      up=True)
        core_xl = self.grid.track_to_coord(vm_layer, core_inp_idx) - dc_core_noflip.left_vm_margin
        cur_col = max(cur_col + sep,
                      self.arr_info.coord_to_col(core_xl, round_mode=RoundMode.GREATER_EQ))
        dc_core_list = []
        dc_col_list = []
        for core_idx in range(num_core):
            dc_col_list.append(cur_col)
            dc_core_list.append(self.add_tile(dc_core_master, 0, cur_col))
            cur_col += dc_ncol + sep
        cur_col -= sep

        # 4. set size
        num_cols = cur_col + r_offset
        self.set_mos_size(num_cols)

        # 5. add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            for i in range(self.num_tile_rows):
                self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=i)
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            for i in range(self.num_tile_rows):
                self.add_tap(num_cols, tap_vdd_list, tap_vss_list, flip_lr=True, tile_idx=i)

        # --- Routing --- #
        # 1. vdd/vss
        vdd_list, vss0_list, vss1_list = [], [], []
        for inst in dc_core_list:
            vdd_list += inst.get_all_port_pins('VDD')
            if substrate_row:
                vss0_list += inst.get_all_port_pins('VSS0')
                vss1_list += inst.get_all_port_pins('VSS1')

                self.reexport(inst.get_port('VDD_sub'))
                self.reexport(inst.get_port('VSS_sub'))
            else:
                vss0_list += inst.get_all_port_pins('VSS')

        vdd_list += bk_inv0_inst.get_all_port_pins('VDD')
        vdd_list += bk_inv1_inst.get_all_port_pins('VDD')
        if substrate_row:
            vss0_list += bk_inv0_inst.get_all_port_pins('VSS')
            vss1_list += bk_inv1_inst.get_all_port_pins('VSS')
        else:
            vss0_list += bk_inv0_inst.get_all_port_pins('VSS')
            vss0_list += bk_inv1_inst.get_all_port_pins('VSS')

        if substrate_row:
            self._substrate_row_intvl.append((inv_col, bk_ncol))
            for dc_col in dc_col_list:
                self._substrate_row_intvl.append((dc_col, dc_ncol))
            for _tidx in range(self.num_tile_rows):
                _pinfo = self.get_tile_pinfo(_tidx)
                for _ridx in range(_pinfo.num_rows):
                    rtype = _pinfo.get_row_place_info(_ridx).row_info.row_type
                    if rtype.is_substrate:
                        warrs = self.add_substrate_contact(_ridx, inv_col, tile_idx=_tidx,
                                                           seg=bk_inv_master.num_cols)
                        sup_name = 'VDD' if rtype is MOSType.ntap else 'VSS'
                        self.add_pin(f'{sup_name}_sub', warrs, label=f'{sup_name}:')
            self.add_pin('VDD', vdd_list, label='VDD:')
            self.add_pin('VSS0', vss0_list, label='VSS:')
            self.add_pin('VSS1', vss1_list, label='VSS:')
        else:
            vdd = self.connect_wires(vdd_list)
            vss = self.connect_wires(vss0_list)

            if tap_vdd_list:
                self.connect_to_track_wires(tap_vdd_list, vdd)
            if tap_vss_list:
                self.connect_to_track_wires(tap_vss_list, vss)

            self.add_pin('VDD', vdd, label='VDD:')
            self.add_pin('VSS', vss, label='VSS:')

        # 2. Export delay_cell_core pins
        if flip_nand_vm:
            start_idx = num_core - 1
            stop_idx = 0
            step = -1
        else:
            start_idx = 0
            stop_idx = num_core - 1
            step = 1
        self.reexport(dc_core_list[start_idx].get_port('in_p'))
        self.reexport(dc_core_list[stop_idx].get_port('co_p'))
        self.reexport(dc_core_list[stop_idx].get_port('ci_p'))
        self.reexport(dc_core_list[start_idx].get_port('out_p'))

        # 3. Connect output of bk_inv0 to input of bk_inv1
        vm_wire = self.connect_to_tracks([bk_inv0_inst.get_pin('pout'),
                                          bk_inv0_inst.get_pin('nout'), inv_bk1],
                                         TrackID(vm_layer, vm_inv_r_idx))
        self.add_pin('vm_l', vm_wire, hide=True)
        self.reexport(dc_core_list[-1].get_port('vm_r'))

        # 3.5. If num_core > 1, connect between delay cell cores on xm_layer
        if num_core > 1 and not is_dum:
            in_vm = dc_core_list[0].get_pin('in_p')
            in_xm = self.grid.coord_to_track(xm_layer, in_vm.middle, mode=RoundMode.NEAREST)

            out_vm = dc_core_list[0].get_pin('out_p')
            out_xm = self.grid.coord_to_track(xm_layer, out_vm.middle, mode=RoundMode.NEAREST)

            for idx in range(start_idx, stop_idx, step):
                cur_co = dc_core_list[idx].get_pin('co_p')
                next_in = dc_core_list[idx + step].get_pin('in_p')
                self.connect_to_tracks([cur_co, next_in], TrackID(xm_layer, in_xm))

                cur_ci = dc_core_list[idx].get_pin('ci_p')
                next_out = dc_core_list[idx + step].get_pin('out_p')
                self.connect_to_tracks([cur_ci, next_out], TrackID(xm_layer, out_xm))

        # 4. Connect bk1 from bk_inv1 to delay_cell_core
        bk1_top_list = [dc_core_list[idx].get_pin('bk1_in') for idx in range(num_core)]
        self.connect_to_tracks([bk_inv1_inst.get_pin('pout'), bk_inv1_inst.get_pin('nout')] +
                               bk1_top_list, TrackID(vm_layer, vm_inv_l_idx))

        bk1_bot_list = [dc_core_list[idx].get_pin('bk1_out') for idx in range(num_core)]
        self.connect_wires(bk1_bot_list)

        # set properties
        self.sch_params = dict(
            bk_inv_params=bk_inv_master.sch_params,
            dc_core_params=dc_core_master.sch_params,
            num_core=num_core,
            is_dum=is_dum,
        )


class DelayLine(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_dlyline

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            num_rows='Number of rows of delay cells in delay line',
            num_cols='Number of columns of delay cells in delay line',
            num_insts='Number of delay cells',
            substrate_row='True to have dedicated substrate row.',
            tile0='Tile index of logic tile 0',
            tile1='Tile index of logic tile 1',
            tile_vss='Tile index of ptap tile',
            tile_vdd='Tile index of ntap tile',
            flop='True to have flop in delay_cell',
            flop_char='True to add flop characterization pins.',
            stack_nand='Number of stacks in NAND gates of DelayCellCore',
            num_core='Number of delay cell cores in one delay cell',
            output_sr_pins='True to output measurement pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_n=0,
            ridx_p=-1,
            draw_taps='NONE',
            num_insts=-1,
            substrate_row=False,
            tile0=0,
            tile1=1,
            tile_vss=0,
            tile_vdd=1,
            flop=True,
            flop_char=False,
            stack_nand=1,
            num_core=1,
            output_sr_pins=False,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, mirror=False)

        seg_dict: Dict[str, Any] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        num_rows: int = self.params['num_rows']
        num_cols: int = self.params['num_cols']
        num_insts: int = self.params['num_insts']
        if num_insts == -1:
            num_insts = num_cols * num_rows
            num_dum = 0
        else:
            num_dum = num_cols * num_rows - num_insts
        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        substrate_row: bool = self.params['substrate_row']
        tile0: int = self.params['tile0']
        tile1: int = self.params['tile1']
        tile_vss: int = self.params['tile_vss']
        tile_vdd: int = self.params['tile_vdd']
        flop: bool = self.params['flop']
        flop_char: bool = self.params['flop_char']
        stack_nand: int = self.params['stack_nand']
        num_core: int = self.params['num_core']
        output_sr_pins: bool = self.params['output_sr_pins']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        # make masters
        params = dict(
            pinfo=pinfo,
            ridx_n=ridx_n,
            ridx_p=ridx_p,
            show_pins=False,
            seg_dict=seg_dict,
            sp_vm_tracks=num_rows + 1 if flop else num_rows,
            draw_taps='NONE',
            substrate_row=substrate_row,
            tile0=tile0,
            tile1=tile1,
            flop_char=flop_char,
            stack_nand=stack_nand,
            output_sr_pins=output_sr_pins,
            num_core=num_core,
        )
        if flop:
            dly_cell_master = self.new_template(DelayCell, params=params)
        else:
            dly_cell_master = self.new_template(DelayCellNoFlop, params=params)
        dc_ncol = dly_cell_master.num_cols
        dc_ncol += dc_ncol % 2
        dc_ntile = dly_cell_master.num_tile_rows
        sep = max(self.min_sep_col, self.get_hm_sp_le_sep_col())
        vm_sep = self.grid.get_line_end_sep_tracks(Direction.UPPER, hm_layer, 1, 1,
                                                   half_space=False).dbl_value // 2

        # --- Placement --- #
        # 1. Compute number of columns for taps
        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col
        l_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH else 0
        r_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH else 0

        # 2. set size
        tot_cols = l_offset + dc_ncol * num_cols + sep * (num_cols - 1) + r_offset
        self.set_mos_size(tot_cols, num_rows * dc_ntile + 1)

        # 2.5 add top substrate row
        vdd_list, vss_list = [], []

        dc_sub_intvl = dly_cell_master.substrate_row_intvl
        offset = l_offset

        for col in range(num_cols):
            for start, num_sub in dc_sub_intvl:
                self.add_substrate_contact(0, offset + start, tile_idx=-1, seg=num_sub)
            offset += dc_ncol + sep

        top_sup_tid = self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=-1)

        # 3. Place delay cells
        dly_cell_list: List[List[PyLayInstance]] = []
        sup_vm_idx_list = []
        rstb_list, se_list, ck_list = [], [], []
        rstb_vm_list, se_vm_list, ck_vm_list = [], [], []
        bk_dict: Dict[str, WireArray] = {}
        prev_row_vss1_list = []
        for row in range(num_rows):
            params['row_idx'] = row
            params['flip_nand_vm'] = row % 2 == 1
            dum_params = params.copy()
            dum_params.update(dict(is_dum=True, output_sr_pins=False))
            if flop:
                dly_cell_master = self.new_template(DelayCell, params=params)
                dum_master = self.new_template(DelayCell, params=dum_params)
            else:
                dly_cell_master = self.new_template(DelayCellNoFlop, params=params)
                dum_master = self.new_template(DelayCellNoFlop, params=dum_params)
            dly_cell_row: List[PyLayInstance] = []
            cur_col = l_offset
            row_vdd_list, row_vss0_list, row_vss1_list = [], [], []
            row_vdd_tid, row_vss_tid = None, None
            for col in range(num_cols):
                # --- local routing --- #
                bk_idx = row * num_cols + (col if row % 2 == 0 else num_cols - col - 1)
                is_dum = bk_idx >= num_insts
                is_last = bk_idx == num_insts - 1

                inst = self.add_tile(dum_master if is_dum else dly_cell_master,
                                     dc_ntile * row, cur_col)
                cur_col += dc_ncol + sep

                if flop and flop_char and not is_dum:
                    self.reexport(inst.get_port('bk1'), net_name=f'flop_q<{bk_idx}>')

                row_vdd_list += inst.get_all_port_pins('VDD')
                if row_vdd_tid is None:
                    row_vdd_tid = self.get_track_id(0, MOSWireType.DS, 'sup',
                                                    tile_idx=row * dc_ntile + tile_vdd)

                row_vss0_list += inst.get_all_port_pins('VSS0')
                row_vss1_list += inst.get_all_port_pins('VSS1')
                if row_vss_tid is None:
                    row_vss_tid = self.get_track_id(0, MOSWireType.DS, 'sup',
                                                    tile_idx=row * dc_ntile + tile_vss)

                if is_dum:
                    if flop:
                        name0_list = ['si', 'ci_p', 'RSTb_vm']
                        name1_list = ['in_p', 'bk', 'ck_vm', 'SE_vm']
                    else:
                        name0_list = ['ci_p']
                        name1_list = ['in_p', 'bk']
                    for name in name0_list:
                        row_vss0_list.append(inst.get_pin(name))
                    for name in name1_list:
                        row_vss1_list.append(inst.get_pin(name))
                else:
                    bk_stub = self.extend_wires(inst.get_pin('bk'), lower=0,
                                                upper=self.bound_box.h)[0]
                    if row == 0 and col > 0:
                        next_vm_tidx = self.tr_manager.get_next_track(vm_layer,
                                                                      bk_stub.track_id.base_index,
                                                                      'sig', 'sup', up=False)
                        sup_vm_idx_list.append(next_vm_tidx)
                    bk_dict[f'{bk_idx}'] = bk_stub

                    out_cur = inst.get_pin('out_p')
                    if flop:
                        rstb_list.append(inst.get_pin('RSTb'))
                        rstb_vm_list.append(inst.get_pin('RSTb_vm'))
                        se_list.append(inst.get_pin('SE'))
                        se_vm_list.append(inst.get_pin('SE_vm'))
                        ck_list.append(inst.get_pin('ck'))
                        ck_vm_list.append(inst.get_pin('ck_vm'))

                        si_cur = inst.get_pin('si')
                        si_xm0 = self.grid.coord_to_track(xm_layer, si_cur.middle,
                                                          mode=RoundMode.NEAREST)
                        o_xm = self.tr_manager.get_next_track(xm_layer, si_xm0, 'sig', 'sig',
                                                              up=True)
                        si_xm1 = self.tr_manager.get_next_track(xm_layer, o_xm, 'sig', 'sig',
                                                                up=True)
                        si_xm2 = self.tr_manager.get_next_track(xm_layer, si_xm0, 'sig', 'sig',
                                                                up=vm_sep + 1)
                        si_xm1 = max(si_xm1, si_xm2)
                    else:
                        o_xm = self.grid.coord_to_track(xm_layer, out_cur.middle,
                                                        mode=RoundMode.NEAREST)
                        si_cur = None
                        si_xm0 = None
                        si_xm1 = None

                    in_cur = inst.get_pin('in_p')
                    in_xm = self.grid.coord_to_track(xm_layer, in_cur.middle,
                                                     mode=RoundMode.NEAREST)
                    if row == 0 and col == 0:
                        if flop:
                            isi = self.connect_to_tracks(si_cur, TrackID(xm_layer, si_xm0),
                                                         track_lower=0)
                            self.add_pin('iSI', isi)
                        dlyin = self.connect_to_tracks(in_cur, TrackID(xm_layer, in_xm),
                                                       track_lower=0)
                        self.add_pin('dlyin', dlyin)
                        dlyout = self.connect_to_tracks(out_cur, TrackID(xm_layer, o_xm),
                                                        track_lower=0)
                        self.add_pin('dlyout', dlyout)
                    elif row % 2 == 1:
                        si_xm = si_xm0 if col % 2 == 1 else si_xm1

                        if is_last:
                            ci_p = self.connect_to_tracks(inst.get_pin('ci_p'),
                                                          TrackID(xm_layer, o_xm),
                                                          track_lower=0)
                            self.add_pin(f'b{num_insts - 1}', ci_p)
                            co_p = self.connect_to_tracks(inst.get_pin('co_p'),
                                                          TrackID(xm_layer, in_xm),
                                                          track_lower=0)
                            self.add_pin(f'a{num_insts - 1}', co_p)

                            if flop:
                                so = self.connect_to_tracks(inst.get_pin('so'),
                                                            TrackID(xm_layer, si_xm),
                                                            track_lower=0)
                                self.add_pin('SOOUT', so)

                        if col == 0:
                            pass
                        else:
                            if not is_last:
                                if flop:
                                    self.connect_to_tracks([dly_cell_row[-1].get_pin('si'),
                                                            inst.get_pin('so')],
                                                           TrackID(xm_layer, si_xm))

                                self.connect_to_tracks([dly_cell_row[-1].get_pin('in_p'),
                                                        inst.get_pin('co_p')],
                                                       TrackID(xm_layer, in_xm))
                                self.connect_to_tracks([dly_cell_row[-1].get_pin('out_p'),
                                                        inst.get_pin('ci_p')],
                                                       TrackID(xm_layer, o_xm))

                            if col == num_cols - 1:
                                lower_inst = dly_cell_list[row - 1][-1]
                                if flop:
                                    si_prev = lower_inst.get_pin('si')
                                    si_xm = self.grid.coord_to_track(xm_layer, si_prev.middle,
                                                                     mode=RoundMode.NEAREST)
                                    tid = self.tr_manager.get_next_track_obj(TrackID(xm_layer,
                                                                                     si_xm),
                                                                             'sig', 'sig',
                                                                             vm_sep + 1)
                                    self.connect_to_tracks([lower_inst.get_pin('so'), si_cur], tid)

                                self.connect_wires([lower_inst.get_pin('co_p'), in_cur])
                                self.connect_wires([lower_inst.get_pin('ci_p'), out_cur])
                    else:
                        if col == 0:
                            lower_inst = dly_cell_list[row - 1][0]
                            if flop:
                                si_prev = lower_inst.get_pin('si')
                                si_xm = self.grid.coord_to_track(xm_layer, si_prev.middle,
                                                                 mode=RoundMode.NEAREST)
                                tid = self.tr_manager.get_next_track_obj(TrackID(xm_layer, si_xm),
                                                                         'sig', 'sig', vm_sep + 1)
                                self.connect_to_tracks([lower_inst.get_pin('so'), si_cur], tid)

                            self.connect_wires([lower_inst.get_pin('co_p'), in_cur])
                            self.connect_wires([lower_inst.get_pin('ci_p'), out_cur])
                        else:
                            if is_last:
                                ci_p = self.connect_to_tracks(inst.get_pin('ci_p'),
                                                              TrackID(xm_layer, o_xm),
                                                              track_upper=self.bound_box.w)
                                self.add_pin(f'b{num_insts - 1}', ci_p)
                                co_p = self.connect_to_tracks(inst.get_pin('co_p'),
                                                              TrackID(xm_layer, in_xm),
                                                              track_upper=self.bound_box.w)
                                self.add_pin(f'a{num_insts - 1}', co_p)

                                if flop:
                                    so = self.connect_to_tracks(inst.get_pin('so'),
                                                                TrackID(xm_layer, si_xm0),
                                                                track_upper=self.bound_box.w)
                                    self.add_pin('SOOUT', so)

                            if flop:
                                self.connect_to_tracks([dly_cell_row[-1].get_pin('so'), si_cur],
                                                       TrackID(xm_layer, si_xm0))

                            self.connect_to_tracks([dly_cell_row[-1].get_pin('co_p'), in_cur],
                                                   TrackID(xm_layer, in_xm))
                            self.connect_to_tracks([dly_cell_row[-1].get_pin('ci_p'), out_cur],
                                                   TrackID(xm_layer, o_xm))

                dly_cell_row.append(inst)
            row_vdd = self.connect_to_tracks(row_vdd_list, row_vdd_tid, track_lower=0,
                                             track_upper=self.bound_box.w)
            vdd_list.append(row_vdd)

            row_vss = self.connect_to_tracks(row_vss0_list + prev_row_vss1_list, row_vss_tid,
                                             track_lower=0, track_upper=self.bound_box.w)
            vss_list.append(row_vss)
            prev_row_vss1_list = row_vss1_list.copy()
            dly_cell_list.append(dly_cell_row)
        vss_list.append(self.connect_to_tracks(prev_row_vss1_list, top_sup_tid, track_lower=0,
                                               track_upper=self.bound_box.w))

        # 4. add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            for i in range(num_rows * 2):
                self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=i)
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            for i in range(num_rows * 2):
                self.add_tap(tot_cols, tap_vdd_list, tap_vss_list, flip_lr=True, tile_idx=i)

        # --- Routing --- #
        # 1. Supplies
        vdd = vdd_list
        vss = vss_list

        if tap_vdd_list:
            vdd = self.connect_to_track_wires(tap_vdd_list, vdd)
        if tap_vss_list:
            vss = self.connect_to_track_wires(tap_vss_list, vss)

        # get vdd and vss on vm_layer and xm_layer
        vdd_xm, vss_xm = [], []
        tr_w_v_sup = self.tr_manager.get_width(vm_layer, 'sup')
        tr_w_x_sup = self.tr_manager.get_width(xm_layer, 'sup')
        sup_vm_tid = TrackID(vm_layer, sup_vm_idx_list[0], width=tr_w_v_sup,
                             num=len(sup_vm_idx_list),
                             pitch=sup_vm_idx_list[1] - sup_vm_idx_list[0])
        for vdd_ind in vdd:
            cur_tidx = vdd_ind.track_id.base_index
            vdd_vm = self.connect_to_tracks(vdd_ind, sup_vm_tid, min_len_mode=MinLenMode.MIDDLE)
            xm_idx = self.grid.coord_to_track(xm_layer,
                                              self.grid.track_to_coord(hm_layer, cur_tidx),
                                              mode=RoundMode.NEAREST)
            vdd_xm.append(self.connect_to_tracks(vdd_vm, TrackID(xm_layer, xm_idx,
                                                                 width=tr_w_x_sup), track_lower=0,
                                                 track_upper=self.bound_box.w))
        for vss_ind in vss:
            cur_tidx = vss_ind.track_id.base_index
            vss_vm = self.connect_to_tracks(vss_ind, sup_vm_tid, min_len_mode=MinLenMode.MIDDLE)
            xm_idx = self.grid.coord_to_track(xm_layer,
                                              self.grid.track_to_coord(hm_layer, cur_tidx),
                                              mode=RoundMode.NEAREST)
            vss_xm.append(self.connect_to_tracks(vss_vm, TrackID(xm_layer, xm_idx,
                                                                 width=tr_w_x_sup), track_lower=0,
                                                 track_upper=self.bound_box.w))

        self.add_pin('VDD', vdd_xm, label='VDD:')
        self.add_pin('VSS', vss_xm, label='VSS:')

        # 2. get bk on xm_layer
        xm_sep = self.tr_manager.get_sep(xm_layer, ('sig', 'sig'))
        for _idx in range(num_rows):
            xm0_idx = self.tr_manager.get_next_track(xm_layer, vss_xm[_idx].track_id.base_index,
                                                     'sup', 'sig', up=True)
            xm1_idx = self.tr_manager.get_next_track(xm_layer, vdd_xm[_idx].track_id.base_index,
                                                     'sup', 'sig', up=False)
            avail_list = self.get_available_tracks(xm_layer, xm0_idx, xm1_idx, lower=0,
                                                   upper=self.bound_box.w, sep=xm_sep,
                                                   include_last=True)

            xm2_idx = self.tr_manager.get_next_track(xm_layer, vdd_xm[_idx].track_id.base_index,
                                                     'sup', 'sig', up=True)
            xm3_idx = self.tr_manager.get_next_track(xm_layer,
                                                     vss_xm[_idx + 1].track_id.base_index,
                                                     'sup', 'sig', up=False)
            avail_list.extend(self.get_available_tracks(xm_layer, xm2_idx, xm3_idx, lower=0,
                                                        upper=self.bound_box.w, sep=xm_sep,
                                                        include_last=True))

            num_avail = len(avail_list)
            if num_avail < num_cols + 3:
                raise ValueError(f'There are {num_cols} bk signals and 3 extra signals, but only '
                                 f'{num_avail} tracks on layer {xm_layer}. Recheck routing.')
            if _idx == 0 and flop:
                idx_offset = 3
                clk_xm = self.connect_to_tracks(ck_vm_list, TrackID(xm_layer, avail_list[0]),
                                                track_lower=0, track_upper=self.bound_box.w)
                self.add_pin('CLKIN', clk_xm)

                rstb_xm = self.connect_to_tracks(rstb_vm_list, TrackID(xm_layer, avail_list[1]),
                                                 track_lower=0, track_upper=self.bound_box.w)
                self.connect_wires(rstb_list)
                self.add_pin('RSTb', rstb_xm)

                se_xm = self.connect_to_tracks(se_vm_list, TrackID(xm_layer, avail_list[2]),
                                               track_lower=0, track_upper=self.bound_box.w)
                self.add_pin('iSE', se_xm)
            else:
                idx_offset = 0
            for cidx in range(idx_offset, num_cols + idx_offset):
                bk_idx = _idx * num_cols + cidx - idx_offset
                try:
                    bk_xm = self.connect_to_tracks(bk_dict[f'{bk_idx}'],
                                                   TrackID(xm_layer, avail_list[cidx]),
                                                   track_lower=0, track_upper=self.bound_box.w)
                    self.add_pin(f'bk<{bk_idx}>', bk_xm)
                except KeyError:
                    break

        # set properties
        self.sch_params = dict(
            dlycell_params=dly_cell_master.sch_params,
            num_insts=num_insts,
            num_dum=num_dum,
            flop=flop,
            flop_char=flop_char,
            output_sr_pins=output_sr_pins
        )

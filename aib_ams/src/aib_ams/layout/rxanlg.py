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

from __future__ import annotations

from typing import Any, Dict, Optional, Type, List, Tuple, Mapping, Sequence

from pybag.enum import RoundMode, MinLenMode, PinMode

from bag.util.immutable import Param, ImmutableList
from bag.design.module import Module
from bag.layout.routing.base import TrackID, WireArray
from bag.layout.template import TemplateDB

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase, SupplyColumnInfo

from bag3_digital.layout.stdcells.gates import InvCore, InvChainCore
from bag3_digital.layout.stdcells.levelshifter import LevelShifterCoreOutBuffer

from ..schematic.aib_rxanlg_core import aib_ams__aib_rxanlg_core
from .se_to_diff import SingleToDiffEnable, DiffBufferEnable
from .util import draw_io_supply_column, draw_io_shifters


class RXAnalog(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_rxanlg_core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            se_params='se_to_diff parameters.',
            match_params='se_to_diff_match parameters.',
            inv_params='async output inverter parameters.',
            data_lv_params='data level shifter parameters.',
            ctrl_lv_params='control signals level shifter parameters.',
            por_lv_params='POR level shifter parameters.',
            buf_ctrl_lv_params='control level shifter input buffer parameters',
            buf_por_lv_params='POR level shifter input buffer parameters.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            core_ncol='Number of columns for receiver core.',
            tap_info_list='Extra substrate taps to draw.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_p=-1,
            ridx_n=0,
            core_ncol=0,
            tap_info_list=ImmutableList(),
        )

    @classmethod
    def get_rx_half_ncol(cls, template: MOSBase, pinfo: MOSBasePlaceInfo,
                         params: Mapping[str, Any]) -> int:
        tmp = cls._make_masters(template, pinfo, params)
        lv_data_master = tmp[0]
        inv_master = tmp[3]
        data_master = tmp[4]
        clk_master = tmp[5]

        min_sep = template.min_sep_col
        lv_data_ncol = lv_data_master.num_cols
        inv_ncol = inv_master.num_cols
        inv_ncol += (inv_ncol & 1)
        inbuf_ncol = max(data_master.num_cols, clk_master.num_cols)
        half_ncol = max(lv_data_ncol + min_sep + inv_ncol, inbuf_ncol)
        return half_ncol

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, flip_tile=True)

        conn_layer = self.conn_layer
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        core_ncol: int = self.params['core_ncol']
        tap_info_list: ImmutableList[Tuple[int, bool]] = self.params['tap_info_list']

        tmp = self._make_masters(self, pinfo, self.params)
        (lv_data_master, lv_ctrl_master, lv_por_master, inv_master, data_master,
         clk_master, buf_ctrl_lv_master, buf_por_lv_master) = tmp

        inbuf_ncol = max(data_master.num_cols, clk_master.num_cols)

        # track definitions
        bot_vss = self.get_track_id(ridx_n, MOSWireType.DS, 'sup', tile_idx=1)
        top_vss = self.get_track_id(ridx_n, MOSWireType.DS, 'sup', tile_idx=2)
        # Placement
        lv_data_ncol = lv_data_master.num_cols
        lv_ctl_ncol = lv_ctrl_master.num_cols
        lv_por_ncol = lv_por_master.num_cols
        min_sep = self.min_sep_col
        sup_info = self.get_supply_column_info(xm_layer)

        inv_ncol = inv_master.num_cols
        inv_ncol += (inv_ncol & 1)
        half_ncol = max(lv_data_ncol + min_sep + inv_ncol, inbuf_ncol, core_ncol)

        # assume 2 inverter chains + margins have smaller width than a single lvl shifter
        if 2 * buf_ctrl_lv_master.num_cols + min_sep > lv_ctl_ncol:
            raise ValueError("buffer too large compared to data level shifter's width")
        if 2 * buf_por_lv_master.num_cols + min_sep > lv_por_ncol:
            raise ValueError("buffer too large compared to data level shifter's width")

        # initialize supply data structures
        lay_range = range(conn_layer, xm_layer + 1)
        vdd_io_table = {lay: [] for lay in lay_range}
        vdd_core_table = {lay: [] for lay in lay_range}
        vss_table = {lay: [] for lay in lay_range}

        # instantiate cells and collect pins
        pin_dict = {'por': [], 'porb': [], 'VDDIO': [], 'VSS': [], 'por_core': [],
                    'porb_core': [], 'VDD': []}
        in_pins = ['din_en', 'por_in', 'ck_en', 'unused']
        cur_col = 0
        cur_col = draw_io_supply_column(self, cur_col, sup_info, vdd_io_table,
                                        vdd_core_table, vss_table, ridx_p, ridx_n, True)
        new_col = draw_io_shifters(self, cur_col, buf_ctrl_lv_master, buf_por_lv_master,
                                   lv_ctrl_master, lv_por_master,
                                   bot_vss, top_vss, in_pins[0], in_pins[1],
                                   True, True, False, pin_dict)
        cur_col = new_col
        cur_col = draw_io_supply_column(self, cur_col, sup_info, vdd_io_table,
                                        vdd_core_table, vss_table, ridx_p, ridx_n, False)
        cur_col += half_ncol
        self._draw_data_path(cur_col, data_master, lv_data_master, inv_master,
                             bot_vss, True, 'data_', pin_dict)
        cur_col += min_sep
        self._draw_data_path(cur_col, clk_master, lv_data_master, inv_master,
                             bot_vss, False, 'clk_', pin_dict)
        cur_col += half_ncol + self.sub_sep_col // 2
        cur_col = draw_io_supply_column(self, cur_col, sup_info, vdd_io_table,
                                        vdd_core_table, vss_table, ridx_p, ridx_n, True)
        cur_col = draw_io_shifters(self, cur_col, buf_ctrl_lv_master, buf_por_lv_master,
                                   lv_ctrl_master, lv_por_master,
                                   bot_vss, top_vss, in_pins[2], in_pins[3],
                                   True, True, False, pin_dict, flip_lr=True)

        # add tapes on bottom tile
        max_col = self._draw_bot_supply_column(tap_info_list, sup_info, vdd_core_table, vss_table,
                                               ridx_p, ridx_n)
        max_col = max(cur_col, max_col)
        self.set_mos_size(num_cols=max_col)
        xh = self.bound_box.xh

        # connect and export supply pins
        vss_table[hm_layer].extend(pin_dict['VSS'])
        vdd_core_table[hm_layer].extend(pin_dict['VDD'])
        vdd_io_table[hm_layer].extend(pin_dict['VDDIO'])
        vss_list = vdd_io_list = vdd_core_list = []
        for lay in range(hm_layer, xm_layer + 1, 2):
            vss_list = self.connect_wires(vss_table[lay], upper=xh)
            vdd_core_list = self.connect_wires(vdd_core_table[lay], upper=xh)
            vdd_io_list = self.connect_wires(vdd_io_table[lay], upper=xh)

        vss_vm = vss_table[vm_layer]
        self.add_pin('VDDIO', vdd_io_list)
        self.add_pin('VDDCore', vdd_core_list)
        self.add_pin('VSS', vss_list)
        self.add_pin('VDDIO_vm', vdd_io_table[vm_layer], hide=True)
        self.add_pin('VDDCore_vm', vdd_core_table[vm_layer], hide=True)
        self.add_pin('VSS_vm', vss_vm, hide=True)
        self.add_pin('VDDIO_conn', vdd_io_table[conn_layer], hide=True)
        self.add_pin('VDDCore_conn', vdd_core_table[conn_layer], hide=True)
        self.add_pin('VSS_conn', vss_table[conn_layer], hide=True)

        # connect VDDCore signals
        grid = self.grid
        tr_manager = self.tr_manager
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        xm_w = tr_manager.get_width(xm_layer, 'sig')
        xm_w_hs = tr_manager.get_width(xm_layer, 'sig_hs')
        # VDD, por_buf, clkn, clkp, data, data_async, porb_buf, data_en, clk_in, por_in
        idx_list = tr_manager.place_wires(xm_layer, ['sup', 'sig', 'sig_hs', 'sig_hs', 'sig_hs',
                                                     'sig_hs', 'sig', 'sig', 'sig', 'sig'],
                                          align_track=vdd_core_list[0].track_id.base_index)[1]
        # POR
        por_core, porb_core = pin_dict['por_in_buf']
        por_list = pin_dict['por_core']
        porb_list = pin_dict['porb_core']
        por_list.append(por_core)
        porb_list.append(porb_core)
        por_core, porb_core = self.connect_differential_tracks(por_list, porb_list, xm_layer,
                                                               idx_list[1], idx_list[6],
                                                               width=xm_w)
        # extend POR signals so they are symmetric with respect to center
        self.extend_wires([por_core, porb_core], upper=2 * self.bound_box.xm - por_core.lower)

        # VDDCore data and control signals
        match_wires = [pin_dict['data_out'], pin_dict['data_async'],
                       pin_dict['clk_outp'], pin_dict['clk_outn']]
        results = self.connect_matching_tracks(match_wires, xm_layer,
                                               [idx_list[4], idx_list[5], idx_list[3], idx_list[2]],
                                               width=xm_w_hs, track_lower=0)
        self.add_pin('odat', results[0], mode=PinMode.LOWER)
        self.add_pin('odat_async', results[1], mode=PinMode.LOWER)
        self.add_pin('oclkp', results[2], mode=PinMode.LOWER)
        self.add_pin('oclkn', results[3], mode=PinMode.LOWER)

        in_info_list = [('din_en', 'data_en', idx_list[7]), ('por_in', 'por', idx_list[9]),
                        ('unused', 'unused', -1), ('ck_en', 'clk_en', idx_list[8])]
        for buf_idx, (key, port_name, tidx) in enumerate(in_info_list):
            hm_warr = pin_dict[key][0]
            vm_tidx_ref = pin_dict[key + '_buf'][1].track_id.base_index
            vm_tidx = tr_manager.get_next_track(vm_layer, vm_tidx_ref, 'sig', 'sig',
                                                up=((buf_idx & 1) == 0))
            vm_warr = self.connect_to_tracks(hm_warr, TrackID(vm_layer, vm_tidx, width=vm_w),
                                             min_len_mode=MinLenMode.LOWER)
            if key == 'unused':
                self.connect_to_tracks(vm_warr, bot_vss)
            else:
                self.add_pin(port_name,
                             self.connect_to_tracks(vm_warr, TrackID(xm_layer, tidx, width=xm_w),
                                                    track_lower=0),
                             mode=PinMode.LOWER)

        # connect VDDIO signals
        idx_list = tr_manager.place_wires(xm_layer, ['sig', 'sig', 'sup', 'sig', 'sig'],
                                          align_track=vdd_io_list[0].track_id.base_index,
                                          align_idx=2)[1]
        for name, ename in [('din_en', 'data'), ('ck_en', 'clk')]:
            wp, wn = self.connect_differential_tracks(pin_dict[f'{name}_out'],
                                                      pin_dict[f'{name}_outb'],
                                                      xm_layer, idx_list[1], idx_list[3],
                                                      width=xm_w)
            self.connect_to_track_wires(wp, pin_dict[f'{ename}_en'])
            self.connect_to_track_wires(wn, pin_dict[f'{ename}_enb'])
        rp, rn = self.connect_differential_tracks(pin_dict['por_in_out'], pin_dict['por_in_outb'],
                                                  xm_layer, idx_list[0], idx_list[4], width=xm_w)
        self.connect_differential_wires(pin_dict['por'], pin_dict['porb'], rp, rn)

        idx_list = tr_manager.place_wires(vm_layer, ['sig', 'sig', 'sup'],
                                          align_track=vdd_io_table[vm_layer][0].track_id.base_index,
                                          align_idx=-1)[1]
        rp, rn = self.connect_differential_tracks(rp, rn, vm_layer, idx_list[0], idx_list[1],
                                                  width=tr_manager.get_width(vm_layer, 'sig'))
        self.add_pin('por_vccl', rp)
        self.add_pin('porb_vccl', rn)

        # connect io pad signals
        in_pins = pin_dict['data_in']
        in_pins.append(pin_dict['clk_inp'][0])
        clkn = pin_dict['clk_inn'][0]
        in_tidx = grid.coord_to_track(xm_layer, in_pins[0].middle, mode=RoundMode.GREATER_EQ)
        ck_tidx = grid.coord_to_track(xm_layer, clkn.middle, mode=RoundMode.LESS_EQ)
        io_w = tr_manager.get_width(xm_layer, 'padout')
        in_warr = self.connect_to_tracks(in_pins, TrackID(xm_layer, in_tidx, width=io_w),
                                         min_len_mode=MinLenMode.MIDDLE)
        clk_warr = self.connect_to_tracks(clkn, TrackID(xm_layer, ck_tidx, width=io_w),
                                          track_lower=in_warr.lower, track_upper=in_warr.upper)
        self.add_pin('iopad', in_warr)
        self.add_pin('iclkn', clk_warr)

        # setup schematic parameters
        rm_keys = ['dual_output', 'invert_out']
        data_lv_sch_params = lv_data_master.sch_params.copy(remove=rm_keys)
        ctrl_lv_sch_params = lv_ctrl_master.sch_params.copy(remove=rm_keys)
        por_lv_sch_params = lv_por_master.sch_params.copy(remove=rm_keys)
        self.sch_params = dict(
            data_params=data_master.sch_params,
            clk_params=clk_master.sch_params,
            data_lv_params=data_lv_sch_params,
            ctrl_lv_params=ctrl_lv_sch_params,
            por_lv_params=por_lv_sch_params,
            buf_ctrl_lv_params=buf_ctrl_lv_master.sch_params.copy(remove=['dual_output']),
            buf_por_lv_params=buf_por_lv_master.sch_params.copy(remove=['dual_output']),
            inv_params=inv_master.sch_params,
        )

    @classmethod
    def _make_masters(cls, template: MOSBase, pinfo: MOSBasePlaceInfo, params: Mapping[str, Any]
                      ) -> Tuple[LevelShifterCoreOutBuffer, LevelShifterCoreOutBuffer,
                                 LevelShifterCoreOutBuffer, InvCore, SingleToDiffEnable,
                                 DiffBufferEnable, InvChainCore, InvChainCore]:
        se_params: Param = params['se_params']
        match_params: Param = params['match_params']
        inv_params: Param = params['inv_params']
        data_lv_params: Param = params['data_lv_params']
        ctrl_lv_params: Param = params['ctrl_lv_params']
        por_lv_params: Param = params['por_lv_params']
        buf_ctrl_lv_params: Param = params['buf_ctrl_lv_params']
        buf_por_lv_params: Param = params['buf_por_lv_params']

        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']

        # setup master parameters
        append = dict(pinfo=pinfo, ridx_n=ridx_n, ridx_p=ridx_p)
        data_params = data_lv_params.copy(append=dict(dual_output=True,
                                                      vertical_rst=['rst_outp', 'rst_outn',
                                                                    'rst_casc'],
                                                      **append))
        ctrl_params = ctrl_lv_params.copy(append=dict(dual_output=True, **append))
        inv_params = inv_params.copy(append=append)
        data_buf_params = dict(vertical_out=False, **append, **se_params)
        clk_buf_params = dict(vertical_out=False, **append, **match_params)

        # create masters
        lv_data_master = template.new_template(LevelShifterCoreOutBuffer, params=data_params)
        lv_ctrl_master = template.new_template(LevelShifterCoreOutBuffer, params=ctrl_params)
        inv_master = template.new_template(InvCore, params=inv_params)
        data_master = template.new_template(SingleToDiffEnable, params=data_buf_params)
        clk_master = template.new_template(DiffBufferEnable, params=clk_buf_params)
        buf_ctrl_lv_master = cls._get_buf_lv_master(template, buf_ctrl_lv_params, append)
        buf_por_lv_master = cls._get_buf_lv_master(template, buf_por_lv_params, append)

        # NOTE: Set POR level shifter to 4 columns less than normal level shifter.
        # Since both level shifter are dual output and symmetric, this means that the POR level
        # shifter is 2 columns shorter than norma level shifter on left and right side, which
        # will conveniently shift the vm_layer output wires of POR to not collide with
        # the control level shifters.
        # Also, since POR level shifter is usually small (because it doesn't have to be big),
        # this make sure we have enough routing tracks for differential inputs on vm_layer
        por_ncol = lv_ctrl_master.num_cols - 4
        por_params = por_lv_params.copy(append=dict(dual_output=True, has_rst=False,
                                                    num_col_tot=por_ncol, **append))
        lv_por_master = template.new_template(LevelShifterCoreOutBuffer, params=por_params)

        return (lv_data_master, lv_ctrl_master, lv_por_master, inv_master, data_master,
                clk_master, buf_ctrl_lv_master, buf_por_lv_master)

    def _draw_data_path(self, anchor: int, se_master: MOSBase,
                        lv_data_master: MOSBase, inv_master: MOSBase,
                        vss_tid: TrackID, flip_lr: bool, prefix: str,
                        pin_dict: Dict[str, List[WireArray]]) -> None:
        tr_manager = self.tr_manager
        grid = self.grid
        vm_layer = self.conn_layer + 2

        dir_sign = 1 - 2 * int(flip_lr)
        inv_ncol = inv_master.num_cols
        inv_ncol += (inv_ncol & 1)
        inv = self.add_tile(inv_master, 0, anchor + dir_sign * inv_ncol, flip_lr=not flip_lr)
        se = self.add_tile(se_master, 1, anchor, flip_lr=flip_lr)
        min_sep = self.min_sep_col
        min_sep += (min_sep & 1)
        if flip_lr:
            lv_col = anchor - inv_ncol - min_sep
        else:
            lv_col = anchor + inv_ncol + min_sep

        lv = self.add_tile(lv_data_master, 0, lv_col, flip_lr=flip_lr)

        # figure out wire location for connecting data buffer to level shifter
        test_pin = se.get_pin('outp')
        vm_test = lv.get_pin('midl')
        test_coord = test_pin.lower if flip_lr else test_pin.upper
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        test_tidx = tr_manager.get_next_track(vm_layer, vm_test.track_id.base_index,
                                              'sig', 'sig', up=-2 * dir_sign)
        test_wbnd = grid.get_wire_bounds(vm_layer, test_tidx, vm_w)[flip_lr]
        if (((test_wbnd < test_coord) and not flip_lr) or
                ((test_wbnd > test_coord) and flip_lr)):
            # closest wires don't work
            vm_ref = lv.get_pin('midr')
            tidx_p = tr_manager.get_next_track(vm_layer, vm_ref.track_id.base_index,
                                               'sig', 'sig', up=dir_sign)

        else:
            tidx_p = test_tidx
        tidx_n = tr_manager.get_next_track(vm_layer, tidx_p,
                                           'sig', 'sig', up=dir_sign)

        outp, outn = self.connect_differential_tracks(se.get_all_port_pins('outp'),
                                                      se.get_all_port_pins('outn'),
                                                      vm_layer, tidx_p, tidx_n, width=vm_w)
        self.connect_differential_wires(outp, outn, lv.get_pin('in'), lv.get_pin('inb'))
        self.connect_to_tracks(lv.get_pin('rst_outb'), vss_tid)

        pin_dict[prefix + 'en'] = [se.get_pin('en')]
        pin_dict[prefix + 'enb'] = [se.get_pin('enb')]
        pin_dict['por_core'].append(lv.get_pin('rst_out'))
        pin_dict['porb_core'].append(lv.get_pin('rst_casc'))

        vm_tidx_ref = inv.get_pin('out').track_id.base_index
        in_vm_tidx = tr_manager.get_next_track(vm_layer, vm_tidx_ref, 'sig', 'sig', up=False)
        in_vm = self.connect_to_tracks(inv.get_pin('in'), TrackID(vm_layer, in_vm_tidx, width=vm_w))
        if se.has_port('inp'):
            # clock path
            pin_dict[prefix + 'inp'] = [se.get_pin('inp')]
            pin_dict[prefix + 'inn'] = [se.get_pin('inn')]
            pin_dict[prefix + 'outp'] = [lv.get_pin('out')]
            pin_dict[prefix + 'outn'] = [lv.get_pin('outb')]
            self.connect_to_tracks(in_vm, vss_tid)
        else:
            # data path
            pin_dict[prefix + 'in'] = [se.get_pin('in')]
            pin_dict[prefix + 'out'] = [lv.get_pin('out')]
            pin_dict[prefix + 'async'] = [inv.get_pin('out')]
            self.connect_to_track_wires([lv.get_pin('poutb'), lv.get_pin('noutb')], in_vm)

    def _draw_bot_supply_column(self, col_list: Sequence[Tuple[int, bool]],
                                sup_info: SupplyColumnInfo,
                                vdd_core_table: Dict[int, List[WireArray]],
                                vss_table: Dict[int, List[WireArray]],
                                ridx_p: int, ridx_n: int) -> int:
        ncol = sup_info.ncol
        max_col = 0
        for idx, (cur_col, flip_lr) in enumerate(col_list):
            anchor = cur_col + int(flip_lr) * ncol
            self.add_supply_column(sup_info, anchor, vdd_core_table, vss_table,
                                   ridx_p=ridx_p, ridx_n=ridx_n, extend_vdd=False, flip_lr=flip_lr,
                                   extend_vss=False, min_len_mode=MinLenMode.MIDDLE)
            max_col = max(max_col, cur_col + ncol)
        return max_col

    @classmethod
    def _get_buf_lv_master(cls, template: MOSBase, buf_params: Param, append: Mapping[str, Any]
                           ) -> InvChainCore:
        buf_params = buf_params.copy(append=dict(
            dual_output=True,
            vertical_output=True,
            **append
        ), remove=['sig_locs'])
        buf_master = template.new_template(InvChainCore, params=buf_params)

        vm_layer = template.conn_layer + 2
        out_tidx = buf_master.get_port('out').get_pins()[0].track_id.base_index
        prev_tidx = template.tr_manager.get_next_track(vm_layer, out_tidx, 'sig', 'sig', up=False)
        buf_master = buf_master.new_template_with(sig_locs=dict(outb=prev_tidx))
        return buf_master

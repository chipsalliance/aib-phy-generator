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

from typing import Dict, List, Tuple, Union, Optional, cast

from pybag.enum import MinLenMode, RoundMode

from bag.util.math import HalfInt
from bag.layout.template import PyLayInstance
from bag.layout.routing.base import WireArray, TrackID

from xbase.layout.mos.base import MOSBase, SupplyColumnInfo

from bag3_digital.layout.stdcells.levelshifter import LevelShifterCoreOutBuffer


def draw_io_supply_column(template: MOSBase, col: int, sup_info: SupplyColumnInfo,
                          vdd_io_table: Dict[int, List[WireArray]],
                          vdd_core_table: Dict[int, List[WireArray]],
                          vss_table: Dict[int, List[WireArray]],
                          ridx_p: int, ridx_n: int, flip_lr: bool) -> int:
    ncol = sup_info.ncol
    sup_col = col + int(flip_lr) * ncol
    # draw vdd core columns
    template.add_supply_column(sup_info, sup_col, vdd_core_table, vss_table, ridx_p=ridx_p,
                               ridx_n=ridx_n, flip_lr=flip_lr, extend_vdd=False,
                               extend_vss=False, min_len_mode=MinLenMode.MIDDLE)
    # draw vdd_io columns
    for tile in range(1, 3):
        template.add_supply_column(sup_info, sup_col, vdd_io_table, vss_table, ridx_p=ridx_p,
                                   ridx_n=ridx_n, tile_idx=tile, flip_lr=flip_lr,
                                   extend_vdd=False)
    return col + ncol + (template.sub_sep_col // 2)


def get_io_shifters_ncol(sub_sep_col: int, bot_lv_master: MOSBase,
                         top_lv_master: MOSBase) -> int:
    top_ncol = top_lv_master.num_cols
    bot_ncol = bot_lv_master.num_cols
    top_center = cast(LevelShifterCoreOutBuffer, top_lv_master).center_col
    bot_center = cast(LevelShifterCoreOutBuffer, bot_lv_master).center_col

    delta = max(top_center, bot_center)
    col_mid = delta
    col_end = col_mid + max(top_ncol - top_center, bot_ncol - bot_center)
    return col_end + (sub_sep_col // 2)


def draw_io_shifters(template: MOSBase, col: int, buf_bot_master: MOSBase, buf_top_master: MOSBase,
                     bot_lv_master: LevelShifterCoreOutBuffer,
                     top_lv_master: LevelShifterCoreOutBuffer, bot_vss: TrackID,
                     top_vss: TrackID, bot_name: str, top_name: str, bot_rst_out: bool,
                     top_rst_out: bool, is_tx_din: bool, pin_dict: Dict[str, List[WireArray]],
                     flip_lr: bool = False) -> int:
    # NOTE: the level shifters are arranged so that the rst/rstb signals of the top/bottom
    # level shifters can be shorted together.
    top_ncol = top_lv_master.num_cols
    bot_ncol = bot_lv_master.num_cols
    top_buf_ncol = buf_top_master.num_cols
    bot_buf_ncol = buf_bot_master.num_cols
    top_center = cast(LevelShifterCoreOutBuffer, top_lv_master).center_col
    bot_center = cast(LevelShifterCoreOutBuffer, bot_lv_master).center_col

    delta = max(top_center, bot_center)
    col_mid = col + delta
    col_end = col_mid + max(top_ncol - top_center, bot_ncol - bot_center)
    if flip_lr:
        col_mid = col_end - delta
        sign = -1
        midr_name = 'midl'
        midl_name = 'midr'
        bufl_ncol = bot_buf_ncol
        bufr_ncol = top_buf_ncol
    else:
        sign = 1
        midr_name = 'midr'
        midl_name = 'midl'
        bufl_ncol = top_buf_ncol
        bufr_ncol = bot_buf_ncol

    bot_lv = template.add_tile(bot_lv_master, 1, col_mid - sign * bot_center, flip_lr=flip_lr)
    top_lv = template.add_tile(top_lv_master, 2, col_mid - sign * top_center, flip_lr=flip_lr)

    # get buffer location
    min_sep2 = template.min_sep_col // 2
    bufl_col = col_mid - min_sep2 - (bufl_ncol + (bufl_ncol & 1))
    bufr_col = col_mid + min_sep2 + (bufr_ncol + (bufr_ncol & 1))

    if bot_lv_master.mid_vertical or top_lv_master.mid_vertical:
        arr_info = template.arr_info
        tr_manager = template.tr_manager
        vm_layer = template.conn_layer + 2
        if bot_lv_master.mid_vertical:
            midl_tidx = bot_lv.get_pin(midl_name).track_id.base_index
            midr_tidx = bot_lv.get_pin(midr_name).track_id.base_index
            if top_lv_master.mid_vertical:
                midl_tidx = min(top_lv.get_pin(midl_name).track_id.base_index, midl_tidx)
                midr_tidx = max(top_lv.get_pin(midr_name).track_id.base_index, midr_tidx)
        else:
            midl_tidx = top_lv.get_pin(midl_name).track_id.base_index
            midr_tidx = top_lv.get_pin(midr_name).track_id.base_index

        bufl_tidx = tr_manager.get_next_track(vm_layer, midl_tidx, 'sig', 'sig', up=False)
        bufr_tidx = tr_manager.get_next_track(vm_layer, midr_tidx, 'sig', 'sig', up=True)
        coll_idx = arr_info.track_to_col(vm_layer, bufl_tidx, mode=RoundMode.LESS_EQ)
        colr_idx = arr_info.track_to_col(vm_layer, bufr_tidx, mode=RoundMode.GREATER_EQ)
        bufl_col = min(bufl_col, coll_idx - (bufl_ncol + (bufl_ncol & 1)))
        bufr_col = max(bufr_col, colr_idx + (bufr_ncol + (bufr_ncol & 1)))
    if flip_lr:
        top_buf_col = bufr_col
        bot_buf_col = bufl_col
    else:
        top_buf_col = bufl_col
        bot_buf_col = bufr_col

    top_buf = template.add_tile(buf_top_master, 0, top_buf_col, flip_lr=flip_lr)
    bot_buf = template.add_tile(buf_bot_master, 0, bot_buf_col, flip_lr=not flip_lr)

    vdd_list = pin_dict['VDD']
    vss_list = pin_dict['VSS']
    vdd_io_list = pin_dict['VDDIO']
    vdd_list.extend(top_buf.port_pins_iter('VDD'))
    vdd_list.extend(bot_buf.port_pins_iter('VDD'))
    vdd_io_list.extend(top_lv.port_pins_iter('VDD'))
    vdd_io_list.extend(bot_lv.port_pins_iter('VDD'))
    vss_list.extend(top_buf.port_pins_iter('VSS'))
    vss_list.extend(bot_buf.port_pins_iter('VSS'))
    vss_list.extend(top_lv.port_pins_iter('VSS'))
    vss_list.extend(bot_lv.port_pins_iter('VSS'))

    # NOTE: here we assume that the last stage of the inverter chain is large
    # enough, so inverter output wires shouldn't run into level shifter middle wires.
    out_top = top_buf.get_pin('out')
    outb_top = top_buf.get_pin('outb')
    out_bot = bot_buf.get_pin('out')
    outb_bot = bot_buf.get_pin('outb')
    template.connect_differential_wires(out_top, outb_top, top_lv.get_pin('in'),
                                        top_lv.get_pin('inb'))
    template.connect_differential_wires(out_bot, outb_bot, bot_lv.get_pin('in'),
                                        bot_lv.get_pin('inb'))

    pin_dict[top_name] = [top_buf.get_pin('in')]
    pin_dict[bot_name] = [bot_buf.get_pin('in')]
    pin_dict[top_name + '_buf'] = [out_top, outb_top]
    pin_dict[bot_name + '_buf'] = [out_bot, outb_bot]
    # connect resets to vm layer
    # NOTE: here we assume that the inverter chain last stage is small enough, and the
    # nmos of the level shifters are are enough, so the track immediately adjacent to the
    # outputs of the invert chain can be used to connect rstb signals together without
    # running into rst signal's hm layer wires.
    if flip_lr:
        rstb_l_tidx = _get_rstb_vm_tidx(template, out_bot, outb_bot, True)
        rstb_r_tidx = _get_rstb_vm_tidx(template, out_top, outb_top, False)
    else:
        rstb_l_tidx = _get_rstb_vm_tidx(template, out_top, outb_top, True)
        rstb_r_tidx = _get_rstb_vm_tidx(template, out_bot, outb_bot, False)

    xl_b, xh_b = _get_por_vm_coords(bot_lv)
    xl_t, xh_t = _get_por_vm_coords(top_lv)
    bot_rst_l_tidx = _get_rst_vm_tidx(template, xl_b, rstb_l_tidx, True)
    bot_rst_r_tidx = _get_rst_vm_tidx(template, xh_b, rstb_r_tidx, False)
    top_rst_l_tidx = _get_rst_vm_tidx(template, xl_t, rstb_l_tidx, True)
    top_rst_r_tidx = _get_rst_vm_tidx(template, xh_t, rstb_r_tidx, False)
    if is_tx_din:
        # NOTE: here, we use the fact that we only have inverter buffer on the right, and the
        # rst signals on the right are connected to VSS (meaning they don't need to cross into
        # the other tile).
        # This means that the rst signals on the left can be shorted with the left-most
        # vertical track and we won't run into potential vertical wires from
        # inverter buffers (because they're guaranteed to not be there), and the rst signals on
        # the right can use their own vertical tracks to short to VSS.
        rst_idx_list = [min(bot_rst_l_tidx, top_rst_l_tidx), rstb_l_tidx,
                        rstb_r_tidx, bot_rst_r_tidx]
        _record_lv_pins(template, bot_lv, bot_vss, pin_dict, bot_name, bot_rst_out, rst_idx_list)
        rst_idx_list[3] = top_rst_r_tidx
        _record_lv_pins(template, top_lv, top_vss, pin_dict, top_name, top_rst_out, rst_idx_list)
    else:
        # NOTE: we know the core of top and bottom level shifters are identical, so we use the
        # same vertical tracks
        rst_idx_list = [bot_rst_l_tidx, rstb_l_tidx, rstb_r_tidx, bot_rst_r_tidx]

        _record_lv_pins(template, bot_lv, bot_vss, pin_dict, bot_name, bot_rst_out, rst_idx_list)
        _record_lv_pins(template, top_lv, top_vss, pin_dict, top_name, top_rst_out, rst_idx_list)

    return col_end + (template.sub_sep_col // 2)


def _get_rstb_vm_tidx(template: MOSBase, out: WireArray, outb: WireArray, left: bool) -> HalfInt:
    tr_manager = template.tr_manager
    vm_layer = template.conn_layer + 2

    if left:
        sel_fun = min
        up = False
    else:
        sel_fun = max
        up = True

    out_tidx = sel_fun(out.track_id.base_index, outb.track_id.base_index)
    rstb_tidx = tr_manager.get_next_track(vm_layer, out_tidx, 'sig', 'sig', up=up)
    return rstb_tidx


def _get_rst_vm_tidx(template: MOSBase, coord: Optional[int], rstb_tidx: HalfInt, left: bool
                     ) -> Optional[HalfInt]:
    if coord is None:
        return None
    grid = template.grid
    tr_manager = template.tr_manager
    vm_layer = template.conn_layer + 2
    vm_w = tr_manager.get_width(vm_layer, 'sig')

    if left:
        rst_mode = RoundMode.LESS_EQ
        sel_fun = min
        up = False
    else:
        rst_mode = RoundMode.GREATER_EQ
        sel_fun = max
        up = True

    rst_tidx = grid.find_next_track(vm_layer, coord, tr_width=vm_w, mode=rst_mode)
    rst_tidx = sel_fun(rst_tidx,
                       tr_manager.get_next_track(vm_layer, rstb_tidx, 'sig', 'sig', up=up))

    return rst_tidx


def _record_lv_pins(template: MOSBase, inst: PyLayInstance, vss_tid: TrackID,
                    pin_dict: Dict[str, List[WireArray]], pin_name: str, por_to_out: bool,
                    rst_idx_list: List[HalfInt]) -> None:
    if inst.has_port('rst_out'):
        rst_out = inst.get_pin('rst_out')
        rst_casc = inst.get_pin('rst_casc')
        rst_outb = inst.get_pin('rst_outb')
        if rst_out.middle < rst_outb.middle:
            rstc_idx = 1 if por_to_out else 2
            tmp = _connect_lv_por_vm(template, rst_out, rst_outb, rst_casc, rst_idx_list, rstc_idx)
            rst_out, rst_outb, rst_casc = tmp
        else:
            rstc_idx = 2 if por_to_out else 1
            tmp = _connect_lv_por_vm(template, rst_outb, rst_out, rst_casc, rst_idx_list, rstc_idx)
            rst_outb, rst_out, rst_casc = tmp

        if por_to_out:
            template.connect_to_tracks(rst_outb, vss_tid)
            pin_dict['por'].append(rst_out)
            pin_dict['porb'].append(rst_casc)
        else:
            template.connect_to_tracks(rst_out, vss_tid)
            pin_dict['por'].append(rst_outb)
            pin_dict['porb'].append(rst_casc)

    master = cast(LevelShifterCoreOutBuffer, inst.master)
    key = pin_name + '_out'
    if master.dual_output:
        pin_dict[key] = [inst.get_pin('out')]
        pin_dict[key + 'b'] = [inst.get_pin('outb')]
    elif master.outr_inverted:
        pin_dict[key] = [inst.get_pin('outb')]
    else:
        pin_dict[key] = [inst.get_pin('out')]


def _connect_lv_por_vm(template: MOSBase, rstl: WireArray, rstr: WireArray, rstc: WireArray,
                       rst_idx_list: List[HalfInt], rstc_idx: int
                       ) -> Tuple[WireArray, WireArray, WireArray]:
    vm_layer = template.conn_layer + 2
    vm_w = template.tr_manager.get_width(vm_layer, 'sig')

    rstl = template.connect_to_tracks(rstl, TrackID(vm_layer, rst_idx_list[0], width=vm_w))
    rstc = template.connect_to_tracks(rstc, TrackID(vm_layer, rst_idx_list[rstc_idx], width=vm_w))
    rstr = template.connect_to_tracks(rstr, TrackID(vm_layer, rst_idx_list[3], width=vm_w))
    return rstl, rstr, rstc


def _get_por_vm_coords(inst: PyLayInstance) -> Union[Tuple[None, None], Tuple[int, int]]:
    if not inst.has_port('rst_out'):
        return None, None
    rst_out = inst.get_pin('rst_out')
    rst_outb = inst.get_pin('rst_outb')
    if rst_out.middle < rst_outb.middle:
        return rst_out.upper, rst_outb.lower
    else:
        return rst_outb.upper, rst_out.lower

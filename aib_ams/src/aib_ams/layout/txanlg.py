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

from typing import Any, Dict, List, Sequence, Optional, Type, Tuple, Mapping

from pybag.enum import MinLenMode, PinMode

from bag.util.immutable import Param, ImmutableList
from bag.design.module import Module
from bag.layout.template import TemplateDB, PyLayInstance
from bag.layout.routing.base import WireArray, TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase, SupplyColumnInfo

from bag3_digital.layout.stdcells.gates import InvChainCore
from bag3_digital.layout.stdcells.levelshifter import LevelShifterCoreOutBuffer

from ..schematic.aib_txanlg_core import aib_ams__aib_txanlg_core
from .driver import AIBOutputDriver
from .util import draw_io_shifters, draw_io_supply_column, get_io_shifters_ncol


class TXAnalog(MOSBase):
    """
    TX part of AIB driver, follows Ayar's verilog models
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._core_ncol: int = 0
        self._lvshift_right_ncol: int = 0
        self._tap_info: ImmutableList[Tuple[int, bool]] = ImmutableList()

    @property
    def core_ncol(self) -> int:
        return self._core_ncol

    @property
    def lvshift_right_ncol(self) -> int:
        return self._lvshift_right_ncol

    @property
    def drv_tap_info(self) -> ImmutableList[Tuple[int, bool]]:
        return self._tap_info

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return aib_ams__aib_txanlg_core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            drv_params='output driver parameters.',
            data_lv_params='data level shifter parameters.',
            ctrl_lv_params='control signals level shifter parameters.',
            buf_data_lv_params='data level shifter input buffer parameters',
            buf_ctrl_lv_params='control level shifter input buffer parameters',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            rxhalf_ncol='number of columns of rxhalf.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_p=-1,
            ridx_n=0,
            rxhalf_ncol=0,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, flip_tile=True)

        grid = self.grid
        tr_manager = self.tr_manager

        drv_params: Param = self.params['drv_params']
        data_lv_params: Param = self.params['data_lv_params']
        ctrl_lv_params: Param = self.params['ctrl_lv_params']
        buf_data_lv_params: Param = self.params['buf_data_lv_params']
        buf_ctrl_lv_params: Param = self.params['buf_ctrl_lv_params']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        rxhalf_ncol: int = self.params['rxhalf_ncol']

        conn_layer = self.conn_layer
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        # setup master parameters
        append = dict(pinfo=pinfo, ridx_p=ridx_p, ridx_n=ridx_n)
        din_params = data_lv_params.copy(append=dict(
            dual_output=False,
            **append,
        ))
        itx_en_params = ctrl_lv_params.copy(append=dict(
            dual_output=True,
            **append,
        ))
        ctrl_params = itx_en_params.copy(append=dict(dual_output=False))
        ndrv_params = ctrl_params.copy(append=dict(invert_out=True))
        drv_params = drv_params.copy(append=append)

        # create masters
        lv_din_master = self.new_template(LevelShifterCoreOutBuffer, params=din_params)
        lv_itx_en_master = self.new_template(LevelShifterCoreOutBuffer, params=itx_en_params)
        lv_ndrv_master = self.new_template(LevelShifterCoreOutBuffer, params=ndrv_params)
        lv_ctrl_master = self.new_template(LevelShifterCoreOutBuffer, params=ctrl_params)
        drv_master: AIBOutputDriver = self.new_template(AIBOutputDriver, params=drv_params)
        buf_data_lv_master = self._get_buf_lv_master(buf_data_lv_params, append)
        buf_ctrl_lv_master = self._get_buf_lv_master(buf_ctrl_lv_params, append)

        # shift output tracks of weak_pullupenb, so it can cross into the other tile.
        tmp_tid = lv_ctrl_master.get_port('out').get_pins()[0].track_id
        pub_tidx = tr_manager.get_next_track(tmp_tid.layer_id, tmp_tid.base_index,
                                             'sig', 'sig', up=True)
        pub_params = ctrl_params.copy(append=dict(sig_locs=dict(outr=pub_tidx)))
        lv_pub_master = self.new_template(LevelShifterCoreOutBuffer, params=pub_params)

        # track definitions
        bot_vss = self.get_track_id(ridx_n, MOSWireType.DS, 'sup', tile_idx=1)
        top_vss = self.get_track_id(ridx_n, MOSWireType.DS, 'sup', tile_idx=2)
        in_pins = ['ipdrv_buf<0>', 'itx_en_buf',
                   'weak_pulldownen', 'weak_pullupenb',
                   'indrv_buf<0>', 'indrv_buf<1>',
                   'din', 'ipdrv_buf<1>']

        # Placement
        din_ncol = lv_din_master.num_cols
        ctrl_ncol = lv_ctrl_master.num_cols
        min_sep = self.min_sep_col
        sup_info = self.get_supply_column_info(xm_layer)

        # assume 2 inverter chains + margins have smaller width than a single lvl shifter
        if 2 * buf_data_lv_master.num_cols + min_sep > din_ncol:
            raise ValueError("input buffer too large compared to data level shifter's width")
        if 2 * buf_ctrl_lv_master.num_cols + min_sep > ctrl_ncol:
            raise ValueError("input buffer too large compared to control level shifter's width")

        # initialize supply data structures
        lay_range = range(conn_layer, xm_layer + 1)
        vdd_io_table = {lay: [] for lay in lay_range}
        vdd_core_table = {lay: [] for lay in lay_range}
        vss_table = {lay: [] for lay in lay_range}

        # instantiate cells and collect pins
        pin_dict = {'por': [], 'porb': [], 'VDDIO': [], 'VDD': [], 'VSS': []}
        cur_col = 0
        cur_col = draw_io_supply_column(self, cur_col, sup_info, vdd_io_table,
                                        vdd_core_table, vss_table, ridx_p, ridx_n, True)
        #  pdrv<0>, itx_en
        new_col = draw_io_shifters(self, cur_col, buf_ctrl_lv_master, buf_ctrl_lv_master,
                                   lv_ctrl_master, lv_itx_en_master,
                                   bot_vss, top_vss, in_pins[0], in_pins[1],
                                   True, True, False, pin_dict)
        ctrl_dual_ncol = new_col - cur_col
        cur_col = new_col
        cur_col = draw_io_supply_column(self, cur_col, sup_info, vdd_io_table,
                                        vdd_core_table, vss_table, ridx_p, ridx_n, False)
        core_ncol = get_io_shifters_ncol(self.sub_sep_col, lv_ctrl_master, lv_pub_master)
        self._core_ncol = max(core_ncol, rxhalf_ncol)
        # puen, puenb
        draw_io_shifters(self, cur_col, buf_ctrl_lv_master, buf_ctrl_lv_master,
                         lv_ctrl_master, lv_pub_master,
                         bot_vss, top_vss, in_pins[2], in_pins[3],
                         False, False, False, pin_dict)
        cur_col += self._core_ncol + min_sep + self.sub_sep_col // 2
        # ndrv<0>, ndrv<1>
        cur_col = draw_io_shifters(self, cur_col + self._core_ncol - core_ncol,
                                   buf_ctrl_lv_master, buf_ctrl_lv_master,
                                   lv_ndrv_master, lv_ndrv_master,
                                   bot_vss, top_vss, in_pins[4], in_pins[5],
                                   True, True, False, pin_dict)
        cur_col = draw_io_supply_column(self, cur_col, sup_info, vdd_io_table,
                                        vdd_core_table, vss_table, ridx_p, ridx_n, True)
        # din, pdrv<1>
        din_ncol_tot = get_io_shifters_ncol(self.sub_sep_col, lv_din_master, lv_ctrl_master)
        din_ncol_max = max(din_ncol_tot, ctrl_dual_ncol)
        self._lvshift_right_ncol = din_ncol_max
        cur_col = draw_io_shifters(self, cur_col + din_ncol_max - din_ncol_tot,
                                   buf_data_lv_master, buf_ctrl_lv_master,
                                   lv_din_master, lv_ctrl_master,
                                   bot_vss, top_vss, in_pins[6], in_pins[7],
                                   True, True, True, pin_dict)
        drv_inst = self.add_tile(drv_master, 1, cur_col)
        # add tapes on bottom tile
        max_col = self._draw_bot_supply_column(cur_col, drv_master.tap_columns, sup_info,
                                               vdd_core_table, vss_table, ridx_p, ridx_n)

        self.set_mos_size(num_cols=max_col)

        # connect and export supply pins
        vss_list = vdd_io_list = vdd_core_list = []
        for lay in range(hm_layer, xm_layer + 1, 2):
            vss = vss_table[lay]
            vdd_io = vdd_io_table[lay]
            vss.extend(drv_inst.port_pins_iter(f'VSS_{lay}'))
            vdd_io.extend(drv_inst.port_pins_iter(f'VDD_{lay}'))
            vss_list = self.connect_wires(vss)
            vdd_io_list = self.connect_wires(vdd_io)
            vdd_core_list = self.connect_wires(vdd_core_table[lay], upper=vdd_io[0].upper)

        self.add_pin('VDDIO', vdd_io_list)
        self.add_pin('VDDCore', vdd_core_list)
        self.add_pin('VSS', vss_list)
        self.add_pin('VDDIO_vm', vdd_io_table[vm_layer], hide=True)
        self.add_pin('VDDCore_vm', vdd_core_table[vm_layer], hide=True)
        self.add_pin('VSS_vm', vss_table[vm_layer], hide=True)
        self.add_pin('VDDIO_conn', vdd_io_table[conn_layer], hide=True)
        self.add_pin('VDDCore_conn', vdd_core_table[conn_layer], hide=True)
        self.add_pin('VSS_conn', vss_table[conn_layer], hide=True)
        self.reexport(drv_inst.get_port('VDD_vm'), net_name='VDDIO_vm')
        self.reexport(drv_inst.get_port('VSS_vm'))
        self.reexport(drv_inst.get_port('VDD_conn'), net_name='VDDIO_conn')
        self.reexport(drv_inst.get_port('VSS_conn'))

        # route input pins to the edge
        in0_tidx = tr_manager.get_next_track(xm_layer, vdd_core_list[0].track_id.base_index,
                                             'sup', 'sig', up=True)
        in_tidx_list = tr_manager.place_wires(xm_layer, ['sig'] * len(in_pins),
                                              align_track=in0_tidx)[1]
        tr_xm_w = tr_manager.get_width(xm_layer, 'sig')
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        for buf_idx, (tidx, name) in enumerate(zip(in_tidx_list, in_pins)):
            hm_warr = pin_dict[name][0]
            vm_tidx_ref = pin_dict[name + '_buf'][1].track_id.base_index
            vm_tidx = tr_manager.get_next_track(vm_layer, vm_tidx_ref, 'sig', 'sig',
                                                up=((buf_idx & 1) == 0))
            vm_warr = self.connect_to_tracks(hm_warr, TrackID(vm_layer, vm_tidx, width=vm_w),
                                             min_len_mode=MinLenMode.LOWER)
            self.add_pin(name,
                         self.connect_to_tracks(vm_warr, TrackID(xm_layer, tidx, width=tr_xm_w),
                                                track_lower=0),
                         mode=PinMode.LOWER)

        # connect POR input
        # connect POR/POR_b
        por_tid = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', wire_idx=2, tile_idx=2)
        porb_tid = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', wire_idx=2, tile_idx=1)
        por = self.connect_to_tracks(pin_dict['por'], por_tid)
        porb = self.connect_to_tracks(pin_dict['porb'], porb_tid)
        self.add_pin('por_vccl', por)
        self.add_pin('porb_vccl', porb)

        # connect driver inputs
        # compute track indices
        # track order: weak_pden, n_enb_drv<0>, pad, p_en_drv<0>, weak_puenb, din
        # n_enb_drv<1>, tristateb, pad, tristate, p_en_drv<1>
        vdd_io_tidx = vdd_io_list[0].track_id.base_index
        pad_tr_w = tr_manager.get_width(xm_layer, 'padout')
        xm_hs_w = tr_manager.get_width(xm_layer, 'sig_hs')
        pad_0_tidx = grid.get_middle_track(vss_list[0].track_id[0].base_index, vdd_io_tidx,
                                           round_up=False)
        pad_1_tidx = grid.get_middle_track(vss_list[0].track_id[1].base_index, vdd_io_tidx,
                                           round_up=True)
        pad_2_tidx = grid.get_middle_track(vss_list[0].track_id[1].base_index,
                                           vdd_io_list[1].track_id.base_index,
                                           round_up=False)
        bb_tidx_list = tr_manager.place_wires(xm_layer, ['sig', 'sig', 'padout'],
                                              align_track=pad_0_tidx, align_idx=-1)[1]
        bt_tidx_list = tr_manager.place_wires(xm_layer, ['padout', 'sig', 'sig', 'sig_hs'],
                                              align_track=pad_0_tidx)[1]
        tb_tidx_list = tr_manager.place_wires(xm_layer, ['sig', 'sig', 'padout'],
                                              align_track=pad_1_tidx, align_idx=-1)[1]
        tt_tidx_list = tr_manager.place_wires(xm_layer, ['padout', 'sig', 'sig'],
                                              align_track=pad_1_tidx)[1]
        # connect wires
        self._connect_lv_drv(pin_dict, drv_inst, 'weak_pulldownen_out', 'weak_pden',
                             TrackID(xm_layer, bb_tidx_list[0], width=tr_xm_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'indrv_buf<0>_out', 'n_enb_drv<0>',
                             TrackID(xm_layer, bb_tidx_list[1], width=tr_xm_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'ipdrv_buf<0>_out', 'p_en_drv<0>',
                             TrackID(xm_layer, bt_tidx_list[1], width=tr_xm_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'weak_pullupenb_out', 'weak_puenb',
                             TrackID(xm_layer, bt_tidx_list[2], width=tr_xm_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'din_out', 'din',
                             TrackID(xm_layer, bt_tidx_list[3], width=xm_hs_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'indrv_buf<1>_out', 'n_enb_drv<1>',
                             TrackID(xm_layer, tb_tidx_list[0], width=tr_xm_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'itx_en_buf_out', 'tristateb',
                             TrackID(xm_layer, tb_tidx_list[1], width=tr_xm_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'itx_en_buf_outb', 'tristate',
                             TrackID(xm_layer, tt_tidx_list[1], width=tr_xm_w))
        self._connect_lv_drv(pin_dict, drv_inst, 'ipdrv_buf<1>_out', 'p_en_drv<1>',
                             TrackID(xm_layer, tt_tidx_list[2], width=tr_xm_w))

        # connect driver outputs
        pad_pitch = pad_2_tidx - pad_0_tidx
        pad_tid0 = TrackID(xm_layer, pad_0_tidx, width=pad_tr_w, num=2, pitch=pad_pitch)
        pad_tid1 = TrackID(xm_layer, pad_1_tidx, width=pad_tr_w, num=2, pitch=pad_pitch)
        pad_pin = drv_inst.get_pin('txpadout')
        self.add_pin('txpadout', [self.connect_to_tracks(pad_pin, pad_tid0),
                                  self.connect_to_tracks(pad_pin, pad_tid1)])
        self.add_pin('txpadout_vm', pad_pin, hide=True)

        # setup schematic parameters
        rm_keys = ['dual_output', 'invert_out']
        data_lv_sch_params = lv_din_master.sch_params.copy(remove=rm_keys)
        ctrl_lv_sch_params = lv_ctrl_master.sch_params.copy(remove=rm_keys)
        self.sch_params = dict(
            drv_params=drv_master.sch_params,
            data_lv_params=dict(
                lev_params=data_lv_sch_params,
                buf_params=buf_data_lv_master.sch_params.copy(remove=['dual_output']),
            ),
            ctrl_lv_params=dict(
                lev_params=ctrl_lv_sch_params,
                buf_params=buf_ctrl_lv_master.sch_params.copy(remove=['dual_output']),
            ),
        )

    def _draw_bot_supply_column(self, col: int, col_list: Sequence[Tuple[int, bool]],
                                sup_info: SupplyColumnInfo,
                                vdd_core_table: Dict[int, List[WireArray]],
                                vss_table: Dict[int, List[WireArray]],
                                ridx_p: int, ridx_n: int) -> int:
        tap_info_list = []
        ncol = sup_info.ncol
        max_col = col
        for idx, (cur_col, flip_lr) in enumerate(col_list):
            col_offset = col + cur_col
            anchor = col_offset + int(flip_lr) * ncol
            tap_info_list.append((col_offset, flip_lr))
            self.add_supply_column(sup_info, anchor, vdd_core_table, vss_table,
                                   ridx_p=ridx_p, ridx_n=ridx_n, extend_vdd=False, flip_lr=flip_lr,
                                   extend_vss=False, min_len_mode=MinLenMode.MIDDLE)
            max_col = max(max_col, col_offset + ncol)
        self._tap_info = ImmutableList(tap_info_list)
        return max_col

    def _connect_lv_drv(self, pin_dict: Dict[str, List[WireArray]], drv_inst: PyLayInstance,
                        name: str, drv_pin: str, tid: TrackID) -> None:
        warr_list = pin_dict[name]
        warr_list.extend(drv_inst.port_pins_iter(drv_pin))
        self.connect_to_tracks(warr_list, tid)

    def _get_buf_lv_master(self, buf_params: Param, append: Mapping[str, Any]) -> InvChainCore:
        buf_params = buf_params.copy(append=dict(
            dual_output=True,
            vertical_output=True,
            **append
        ), remove=['sig_locs'])
        buf_master = self.new_template(InvChainCore, params=buf_params)

        vm_layer = self.conn_layer + 2
        out_tidx = buf_master.get_port('out').get_pins()[0].track_id.base_index
        prev_tidx = self.tr_manager.get_next_track(vm_layer, out_tidx, 'sig', 'sig', up=False)
        buf_master = buf_master.new_template_with(sig_locs=dict(outb=prev_tidx))
        return buf_master

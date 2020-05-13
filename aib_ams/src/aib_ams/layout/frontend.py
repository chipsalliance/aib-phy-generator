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

from typing import Any, Dict, Tuple, Optional, Type, List

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.layout.routing.base import TrackID, WireArray
from bag.layout.template import TemplateDB, PyLayInstance

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from .txanlg import TXAnalog
from .rxanlg import RXAnalog


class Frontend(MOSBase):
    """The transmitter and receiver integrated together.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_frontend_core')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            por_lv_params='POR level shifter parameters.',
            ctrl_lv_params='control signals level shifter parameters.',
            buf_data_lv_params='data level shifter input buffer parameters',
            buf_ctrl_lv_params='ctrl level shifter input buffer parameters',
            buf_por_lv_params='por level shifter input buffer parameters',
            tx_lv_params='TX data level shifter parameters.',
            drv_params='TX output driver parameters.',
            rx_lv_params='RX data level shifter parameters.',
            se_params='se_to_diff parameters.',
            match_params='se_to_diff_match parameters.',
            inv_params='RX async output inverter parameters.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            num_sup_clear='Number of supply tracks to clear out.',
            rx_in_pitch='RX input pin pitch',
            rx_in_off='RX input pin offset.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_p=-1,
            ridx_n=0,
            num_sup_clear=0,
            rx_in_pitch=0.5,
            rx_in_off=0,
        )

    def draw_layout(self) -> None:
        params = self.params
        ridx_n: int = params['ridx_n']
        num_sup_clear: int = params['num_sup_clear']
        rx_in_pitch: HalfInt = HalfInt.convert(params['rx_in_pitch'])
        rx_in_off: HalfInt = HalfInt.convert(params['rx_in_off'])

        rx_master, tx_master = self._make_masters()

        grid = self.grid
        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        ym_layer = vm_layer + 2

        tx = self.add_tile(tx_master, 0, 0)
        rx = self.add_tile(rx_master, 5, 0)
        self.set_mos_size(num_cols=tx_master.num_cols)

        vss_hm_top_tidx = self.get_track_index(ridx_n, MOSWireType.DS, 'sup', tile_idx=-1)
        vss_hm_top_coord = grid.track_to_coord(hm_layer, vss_hm_top_tidx)

        # connect supplies on M3
        for name in ['VDDIO_vm', 'VSS_vm']:
            warr_list = tx.get_all_port_pins(name)
            warr_list.extend(rx.port_pins_iter(name))
            warrs = self.connect_wires(warr_list)
            self.add_pin(name, warrs, hide=True)
        self.reexport(rx.get_port('VDDCore_vm'), net_name='VDDCore_top_vm')
        self.reexport(tx.get_port('VDDCore_vm'), net_name='VDDCore_bot_vm')

        # reexport pins
        for name in ['din', 'ipdrv_buf<1>', 'ipdrv_buf<0>',
                     'indrv_buf<1>', 'indrv_buf<0>', 'itx_en_buf', 'weak_pulldownen',
                     'weak_pullupenb']:
            self.reexport(tx.get_port(name))

        for name in ['clk_en', 'data_en', 'por', 'odat', 'odat_async',
                     'oclkp', 'oclkn']:
            self.reexport(rx.get_port(name))

        # POR routing
        self.connect_differential_wires(rx.get_pin('por_vccl'), rx.get_pin('porb_vccl'),
                                        tx.get_pin('por_vccl'), tx.get_pin('porb_vccl'))

        sup_w = tr_manager.get_width(ym_layer, 'sup')
        hs_w = tr_manager.get_width(ym_layer, 'sig_hs')

        # iopad/iclkn routing
        iopad = rx.get_pin('iopad')
        iclkn = rx.get_pin('iclkn')
        rx_idx_list = tr_manager.place_wires(ym_layer, ['sig_hs', 'sig_hs'],
                                             center_coord=iopad.middle)[1]
        rx_in_p2 = rx_in_pitch.dbl_value
        q_iclkn = ((rx_idx_list[0].dbl_value - rx_in_off.dbl_value) // rx_in_p2)
        tidx_iclkn = HalfInt(q_iclkn * rx_in_p2 + rx_in_off.dbl_value)
        tidx_iopad = rx_idx_list[1] + (rx_idx_list[0] - tidx_iclkn)
        iopad, iclkn = self.connect_differential_tracks(iopad, iclkn, ym_layer, tidx_iopad,
                                                        tidx_iclkn, width=hs_w,
                                                        track_upper=vss_hm_top_coord)
        self.add_pin('rxpadin', iopad)
        self.add_pin('iclkn', iclkn)

        # tx pad routing
        vdd_io_xm = rx.get_all_port_pins('VDDIO')
        vdd_io_xm.extend(tx.port_pins_iter('VDDIO'))
        vdd_xm = [tx.get_pin('VDDCore'), rx.get_pin('VDDCore')]
        vss_xm = rx.get_all_port_pins('VSS')
        vss_xm.extend(tx.port_pins_iter('VSS'))
        self.add_pin('VDDIO_xm', vdd_io_xm, hide=True)
        self.add_pin('VDDCore_xm', vdd_xm, hide=True)
        self.add_pin('VSS_xm', vss_xm, hide=True)

        vdd_idx_list = []
        vdd_core_idx_list = []
        vss_idx_list = []
        txpad, sup_l, sup_r = self._route_tx_pad(tx, vss_xm, vdd_io_xm, vdd_idx_list, vss_idx_list)
        vss_idx_list.append(sup_r)
        self.add_pin('txpadout', txpad)

        # supply routing
        sig_sup_sep = tr_manager.get_sep(ym_layer, ('sig_hs', 'sup'))
        sup_sep = tr_manager.get_sep(ym_layer, ('sup', 'sup'))
        sub_sep2 = sup_sep.div2(round_up=True)
        top_idx_inc = grid.coord_to_track(ym_layer, self.bound_box.xh) - sub_sep2
        num_wires = _append_supply_indices(vdd_idx_list, vss_idx_list, sup_r, sup_sep, top_idx_inc,
                                           True, True)
        bot_idx_inc = tidx_iopad + sig_sup_sep
        vss_idx_list.append(sup_l)
        _append_supply_indices(vdd_idx_list, vss_idx_list, sup_l, sup_sep, bot_idx_inc,
                               False, True, num_max=num_wires)
        top_idx_inc = min(vss_idx_list[-1], vdd_idx_list[-1]) - sup_sep
        _append_supply_indices(vdd_core_idx_list, vss_idx_list, bot_idx_inc, sup_sep,
                               top_idx_inc, True, False)
        bot_idx_inc = grid.coord_to_track(ym_layer, 0) + sig_sup_sep + sup_sep * num_sup_clear
        next_io_idx = tr_manager.get_next_track(ym_layer, tidx_iclkn, 'sig_hs', 'sup', up=False)
        _append_supply_indices(vdd_core_idx_list, vss_idx_list, next_io_idx, sup_sep, bot_idx_inc,
                               False, False)

        sup_yl = sup_yh = 0
        for parity, idx in enumerate(vdd_core_idx_list):
            cur_tid = TrackID(ym_layer, idx, width=sup_w)
            if (parity & 1) == 0:
                warr = self.connect_to_tracks(vdd_xm, cur_tid)
                sup_yl = warr.lower
                sup_yh = warr.upper
                self.add_pin('VDDCore', warr)
            else:
                self._export_vdd_io(vdd_io_xm, cur_tid, sup_yl, sup_yh)

        for idx in vdd_idx_list:
            cur_tid = TrackID(ym_layer, idx, width=sup_w)
            self._export_vdd_io(vdd_io_xm, cur_tid, sup_yl, sup_yh)

        for idx in vss_idx_list:
            cur_tid = TrackID(ym_layer, idx, width=sup_w)
            warr = self.connect_to_tracks(vss_xm, cur_tid, track_lower=sup_yl, track_upper=sup_yh)
            self.add_pin('VSS', warr)

        self.sch_params = dict(
            tx_params=tx_master.sch_params,
            rx_params=rx_master.sch_params,
        )

    def _export_vdd_io(self, vdd_io_xm: List[WireArray], tid: TrackID,
                       sup_yl: int, sup_yh: int) -> None:
        warr = self.connect_to_tracks(vdd_io_xm, tid, track_lower=sup_yl, track_upper=sup_yh)
        self.add_pin('VDDIO', warr)

    def _route_tx_pad(self, tx: PyLayInstance, vss_warrs: List[WireArray],
                      vdd_warrs: List[WireArray], vdd_idx_list: List[HalfInt],
                      vss_idx_list: List[HalfInt]) -> Tuple[WireArray, HalfInt, HalfInt]:
        grid = self.grid
        tr_manager = self.tr_manager
        vm_layer = self.conn_layer + 2
        ym_layer = vm_layer + 2

        # Note: we designed TX driver so it is pitch align with ym layer
        tx_vm = tx.get_pin('txpadout_vm')
        tid = tx_vm.track_id
        pad_coord0 = grid.track_to_coord(vm_layer, tid.base_index)
        pad_tidx0 = grid.coord_to_track(ym_layer, pad_coord0)
        if tid.num == 1:
            pad_tidx_list = [pad_tidx0]
            ym_pitch = 0
        else:
            pad_coord1 = grid.track_to_coord(vm_layer, tid.base_index + tid.pitch)
            pad_tidx1 = grid.coord_to_track(ym_layer, pad_coord1)
            ym_pitch = pad_tidx1 - pad_tidx0
            pad_tidx_list = [pad_tidx0 + idx_ * ym_pitch for idx_ in range(tid.num)]

            num_sup = tr_manager.get_num_wires_between(ym_layer, 'padout', pad_tidx_list[0],
                                                       'padout', pad_tidx_list[1], 'sup')
            center_coord = (pad_coord0 + pad_coord1) // 2
            sup_idx_list = tr_manager.place_wires(ym_layer, ['sup'] * num_sup,
                                                  center_coord=center_coord)[1]
            delta = pad_tidx_list[1] - pad_tidx_list[0]
            sup_w = tr_manager.get_width(ym_layer, 'sup')
            for bot_tidx in range(tid.num - 1):
                offset = bot_tidx * delta
                vdd_first = (bot_tidx & 1)
                for parity, sup_tidx in enumerate(sup_idx_list):
                    cur_tr_idx = offset + sup_tidx
                    cur_tid = TrackID(ym_layer, cur_tr_idx, width=sup_w)
                    if vdd_first ^ (parity & 1):
                        warr = self.connect_to_tracks(vdd_warrs, cur_tid)
                        vdd_idx_list.append(cur_tr_idx)
                        self.add_pin('VDDIO', warr)
                    else:
                        warr = self.connect_to_tracks(vss_warrs, cur_tid)
                        vss_idx_list.append(cur_tr_idx)
                        self.add_pin('VSS', warr)

        pad_sup_sep = tr_manager.get_sep(ym_layer, ('padout', 'sup'))
        pad_w = tr_manager.get_width(ym_layer, 'padout')
        pad_tid = TrackID(ym_layer, pad_tidx_list[0], width=pad_w, num=tid.num, pitch=ym_pitch)
        txpad = self.connect_to_tracks(tx.get_all_port_pins('txpadout'), pad_tid)
        return txpad, pad_tidx_list[0] - pad_sup_sep, pad_tidx_list[-1] + pad_sup_sep

    def _make_masters(self) -> Tuple[RXAnalog, TXAnalog]:
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, params['pinfo'])
        self.draw_base(pinfo, flip_tile=True)

        por_lv_params: Param = params['por_lv_params']
        ctrl_lv_params: Param = params['ctrl_lv_params']
        buf_data_lv_params: Param = params['buf_data_lv_params']
        buf_ctrl_lv_params: Param = params['buf_ctrl_lv_params']
        buf_por_lv_params: Param = params['buf_por_lv_params']
        tx_lv_params: Param = params['tx_lv_params']
        drv_params: Param = params['drv_params']
        rx_lv_params: Param = params['rx_lv_params']
        se_params: Param = params['se_params']
        match_params: Param = params['match_params']
        inv_params: Param = params['inv_params']
        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']

        rx_params = dict(
            pinfo=pinfo,
            se_params=se_params,
            match_params=match_params,
            inv_params=inv_params,
            data_lv_params=rx_lv_params,
            ctrl_lv_params=ctrl_lv_params,
            por_lv_params=por_lv_params,
            buf_ctrl_lv_params=buf_ctrl_lv_params,
            buf_por_lv_params=buf_por_lv_params,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
        )

        rxhalf_ncol = RXAnalog.get_rx_half_ncol(self, pinfo, rx_params)

        tx_params = dict(
            pinfo=pinfo,
            drv_params=drv_params,
            data_lv_params=tx_lv_params,
            ctrl_lv_params=ctrl_lv_params,
            buf_data_lv_params=buf_data_lv_params,
            buf_ctrl_lv_params=buf_ctrl_lv_params,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            rxhalf_ncol=rxhalf_ncol,
        )

        tx_master = self.new_template(TXAnalog, params=tx_params)
        rx_params['core_ncol'] = tx_master.core_ncol
        rx_params['tap_info_list'] = tx_master.drv_tap_info
        rx_master = self.new_template(RXAnalog, params=rx_params)
        return rx_master, tx_master


def _append_supply_indices(vdd_list: List[HalfInt], vss_list: List[HalfInt], cur_idx: HalfInt,
                           sep: HalfInt, stop_idx_inc: HalfInt, up: bool, vdd_first: bool,
                           num_max: Optional[int] = None) -> int:
    if vdd_first:
        sel_list = (vdd_list, vss_list)
    else:
        sel_list = (vss_list, vdd_list)

    num_wires = 0
    sign = int(up) * 2 - 1
    while num_max is None or num_wires < num_max:
        new_idx = cur_idx + sign * sep
        if (up and new_idx <= stop_idx_inc) or (not up and new_idx >= stop_idx_inc):
            sel_list[num_wires & 1].append(new_idx)
            cur_idx = new_idx
            num_wires += 1
        else:
            return num_wires

    return num_wires

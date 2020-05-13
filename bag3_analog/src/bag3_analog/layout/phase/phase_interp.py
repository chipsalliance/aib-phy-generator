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

from typing import Any, Dict, Union, List, Optional, Type

from bag.util.immutable import Param
from bag.design.module import Module
from bag.layout.routing.base import TrackID, WireArray
from bag.layout.template import TemplateDB
from pybag.enum import RoundMode

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.enum import MOSWireType

from bag3_digital.layout.stdcells.gates import InvTristateCore, InvCore

from ...schematic.phase_interp import bag3_analog__phase_interp


class PhaseInterpUnit(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._col_margin = 0

    @property
    def col_margin(self) -> int:
        return self._col_margin

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            w_p='pmos width.',
            w_n='nmos width.',
            stack_p='pmos stack',
            stack_n='nmos stack',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(w_p=0, w_n=0, stack_p=1, stack_n=1)

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        # create masters
        tri_params = self.params.copy(append=dict(pinfo=pinfo, ridx_p=-1, ridx_n=0,
                                                  vertical_out=True, vertical_sup=True))
        tri_master = self.new_template(InvTristateCore, params=tri_params)
        # floorplanning
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        grid = self.grid
        arr_info = self.arr_info
        tr_manager = self.tr_manager
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        vm_w_sup = tr_manager.get_width(vm_layer, 'sup')
        vm_l = grid.get_next_length(vm_layer, vm_w, 0, even=True)
        vm_l_sup = grid.get_next_length(vm_layer, vm_w_sup, 0, even=True)

        tri_ncol = tri_master.num_cols
        loc_list = tr_manager.place_wires(vm_layer, ['sup', 'sig', 'sig_hs', 'sig', 'sup'])[1]
        ntr = loc_list[4] - loc_list[0] + tr_manager.get_sep(vm_layer, ('sup', 'sup'))
        ncol_tot = max(tri_ncol, arr_info.get_column_span(vm_layer, ntr))
        blk_ncol = arr_info.get_block_ncol(vm_layer, half_blk=True)
        ncol_tot = -(-ncol_tot // blk_ncol) * blk_ncol
        col_diff = ncol_tot - tri_ncol
        if col_diff & 1:
            if blk_ncol & 1 == 0:
                raise ValueError('Cannot center tristate inverter with number of tracks.')
            ncol_tot += blk_ncol
            col_diff += blk_ncol

        self._col_margin = col_diff // 2
        core = self.add_tile(tri_master, 0, self._col_margin)
        self.set_mos_size(num_cols=ncol_tot)

        # routing
        out = core.get_pin('out')
        tr_off = out.track_id.base_index - loc_list[2]
        sig_l = out.middle - vm_l // 2
        sup_l = out.middle - vm_l_sup // 2
        sup_u = sup_l + vm_l_sup
        enl = self.add_wires(vm_layer, loc_list[1] + tr_off, sig_l, sig_l + vm_l, width=vm_w)
        enr = self.add_wires(vm_layer, loc_list[3] + tr_off, enl.lower, enl.upper, width=vm_w)
        vss = self.add_wires(vm_layer, loc_list[0] + tr_off, sup_l, sup_u, width=vm_w_sup)
        vdd = self.add_wires(vm_layer, loc_list[4] + tr_off, sup_l, sup_u, width=vm_w_sup)
        self.add_pin('out', out)
        self.add_pin('enl', enl)
        self.add_pin('enr', enr)
        self.add_pin('VSS_vm', vss)
        self.add_pin('VDD_vm', vdd)

        for name in ['nin', 'en', 'enb', 'VDD', 'VSS', 'pout', 'nout']:
            self.reexport(core.get_port(name))

        self.sch_params = tri_master.sch_params


class PhaseInterpolator(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._col_margin = 0

    @property
    def col_margin(self) -> int:
        return self._col_margin

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_analog__phase_interp

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The layout information object.',
            unit_params='tristate inverter unit cell parameters.',
            inv_params='output inverter parameters.',
            nbits='number of control bits.',
            flip_b_en='True to flip vm_layer enable wires for b inputs.',
            draw_sub='Draw substrate connection row',
            export_outb='True to export input of output buffer',
            abut_tristates='True to abut the tristates next to each other'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(flip_b_en=False, draw_sub=False, export_outb=False)

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, mirror=False)

        unit_params: Param = self.params['unit_params']
        inv_params: Param = self.params['inv_params']
        nbits: int = self.params['nbits']
        flip_b_en: bool = self.params['flip_b_en']
        draw_sub: bool = self.params['draw_sub']
        export_outb: bool = self.params['export_outb']
        abut_tristates: bool = self.params['abut_tristates']

        # create masters
        logic_pinfo = self.get_tile_pinfo(1)
        unit_params = unit_params.copy(append=dict(pinfo=logic_pinfo, vertical_sup=True,
                                                   vertical_out=False))
        unit_cls = InvTristateCore if abut_tristates else PhaseInterpUnit
        unit_master: Union[PhaseInterpUnit, InvTristateCore] = self.new_template(unit_cls,
                                                                                 params=unit_params)

        seg = inv_params['seg']
        if seg == 0:
            raise ValueError('Only seg is supported in inv_params.')
        if seg & 1:
            raise ValueError('Output inverter must have even number of segments.')
        if abut_tristates and unit_params['seg'] & 1:
            raise ValueError('Tristates must have even segments when abutting tristates')
        seg_half = seg // 2
        sig_locs = {'in': unit_master.get_track_index(1, MOSWireType.G, 'sig', wire_idx=0)}
        inv_params = inv_params.copy(append=dict(pinfo=logic_pinfo, seg=seg_half,
                                                 vertical_sup=True, sig_locs=sig_locs),
                                     remove=['seg_p', 'seg_n'])
        inv_master = self.new_template(InvCore, params=inv_params)
        if abut_tristates:
            self._col_margin = self.get_hm_sp_le_sep_col()
        else:
            self._col_margin = unit_master.col_margin

        a_dict = self._draw_row(unit_master, inv_master, 1, nbits, False, draw_sub, 'a', 0, 2,
                                abut_tristates)
        b_dict = self._draw_row(unit_master, inv_master, 3, nbits, flip_b_en, draw_sub, 'b', 4, 2,
                                abut_tristates)
        self.set_mos_size(num_tiles=5)

        outb = self.connect_wires(a_dict['mid'] + b_dict['mid'])
        self.add_pin('outb', outb, hide=not export_outb)
        out = self.connect_wires([a_dict['out'], b_dict['out']])
        self.add_pin('out', out)
        for name in ['VDD', 'VSS', 'VDD_hm']:
            self.add_pin(name, self.connect_wires(a_dict[name] + b_dict[name]))
        self.add_pin('VSS0', a_dict['VSS_hm'], hide=True)
        self.add_pin('VSS1', b_dict['VSS_hm'], hide=True)

        if draw_sub:
            vss_hm = [a_dict['VSS_hm'], b_dict['VSS_hm']]
            if isinstance(a_dict['VDD_hm'], List):
                vdd_hm = self.connect_wires(a_dict['VDD_hm']+b_dict['VDD_hm'])
            else:
                vdd_hm = self.connect_wires([a_dict['VDD_hm'], b_dict['VDD_hm']])
            ncol_tot = self.num_cols
            sub = self.add_substrate_contact(0, 0, tile_idx=0, seg=ncol_tot)
            self.connect_to_track_wires(sub, vss_hm[0])
            sub = self.add_substrate_contact(0, 0, tile_idx=2, seg=ncol_tot)
            self.connect_to_track_wires(sub, vdd_hm)
            sub = self.add_substrate_contact(0, 0, tile_idx=4, seg=ncol_tot)
            self.connect_to_track_wires(sub, vss_hm[1])

        self.sch_params = dict(
            tri_params=unit_master.sch_params,
            inv_params=inv_master.sch_params.copy(append=dict(seg_p=seg, seg_n=seg)),
            nbits=nbits,
            export_outb=export_outb,
        )

    def _draw_row(self, unit_master: PhaseInterpUnit, inv_master: InvCore, tile_idx: int,
                  nbits: int, flip_en: bool, draw_sub: bool, prefix: str, vss_tile: int,
                  vdd_tile: int, abut_tristates: bool) -> Dict[str, Union[List[WireArray], WireArray]]:
        vm_layer = self.conn_layer + 2
        vm_sup_w = self.tr_manager.get_width(vm_layer, 'sup')
        vm_sup_l = self.grid.get_next_length(vm_layer, vm_sup_w, 0, even=True)

        if abut_tristates:
            unit_sep = 0
            col_margin = 0
        else:
            col_margin = unit_master.col_margin
            min_sep = self.min_sep_col
            min_sep += (min_sep & 1)
            unit_sep = max(min_sep - 2 * col_margin, 0)
            unit_sep += (unit_sep & 1)

        # gather wires, connect/export enable signals
        pin_list = ['pout', 'nout', 'nin', 'VDD', 'VSS', 'VDD_vm', 'VSS_vm']
        pin_dict = {name: [] for name in pin_list}
        cur_col = 1 if draw_sub else 0
        unit_ncol = unit_master.num_cols
        for idx in range(nbits):
            inst = self.add_tile(unit_master, tile_idx, cur_col)
            if abut_tristates:
                en = inst.get_pin('en')
                enb = inst.get_pin('enb')
                en_tr_idx = self.grid.coord_to_track(vm_layer, en.middle, RoundMode.LESS)
                enb_tr_idx = self.grid.coord_to_track(vm_layer, en.middle, RoundMode.GREATER)
                sup_tr_idx = self.tr_manager.get_next_track(vm_layer, en_tr_idx, 'sig', 'sup',
                                                            up=False)
                if flip_en:
                    en, enb = self.connect_differential_tracks(en, enb, vm_layer, en_tr_idx,
                                                               enb_tr_idx)
                else:
                    en, enb = self.connect_differential_tracks(en, enb, vm_layer, enb_tr_idx,
                                                               en_tr_idx)
                sup_w = self.add_wires(vm_layer, sup_tr_idx, en.middle - vm_sup_l // 2,
                                       en.middle + vm_sup_l // 2, width=vm_sup_w)
                if idx & 1:
                    pin_dict['VDD_vm'].append(sup_w)
                else:
                    pin_dict['VSS_vm'].append(sup_w)
                for pin in pin_list:
                    if pin not in ['VDD_vm', 'VSS_vm']:
                        pin_dict[pin].append(inst.get_pin(pin))
            else:
                for name in pin_list:
                    pin_dict[name].append(inst.get_pin(name))
                enl = inst.get_pin('enl')
                enr = inst.get_pin('enr')
                en = inst.get_pin('en')
                enb = inst.get_pin('enb')
                if flip_en:
                    en, enb = self.connect_differential_wires(en, enb, enr, enl)
                else:
                    en, enb = self.connect_differential_wires(en, enb, enl, enr)

            # NOTE: LSB closest to the output to help with nonlinearity
            bit_idx = nbits - 1 - idx
            self.add_pin(f'{prefix}_en<{bit_idx}>', en)
            self.add_pin(f'{prefix}_enb<{bit_idx}>', enb)
            cur_col += unit_ncol + unit_sep

        # short hm_layer wires
        pout = self.connect_wires(pin_dict['pout'])[0]
        nout = self.connect_wires(pin_dict['nout'])[0]
        in_warr = self.connect_wires(pin_dict['nin'])[0]
        self.add_pin(prefix + '_in', in_warr)

        # connect output inverter
        cur_col += self.min_sep_col
        vm_w = self.tr_manager.get_width(vm_layer, 'sig_hs')
        inst = self.add_tile(inv_master, tile_idx, cur_col)
        buf_out = inst.get_pin('out')
        vm_tidx = self.tr_manager.get_next_track(vm_layer, buf_out.track_id.base_index,
                                                 'sig', 'sig_hs', up=False)
        inv_in = self.connect_to_tracks([pout, nout, inst.get_pin('in')],
                                        TrackID(vm_layer, vm_tidx, width=vm_w))
        pin_dict['out'] = [inv_in]
        pin_dict['VDD'].extend(inst.port_pins_iter('VDD'))
        pin_dict['VSS'].extend(inst.port_pins_iter('VSS'))

        # connect supplies
        vss_tid = self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=vss_tile)
        vdd_tid = self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=vdd_tile)
        vss = self.connect_to_tracks(pin_dict['VSS'], vss_tid)
        vdd = self.connect_to_tracks(pin_dict['VDD'], vdd_tid)
        vss_vm_list = []
        vdd_vm_list = []
        for vss_vm, vdd_vm in zip(pin_dict['VSS_vm'], pin_dict['VDD_vm']):
            vss_vm, vdd_vm = self.connect_differential_wires(vss, vdd, vss_vm, vdd_vm)
            vss_vm_list.append(vss_vm)
            vdd_vm_list.append(vdd_vm)
        return {'mid': pin_dict['out'], 'out': buf_out, 'VSS': vss_vm_list, 'VDD': vdd_vm_list,
                'VSS_hm': vss, 'VDD_hm': [vdd]}

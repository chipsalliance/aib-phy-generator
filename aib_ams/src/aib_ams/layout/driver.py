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

from typing import Any, Dict, Mapping, Union, List, Optional, Type, Iterable, Tuple

from pybag.enum import MinLenMode, RoundMode

from bag.util.math import HalfInt
from bag.util.immutable import Param, ImmutableList
from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.layout.template import TemplateDB, PyLayInstance
from bag.layout.routing.base import TrackID, WireArray

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase, SupplyColumnInfo

from bag3_digital.layout.stdcells.gates import NAND2Core, NOR2Core


class PullUpDown(MOSBase):
    """pull up/pull down output driver
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_driver_pu_pd')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_p='the number of fingers of pull up',
            seg_n='the number of fingers of pull down',
            w_p='pmos width.',
            w_n='nmos width.',
            stack='transistor stack number.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            stack=1,
            ridx_p=-1,
            ridx_n=0,
            sig_locs={},
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack: int = self.params['stack']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        #if (seg_p - seg_n) % 2 != 0:
        #    raise ValueError('seg_p - seg_n should be even')
        if w_p == 0:
            w_p = self.place_info.get_row_place_info(ridx_p).row_info.width
        if w_n == 0:
            w_n = self.place_info.get_row_place_info(ridx_n).row_info.width

        pin_tidx = sig_locs.get('pin',
                                self.get_track_index(ridx_p, MOSWireType.G,
                                                     wire_name='sig', wire_idx=-1))
        nin_tidx = sig_locs.get('nin',
                                self.get_track_index(ridx_n, MOSWireType.G,
                                                     wire_name='sig', wire_idx=0))
        nout_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='padout')
        pout_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='padout')

        fg_p = seg_p * stack
        fg_n = seg_n * stack
        num_col = max(fg_p, fg_n)
        p_col = (num_col - fg_p) // 2
        n_col = (num_col - fg_n) // 2
        pmos = self.add_mos(ridx_p, p_col, seg_p, w=w_p, stack=stack)
        nmos = self.add_mos(ridx_n, n_col, seg_n, w=w_n, stack=stack)
        self.set_mos_size(num_cols=num_col)

        # connect supplies
        vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
        vss_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
        vdd = self.connect_to_tracks(pmos.s, vdd_tid)
        vss = self.connect_to_tracks(nmos.s, vss_tid)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        # connect inputs
        hm_w = self.tr_manager.get_width(hm_layer, 'sig')
        pden = self.connect_to_tracks(nmos.g, TrackID(hm_layer, nin_tidx, width=hm_w),
                                      min_len_mode=MinLenMode.MIDDLE)
        puenb = self.connect_to_tracks(pmos.g, TrackID(hm_layer, pin_tidx, width=hm_w),
                                       min_len_mode=MinLenMode.MIDDLE)
        self.add_pin('pden', pden)
        self.add_pin('puenb', puenb)

        # connect output
        vm_w_pad = self.tr_manager.get_width(vm_layer, 'padout')
        nout = self.connect_to_tracks(nmos.d, nout_tid, min_len_mode=MinLenMode.MIDDLE)
        pout = self.connect_to_tracks(pmos.d, pout_tid, min_len_mode=MinLenMode.MIDDLE)
        vm_tidx = sig_locs.get('out', self.grid.coord_to_track(vm_layer, nout.middle,
                                                               mode=RoundMode.NEAREST))
        out = self.connect_to_tracks([nout, pout], TrackID(vm_layer, vm_tidx, width=vm_w_pad))
        self.add_pin('out', out)
        self.add_pin('nout', nout, hide=True)
        self.add_pin('pout', pout, hide=True)

        # compute schematic parameters.
        th_p = self.place_info.get_row_place_info(ridx_p).row_info.threshold
        th_n = self.place_info.get_row_place_info(ridx_n).row_info.threshold
        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=self.place_info.lch,
            w_p=w_p,
            w_n=w_n,
            th_p=th_p,
            th_n=th_n,
            stack_p=stack,
            stack_n=stack,
        )


class OutputDriverCore(MOSBase):
    """tristate buffer with nand/nor gates.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_driver_output_unit_cell')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_p='number of segments for output pmos.',
            seg_n='number of segments for output nmos.',
            seg_nand='number of segments for nand.',
            seg_nor='number of segments for nor.',
            w_p='output pmos width.',
            w_n='output nmos width.',
            w_p_nand='NAND pmos width.',
            w_n_nand='NAND nmos width.',
            w_p_nor='NOR pmos width.',
            w_n_nor='NOR nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            export_pins='Defaults to False.  True to export simulation pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p_nand=0,
            w_n_nand=0,
            w_p_nor=0,
            w_n_nor=0,
            ridx_p=-1,
            ridx_n=0,
            export_pins=False,
        )

    def draw_layout(self):
        grid = self.grid
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(grid, params['pinfo'])
        self.draw_base(pinfo)

        tr_manager = self.tr_manager

        seg_p: int = params['seg_p']
        seg_n: int = params['seg_n']
        seg_nand: int = params['seg_nand']
        seg_nor: int = params['seg_nor']
        w_p: int = params['w_p']
        w_n: int = params['w_n']
        w_p_nand: int = params['w_p_nand']
        w_n_nand: int = params['w_n_nand']
        w_p_nor: int = params['w_p_nor']
        w_n_nor: int = params['w_n_nor']
        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']
        export_pins: bool = params['export_pins']

        vm_layer = self.conn_layer + 2

        #if (seg_p - seg_n) % 2 != 0:
        #    raise ValueError('seg_p - seg_n should be even')

        in0_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        nout_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pout_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=-1)
        pin_drv_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        nin_drv_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        sig_locs_gate = dict(nin=in0_tidx, nout=nout_tidx, pout=pout_tidx)
        sig_locs_drv = dict(pin=pin_drv_tidx, nin=nin_drv_tidx)
        nand_params = dict(
            pinfo=pinfo,
            seg=seg_nand,
            vertical_out=False,
            vertical_in=False,
            is_guarded=True,
            w_p=w_p_nand,
            w_n=w_n_nand,
            ridx_n=ridx_n,
            ridx_p=ridx_p,
            sig_locs=sig_locs_gate,
            min_len_mode=dict(
                in0=MinLenMode.LOWER,
                in1=MinLenMode.LOWER,
            ),
        )
        nor_params = nand_params.copy()
        nor_params['seg'] = seg_nor
        nor_params['w_p'] = w_p_nor
        nor_params['w_n'] = w_n_nor
        drv_params = dict(
            pinfo=pinfo,
            seg_p=seg_p,
            seg_n=seg_n,
            w_p=w_p,
            w_n=w_n,
            stack=1,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            sig_locs=sig_locs_drv,
        )

        # --- Placement --- #
        nand_master = self.new_template(NAND2Core, params=nand_params)
        nor_master = self.new_template(NOR2Core, params=nor_params)
        drv_master = self.new_template(PullUpDown, params=drv_params)

        nand_ncols = nand_master.num_cols
        nor_ncols = nor_master.num_cols
        drv_ncols = drv_master.num_cols
        min_sep = self.min_sep_col
        ncol_tot = nand_ncols + drv_ncols + nor_ncols + 2 * min_sep

        nand_inst = self.add_tile(nand_master, 0, nand_ncols, flip_lr=True)
        drv_inst = self.add_tile(drv_master, 0, nand_ncols + min_sep)
        nor_inst = self.add_tile(nor_master, 0, ncol_tot - nor_ncols)
        self.set_mos_size(num_cols=ncol_tot)

        # Routing
        pin = self.connect_wires([nor_inst.get_pin('pin<0>'), nand_inst.get_pin('pin<0>')])
        nin = self.connect_wires([nor_inst.get_pin('nin<0>'), nand_inst.get_pin('nin<0>')])
        self.add_pin('pin', pin, label='in:')
        self.add_pin('nin', nin, label='in:')

        bnd_box = self.bound_box
        vm_tr_w = tr_manager.get_width(vm_layer, 'sig')
        en_vm_tidx = grid.coord_to_track(vm_layer, bnd_box.xl, mode=RoundMode.GREATER_EQ)
        enb_vm_tidx = grid.coord_to_track(vm_layer, bnd_box.xh, mode=RoundMode.LESS_EQ)
        puenb_vm_tidx = tr_manager.get_next_track(vm_layer, en_vm_tidx, 'sig', 'sig', up=True)
        pden_vm_tidx = tr_manager.get_next_track(vm_layer, enb_vm_tidx, 'sig', 'sig', up=False)
        puenb = drv_inst.get_pin('puenb')
        pden = drv_inst.get_pin('pden')
        if export_pins:
            self.add_pin('nand_pu', puenb)
            self.add_pin('nor_pd', pden)
        self.connect_to_tracks([nand_inst.get_pin('pout'), nand_inst.get_pin('nout'), puenb],
                               TrackID(vm_layer, puenb_vm_tidx, width=vm_tr_w))
        self.connect_to_tracks([nor_inst.get_pin('pout'), nor_inst.get_pin('nout'), pden],
                               TrackID(vm_layer, pden_vm_tidx, width=vm_tr_w))
        en = self.connect_to_tracks([nand_inst.get_pin('pin<1>'), nand_inst.get_pin('nin<1>')],
                                    TrackID(vm_layer, en_vm_tidx, width=vm_tr_w))
        enb = self.connect_to_tracks([nor_inst.get_pin('pin<1>'), nor_inst.get_pin('nin<1>')],
                                     TrackID(vm_layer, enb_vm_tidx, width=vm_tr_w))
        self.add_pin('en', en)
        self.add_pin('enb', enb)
        self.reexport(drv_inst.get_port('out'))
        self.reexport(drv_inst.get_port('pout'))
        self.reexport(drv_inst.get_port('nout'))

        # connect vss and vdd
        vdd = self.connect_wires([nand_inst.get_pin('VDD'), nor_inst.get_pin('VDD'),
                                  drv_inst.get_pin('VDD')])
        vss = self.connect_wires([nand_inst.get_pin('VSS'), nor_inst.get_pin('VSS'),
                                  drv_inst.get_pin('VSS')])
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        # get schematic parameters
        self.sch_params = dict(
            nand_params=nand_master.sch_params,
            nor_params=nor_master.sch_params,
            pupd_params=drv_master.sch_params,
            export_pins=export_pins,
        )


class AIBOutputDriver(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._tap_columns: ImmutableList[Tuple[int, bool]] = ImmutableList()

    @property
    def tap_columns(self) -> ImmutableList[Tuple[int, bool]]:
        return self._tap_columns

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_driver_output_driver')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            pupd_params='weak pull up pull down parameters',
            unit_params='unit cell parameters',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_p=-1,
            ridx_n=0,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        arr_info = self.arr_info
        tr_manager = self.tr_manager

        pupd_params: Param = self.params['pupd_params']
        unit_params: Param = self.params['unit_params']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']

        conn_layer = self.conn_layer
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        # create masters
        unit_params = unit_params.copy(append=dict(pinfo=pinfo, ridx_p=ridx_p, ridx_n=ridx_n))
        unit_master = self.new_template(OutputDriverCore, params=unit_params)
        vm_tidx = unit_master.get_port('out').get_pins()[0].track_id.base_index
        pupd_params = pupd_params.copy(append=dict(pinfo=pinfo, ridx_p=ridx_p, ridx_n=ridx_n,
                                                   sig_locs=dict(out=vm_tidx)))
        pupd_master = self.new_template(PullUpDown, params=pupd_params)
        # placement
        # initialize supply data structures
        lay_range = range(conn_layer, xm_layer + 1)
        vdd_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        vss_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        sup_info = self.get_supply_column_info(xm_layer)
        sub_sep = self.sub_sep_col
        sub_sep2 = sub_sep // 2

        # instantiate cells and collect pins
        pin_dict = {'din': [], 'out': [], 'out_hm': [], 'en': [], 'enb': []}
        cur_col = 0
        din_type = 'sig_hs'
        tap_list = []
        sup_unit_sep = sub_sep2

        # first column
        tap_list.append((cur_col, False))
        cur_col = self._draw_supply_column(cur_col, sup_info, vdd_table, vss_table,
                                           ridx_p, ridx_n, False)
        # make sure we have room for din wire
        sup_tidx = max(vdd_table[vm_layer][-1].track_id.base_index,
                       vss_table[vm_layer][-1].track_id.base_index)
        din_tidx = tr_manager.get_next_track(vm_layer, sup_tidx, 'sup', din_type, up=True)
        en_tidx = tr_manager.get_next_track(vm_layer, din_tidx, din_type, 'sig', up=True)
        unit_col = arr_info.coord_to_col(self.grid.track_to_coord(vm_layer, en_tidx),
                                         round_mode=RoundMode.GREATER_EQ)
        sup_unit_sep = max(sup_unit_sep, unit_col - cur_col)

        # make sure unit element is in pitch on ym_layer
        ncol_core = unit_master.num_cols + sup_info.ncol
        ncol_unit = 2 * sup_unit_sep + ncol_core
        if self.top_layer >= ym_layer:
            ncol_unit = arr_info.round_up_to_block_size(ym_layer, ncol_unit, even_diff=True,
                                                        half_blk=False)
        sup_unit_sep = (ncol_unit - ncol_core) // 2

        cur_col += sup_unit_sep
        cur_col = self._draw_unit_cells(cur_col, unit_master, pin_dict, range(4))
        cur_col += sup_unit_sep

        # second column
        tap_list.append((cur_col, False))
        cur_col = self._draw_supply_column(cur_col, sup_info, vdd_table, vss_table,
                                           ridx_p, ridx_n, False)
        cur_col += sup_unit_sep
        new_col = self._draw_unit_cells(cur_col, unit_master, pin_dict, range(1, 3))

        # pull up/pull down
        pupd_inst = self.add_tile(pupd_master, 0, cur_col)
        self._record_pins(pupd_inst, pin_dict, False)
        cur_col = max(new_col + sup_unit_sep, cur_col + pupd_master.num_cols + sub_sep2)

        # last supply column
        tap_list.append((cur_col, True))
        cur_col = self._draw_supply_column(cur_col, sup_info, vdd_table, vss_table,
                                           ridx_p, ridx_n, True)
        self._tap_columns = ImmutableList(tap_list)
        self.set_mos_size(num_cols=cur_col)

        # routing
        # supplies
        vss = vdd = None
        for lay in range(hm_layer, xm_layer + 1, 2):
            vss = vss_table[lay]
            vdd = vdd_table[lay]
            if lay == hm_layer:
                vss.append(pupd_inst.get_pin('VSS'))
                vdd.append(pupd_inst.get_pin('VDD'))
            vdd = self.connect_wires(vdd)
            vss = self.connect_wires(vss)
            self.add_pin(f'VDD_{lay}', vdd, hide=True)
            self.add_pin(f'VSS_{lay}', vss, hide=True)
        vdd = vdd[0]
        vss = vss[0]
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)
        self.add_pin('VDD_vm', self.connect_wires(vdd_table[vm_layer]), hide=True)
        self.add_pin('VSS_vm', vss_table[vm_layer], hide=True)
        self.add_pin('VDD_conn', vdd_table[conn_layer], hide=True)
        self.add_pin('VSS_conn', vss_table[conn_layer], hide=True)

        # signals
        self.connect_wires(pin_dict['out_hm'])
        din = self.connect_wires(pin_dict['din'])
        out_list = pin_dict['out']
        out_upper = max((warr.upper for warr in out_list))
        out = self.connect_wires(out_list, upper=out_upper)[0]
        en_list = pin_dict['en']
        enb_list = pin_dict['enb']
        p_en_1 = self.connect_wires(en_list[4:])[0]
        n_enb_1 = self.connect_wires(enb_list[4:])[0]
        din = self.connect_to_tracks(din, TrackID(vm_layer, din_tidx,
                                                  width=tr_manager.get_width(vm_layer, din_type)))
        pub = self.connect_to_tracks(pupd_inst.get_pin('puenb'), p_en_1.track_id,
                                     min_len_mode=MinLenMode.LOWER)
        pd = self.connect_to_tracks(pupd_inst.get_pin('pden'), n_enb_1.track_id,
                                    min_len_mode=MinLenMode.UPPER)

        self.add_pin('din', din)
        self.add_pin('p_en_drv<0>', en_list[0])
        self.add_pin('n_enb_drv<0>', enb_list[0])
        self.add_pin('tristateb', self.connect_wires(en_list[1:4]))
        self.add_pin('tristate', self.connect_wires(enb_list[1:4]))
        self.add_pin('p_en_drv<1>', p_en_1)
        self.add_pin('n_enb_drv<1>', n_enb_1)
        self.add_pin('weak_puenb', pub)
        self.add_pin('weak_pden', pd)
        self.add_pin('txpadout', out)
        self.sch_params = dict(
            unit_params=unit_master.sch_params,
            pupd_params=pupd_master.sch_params,
        )

    def _draw_supply_column(self, col: int, sup_info: SupplyColumnInfo,
                            vdd_table: Dict[int, List[WireArray]],
                            vss_table: Dict[int, List[WireArray]],
                            ridx_p: int, ridx_n: int, flip_lr: bool) -> int:
        ncol = sup_info.ncol
        anchor = col + int(flip_lr) * ncol
        for tile in range(4):
            self.add_supply_column(sup_info, anchor, vdd_table, vss_table, ridx_p=ridx_p,
                                   ridx_n=ridx_n, tile_idx=tile, flip_lr=flip_lr,
                                   extend_vdd=False)
        return col + ncol

    def _draw_unit_cells(self, col: int, unit_master: OutputDriverCore,
                         pin_dict: Dict[str, List[WireArray]], tile_range: Iterable[int]) -> int:
        for tidx in tile_range:
            inst = self.add_tile(unit_master, tile_idx=tidx, col_idx=col)
            self._record_pins(inst, pin_dict, True)
        return col + unit_master.num_cols

    @staticmethod
    def _record_pins(inst: PyLayInstance, pin_dict: Dict[str, List[WireArray]],
                     is_unit: bool) -> None:
        out_hm_list = pin_dict['out_hm']
        out_hm_list.append(inst.get_pin('pout'))
        out_hm_list.append(inst.get_pin('nout'))
        pin_dict['out'].append(inst.get_pin('out'))

        if is_unit:
            din_list = pin_dict['din']
            pin_dict['en'].append(inst.get_pin('en'))
            pin_dict['enb'].append(inst.get_pin('enb'))
            din_list.append(inst.get_pin('pin'))
            din_list.append(inst.get_pin('nin'))

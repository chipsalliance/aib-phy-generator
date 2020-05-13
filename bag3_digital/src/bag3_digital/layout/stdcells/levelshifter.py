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

from typing import Any, Dict, List, Mapping, Optional, Union, Type, cast

from pybag.enum import RoundMode, MinLenMode

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.template import TemplateDB, PyLayInstance
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from bag3_digital.layout.stdcells.gates import InvChainCore


class LevelShifterCore(MOSBase):
    """Core of level shifter, with differential inputs and no output buffers.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._center_col: int = -1

    @property
    def center_col(self) -> int:
        """int: The centerline column index."""
        return self._center_col

    @property
    def out_vertical(self) -> bool:
        """bool: True if outputs are on vm_layer."""
        return self.params['has_rst'] or self.params['is_guarded']

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'lvshift_core')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='dictionary of number of segments.',
            w_dict='dictionary of number of fins.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            vertical_rst='List of reset pins to draw on vm layer.',
            is_guarded='True if it there should be guard ring around the cell',
            in_upper='True to make the input connected to the upper transistor in the stack',
            inp_on_right='True to connect inp pin on right.',
            has_rst='True to enable reset pins.',
            stack_p='PMOS number of stacks.',
            sig_locs='Optional dictionary of user defined signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_p=-1,
            ridx_n=0,
            vertical_rst=[],
            is_guarded=False,
            in_upper=False,
            inp_on_right=False,
            has_rst=True,
            stack_p=1,
            sig_locs={},
        )

    def draw_layout(self):
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, params['pinfo'])
        self.draw_base(pinfo)

        tr_manager = self.tr_manager

        seg_dict: Mapping[str, int] = params['seg_dict']
        w_dict: Mapping[str, int] = params['w_dict']
        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']
        vertical_rst: List[str] = params['vertical_rst']
        is_guarded: bool = params['is_guarded']
        in_upper: bool = params['in_upper']
        inp_on_right: bool = params['inp_on_right']
        has_rst: bool = params['has_rst']
        stack_p: int = params['stack_p']
        sig_locs: Mapping[str, Union[float, HalfInt]] = params['sig_locs']

        if stack_p != 1 and stack_p != 2:
            raise ValueError('Only support stack_p = 1 or stack_p = 2')
        if stack_p == 2:
            if not has_rst:
                raise ValueError('stack_p = 2 only allowed if has_rst = True')
            if not in_upper:
                raise ValueError('stack_p = 2 only allowed if in_upper = True')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        sig_w_hm = tr_manager.get_width(hm_layer, 'sig')

        seg_pu = seg_dict['pu']
        seg_pd = seg_dict['pd']
        seg_rst = seg_dict.get('rst', 0)
        seg_prst = seg_dict.get('prst', 0)

        default_wp = self.get_row_info(ridx_p).width
        default_wn = self.get_row_info(ridx_n).width
        w_pd = w_dict.get('pd', default_wn)
        w_pu = w_dict.get('pu', default_wp)
        sch_w_dict = dict(pd=w_pd, pu=w_pu)

        vss_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sup')
        vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sup')
        pg_midl_tidx = self.get_track_index(ridx_p, MOSWireType.G, 'sig', wire_idx=-2)
        pg_midr_tidx = self.get_track_index(ridx_p, MOSWireType.G, 'sig', wire_idx=-1)
        nd_mid_tidx = self.get_track_index(ridx_n, MOSWireType.DS, 'sig', wire_idx=-1)
        inn_tidx = self.get_track_index(ridx_n, MOSWireType.G, 'sig', wire_idx=0)
        inp_tidx = self.get_track_index(ridx_n, MOSWireType.G, 'sig', wire_idx=1)
        inn_tidx = sig_locs.get('inb', inn_tidx)
        inp_tidx = sig_locs.get('in', inp_tidx)
        inn_tid = TrackID(hm_layer, inn_tidx, width=sig_w_hm)
        inp_tid = TrackID(hm_layer, inp_tidx, width=sig_w_hm)
        if in_upper:
            rst_tid = inn_tid
        else:
            rst_tid = None
        nd_mid_tid = TrackID(hm_layer, nd_mid_tidx)

        # floorplanning number of columns
        min_sep_odd = self.min_sep_col
        min_sep_even = min_sep_odd + (min_sep_odd & 1)
        mid_sep = min_sep_even
        mid_sep2 = mid_sep // 2
        pmos_fg = seg_pu * stack_p
        pmos_col = pmos_fg + (pmos_fg & 1)
        if has_rst:
            rst_sep = min_sep_odd
            nmos_fg = 2 * seg_pd
            nmos_col = seg_rst + rst_sep + nmos_fg
            if nmos_col & 1:
                rst_sep += 1
                nmos_col += 1

            if stack_p == 2:
                if pmos_fg > nmos_fg:
                    # TODO: remove lazy hack
                    raise ValueError('pmos reset placement code will break in this case')
            num_core_col = max(nmos_col, pmos_col)
        else:
            rst_sep = 0
            nmos_fg = seg_pd
            num_core_col = max(seg_pd + (seg_pd & 1), pmos_col)

        self._center_col = col_mid = num_core_col + mid_sep2
        seg_tot = 2 * col_mid
        self.set_mos_size(num_cols=seg_tot)
        # --- Placement --- #
        # rst
        export_mid = sep_g_pmos = (stack_p != 1)
        load_l = self.add_mos(ridx_p, col_mid - mid_sep2, seg_pu, g_on_s=True, w=w_pu,
                              stack=stack_p, sep_g=sep_g_pmos, export_mid=export_mid, flip_lr=True)
        load_r = self.add_mos(ridx_p, col_mid + mid_sep2, seg_pu, g_on_s=True, w=w_pu,
                              stack=stack_p, sep_g=sep_g_pmos, export_mid=export_mid)
        vdd_list = [load_l.s, load_r.s]
        if has_rst:
            if seg_rst == 0:
                raise ValueError('seg_rst cannot be 0')

            w_rst = w_dict.get('rst', default_wn)
            sch_w_dict['rst'] = w_rst
            rst_delta = mid_sep2 + nmos_fg + rst_sep + seg_rst
            rst_l = self.add_mos(ridx_n, col_mid - rst_delta, seg_rst, w=w_rst)
            rst_r = self.add_mos(ridx_n, col_mid + rst_delta, seg_rst, w=w_rst, flip_lr=True)
            in_l = self.add_mos(ridx_n, col_mid - mid_sep2, seg_pd, g_on_s=True, w=w_pd,
                                stack=2, sep_g=True, flip_lr=True)
            in_r = self.add_mos(ridx_n, col_mid + mid_sep2, seg_pd, g_on_s=True, w=w_pd,
                                stack=2, sep_g=True)
            vss_list = [in_l.s, in_r.s, rst_r.s, rst_l.s]
            nd_l = [rst_l.d, in_l.d]
            nd_r = [rst_r.d, in_r.d]

            if stack_p == 2:
                if seg_prst == 0:
                    raise ValueError('seg_prst cannot be 0')
                prst_sep = min_sep_odd + ((min_sep_odd & 1) ^ ((seg_prst & 1) == 0))
                prst_delta = mid_sep2 + pmos_fg + prst_sep + seg_prst
                prst_l = self.add_mos(ridx_p, col_mid - prst_delta, seg_prst, w=w_pu)
                prst_r = self.add_mos(ridx_p, col_mid + prst_delta, seg_prst, w=w_pu, flip_lr=True)
            else:
                prst_l = prst_r = None
        else:
            prst_l = prst_r = rst_l = rst_r = None
            in_l = self.add_mos(ridx_n, col_mid - mid_sep2, seg_pd, g_on_s=True, w=w_pd,
                                flip_lr=True)
            in_r = self.add_mos(ridx_n, col_mid + mid_sep2, seg_pd, g_on_s=True, w=w_pd)
            vss_list = [in_l.s, in_r.s]
            nd_l = [in_l.d]
            nd_r = [in_r.d]

        # --- Routing --- #
        # vdd/vss
        vdd = self.connect_to_tracks(vdd_list, vdd_tid)
        vss = self.connect_to_tracks(vss_list, vss_tid)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        if has_rst or is_guarded:
            # use vm_layer to connect nmos and pmos drains
            left_coord = self.grid.track_to_coord(self.conn_layer, in_l.s[0].track_id.base_index)
            right_coord = self.grid.track_to_coord(self.conn_layer, in_r.s[0].track_id.base_index)
            dleft_tidx = self.grid.coord_to_track(vm_layer, left_coord, RoundMode.NEAREST)
            dright_tidx = self.grid.coord_to_track(vm_layer, right_coord, RoundMode.NEAREST)
            dleft_tidx = tr_manager.get_next_track(vm_layer, dleft_tidx, 'sig', 'sig', up=False)
            dright_tidx = tr_manager.get_next_track(vm_layer, dright_tidx, 'sig', 'sig', up=True)

            # connect nmos drains together
            nd_midl = self.connect_to_tracks(nd_l, nd_mid_tid)
            nd_midr = self.connect_to_tracks(nd_r, nd_mid_tid)

            # pmos cross coupling connection
            if stack_p == 1:
                pg_midl, pg_midr = self.connect_differential_tracks(load_l.d, load_r.d, hm_layer,
                                                                    pg_midl_tidx, pg_midr_tidx)
                pg_midl, pg_midr = self.connect_differential_wires(load_r.g, load_l.g,
                                                                   pg_midl, pg_midr)
                hm_midl_list = [nd_midl, pg_midl]
                hm_midr_list = [nd_midr, pg_midr]
            else:
                pm_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0)
                pd_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=1)
                pd_midl = self.connect_to_tracks([prst_l.d, load_l.d], pd_tid)
                pd_midr = self.connect_to_tracks([prst_r.d, load_r.d], pd_tid)
                self.connect_to_tracks([prst_l.s, load_l.m], pm_tid)
                self.connect_to_tracks([prst_r.s, load_r.m], pm_tid)

                self.connect_wires([load_l.g[1::2], in_l.g[1::2]])
                self.connect_wires([load_r.g[1::2], in_r.g[1::2]])
                pg_midl, pg_midr = self.connect_differential_tracks(
                    load_r.g[0::2], load_l.g[0::2], hm_layer, pg_midl_tidx, pg_midr_tidx,
                    track_lower=pd_midl.lower, track_upper=pd_midr.upper)

                hm_midl_list = [nd_midl, pd_midl, pg_midl]
                hm_midr_list = [nd_midr, pd_midr, pg_midr]

            vm_w = tr_manager.get_width(vm_layer, 'sig')
            midl_vm_tid = TrackID(vm_layer, dleft_tidx, width=vm_w)
            midr_vm_tid = TrackID(vm_layer, dright_tidx, width=vm_w)
            midl = self.connect_to_tracks(hm_midl_list, midl_vm_tid)
            midr = self.connect_to_tracks(hm_midr_list, midr_vm_tid)
            midl, midr = self.extend_wires([midl, midr], lower=min(midl.lower, midr.lower),
                                           upper=max(midl.upper, midr.upper))

            if has_rst:
                # reset connections
                if in_upper:
                    inl = in_l.g1
                    inr = in_r.g1
                    rst_midl_b = in_l.g0
                    rst_midr_b = in_r.g0
                else:
                    inl = in_l.g0
                    inr = in_r.g0
                    rst_midl_b = in_l.g1
                    rst_midr_b = in_r.g1

                # connect rst gates
                rst_b_tidx = self.get_track_index(ridx_n, MOSWireType.G, 'sig',  wire_idx=2)
                rst_b_tidx = sig_locs.get('rst_casc', rst_b_tidx)
                rst_b_tid = TrackID(hm_layer, rst_b_tidx, width=sig_w_hm)
                if rst_tid is None:
                    rst_tid = rst_b_tid

                rst_b_wires = [rst_midl_b, rst_midr_b]
                if prst_l is not None:
                    # TODO: check for line-end spacing errors, rst_b_tid may not be a good track.
                    rst_b_wires.append(prst_l.g)
                    rst_b_wires.append(prst_r.g)

                rst_b_warr = self.connect_to_tracks(rst_b_wires, rst_b_tid)

                # rst_tid has some value now, convert that to rst_outn_tid / rst_outp_tid based
                # on sig_locs
                rst_outn_tidx = sig_locs.get('rst_outb', None)
                if rst_outn_tidx:
                    rst_outn_tid = TrackID(hm_layer, rst_outn_tidx, width=sig_w_hm)
                else:
                    rst_outn_tid = rst_tid

                rst_outp_tidx = sig_locs.get('rst_out', None)
                if rst_outp_tidx:
                    rst_outp_tid = TrackID(hm_layer, rst_outp_tidx, width=sig_w_hm)
                else:
                    rst_outp_tid = rst_tid

                rst_outp_gwarrs = rst_l.g if inp_on_right else rst_r.g
                rst_outn_gwarrs = rst_r.g if inp_on_right else rst_l.g

                rst_outp_warr = self.connect_to_tracks(rst_outp_gwarrs, rst_outp_tid)
                rst_outn_warr = self.connect_to_tracks(rst_outn_gwarrs, rst_outn_tid)
                rst_list = [('rst_outn', rst_outn_warr), ('rst_outp', rst_outp_warr),
                            ('rst_casc', rst_b_warr)]

                for name, wire in rst_list:
                    if name in vertical_rst:
                        if name == 'rst_casc':
                            vm_tid = self.grid.coord_to_track(vm_layer, wire.middle,
                                                              RoundMode.NEAREST)
                        elif (name == 'rst_outn') ^ inp_on_right:
                            vm_tid = self.grid.coord_to_track(vm_layer, wire.upper,
                                                              RoundMode.GREATER_EQ)
                        else:
                            vm_tid = self.grid.coord_to_track(vm_layer, wire.lower,
                                                              RoundMode.LESS_EQ)

                        wire = self.connect_to_tracks(wire, TrackID(vm_layer, vm_tid),
                                                      min_len_mode=MinLenMode.MIDDLE)
                    self.add_pin(name, wire)
            else:
                inl = in_l.g
                inr = in_r.g
        else:
            # use conn_layer to connect nmos and pmos drains
            pg_midl, pg_midr = self.connect_differential_tracks(nd_l, nd_r, hm_layer,
                                                                pg_midl_tidx, pg_midr_tidx)
            pg_midl, pg_midr = self.connect_differential_wires(load_l.d, load_r.d, pg_midl, pg_midr)
            pg_midl, pg_midr = self.connect_differential_wires(load_r.g, load_l.g, pg_midl, pg_midr)
            midl = pg_midl
            midr = pg_midr
            inl = in_l.g
            inr = in_r.g

        # connect and export input and output pins
        self.add_pin('poutr', pg_midr, hide=True)
        self.add_pin('poutl', pg_midl, hide=True)
        if inp_on_right:
            inp, inn = self.connect_differential_tracks(inr, inl, inp_tid.layer_id,
                                                        inp_tid.base_index, inn_tid.base_index,
                                                        width=inp_tid.width)
            self.add_pin('outn', midr)
            self.add_pin('outp', midl)
        else:
            inp, inn = self.connect_differential_tracks(inl, inr, inp_tid.layer_id,
                                                        inp_tid.base_index, inn_tid.base_index,
                                                        width=inp_tid.width)
            self.add_pin('outp', midr)
            self.add_pin('outn', midl)
        self.add_pin('inp', inp)
        self.add_pin('inn', inn)

        # compute schematic parameters
        default_thp = self.get_row_info(ridx_p).threshold
        default_thn = self.get_row_info(ridx_n).threshold
        self.sch_params = dict(
            lch=self.place_info.lch,
            seg_dict=seg_dict,
            w_dict=sch_w_dict,
            intent_dict=dict(
                nch=default_thn,
                pch=default_thp,
            ),
            in_upper=in_upper,
            has_rst=has_rst,
            stack_p=stack_p,
        )


class LevelShifterCoreOutBuffer(MOSBase):
    """Level shifter with output buffers.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._center_col: int = -1
        self._outr_inverted: bool = False
        self._mid_vertical: bool = False

    @property
    def center_col(self) -> int:
        """int: The centerline column index."""
        return self._center_col

    @property
    def outr_inverted(self) -> bool:
        return self._outr_inverted

    @property
    def mid_vertical(self) -> bool:
        """bool: True if level shifter core outputs are on vm_layer."""
        return self._mid_vertical

    @property
    def dual_output(self) -> bool:
        return self.params['dual_output']

    @property
    def is_guarded(self) -> bool:
        return self.params['is_guarded']

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'lvshift_core_w_drivers')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='dictionary of number of segments.',
            buf_seg_list='list of number of segments for output buffers.',
            buf_segn_list='list of number of segments for output buffers.',
            buf_segp_list='list of number of segments for output buffers.',
            w_dict='dictionary of number of fins.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            vertical_rst='whether rst pins is vertical',
            is_guarded='True if it there should be guard ring around the cell',
            dual_output='True to have complementary outputs.',
            invert_out='True to export flip output parity.',
            vertical_out='True to have inverter chain output on vertical metal',
            in_upper='True to make the input connected to the upper transistor in the stack',
            has_rst='True to enable reset pins.',
            stack_p='PMOS number of stacks.',
            sig_locs='Optional dictionary of user defined signal locations',
            num_col_tot='Total number of columns.',
            export_pins='Defaults to False.  True to export simulation pins.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_p=-1,
            ridx_n=0,
            vertical_rst=[],
            buf_seg_list=[],
            buf_segn_list=[],
            buf_segp_list=[],
            is_guarded=False,
            dual_output=True,
            invert_out=False,
            vertical_out=True,
            in_upper=False,
            has_rst=True,
            stack_p=1,
            sig_locs={},
            num_col_tot=0,
            export_pins=False,
        )

    def draw_layout(self):
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Mapping[str, int] = params['seg_dict']
        buf_seg_list: List[int] = params['buf_seg_list']
        buf_segn_list: List[int] = params['buf_segn_list']
        buf_segp_list: List[int] = params['buf_segp_list']
        w_dict: Mapping[str, int] = params['w_dict']
        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']
        vertical_rst: List[str] = params['vertical_rst']
        is_guarded: bool = params['is_guarded']
        dual_output: bool = params['dual_output']
        invert_out: bool = params['invert_out']
        vertical_out: bool = params['vertical_out']
        in_upper: bool = params['in_upper']
        has_rst: bool = params['has_rst']
        stack_p: int = params['stack_p']
        sig_locs: Mapping[str, Union[float, HalfInt]] = params['sig_locs']
        num_col_tot: int = params['num_col_tot']
        export_pins: bool = params['export_pins']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # placement and track computations
        # create level shifter core
        if not buf_segn_list or not buf_segp_list:
            if not buf_seg_list:
                raise RuntimeError('Not segment list provided')
            buf_segn_list = buf_seg_list
            buf_segp_list = buf_seg_list

        buf_nstage = len(buf_segn_list)
        buf_invert = (buf_nstage % 2 == 1)
        invert_in = (buf_invert ^ invert_out)
        if buf_invert ^ invert_in:
            self._outr_inverted = True
            outl_name = 'out'
            outr_name = 'outb'
        else:
            self._outr_inverted = False
            outl_name = 'outb'
            outr_name = 'out'

        default_wp = self.get_row_info(ridx_p).width
        default_wn = self.get_row_info(ridx_n).width
        core_params = dict(
            pinfo=pinfo,
            seg_dict=seg_dict,
            w_dict=w_dict,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            vertical_rst=vertical_rst,
            is_guarded=is_guarded,
            in_upper=in_upper,
            has_rst=has_rst,
            stack_p=stack_p,
            inp_on_right=invert_in,
            sig_locs=sig_locs,
        )
        core_master: LevelShifterCore = self.new_template(LevelShifterCore, params=core_params)
        self._mid_vertical = core_master.out_vertical

        # get buffer track indices
        buf_inl_tidx = core_master.get_port('poutl').get_pins()[0].track_id.base_index
        buf_inr_tidx = core_master.get_port('poutr').get_pins()[0].track_id.base_index
        buf_pout0_tidx = self.get_track_index(ridx_p, MOSWireType.DS, 'sig', wire_idx=1)
        buf_pout1_tidx = self.get_track_index(ridx_p, MOSWireType.DS, 'sig', wire_idx=0)
        buf_nout0_tidx = self.get_track_index(ridx_n, MOSWireType.DS, 'sig', wire_idx=-2)
        buf_nout1_tidx = self.get_track_index(ridx_n, MOSWireType.DS, 'sig', wire_idx=-1)

        # create buffer master
        sig_locs_l = dict(
            pout0=buf_pout0_tidx,
            pout1=buf_pout1_tidx,
            nout0=buf_nout0_tidx,
            nout1=buf_nout1_tidx,
        )
        sig_locs_r = sig_locs_l.copy()
        if is_guarded:
            # TODO: this code work with InvChainCore's gate index hack
            sig_locs_l['pin1'] = sig_locs_r['pin0'] = buf_inl_tidx
            sig_locs_l['pin0'] = sig_locs_r['pin1'] = buf_inr_tidx
        else:
            sig_locs_l['nin0'] = sig_locs_r['nin1'] = buf_inl_tidx
            sig_locs_l['nin1'] = sig_locs_r['nin0'] = buf_inr_tidx

        w_invp = w_dict.get('invp', default_wp)
        w_invn = w_dict.get('invn', default_wn)
        invr_params = dict(
            pinfo=pinfo,
            segn_list=buf_segn_list,
            segp_list=buf_segp_list,
            is_guarded=is_guarded,
            ridx_n=ridx_n,
            ridx_p=ridx_p,
            w_n=w_invn,
            w_p=w_invp,
            sig_locs=sig_locs_r,
            vertical_out=vertical_out,
        )
        invr_master = self.new_template(InvChainCore, params=invr_params)
        sch_buf_params = invr_master.sch_params.copy(remove=['dual_output'])

        # place instances
        inv_sep = self.min_sep_col
        inv_sep += (inv_sep & 1)
        inv_col = invr_master.num_cols
        inv_gap = (inv_col & 1)
        inv_col_even = inv_col + inv_gap
        core_col = core_master.num_cols
        vdd_list = []
        vss_list = []
        if dual_output:
            invl_params = invr_params.copy()
            invl_params['sig_locs'] = sig_locs_l
            invl_master = self.new_template(InvChainCore, params=invl_params)
            cur_tot = core_col + 2 * inv_col_even
            num_col_tot = max(num_col_tot, cur_tot + 2 * inv_sep)
            sep_l = (num_col_tot - cur_tot) // 2
            sep_l += (sep_l & 1)
            sep_r = num_col_tot - cur_tot - sep_l
            inv_l = self.add_tile(invl_master, 0, inv_col_even, flip_lr=True, commit=False)
            self._update_buf_inst(inv_l, vm_layer, sig_locs_l, sig_locs, 'l')
            vdd_list.append(inv_l.get_pin('VDD'))
            vss_list.append(inv_l.get_pin('VSS'))

            cur_col = inv_col_even + sep_l
            core = self.add_tile(core_master, 0, cur_col)
            self._center_col = core_master.center_col + cur_col
            cur_col += core_col + sep_r

            self.connect_wires([core.get_pin('poutl'), inv_l.get_pin('pin')])
            self._export_buf(inv_l, vertical_out, outl_name, buf_invert)
        else:
            cur_tot = core_col + inv_col + inv_gap
            num_col_tot = max(num_col_tot, cur_tot + inv_sep)
            sep = num_col_tot - cur_tot
            sep += (sep & 1)
            core = self.add_tile(core_master, 0, 0)
            self._center_col = core_master.center_col
            cur_col = core_col + sep

        vdd_list.append(core.get_pin('VDD'))
        vss_list.append(core.get_pin('VSS'))
        inv_r = self.add_tile(invr_master, 0, cur_col, commit=False)
        self._update_buf_inst(inv_r, vm_layer, sig_locs_r, sig_locs, 'r')
        vdd_list.append(inv_r.get_pin('VDD'))
        vss_list.append(inv_r.get_pin('VSS'))
        self._export_buf(inv_r, vertical_out, outr_name, buf_invert)
        self.set_mos_size(num_cols=cur_col + inv_col_even)

        # export supplies
        self.add_pin('VDD', self.connect_wires(vdd_list))
        self.add_pin('VSS', self.connect_wires(vss_list))

        # connect core output to buffer input
        self.connect_wires([core.get_pin('poutr'), inv_r.get_pin('pin')])

        # reexport core pins
        if core.has_port('rst_casc'):
            self.reexport(core.get_port('rst_casc'))
            self.reexport(core.get_port('rst_outp'), net_name='rst_out')
            self.reexport(core.get_port('rst_outn'), net_name='rst_outb')
        self.reexport(core.get_port('inp'), net_name='in')
        self.reexport(core.get_port('inn'), net_name='inb')
        midp = core.get_pin('outp')
        midn = core.get_pin('outn')
        self.add_pin('midp', midp, hide=not export_pins)
        self.add_pin('midn', midn, hide=not export_pins)
        if invert_in:
            self.add_pin('midr', midn, hide=True)
            self.add_pin('midl', midp, hide=True)
        else:
            self.add_pin('midr', midp, hide=True)
            self.add_pin('midl', midn, hide=True)

        # compute schematic parameters
        self.sch_params = dict(
            core_params=core_master.sch_params,
            buf_params=sch_buf_params,
            dual_output=dual_output,
            invert_out=invert_out,
        )

    def _export_buf(self, inst: Optional[PyLayInstance], vertical_out: bool, name: str,
                    buf_invert: bool) -> None:
        if inst is not None:
            pin_name = 'outb' if buf_invert else 'out'
            if vertical_out:
                self.reexport(inst.get_port(pin_name), net_name=name)
            else:
                self.reexport(inst.get_port(f'p{pin_name}'), net_name=name, connect=True)
                self.reexport(inst.get_port(f'n{pin_name}'), net_name=name, connect=True)

            self.reexport(inst.get_port(f'p{pin_name}'), net_name=f'p{name}')
            self.reexport(inst.get_port(f'n{pin_name}'), net_name=f'n{name}')

    def _update_buf_inst(self, inst: PyLayInstance, vm_layer: int, sig_locs_inst: Dict[str, Any],
                         sig_locs: Mapping[str, Any], suffix: str) -> None:
        xform = inst.transformation.get_inverse()

        test = sig_locs.get('out' + suffix, None)
        if test is not None:
            key = 'outb' if cast(InvChainCore, inst.master).out_invert else 'out'
            sig_locs_inst[key] = self.grid.transform_track(vm_layer, test, xform)
            inst.new_master_with(sig_locs=sig_locs_inst)

        inst.commit()


class LevelShifter(MOSBase):
    """Level shifter with single-ended input and output buffers.

    This generator uses two tiles, with the bottom tile mirrored, to separate the two
    supply domains.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._ridx_p = -1
        self._ridx_n = 0

    @property
    def ridx_p(self) -> int:
        return self._ridx_p

    @property
    def ridx_n(self) -> int:
        return self._ridx_n

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'lvshift')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            lv_params='level shifter with output buffer parameters.',
            in_buf_params='input buffer parameters.',
            export_pins='Defaults to False.  True to export simulation pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            export_pins=False,
        )

    def draw_layout(self):
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, params['pinfo'])
        self.draw_base(pinfo, flip_tile=True)

        lv_params: Param = params['lv_params']
        in_buf_params: Param = params['in_buf_params']
        export_pins = params['export_pins']

        # create masters
        lv_params = lv_params.copy(dict(pinfo=pinfo, export_pins=export_pins))
        lv_master: LevelShifterCoreOutBuffer = self.new_template(LevelShifterCoreOutBuffer,
                                                                 params=lv_params)
        self._ridx_p = lv_master.params['ridx_p']
        self._ridx_n = lv_master.params['ridx_n']

        in_buf_params = in_buf_params.copy(dict(pinfo=pinfo, dual_output=True,
                                                vertical_output=True,
                                                is_guarded=lv_master.is_guarded),
                                           remove=['sig_locs'])
        buf_master: InvChainCore = self.new_template(InvChainCore, params=in_buf_params)
        if buf_master.num_stages != 2:
            raise ValueError('Must have exactly two stages in input buffer.')

        # make sure buffer outb output is next to out, to avoid collision
        vm_layer = self.conn_layer + 2
        out_tidx = buf_master.get_port('out').get_pins()[0].track_id.base_index
        prev_tidx = self.tr_manager.get_next_track(vm_layer, out_tidx, 'sig', 'sig', up=False)
        buf_master = cast(InvChainCore, buf_master.new_template_with(sig_locs=dict(outb=prev_tidx)))

        # placement
        lv = self.add_tile(lv_master, 1, 0)
        buf_ncol = buf_master.num_cols
        if lv_master.mid_vertical:
            tid = lv.get_pin('midr').track_id
            tidx = self.tr_manager.get_next_track(vm_layer, tid.base_index, 'sig', 'sig', up=True)
            col_idx = self.arr_info.track_to_col(vm_layer, tidx, mode=RoundMode.GREATER_EQ)
            buf_idx = col_idx + buf_ncol
        else:
            lv_center = lv_master.center_col
            buf_idx = lv_center + buf_ncol

        # make sure supply on even number
        buf_idx += (buf_idx & 1)
        buf = self.add_tile(buf_master, 0, buf_idx, flip_lr=True)
        self.set_mos_size(num_cols=max(buf_idx, lv_master.num_cols))

        # connect wires
        self.add_pin('VSS', self.connect_wires([lv.get_pin('VSS'), buf.get_pin('VSS')]))
        self.connect_differential_wires(buf.get_pin('out'), buf.get_pin('outb'),
                                        lv.get_pin('in'), lv.get_pin('inb'))

        # reexport pins
        self.reexport(buf.get_port('VDD'), net_name='VDD_in')
        self.reexport(lv.get_port('VDD'))
        self.reexport(buf.get_port('in'))

        if export_pins:
            self.add_pin('inb_buf', buf.get_pin('outb'))
            self.add_pin('in_buf', buf.get_pin('out'))
            self.reexport(lv.get_port('midn'))
            self.reexport(lv.get_port('midp'))

        for name in ['out', 'outb', 'rst_out', 'rst_outb', 'rst_casc']:
            if lv.has_port(name):
                self.reexport(lv.get_port(name))

        buf_sch_params = buf_master.sch_params.copy(remove=['dual_output'])
        lv_sch_params = lv_master.sch_params.copy(remove=['dual_output', 'invert_out'])
        self.sch_params = dict(
            lev_params=lv_sch_params,
            buf_params=buf_sch_params,
            dual_output=lv_master.dual_output,
            invert_out=lv_master.outr_inverted,
            export_pins=export_pins,
        )

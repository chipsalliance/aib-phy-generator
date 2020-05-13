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

from typing import Any, Dict, Sequence, Mapping, Union, Tuple, Optional, Type

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.module import Module
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ...schematic.se_to_diff import bag3_digital__se_to_diff
from .gates import InvCore, PassGateCore, get_adj_tidx_list


class SingleToDiff(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

        self._buf_col_list: Sequence[int] = []

    @property
    def buf_col_list(self) -> Sequence[int]:
        return self._buf_col_list

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__se_to_diff

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            invp_params_list='Positive output chain parameters.',
            invn_params_list='Negative output chain parameters.',
            pg_params='passgate parameters.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            is_guarded='True to not route anything on conn_layer to allow space for guard rings',
            swap_tiles='True to swap outp/outn tiles.',
            vertical_out='False to not draw the vertical output wire',
            vertical_in='False to not draw the vertical input wire',
            export_pins='True to export simulation pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_p=-1,
            ridx_n=0,
            sig_locs={},
            is_guarded=False,
            swap_tiles=False,
            vertical_out=True,
            vertical_in=True,
            export_pins=False,
        )

    def draw_layout(self):
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, params['pinfo'])
        self.draw_base(pinfo)

        invp_params_list: Sequence[Param] = params['invp_params_list']
        invn_params_list: Sequence[Param] = params['invn_params_list']
        pg_params: Param = params['pg_params']
        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = params['sig_locs']
        is_guarded: bool = params['is_guarded']
        swap_tiles: bool = params['swap_tiles']
        vertical_out: bool = params['vertical_out']
        vertical_in: bool = params['vertical_in']
        export_pins: bool = params['export_pins']

        if len(invp_params_list) != 2 or len(invn_params_list) != 3:
            raise ValueError('Wrong number of parameters for inverters.')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # create masters
        tmp = self._create_masters(pinfo, ridx_p, ridx_n, is_guarded, invp_params_list,
                                   invn_params_list, pg_params, sig_locs, vertical_out,
                                   vertical_in)
        invp_masters, invn_masters, pg_master = tmp

        # place instances
        # first two columns left-aligned, last column right-aligned
        ncol0 = max(invp_masters[0].num_cols, invn_masters[0].num_cols)
        ncol0 += (ncol0 & 1)
        ncol1 = max(pg_master.num_cols, invn_masters[1].num_cols)
        ncol1 += (ncol1 & 1)
        invp_ncol2 = invp_masters[1].num_cols
        invn_ncol2 = invn_masters[2].num_cols
        invp_ncol2 += (invp_ncol2 & 1)
        invn_ncol2 += (invn_ncol2 & 1)
        ncol2 = max(invp_ncol2, invn_ncol2)
        sep = self.min_sep_col
        col1 = ncol0 + sep
        col_tot = col1 + ncol1 + sep + ncol2
        tile_outp = int(swap_tiles)
        tile_outn = 1 - tile_outp

        invp0 = self.add_tile(invp_masters[0], tile_outp, 0)
        invn0 = self.add_tile(invn_masters[0], tile_outn, 0)
        pg = self.add_tile(pg_master, tile_outp, col1)
        invn1 = self.add_tile(invn_masters[1], tile_outn, col1)
        invp1 = self.add_tile(invp_masters[1], tile_outp, col_tot - invp_ncol2)
        invn2 = self.add_tile(invn_masters[2], tile_outn, col_tot - invn_ncol2)
        self._buf_col_list = [0, col1, col_tot - invn_ncol2]
        self.set_mos_size(num_cols=col_tot)

        if export_pins:
            self.add_pin('midn_pass0', invp0.get_pin('out'))
            self.add_pin('midn_pass1', pg.get_pin('s'))
            self.add_pin('midn_inv', invn0.get_pin('out'))
            self.add_pin('midp', invn1.get_pin('out'))

        # vdd/vss
        vdd_list = []
        vss_list = []
        for inst in [invp0, pg, invp1, invn0, invn1, invn2]:
            vdd_list.extend(inst.port_pins_iter('VDD'))
            vss_list.extend(inst.port_pins_iter('VSS'))
        vdd = self.connect_wires(vdd_list)[0]
        vss = self.connect_wires(vss_list)[0]
        self.add_pin('VDD', vdd, connect=True)
        self.add_pin('VSS', vss, connect=True)

        # connect inverters and pass gates
        tr_manager = self.tr_manager
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        invp0_out = invp0.get_pin('out')
        self.connect_wires([invp0.get_pin('pout'), pg.get_pin('pd')])
        self.connect_wires([invp0.get_pin('nout'), pg.get_pin('nd')])
        self.connect_wires([invp1.get_pin('nin'), pg.get_pin('ns')])
        if is_guarded:
            self.connect_wires([invp1.get_pin('pin'), pg.get_pin('ps')])

            # inputs
            if vertical_in:
                self.add_pin('in', self.connect_wires([invp0.get_pin('in'), invn0.get_pin('in')]))
            else:
                self.reexport(invp0.get_port('in'))
                self.reexport(invn0.get_port('in'))
            # connect first stage
            en_center = invn1.get_pin('in')
            self.connect_to_track_wires([invn0.get_pin('nout'), invn0.get_pin('pout')], en_center)
            self.connect_to_tracks([invp0.get_pin('nout'), invp0.get_pin('pout'),
                                    pg.get_pin('pd'), pg.get_pin('nd')], en_center.track_id)
            # connect second stage
            self.connect_to_track_wires([invn1.get_pin('nout'), invn1.get_pin('pout')],
                                        invn2.get_pin('in'))
            self.connect_to_track_wires([pg.get_pin('ps'), pg.get_pin('ns')],
                                        invp1.get_pin('in'))
        else:
            # inputs
            invn0_out = invn0.get_pin('out')
            if vertical_in:
                vm_ref = min(invp0_out.track_id.base_index, invn0_out.track_id.base_index)
                in_tidx = tr_manager.get_next_track(vm_layer, vm_ref, 'sig', 'sig', up=False)
                self.add_pin('in',
                             self.connect_to_tracks([invp0.get_pin('in'), invn0.get_pin('in')],
                                                    TrackID(vm_layer, in_tidx, width=vm_w)))
            else:
                self.reexport(invp0.get_port('in'))
                self.reexport(invn0.get_port('in'))

            # connect first stage
            self.connect_to_track_wires(invn1.get_pin('in'), invn0_out)
            self.connect_wires([invp0.get_pin('nout'), invp0.get_pin('pout'),
                                pg.get_pin('pd'), pg.get_pin('nd')])
            # connect second stage
            en_center = invn1.get_pin('out')
            self.connect_to_track_wires(invn2.get_pin('in'), en_center)
            self.connect_to_tracks([invp1.get_pin('in'), pg.get_pin('ps'), pg.get_pin('ns')],
                                   en_center.track_id)

        # enables for passgate
        vdd_tidx = tr_manager.get_next_track(vm_layer, en_center.track_id.base_index,
                                             'sig', 'sig', up=False)
        vss_tidx = tr_manager.get_next_track(vm_layer, en_center.track_id.base_index,
                                             'sig', 'sig', up=True)
        self.connect_to_tracks([pg.get_pin('en'), vdd], TrackID(vm_layer, vdd_tidx, width=vm_w))
        self.connect_to_tracks([pg.get_pin('enb'), vss[0]], TrackID(vm_layer, vss_tidx, width=vm_w))

        # export outputs
        if vertical_out:
            self.reexport(invp1.get_port('out'), net_name='outp')
            self.reexport(invn2.get_port('out'), net_name='outn')
        else:
            self.reexport(invn2.get_port('pout'), net_name='outn', label='outn:', hide=False)
            self.reexport(invn2.get_port('nout'), net_name='outn', label='outn:', hide=False)
            self.reexport(invp1.get_port('pout'), net_name='outp', label='outp:', hide=False)
            self.reexport(invp1.get_port('nout'), net_name='outp', label='outp:', hide=False)
        # set schematic parameters
        self.sch_params = dict(
            invp_params_list=[invp_masters[0].sch_params, invp_masters[1].sch_params],
            invn_params_list=[invn_masters[0].sch_params, invn_masters[1].sch_params,
                              invn_masters[2].sch_params],
            pg_params=pg_master.sch_params,
            export_pins=export_pins,
        )

    def _create_masters(self, pinfo: MOSBasePlaceInfo, ridx_p: int, ridx_n: int,
                        is_guarded: bool, invp_params_list: Sequence[Param],
                        invn_params_list: Sequence[Param], pg_params: Param,
                        sig_locs: Mapping[str, Union[float, HalfInt]],
                        vertical_out: bool, vertical_in: bool
                        ) -> Tuple[Sequence[InvCore], Sequence[InvCore], PassGateCore]:
        # configure sig_locs dictionary so we can connect more signals on hm_layer
        nout_tid = get_adj_tidx_list(self, ridx_n, sig_locs, MOSWireType.DS, 'nout', False)
        pout_tid = get_adj_tidx_list(self, ridx_p, sig_locs, MOSWireType.DS, 'pout', True)
        nin_tid = get_adj_tidx_list(self, ridx_n, sig_locs, MOSWireType.G, 'nin', True)
        pin_tid = get_adj_tidx_list(self, ridx_p, sig_locs, MOSWireType.G, 'pin', False)
        # TODO: this gate index hack fixes cases where you cannot short adjacent hm_layer
        # TODO: tracks with vm_layer.  Need more rigorous checking later so we can reduce
        # TODO: gate resistance
        even_gate_index = int(is_guarded)
        # create masters
        append_dict = dict(
            pinfo=pinfo,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            is_guarded=is_guarded,
        )
        sig_locs0 = dict(
            nout=nout_tid[0],
            pout=pout_tid[0],
            nin=nin_tid[even_gate_index],
            pin=pin_tid[even_gate_index],
        )
        sig_locs1 = dict(
            nout=nout_tid[1],
            pout=pout_tid[1],
            nin=nin_tid[1],
            pin=pin_tid[1],
        )
        invp0 = self.new_template(InvCore, params=invp_params_list[0].copy(append=dict(
            sig_locs=sig_locs0,
            vertical_out=not is_guarded,
            vertical_in=vertical_in,
            **append_dict
        )))
        pg = self.new_template(PassGateCore, params=pg_params.copy(append=dict(
            sig_locs=dict(
                nd=nout_tid[0],
                pd=pout_tid[0],
                ns=nout_tid[1],
                ps=pout_tid[1],
                en=nin_tid[1],
                enb=pin_tid[1],
            ),
            pinfo=pinfo,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            is_guarded=is_guarded,
            vertical_out=False,
            vertical_in=False,
        )))
        invp1 = self.new_template(InvCore, params=invp_params_list[1].copy(append=dict(
            sig_locs=sig_locs0,
            vertical_out=vertical_out,
            **append_dict
        )))

        invn0 = self.new_template(InvCore, params=invn_params_list[0].copy(append=dict(
            sig_locs=sig_locs0,
            vertical_out=not is_guarded,
            vertical_in=vertical_in,
            **append_dict
        )))
        invn1 = self.new_template(InvCore, params=invn_params_list[1].copy(append=dict(
            sig_locs=sig_locs1,
            vertical_out=not is_guarded,
            **append_dict
        )))
        invn2 = self.new_template(InvCore, params=invn_params_list[2].copy(append=dict(
            sig_locs=sig_locs0,
            vertical_out=vertical_out,
            **append_dict
        )))

        if is_guarded:
            # make sure vm input pin are on the same track
            n0_in = invn0.get_port('in').get_pins()[0]
            p0_in = invp0.get_port('in').get_pins()[0]
            n0_tidx = n0_in.track_id.base_index
            p0_tidx = p0_in.track_id.base_index
            if n0_tidx < p0_tidx:
                sig_locs0['in'] = n0_tidx
                invp0 = invp0.new_template_with(sig_locs=sig_locs0)
            elif p0_tidx < n0_tidx:
                sig_locs0['in'] = p0_tidx
                invn0 = invn0.new_template_with(sig_locs=sig_locs0)

        return [invp0, invp1], [invn0, invn1, invn2], pg

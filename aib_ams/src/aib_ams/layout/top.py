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

from typing import Any, Dict, Optional, Type, List, cast

from pybag.enum import Direction, Orient2D, RoundMode
from pybag.core import Transform, BBox, COORD_MAX, COORD_MIN

from bag.util.immutable import Param
from bag.util.importlib import import_class
from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.layout.routing.base import TrackID, WireArray
from bag.layout.template import TemplateDB, TemplateBase
from bag.layout.util import IPMarginTemplate

from .frontend import Frontend


class FrontendESD(TemplateBase):
    """The transmitter and receiver integrated together.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_frontend')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            ndio_cls='N diode generator class.',
            pdio_cls='P diode generator class.',
            ndio_params='N diode parameters.',
            pdio_params='P diode parameters.',
            fe_params='frontend parameters.',
            npadout='Number of iopad_out wires.',
            w_dio_min='minimum diode area width.',
            dio_margin='diode wire margin.',
            tb_margin='top/bottom margin, in resolution units.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(w_min=0, dio_margin=-1, tb_margin=0)

    def draw_layout(self) -> None:
        ndio_cls_name: str = self.params['ndio_cls']
        pdio_cls_name: str = self.params['pdio_cls']
        ndio_params: Param = self.params['ndio_params']
        pdio_params: Param = self.params['pdio_params']
        fe_params: Param = self.params['fe_params']
        npadout: int = self.params['npadout']
        w_dio_min: int = self.params['w_dio_min']
        dio_margin: int = self.params['dio_margin']
        tb_margin: int = self.params['tb_margin']

        fe_ip_params = dict(
            cls_name='xbase.layout.mos.top.GenericWrapper',
            params=dict(
                cls_name=Frontend.get_qualified_name(),
                params=fe_params,
                export_hidden=True,
                half_blk_x=False,
            ),
        )

        # make masters
        fe_master: TemplateBase = self.new_template(IPMarginTemplate, params=fe_ip_params)
        ndio_cls = cast(Type[TemplateBase], import_class(ndio_cls_name))
        pdio_cls = cast(Type[TemplateBase], import_class(pdio_cls_name))
        nd_master: TemplateBase = self.new_template(ndio_cls, params=ndio_params)
        pd_master: TemplateBase = self.new_template(pdio_cls, params=pdio_params)

        # floorplan
        grid = self.grid
        top_layer = nd_master.top_layer
        w_blk, h_blk = grid.get_block_size(top_layer)
        tb_margin = -(-tb_margin // h_blk) * h_blk
        fe_box = fe_master.bound_box
        nd_box = nd_master.bound_box
        pd_box = pd_master.bound_box
        h_fe = fe_box.h
        h_nd = nd_box.h
        h_pd = pd_box.h
        w_fe = fe_box.w
        w_nd = nd_box.w
        w_pd = pd_box.w
        w_dio = max(w_dio_min + dio_margin, w_nd + w_pd)
        w_tot = w_fe + w_dio
        w_tot = -(-w_tot // w_blk) * w_blk
        h_tot = max(h_fe, h_nd, h_pd) + 2 * tb_margin
        y_fe = (h_tot - h_fe) // (2 * h_blk) * h_blk
        y_nd = (h_tot - h_nd) // (2 * h_blk) * h_blk
        y_pd = (h_tot - h_pd) // (2 * h_blk) * h_blk
        x_nd = w_fe
        x_pd = w_tot - w_pd

        # instantiate blocks
        fe = self.add_instance(fe_master, inst_name='XFE', xform=Transform(0, y_fe))
        nd = self.add_instance(nd_master, inst_name='XND', xform=Transform(x_nd, y_nd))
        pd = self.add_instance(pd_master, inst_name='XPD', xform=Transform(x_pd, y_pd))

        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot))

        # diode guard ring connections
        vlay = (fe.get_port('VSS').get_single_layer(), 'drawing')
        vdir = Direction.LOWER
        vss_list = []
        vdd_list = []
        vddio_list = []
        vss_bbox: List[BBox] = fe.get_all_port_pins('VSS', layer=vlay[0])
        vdd_bbox: List[BBox] = fe.get_all_port_pins('VDDCore', layer=vlay[0])
        vddio_bbox: List[BBox] = fe.get_all_port_pins('VDDIO', layer=vlay[0])

        vss_pd = pd.get_all_port_pins('gr')
        for bbox in vss_bbox:
            vss_list.extend(self.connect_bbox_to_track_wires(vdir, vlay, bbox, vss_pd))
        vss_list = self.connect_wires(vss_list)

        vddio_nd = nd.get_all_port_pins('gr')
        for bbox in vddio_bbox:
            vddio_list.extend(self.connect_bbox_to_track_wires(vdir, vlay, bbox, vddio_nd))
        vddio_list = self.connect_wires(vddio_list)

        # diode ports
        vddio_pd = pd.get_all_port_pins('sub')
        vss_nd = nd.get_all_port_pins('sub')
        dio_sp_le = grid.get_line_end_space(vss_nd[0].layer_id, vss_nd[0].track_id.width, even=True)
        dio_sp_le2 = dio_sp_le // 2
        if dio_margin < 0:
            dio_margin = dio_sp_le2
        dio_port_xh = w_tot - dio_margin
        x_dio_mid = (nd.bound_box.xh + pd.bound_box.xl) // 2
        vddio_pd = self.extend_wires(vddio_pd, lower=x_dio_mid + dio_sp_le2, upper=dio_port_xh)
        vss_nd = self.extend_wires(vss_nd, lower=x_nd, upper=x_dio_mid - dio_sp_le2)
        self.add_pin('VDDIO', vddio_pd)
        self.add_pin('VSS', vss_nd)
        pad = pd.get_all_port_pins('pad')
        pad.extend(nd.port_pins_iter('pad'))
        pad = self.connect_wires(pad, lower=x_nd, upper=dio_port_xh)[0]
        self.add_pin('iopad', pad)

        # short diode wires so we are LVS clean
        ym_layer = vddio_nd[0].layer_id - 1
        vss_tidx = grid.coord_to_track(ym_layer, x_nd + w_nd, mode=RoundMode.GREATER_EQ)
        vddio_tidx = grid.coord_to_track(ym_layer, x_pd, mode=RoundMode.LESS_EQ)
        iopad_tidx = grid.get_middle_track(vss_tidx, vddio_tidx, round_up=False)
        ym_w = grid.get_min_track_width(ym_layer, top_ntr=vddio_nd[0].track_id.width)
        vddio_pd.extend(vddio_list)
        vss_nd.extend(vss_list)
        self.connect_to_tracks(vddio_pd, TrackID(ym_layer, vddio_tidx, width=ym_w))
        self.connect_to_tracks(vss_nd, TrackID(ym_layer, vss_tidx, width=ym_w))
        self.connect_to_tracks(pad, TrackID(ym_layer, iopad_tidx, width=ym_w))

        # connect iopad and iclkn
        npad = pad.track_id.num
        pad_conn = pad[:npadout]
        iclkn_bnds = [COORD_MAX, COORD_MIN]
        iopad_out_list = []
        for bbox in fe.port_pins_iter('txpadout', layer=vlay[0]):
            warr = self.connect_bbox_to_track_wires(vdir, vlay, bbox, pad_conn)
            iopad_out_list.append(warr)

        for bbox in fe.port_pins_iter('rxpadin', layer=vlay[0]):
            cur_bnds = [0, 0]
            warr = self.connect_bbox_to_track_wires(vdir, vlay, bbox, pad_conn, ret_bnds=cur_bnds)
            iopad_out_list.append(warr)
            iclkn_bnds[0] = min(iclkn_bnds[0], cur_bnds[0])
            iclkn_bnds[1] = max(iclkn_bnds[1], cur_bnds[1])

        # export iopad_out
        pad_conn_tid = pad_conn.track_id
        pad_layer = pad_conn_tid.layer_id
        iopad_out_list = self.connect_wires(iopad_out_list)
        lower = iopad_out_list[0].lower
        pad_bnds = grid.get_wire_bounds(pad_layer, pad_conn_tid.base_index,
                                        width=pad_conn_tid.width)
        pin_len = grid.get_next_length(pad_layer, pad_conn_tid.width, pad_bnds[1] - pad_bnds[0],
                                       even=True)
        ms_params = self.add_res_metal_warr(pad_layer, pad_conn_tid.base_index,
                                            lower + pin_len, lower + 2 * pin_len,
                                            width=pad_conn_tid.width, num=pad_conn_tid.num,
                                            pitch=pad_conn_tid.pitch)

        self.add_pin('iopad_out', WireArray(pad_conn_tid, lower, lower + pin_len))

        bbox_iclkn = fe.get_pin('iclkn', layer=vlay[0])
        bbox_iclkn = bbox_iclkn.set_interval(Orient2D.y, iclkn_bnds[0], iclkn_bnds[1])
        # NOTE: some process require metal to be in same hierarchy as pin
        self.add_rect(vlay, bbox_iclkn)
        self.add_pin_primitive('iclkn', vlay[0], bbox_iclkn)

        # supply connections
        for idx in range(npad):
            tid = vss_nd[idx].track_id
            if idx == 0 or idx == npad - 1:
                cur_list = vdd_list
                cur_bbox = vdd_bbox
            elif idx & 1:
                cur_list = vss_list
                cur_bbox = vss_bbox
            else:
                cur_list = vddio_list
                cur_bbox = vddio_bbox

            for bbox in cur_bbox:
                cur_list.append(self.connect_bbox_to_tracks(vdir, vlay, bbox, tid))

        self.add_pin('VDDCore', self.connect_wires(vdd_list))
        self.add_pin('VDDIO', self.connect_wires(vddio_list))
        self.add_pin('VSS', self.connect_wires(vss_list))
        for name in ['VDDCore', 'VDDIO', 'VSS']:
            self.reexport(fe.get_port(f'{name}_xm'), net_name=name, hide=False)

        # reexport pins
        for name in ['din', 'ipdrv_buf<1>', 'ipdrv_buf<0>',
                     'indrv_buf<1>', 'indrv_buf<0>', 'itx_en_buf', 'weak_pulldownen',
                     'weak_pullupenb']:
            self.reexport(fe.get_port(name))

        for name in ['clk_en', 'data_en', 'por', 'odat', 'odat_async',
                     'oclkp', 'oclkn']:
            self.reexport(fe.get_port(name))

        self.sch_params = dict(
            fe_params=fe_master.sch_params,
            nd_params=nd_master.sch_params,
            pd_params=pd_master.sch_params,
            ms_params=ms_params,
        )
